"""训练曲线诊断 —— 只看最终数值会掩盖训练动态，必须逐条检查曲线本身。

诊断口径（不依赖任何主观读图）：
  * NaN/Inf      训练或验证损失出现非有限值 —— 训练已失效，先走 NaN 排查路径；
  * 发散          训练损失后期显著高于前期（学习率过大/梯度爆炸的典型表现）；
  * 过拟合        训练损失下降而验证损失上升 —— 需要正则化、增大数据量或早停；
                  训练损失停滞（平台期）而验证损失回升同样按过拟合处理
                  （泛化先于训练损失恶化，不能因"训练没降"误判成欠拟合）；
  * 过拟合趋势    训练损失仍在实质下降而验证损失无实质变化（±REL_TOL 内）——
                  验证不再随训练改善、泛化差距扩大中的早期信号；按过拟合
                  早期处理（正则化/早停），不要仅凭训练损失还在降就续训；
  * 欠拟合        训练损失也不下降（或停在 log(V) 随机猜测水平附近）——
                  学习率不当或数据管道本身有问题（标签错位/特征污染）；
  * 训练不足      训练/验证都在下降，但累计只见过不足 1 个 epoch 的训练
                  数据——这是“还没学完”，不能当作收敛或“无需干预”；
  * 早停空间      最终验证损失劣于历史最优的程度——量化"早停能挽回多少"；
  * 评估噪声      日志带 val_loss_ci95 时，小于该抽样误差的变化不算“实质
                  回升/下降”——避免把有限窗口的评估噪声误判成过拟合；
  * 瞬态损失尖峰  单步损失远超邻近点中位数（分段均值会掩盖）——异常批次或
                  一次梯度爆炸的信号，即使趋势正常也应先核对批次与梯度；
  * 坏记录        值/step 无法解析的日志行被跳过并计数，结论只基于其余点。

诊断输出为结构化 JSON（含建议清单），由 run_all 汇总进 summary.md；
也可独立运行：
  python -m lega.curves --run_dir out/tiny/lega [--out curve.json]
"""
import argparse
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple

# 相对变化阈值：低于该幅度视为"没有实质下降/上升"（覆盖常见噪声量级）
REL_TOL = 0.03
# 发散判定：后期均值超过前期均值的该倍数
DIVERGE_FACTOR = 1.5
# 早停提示阈值：最终 val 损失比历史最优差出该比例以上
EARLY_STOP_TOL = 0.02
# 欠拟合疑似数据管道问题：训练损失停在随机猜测水平的该比例以内
RANDOM_LEVEL_TOL = 0.9
# 训练覆盖是否足以谈“收敛”：随机窗口采样下折算的 epoch 数低于该值时，
# 曲线下降只说明“还在学”，不能据此判定收敛或建议“无需干预”。
MIN_TRAIN_EPOCHS = 1.0
# 梯度范数尖峰：最大 pre-clip 范数超过中位数该倍数视为爆炸性尖峰
# （NaN 之前的典型早期信号；即使损失曲线尚未发散也应预警）
GRAD_SPIKE_FACTOR = 10.0
# 瞬态损失尖峰：某点训练损失超过"两侧邻居中位数"该倍数。区间均值/首末值会
# 把单步尖峰平均掉，只看分段趋势可能完全漏掉（尖峰常伴随一次异常批次或
# 梯度爆炸，即使之后恢复也说明训练不稳）。局部比较而非全局中位数：损失
# 整体单调下降时全局中位数会把早期高点误判为尖峰。
LOSS_SPIKE_FACTOR = 8.0
LOSS_SPIKE_WINDOW = 5  # 每侧参与中位数的邻居点数
# 训练中途的过拟合/平台预警：连续该次数评测无实质改善即提示（仅提示，
# 不代替显式开启的早停；曲线诊断在训练结束后给出同样的结论）。
LIVE_WARN_EVALS = 3
# 回升幅度超过该比例时措辞升级为"过拟合信号"（否则视为平台）
LIVE_OVERFIT_GAP = 0.02
# 曲线可观测性下限：点太少时趋势/早停/尖峰检测都没有统计意义。验证点
# 少于 5 个无法区分"早期波动"与"真实回升"；训练日志少于 10 个点无法
# 可靠估计区间趋势。低于下限要在训练启动时告警，而不是等汇总时才发现
# 曲线不可用。
MIN_CURVE_EVAL_POINTS = 5
MIN_CURVE_LOG_POINTS = 10


def curve_observability_warnings(steps, eval_every, log_every) -> List[str]:
    """训练启动前检查曲线采样密度，返回可操作的配置告警（空表=足够）。

    steps/eval_every/log_every 直接决定曲线点数：eval_every 接近 steps 时
    验证曲线只有 1-2 个点，过拟合/早停/验证噪声诊断全部失效；log_every
    过大则训练侧趋势与瞬态尖峰无法检测。这里只告警不阻断——小规模冒烟
    训练有意识地用稀疏采样是合法的，但正式实验必须知道结论会因此受限。
    """
    try:
        steps = int(steps)
        eval_every = int(eval_every)
        log_every = int(log_every)
    except (TypeError, ValueError, OverflowError):
        return ["训练步数/评测/日志间隔不是整数，无法评估曲线可观测性"]
    if steps < 1:
        return [f"steps={steps}，没有训练曲线可诊断"]
    warnings = []
    if eval_every < 1:
        warnings.append(f"eval_every={eval_every} 非法（必须 >=1），无法评测")
    else:
        # 循环内每 eval_every 步评测一次；末尾若不在评测点还会补一次最终评测。
        n_eval = steps // eval_every + (1 if steps % eval_every else 0)
        if n_eval < MIN_CURVE_EVAL_POINTS:
            suggest = max(1, steps // MIN_CURVE_EVAL_POINTS)
            warnings.append(
                f"验证点预计仅 {n_eval} 个（steps={steps}, "
                f"eval_every={eval_every}）：过拟合/早停/验证噪声诊断至少需要 "
                f"{MIN_CURVE_EVAL_POINTS} 个点；建议 eval_every<={suggest} "
                "或增加 steps（冒烟训练可忽略，但结论不能外推）")
    if log_every < 1:
        warnings.append(f"log_every={log_every} 非法（必须 >=1），无法记录训练点")
    else:
        # step 0 先打一个点，之后每个 log_every 步一个点。
        n_log = 1 + steps // log_every
        if n_log < MIN_CURVE_LOG_POINTS:
            suggest = max(1, steps // (MIN_CURVE_LOG_POINTS - 1))
            warnings.append(
                f"训练日志预计仅 {n_log} 个点（steps={steps}, "
                f"log_every={log_every}）：训练侧趋势/瞬态尖峰检测至少需要 "
                f"{MIN_CURVE_LOG_POINTS} 个点；建议 log_every<={suggest} "
                "或增加 steps（冒烟训练可忽略，但结论不能外推）")
    return warnings


def load_train_log(path: str) -> List[Dict]:
    """读取 train_log.jsonl（容忍个别损坏行，缺文件返回空表）。"""
    if not os.path.exists(path):
        return []
    recs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                # 合法 JSON 但非对象的行（数组/标量）按损坏行跳过
                if isinstance(rec, dict):
                    recs.append(rec)
    except OSError:
        return []
    return recs


def _finite_points(records: List[Dict], key: str) -> Tuple[List[Dict], int, int]:
    """取出 (step, value) 均有限并按 step 排序的记录点；异常记录单独计数。

    返回 (pts, bad, malformed)：pts 为有限点列表，bad 为非有限值个数，
    malformed 为 step/值无法解析的记录数。
    记录按 step 排序（train/val 两类点在 jsonl 中交错写入，不能假设有序）。
    单条记录的值可能是字符串/对象（日志被中断或手工编辑）：整个诊断不能
    因此崩溃，坏记录跳过并计数，结论只基于其余点。
    """
    pts, bad, malformed = [], 0, 0
    if not isinstance(records, (list, tuple)):
        records = []
    for r in records:
        if not isinstance(r, dict):
            malformed += 1
            continue
        v, s = r.get(key), r.get("step")
        if v is None or s is None:
            continue
        try:
            v, s = float(v), int(s)
        except (TypeError, ValueError, OverflowError):
            # step 可能是 JSON 里的 Infinity/1e999（json.loads 接受非标准
            # 字面量）：int(inf) 抛 OverflowError，同样必须按坏记录跳过。
            malformed += 1
            continue
        if math.isfinite(v):
            pts.append({"step": s, "value": v})
        else:
            bad += 1
    pts.sort(key=lambda p: p["step"])
    return pts, bad, malformed


def _detect_loss_spikes(train_pts: List[Dict]) -> List[Dict]:
    """检出瞬态训练损失尖峰：值 ≥ LOSS_SPIKE_FACTOR × 两侧邻居中位数。

    只比较局部邻居（每侧 LOSS_SPIKE_WINDOW 个点）而非全局中位数/均值：
    损失整体单调下降时，全局统计会把早期高点误报为尖峰，而局部比较对
    正常下降免疫。点太少（<2W+1）时无法判定，返回空表。
    """
    n = len(train_pts)
    w = LOSS_SPIKE_WINDOW
    spikes = []
    for i in range(w, n - w):
        neigh = [p["value"] for p in train_pts[i - w:i]] \
            + [p["value"] for p in train_pts[i + 1:i + 1 + w]]
        med = statistics.median(neigh)
        if med <= 0:
            continue
        ratio = train_pts[i]["value"] / med
        if ratio >= LOSS_SPIKE_FACTOR:
            spikes.append({"step": train_pts[i]["step"],
                           "value": train_pts[i]["value"],
                           "local_median": med, "ratio": ratio})
    return spikes


def _seg_mean(pts: List[Dict], lo: float, hi: float) -> Optional[float]:
    """按 step 区间 [lo, hi)（比例）取均值；空区间返回 None。"""
    if not pts:
        return None
    steps = [p["step"] for p in pts]
    lo_s = steps[0] + lo * (steps[-1] - steps[0])
    hi_s = steps[0] + hi * (steps[-1] - steps[0])
    vals = [p["value"] for p in pts if lo_s <= p["step"] < hi_s
            or (hi == 1.0 and p["step"] == steps[-1])]
    return float(sum(vals) / len(vals)) if vals else None


def _val_ci_rel(records: List[Dict]) -> Optional[float]:
    """验证损失点级 95% CI 的相对半宽中位数（无可用区间时返回 None）。

    单次 run 只评测 eval_iters 个窗口；train.py/evaluate.py 已把窗口抽样
    误差写入 val_loss_ci95，但曲线诊断此前只比较点估计，2% 的“验证回升”
    可能完全落在该误差内。这里取所有带区间的验证点中相对半宽的中位数作为
    噪声量级（点级、偏保守：分段均值实际会进一步降低该项噪声）。
    """
    vals = []
    for r in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(r, dict):
            continue
        ci, v = r.get("val_loss_ci95"), r.get("val_loss")
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            continue
        try:
            lo, hi, v = float(ci[0]), float(ci[1]), float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(v)
                and v > 0 and hi >= lo):
            vals.append((hi - lo) / 2.0 / v)
    return float(statistics.median(vals)) if vals else None


def _rel_change(first: Optional[float], last: Optional[float]) -> Optional[float]:
    """last 相对 first 的相对变化（正=上升）；first 接近 0 时返回 None
    （相对变化无定义，不回退成绝对差以免口径混淆）。"""
    if first is None or last is None:
        return None
    if abs(first) < 1e-12:
        return None
    return (last - first) / abs(first)


def _fmt_num(x: Optional[float], spec: str = ".4f") -> str:
    """数值缺失（区间内无点时 _seg_mean 返回 None）显式显示 '-'。"""
    return "-" if x is None else format(x, spec)


def _fmt_rel(x: Optional[float]) -> str:
    """相对变化缺失时显式显示 '-'（不能把 None 交给百分号格式化）。"""
    return "-" if x is None else f"{x:+.1%}"


def _train_trend(train_recent) -> Optional[float]:
    """最近训练日志点的相对变化（正=上升）；不足两点/非有限返回 None。"""
    if not isinstance(train_recent, (list, tuple)):
        return None
    vals = []
    for x in train_recent:
        try:
            x = float(x)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(x):
            vals.append(x)
    if len(vals) < 2 or abs(vals[0]) < 1e-12:
        return None
    return (vals[-1] - vals[0]) / abs(vals[0])


def live_overfit_hint(evals_since_improve, best_val, val,
                      patience: int = 0, train_recent=None,
                      val_ci95=None) -> Optional[str]:
    """训练中途的曲线预警：连续多次评测无实质改善时返回提示文案。

    只看最终数值会漏掉"训练 loss 仍降、验证早已停步"的过程信号；早停关闭
    （patience<=0，默认）时没有任何机制会在训练中提醒。该函数与
    diagnose_curve 的"过拟合→早停"判定同口径（best 由严格最小维护），
    但只提示不改流程。返回 None 表示无预警或早停已开启（由早停自身处理）。

    train_recent: 最近若干训练日志点的 loss（可选）。验证停步时结合训练侧
        趋势区分"验证回升但训练仍降"（过拟合）、"两边都停滞/上升"（更像
        学习率/容量/数据管道问题）——否则会把欠拟合误报为过拟合，诱导
        用户加正则化而进一步恶化。
    val_ci95: 当前验证损失的 95% 区间（可选）。回升幅度落在抽样误差内时
        只提示"仍在噪声内、继续观察"，不升级为过拟合结论。
    """
    if patience and int(patience) > 0:
        return None
    try:
        n = int(evals_since_improve or 0)
    except (TypeError, ValueError):
        return None
    if n < LIVE_WARN_EVALS or best_val is None or val is None:
        return None
    try:
        best_f, val_f = float(best_val), float(val)
        gap = (val_f - best_f) / best_f
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(gap):
        return None
    # 验证抽样噪声：区间半宽相对当前 val。gap 小于该半宽时，"回升"与
    # "平台"都无法和有限窗口抽样误差区分。
    ci_rel = None
    if isinstance(val_ci95, (list, tuple)) and len(val_ci95) == 2:
        try:
            lo, hi = float(val_ci95[0]), float(val_ci95[1])
            if (math.isfinite(lo) and math.isfinite(hi) and hi >= lo
                    and val_f > 0):
                ci_rel = (hi - lo) / 2.0 / val_f
        except (TypeError, ValueError, OverflowError):
            ci_rel = None
    if ci_rel is not None and abs(gap) <= ci_rel + 1e-12:
        label = "回升" if gap > LIVE_OVERFIT_GAP else "平台"
        return (f"验证损失{label}（较历史最优 {gap:+.1%}）仍在 95% 抽样噪声内"
                f"（±{ci_rel:.1%}），已连续 {n} 次评测未改善：先继续观察，"
                "不要据此加正则化或触发早停；可增大 eval_iters/验证集。")

    trend = _train_trend(train_recent)
    trend_txt = ""
    overfit_tail = "检查训练/验证曲线是否分叉，必要时加正则化/数据或显式开启 --early_stop_patience。"
    underfit_tail = ("先检查学习率/数据管道/模型容量，不要盲目加正则化；"
                     "确认训练损失本身是否还在下降后再决定是否延长训练。")
    if trend is not None:
        if trend < -0.01:
            trend_txt = f"，训练损失仍在下降（{trend:+.1%}）"
            kind = "overfit"
        elif trend > 0.01:
            trend_txt = f"，且训练损失也在上升（{trend:+.1%}）"
            kind = "unstable"
        else:
            trend_txt = f"，训练损失已基本停滞（{trend:+.1%}）"
            kind = "flat"
    else:
        kind = None

    if gap > LIVE_OVERFIT_GAP:
        if kind == "overfit":
            what = f"验证损失在回升（较历史最优 +{gap:.1%}）{trend_txt}，典型过拟合信号"
            tail = overfit_tail
        elif kind == "unstable":
            what = (f"验证损失在回升（较历史最优 +{gap:.1%}）{trend_txt}："
                    "更像优化不稳定/学习率过大，不是典型过拟合")
            tail = underfit_tail
        elif kind == "flat":
            what = (f"验证损失在回升（较历史最优 +{gap:.1%}）{trend_txt}："
                    "训练侧已无改善，优先核对验证集构造与正则化，别只延长训练")
            tail = overfit_tail
        else:
            what = f"验证损失在回升（较历史最优 +{gap:.1%}），典型过拟合信号"
            tail = overfit_tail
    else:
        if kind == "overfit":
            what = (f"验证损失进入平台（较历史最优 +{gap:.1%}）{trend_txt}："
                    "泛化差距扩大，早期过拟合信号")
            tail = overfit_tail
        elif kind == "unstable":
            what = (f"验证损失进入平台（较历史最优 +{gap:.1%}）{trend_txt}："
                    "更像欠拟合/优化不足，不是过拟合")
            tail = underfit_tail
        elif kind == "flat":
            what = (f"训练与验证损失同时停滞（验证较历史最优 +{gap:.1%}）："
                    "更像容量/优化不足（欠拟合），不是过拟合")
            tail = underfit_tail
        else:
            what = f"验证损失进入平台（较历史最优 +{gap:.1%}，无实质改善）"
            tail = overfit_tail
    return f"{what}：已连续 {n} 次评测未改善。{tail}"


def diagnose_curve(records: List[Dict], log_vocab: Optional[float] = None,
                   epochs_seen: Optional[float] = None,
                   train_tokens: Optional[int] = None) -> Dict:
    """从单条 train_log 记录序列给出结构化诊断。

    records: load_train_log 的输出（含 'loss' 训练点与 'val_loss' 验证点）。
    epochs_seen/train_tokens: 来自 eval.json 的训练数据覆盖率；缺失时跳过
    “训练不足”判定（旧结果目录仍可诊断）。
    返回 dict：status / 判定依据 / 关键数值 / 建议清单 recommendations。
    早停终止的 run（final 记录带 early_stop=true）曲线按设计截断，显式标注。
    """
    records = records if isinstance(records, (list, tuple)) else []
    train_pts, train_bad, train_malformed = _finite_points(records, "loss")
    val_pts, val_bad, val_malformed = _finite_points(records, "val_loss")
    # 早停终止标记：训练按设计提前结束（区别于日志缺失/训练中断），
    # 建议不再指向"延长训练"（验证已连续无改善）。
    # step 只采信可解析的有限值；缺失/损坏（外部或截断的日志）时回退到
    # 最后一个有限训练点，既保留早停标注，又不把 "step None" 写进证据。
    early_stop_recs = [r for r in records
                       if isinstance(r, dict) and r.get("early_stop")]
    early_stopped = bool(early_stop_recs)
    early_stop_steps = []
    for r in early_stop_recs:
        try:
            s = int(r.get("step"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(float(s)):
            early_stop_steps.append(s)
    es_step = (early_stop_steps[0] if early_stop_steps
               else (train_pts[-1]["step"] if train_pts else None))
    # 累计门控/统计点上的 NaN 只在对应序列里报；任何序列出现即整体报
    nan_count = train_bad + val_bad
    malformed_count = train_malformed + val_malformed
    # 梯度范数体检：尖峰/非有限是发散与 NaN 的早期信号，损失曲线可能还没反映
    grad_pts, grad_bad, grad_malformed = _finite_points(records, "grad_norm")
    malformed_count += grad_malformed
    # 瞬态损失尖峰：分段均值会把它平均掉，必须按点单独体检（与损失状态独立）
    loss_spikes = _detect_loss_spikes(train_pts)
    grad_vals = [p["value"] for p in grad_pts]
    grad_med = None
    grad_max = None
    grad_spike_ratio = None
    if grad_vals:
        grad_med = float(statistics.median(grad_vals))
        grad_max = float(max(grad_vals))
        if grad_med > 0:
            grad_spike_ratio = grad_max / grad_med
    grad_spike = (grad_spike_ratio is not None
                  and grad_spike_ratio >= GRAD_SPIKE_FACTOR)

    first_train = _seg_mean(train_pts, 0.0, 1 / 3)
    last_train = _seg_mean(train_pts, 2 / 3, 1.0)
    first_val = _seg_mean(val_pts, 0.0, 1 / 3)
    last_val = _seg_mean(val_pts, 2 / 3, 1.0)

    train_delta = _rel_change(first_train, last_train)
    val_delta = _rel_change(first_val, last_val)
    train_down = train_delta is not None and train_delta < -REL_TOL
    train_up = train_delta is not None and train_delta > REL_TOL
    # 验证侧判定必须扣除评估抽样噪声：val_loss_ci95 存在时，变化幅度不超过
    # 该点级 CI 半宽就不算“实质回升/下降”。无 CI 的旧日志保持原口径。
    val_ci_rel = _val_ci_rel(records)
    val_tol = max(REL_TOL, val_ci_rel or 0.0)
    val_down = val_delta is not None and val_delta < -val_tol
    val_up = val_delta is not None and val_delta > val_tol
    val_unresolved = (val_delta is not None and abs(val_delta) <= val_tol)

    best_val, best_step = None, None
    for p in val_pts:
        if best_val is None or p["value"] < best_val:
            best_val, best_step = p["value"], p["step"]
    final_val = val_pts[-1]["value"] if val_pts else None
    final_step = val_pts[-1]["step"] if val_pts else None
    early_stop_gain = _rel_change(best_val, final_val)  # >0 表示最终劣于最优
    # 早停空间同样要扣除评估噪声：final 比 best 差 2% 但单点 CI 就有 ±3% 时，
    # 不能据此断言“已经过拟合、早停能挽回”。
    gain_tol = max(EARLY_STOP_TOL, val_ci_rel or 0.0)
    gain_sig = (early_stop_gain or 0) > gain_tol

    status, evidence, recommendations = "数据不足", [], []
    n_train, n_val = len(train_pts), len(val_pts)

    if nan_count:
        status = "NaN/Inf"
        evidence.append(f"训练日志出现 {nan_count} 个非有限损失值"
                        f"（训练 {train_bad}、验证 {val_bad}）")
        recommendations.append(
            "训练已失效：按数据 → 代码(log(0)/除零/softmax 溢出) → 混合精度下溢 "
            "→ 梯度爆炸/学习率的顺序排查（见该 run 目录 nan_report.json）")
    elif n_train < 3 or n_val < 2:
        status = "数据不足"
        evidence.append(f"曲线点过少（训练 {n_train}、验证 {n_val}），只能描述统计")
        recommendations.append("增加 eval_every 密度或训练步数后再诊断")
    else:
        if train_up and (last_train or 0) > (first_train or 0) * DIVERGE_FACTOR:
            status = "发散"
            evidence.append(
                f"训练损失后期均值 {last_train:.4f} 为前期 {first_train:.4f} 的 "
                f"{last_train / first_train:.2f} 倍")
            recommendations.append("疑似发散：优先降低学习率或加强梯度裁剪，"
                                   "并检查梯度范数是否出现尖峰")
        elif ((train_down and (val_up or gain_sig))
              or (train_delta is not None and not train_down and not train_up
                  and val_up and gain_sig)):
            # 平台期过拟合：训练不再下降而验证回升，同样需要早停/正则化；
            # 若按"训练没降"误判成欠拟合，会给出"延长训练/增大模型"的反向建议。
            status = "过拟合"
            if train_down:
                ev = [f"训练损失 {_fmt_num(first_train)}→{_fmt_num(last_train)}"
                      f"（{_fmt_rel(train_delta)}）而验证损失 "
                      f"{_fmt_num(first_val)}→{_fmt_num(last_val)}"
                      f"（{_fmt_rel(val_delta)}）"]
            else:
                ev = [f"训练损失 {_fmt_num(first_train)}→{_fmt_num(last_train)}"
                      f"（{_fmt_rel(train_delta)}）已停滞/不再实质下降，而验证损失 "
                      f"{_fmt_num(first_val)}→{_fmt_num(last_val)}"
                      f"（{_fmt_rel(val_delta)}）上升——泛化先于训练损失恶化，"
                      "按平台期过拟合处理"]
            if gain_sig:
                ev.append(f"最终验证损失劣于 step {best_step} 的最优值 "
                          f"{best_val:.4f} 达 {early_stop_gain:+.1%}"
                          "（超过评估噪声）")
            evidence += ev
            recommendations += [
                "加正则化（dropout / weight decay）或增大数据量",
                f"早停：val 最优在 step {best_step}，最终 step {final_step} 已过拟合"
                + (f"（早停可挽回 {early_stop_gain:.1%} 验证损失）"
                   if gain_sig else ""),
                "下游对比应使用 val 上选出的 model_best.pt，而不是最终步模型"]
            if not train_down:
                recommendations.append(
                    "训练已停滞而验证恶化：按平台期过拟合处理（早停/正则化），"
                    "并先确认学习率不过小（训练推得动）；不要按欠拟合扩大容量或续训")
        elif (train_down and val_unresolved and not gain_sig):
            # 训练仍在实质下降而验证无实质变化（|Δ|<REL_TOL）：验证不再随
            # 训练改善，泛化差距在扩大——过拟合的早期形态。它比"验证回升"
            # 更早出现；只看最终数值或只等验证回升会漏掉这一信号。
            status = "过拟合趋势"
            evidence.append(
                f"训练损失 {_fmt_num(first_train)}→{_fmt_num(last_train)}"
                f"（{_fmt_rel(train_delta)}）仍在实质下降，而验证损失 "
                f"{_fmt_num(first_val)}→{_fmt_num(last_val)}"
                f"（{_fmt_rel(val_delta)}）无实质变化"
                + (f"（评估噪声 ±{val_ci_rel:.1%}）" if val_ci_rel is not None
                   else "")
                + "：验证不再随训练改善，泛化差距在扩大"
                "（尚不满足“验证回升”的过拟合判定）")
            recommendations += [
                "训练降而验证停滞：按过拟合早期信号处理——启用/加强正则化"
                "（dropout / weight decay），对比使用 val 最优的 model_best.pt",
                "不要仅因训练损失还在下降就延长训练：新增收益只体现在训练集",
                "若验证损失自始未改善且训练降幅也很小：先按欠拟合排查学习率/"
                "数据管道，再考虑扩容",
            ]
        elif not train_down:
            status = "欠拟合/学习不充分"
            trend = "不降反升" if train_up else "没有实质下降"
            evidence.append(
                f"训练损失 {_fmt_num(first_train)}→{_fmt_num(last_train)}"
                f"（{_fmt_rel(train_delta)}），{trend}")
            if (log_vocab is not None and last_train is not None
                    and last_train > RANDOM_LEVEL_TOL * log_vocab):
                evidence.append(
                    f"训练损失停在随机猜测水平 log(V)={log_vocab:.3f} 的 "
                    f"{RANDOM_LEVEL_TOL:.0%} 以内——学习链路基本未起作用")
                recommendations.append(
                    "先排查数据管道（标签是否随机/错位、特征与标签是否对齐），"
                    "再调学习率（过大→震荡，过小→停滞）")
            else:
                recommendations += [
                    "检查学习率是否过小/过大（观察 loss 曲线是否停滞或震荡）",
                    "检查数据管道（标签错位、特征污染）与 batch 组成"]
                if train_up:
                    recommendations.append(
                        "训练损失不降反升：优先降低学习率并检查数据顺序/批组成；"
                        "容量不足通常表现为低损失平台，而非上升")
                elif val_up:
                    recommendations.append(
                        "容量不足应表现为训练/验证同步的低损失平台；这里验证在"
                        "上升而训练不降，先排除过拟合/训练不稳，不要盲目增大模型")
                else:
                    recommendations.append("若容量不足：增大模型或延长训练")
            if train_up:
                if val_up:
                    recommendations.append(
                        "训练与验证损失同步上升：更像学习率过大/训练不稳，"
                        "按发散前期处理而不是欠拟合")
            elif val_up:
                recommendations.append(
                    "训练不降而验证上升：按平台期过拟合处理（早停/正则化），"
                    "不要按欠拟合延长训练")
            elif val_down:
                # 验证仍在实质下降时绝不能套用"验证损失同样不降"的论断
                # （会把在改善的 run 说成停滞）；先自检训练链路再谈容量。
                recommendations.append(
                    "训练不降而验证损失仍在下降：先确认训练损失记录与优化"
                    "链路（日志/梯度）是否正常，再判断是否需要更大容量或"
                    "更长训练")
            else:
                recommendations.append(
                    "训练与验证损失均无实质下降（波动未超噪声阈值），"
                    "与欠拟合判断一致")
        else:
            status = "正常收敛"
            evidence.append(
                f"训练损失 {_fmt_num(first_train)}→{_fmt_num(last_train)}"
                f"（{_fmt_rel(train_delta)}），"
                f"验证损失 {_fmt_num(first_val)}→{_fmt_num(last_val)}"
                f"（{_fmt_rel(val_delta)}）")
            if gain_sig:
                evidence.append(f"注意：最终验证损失仍比 step {best_step} 的最优值差 "
                                f"{early_stop_gain:.1%}（超过评估噪声），"
                                "已按 best-checkpoint 口径评测")
            elif (early_stop_gain or 0) > EARLY_STOP_TOL:
                evidence.append(
                    f"最终验证损失比 step {best_step} 的最优值差 "
                    f"{early_stop_gain:.1%}，但未超过评估噪声 "
                    f"±{val_ci_rel:.1%}：不据此判定过拟合")
            elif not loss_spikes:
                # 有瞬态尖峰时不写"无需干预"：先按尖峰建议排查（下文追加）
                recommendations.append("无需干预；对比实验使用 model_best.pt")
        if gain_sig and status != "过拟合":
            recommendations.append(
                f"早停提示：最终验证损失劣于 step {best_step} 最优值 "
                f"{early_stop_gain:.1%}")

    if val_ci_rel is not None:
        evidence.append(
            f"验证损失点级 95% CI 相对半宽中位数 ±{val_ci_rel:.1%}"
            "（窗口抽样误差；分段均值会进一步降低该项，按保守口径使用）："
            "小于该量级的验证变化按噪声处理，不作为过拟合/收敛证据")

    # 训练数据覆盖率（来自 eval.json）：曲线下降 ≠ 已收敛。只见过不足
    # 1 个 epoch 时，验证损失仍在快速变化，此时比较方法/下“无需干预”的
    # 结论都不稳；必须把“还没学完”与真正的收敛区分开。
    budget_insufficient = (epochs_seen is not None
                           and epochs_seen < MIN_TRAIN_EPOCHS)
    if budget_insufficient:
        tok_s = (f"（{train_tokens:,} 训练 token）"
                 if train_tokens else "")
        evidence.append(
            f"训练覆盖仅约 {epochs_seen:.3f} 个 epoch{tok_s}："
            f"低于 {MIN_TRAIN_EPOCHS:g} 个 epoch，曲线下降主要反映“还没学完”，"
            "不能据此判定收敛或比较方法优劣")
        recommendations[:] = [r for r in recommendations
                              if "无需干预" not in r]
        recommendations.append(
            f"训练数据覆盖不足（约 {epochs_seen:.3f} epoch）：先继续训练/增大"
            "训练覆盖至少 1 个 epoch，并保持当前验证密度；在此之前"
            "不要把验证损失变化解释为收敛或方法差异")
        if status == "正常收敛":
            status = "训练不足（数据覆盖不足）"

    # 早停终止：截断是设计行为（不同于日志缺失），独立于状态标注；
    # "数据不足"分支的"增加步数/评测密度"建议与早停矛盾，须替换。
    if early_stopped:
        es_txt = f"（step {es_step}）" if es_step is not None else ""
        evidence.append(
            f"该 run 由早停终止{es_txt}：验证损失连续无实质改善，"
            "训练按设计提前结束，曲线被截断而非日志缺失/训练中断")
        if status == "数据不足":
            recommendations[:] = [r for r in recommendations
                                  if "训练步数" not in r and "eval_every" not in r]
        recommendations.append(
            "已早停：不要以“曲线未走完”为由延长训练；对比使用 val 选出的 "
            "model_best.pt，先按上述诊断处理（正则化/数据/学习率）再决定是否重跑")

    # 梯度范数体检（与损失状态独立）：非有限/尖峰是梯度爆炸的早期信号，
    # 损失曲线可能尚未发散，只看最终损失会漏掉。
    if grad_bad:
        evidence.append(f"pre-clip 梯度范数出现 {grad_bad} 个非有限值"
                        "（梯度已爆炸，先于或伴随损失 NaN）")
        recommendations.append(
            "梯度爆炸：降低学习率、收紧 grad_clip（如 0.5）、延长 warmup，"
            "并确认 train_log.jsonl 的 grad_norm 不再出现非有限值")
    elif grad_spike:
        evidence.append(
            f"pre-clip 梯度范数最大 {grad_max:.2e} 为中位数 {grad_med:.2e} 的 "
            f"{grad_spike_ratio:.1f} 倍（爆炸性尖峰）")
        recommendations.append(
            f"检测到梯度范数尖峰（×{grad_spike_ratio:.0f}）：这是 NaN 之前的"
            "典型信号，应降低学习率/收紧 grad_clip 后再验证曲线")

    # 瞬态损失尖峰（与分段状态独立）：某一步损失突然跳升又恢复。分段均值/
    # 首末值会把这个点平均掉，只看趋势会漏；它常来自异常批次或一次梯度爆炸。
    if loss_spikes:
        worst_spike = max(loss_spikes, key=lambda s: s["ratio"])
        evidence.append(
            f"训练损失出现 {len(loss_spikes)} 个瞬态尖峰，最大在 step "
            f"{worst_spike['step']}: {worst_spike['value']:.4f} 是邻近点中位数 "
            f"{worst_spike['local_median']:.4f} 的 {worst_spike['ratio']:.1f} 倍"
            "（分段均值会将其掩盖）")
        recommendations.append(
            f"损失尖峰（×{worst_spike['ratio']:.0f} @step {worst_spike['step']}）："
            "先查该步批次是否含异常样本（见 datacheck/该步附近数据）与梯度范数"
            "是否同步尖峰，再考虑收紧 grad_clip 或降低学习率后重跑")

    # 坏记录（值/step 无法解析）：结构化日志被中断或手工编辑的信号；明确
    # 报告已跳过多少点，避免剩余曲线被当作完整曲线解读。
    if malformed_count:
        evidence.append(
            f"训练日志有 {malformed_count} 条记录的 step/值无法解析（已跳过）："
            "日志可能被中断或损坏，上述结论只基于其余记录点")

    gap = None
    if last_train is not None and final_val is not None:
        gap = final_val - last_train

    return {
        "status": status,
        "evidence": evidence,
        "recommendations": recommendations,
        "n_train_points": n_train,
        "n_val_points": n_val,
        "nonfinite_values": int(nan_count),
        "malformed_records": int(malformed_count),
        "train_loss_first": first_train,
        "train_loss_last": last_train,
        "train_rel_change": train_delta,
        "val_loss_first": first_val,
        "val_loss_last": last_val,
        "val_rel_change": val_delta,
        "val_loss_best": best_val,
        "val_best_step": best_step,
        "val_final_step": final_step,
        "early_stop_gain_rel": early_stop_gain,
        # 验证侧噪声量级与“变化是否超出噪声”的显式标记：下游可据此避免
        # 把窗口抽样误差当成过拟合/收敛信号。
        "val_ci_rel": val_ci_rel,
        "val_change_above_noise": (
            None if (val_delta is None or val_ci_rel is None)
            else bool(abs(val_delta) > val_ci_rel)),
        "early_stop_gain_above_noise": (
            None if (early_stop_gain is None or val_ci_rel is None)
            else bool(early_stop_gain > val_ci_rel)),
        "early_stopped": early_stopped,
        "early_stop_step": (es_step if early_stopped else None),
        "epochs_seen": epochs_seen,
        "train_tokens": train_tokens,
        "train_budget_sufficient": (None if epochs_seen is None
                                    else not budget_insufficient),
        "generalization_gap": gap,
        "grad_norm_points": len(grad_pts),
        "grad_norm_nonfinite": int(grad_bad),
        "grad_norm_median": grad_med,
        "grad_norm_max": grad_max,
        "grad_spike_ratio": grad_spike_ratio,
        "has_grad_spike": bool(grad_spike or grad_bad),
        "loss_spikes": loss_spikes,
        "has_loss_spike": bool(loss_spikes),
        "thresholds": {"rel_tol": REL_TOL, "diverge_factor": DIVERGE_FACTOR,
                       "early_stop_tol": EARLY_STOP_TOL,
                       "grad_spike_factor": GRAD_SPIKE_FACTOR,
                       "loss_spike_factor": LOSS_SPIKE_FACTOR,
                       "loss_spike_window": LOSS_SPIKE_WINDOW},
    }


# 状态严重度排序（汇总时取"最差种子"的状态代表整个变体）
STATUS_ORDER = {
    "NaN/Inf": 6, "发散": 5, "过拟合": 4,
    "欠拟合/学习不充分": 3, "训练不足（数据覆盖不足）": 2.5,
    "过拟合趋势": 2, "数据不足": 1, "正常收敛": 0,
}


def worst_status(statuses) -> str:
    """多个种子的诊断状态里取最严重的（汇总表口径）。"""
    return max(statuses, key=lambda s: STATUS_ORDER.get(s, -1)) \
        if statuses else "数据不足"


def diagnose_run_dir(run_dir: str, log_vocab: Optional[float] = None) -> Dict:
    """诊断一个训练 run 目录（train_log.jsonl + eval.json + nan_report.json）。"""
    records = load_train_log(os.path.join(run_dir, "train_log.jsonl"))
    epochs_seen = None
    train_tokens = None
    eval_path = os.path.join(run_dir, "eval.json")
    diag_warnings = []
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            if isinstance(ev, dict):
                e = ev.get("epochs_seen")
                t = ev.get("train_tokens")
                if e is not None and math.isfinite(float(e)):
                    epochs_seen = float(e)
                if t is not None:
                    train_tokens = int(t)
                cw = ev.get("curve_observability_warnings")
                if isinstance(cw, list):
                    diag_warnings = [str(x) for x in cw if str(x).strip()]
                else:
                    diag_warnings = []
        except (OSError, ValueError, TypeError, OverflowError):
            diag_warnings = []
    diag = diagnose_curve(records, log_vocab=log_vocab,
                          epochs_seen=epochs_seen,
                          train_tokens=train_tokens)
    if diag_warnings:
        # 采样过稀是配置级限制：随诊断结果进入 summary，避免只看汇总时
        # 把 2 个验证点的曲线当成可靠趋势。
        diag["curve_observability_warnings"] = diag_warnings
    nan_path = os.path.join(run_dir, "nan_report.json")
    if os.path.exists(nan_path):
        try:
            with open(nan_path, "r", encoding="utf-8") as f:
                diag["nan_report"] = json.load(f)
        except (OSError, ValueError):
            pass
    return diag


def main():
    p = argparse.ArgumentParser(description="训练曲线诊断（过拟合/欠拟合/发散/NaN）")
    p.add_argument("--run_dir", required=True, help="训练输出目录（含 train_log.jsonl）")
    p.add_argument("--log_vocab", type=float, default=None,
                   help="随机猜测水平 log(V)，用于识别'停在随机水平'的欠拟合")
    p.add_argument("--out", default=None, help="诊断 JSON 输出路径（默认打印）")
    args = p.parse_args()
    diag = diagnose_run_dir(args.run_dir, log_vocab=args.log_vocab)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
        print(f"[curves] 诊断 -> {args.out}")
    print(f"[curves] {args.run_dir}: status={diag['status']}")
    for ev in diag["evidence"]:
        print(f"[curves]   - {ev}")
    for rec in diag["recommendations"]:
        print(f"[curves]   * 建议: {rec}")


if __name__ == "__main__":
    main()
