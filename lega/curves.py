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
                  除全程首/末分段外，还比较末段（后 1/3 vs 中 1/3）：验证
                  早先大幅下降、后段才平台/回升时，全程变化仍可能显示下降，
                  只看首末会漏掉这一信号；
  * 欠拟合        训练损失也不下降（或停在 log(V) 随机猜测水平附近）——
                  学习率不当或数据管道本身有问题（标签错位/特征污染）；
  * 验证倒挂      验证损失显著低于训练损失（与泛化差距方向相反）——
                  优先排查数据泄漏/划分错配，或 train/val 评估口径不一致；
                  不能当作“验证更好”的收敛证据；
  * 训练不足      训练/验证都在下降，但累计只见过不足 1 个 epoch 的训练
                  数据——这是“还没学完”，不能当作收敛或“无需干预”；
  * 早停空间      最终验证损失劣于历史最优的程度——量化"早停能挽回多少"；
  * 评估噪声      日志带 val_loss_ci95 时，小于该抽样误差的变化不算“实质
                  回升/下降”——避免把有限窗口的评估噪声误判成过拟合；
  * 瞬态损失尖峰  单步损失远超邻近点中位数（分段均值会掩盖）——异常批次或
                  一次梯度爆炸的信号，即使趋势正常也应先核对批次与梯度；
                  训练日志同时记录 log 区间内单步峰值 loss_max 时，另按峰值
                  检测，避免“一步尖峰被区间均值平均掉”；
  * 坏记录        值/step 无法解析的日志行被跳过并计数，结论只基于其余点。

诊断输出为结构化 JSON（含建议清单），由 run_all 汇总进 summary.md；
也可独立运行：
  python -m lega.curves --run_dir out/tiny/lega [--out curve.json]
"""
import argparse
import json
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np

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
# 门控健康：门控是实现机制本身，不能只看最终 PPL。若 g 长期贴近 0
# （几乎不写入）或贴近 1（退化为标准注意力），方法的因果解释即失效。
# 这些阈值用于把“门控是否真的在学、是否有样本级区分”变成可复核信号。
GATE_SAT_LOW = 0.05
GATE_SAT_HIGH = 0.95
GATE_SAT_FRAC = 0.90
GATE_MIN_STD = 0.01
GATE_DRIFT_TOL = 0.25


def curve_observability_warnings(steps, eval_every, log_every) -> List[str]:
    """训练启动前检查曲线采样密度，返回可操作的配置告警（空表=足够）。

    steps/eval_every/log_every 直接决定曲线点数：eval_every 接近 steps 时
    验证曲线只有 1-2 个点，过拟合/早停/验证噪声诊断全部失效；log_every
    过大则训练侧趋势与瞬态尖峰无法检测。这里只告警不阻断——小规模冒烟
    训练有意识地用稀疏采样是合法的，但正式实验必须知道结论会因此受限。
    """
    normalized = []
    for value in (steps, eval_every, log_every):
        # bool 是 int 子类（True→1），小数不能静默截断成另一个采样间隔；
        # 这里只告警不阻断，但必须明确说“无法评估”而不是给出错误点数。
        if isinstance(value, (bool, np.bool_)):
            return ["训练步数/评测/日志间隔不是整数，无法评估曲线可观测性"]
        if isinstance(value, str) and "_" in value:
            return ["训练步数/评测/日志间隔不是整数，无法评估曲线可观测性"]
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return ["训练步数/评测/日志间隔不是整数，无法评估曲线可观测性"]
        if not math.isfinite(f) or not f.is_integer():
            return ["训练步数/评测/日志间隔不是整数，无法评估曲线可观测性"]
        normalized.append(int(f))
    steps, eval_every, log_every = normalized
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
        # 训练循环在 step==0 以及 (step+1)%log_every==0 时记录。log_every=1
        # 时 step 0 已包含在“每步”里，不能再额外 +1，否则会高估可观测点数。
        n_log = steps // log_every + (0 if log_every == 1 else 1)
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
        # 日志被中断时可能留下半个 UTF-8 字符：按替换字符读入，坏行随后由
        # JSON 解析失败跳过，而不是让整个诊断因 UnicodeDecodeError 崩溃。
        with open(path, "r", encoding="utf-8", errors="replace") as f:
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


def _step_or_none(value) -> Optional[int]:
    """日志 step → 整数；bool/非有限浮点/小数/不可解析值返回 None。

    小数 step 若被 int() 截断（1.5 → 1、True → 1），不同记录会折叠到同一
    步上，曲线与早停标记随之错位；整数值浮点（2.0）仍是合法 step。
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        value = int(value)
    elif isinstance(value, str):
        # int("1_0")==10：下划线写法会让坏日志冒充第 10 步，与其他整数
        # 解析入口（strict_int）的拒绝口径不一致。
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            return None
        value = int(value)
    else:
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(f) or not f.is_integer():
            return None
        value = int(f)
    # 曲线分段均值把 step 差与浮点比例相乘；超过 2^53 的 step 不能精确
    # 表示为 float，两个巨大 step 的差（如 1e308 − (−1e308)）还会让
    # int→float 转换直接 OverflowError。训练步数远小于该上限，按坏记录
    # 跳过而不是让整份曲线诊断崩溃。
    if abs(value) > 2 ** 53:
        return None
    try:
        if not math.isfinite(float(value)):
            return None
    except (OverflowError, ValueError):
        return None
    return value


def _finite_value(value) -> Optional[float]:
    """日志数值 → 有限 float；None/bool/非数值/NaN/Inf → None。

    bool 是 int 子类，float(True)=1.0；若当成 loss/grad_norm/val_loss，
    坏日志会伪造一个“正常”的数值点，污染趋势、尖峰与噪声诊断。
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    # float("1_0")==10.0：下划线数字串不是规范 JSON 数值，不能伪装成观测点。
    if isinstance(value, str) and "_" in value:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


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
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            malformed += 1
            continue
        if isinstance(v, str) and "_" in v:
            malformed += 1
            continue
        try:
            v = float(v)
        except (TypeError, ValueError, OverflowError):
            malformed += 1
            continue
        s = _step_or_none(r.get("step"))
        # 非有限损失是训练失效的直接证据，与 step 是否可解析无关：坏行上的
        # NaN/Inf 仍必须进入 nan_count，不能被“step 缺失”吞成 malformed。
        if not math.isfinite(v):
            bad += 1
            if s is None:
                malformed += 1
            continue
        # 有值但 step 缺失/无法解析同样是损坏记录：不能静默当成“没有这条点”，
        # 否则截断的日志会以更少的曲线点被当作完整日志诊断。
        if s is None:
            malformed += 1
            continue
        pts.append({"step": s, "value": v})
    pts.sort(key=lambda p: p["step"])
    return pts, bad, malformed


def _gradient_points(records: List[Dict]):
    """提取每个 log 点的 pre-clip 梯度范数，优先区间峰值 grad_norm_max。

    训练循环原先只在 log 点记录区间最后一步的 grad_norm：一次落在区间中段
    的梯度爆炸会被静默漏掉。新版日志另写 grad_norm_max（区间内最大 pre-clip
    范数），这里按 step 合并：同一步有峰值就用峰值，旧日志回退 grad_norm。
    返回 (pts, nonfinite_count, malformed_count, nonfinite_steps)。
    """
    by_step: Dict[int, Dict] = {}
    nonfinite_steps = set()
    nonfinite_unknown = 0
    malformed = 0
    for r in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(r, dict):
            malformed += 1
            continue
        step = _step_or_none(r.get("step"))
        chosen = None
        saw_numeric = nonfinite_here = False
        # 两个键都要检查：grad_norm_max 有限但 grad_norm 非有限（或反过来）
        # 仍是训练失效证据；不能因为优先键可解析就静默忽略另一个。
        for key in ("grad_norm_max", "grad_norm"):
            if key not in r or r.get(key) is None:
                continue
            raw = r.get(key)
            if isinstance(raw, (bool, np.bool_)) or (
                    isinstance(raw, str) and "_" in raw):
                malformed += 1
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            saw_numeric = True
            if not math.isfinite(val):
                nonfinite_here = True
                if step is not None:
                    nonfinite_steps.add(step)
                continue
            if chosen is None:
                chosen = val
        if nonfinite_here and step is None:
            # 没有可挂接的 step，但非有限梯度仍是必须报告的失效证据。
            nonfinite_unknown += 1
        if step is None:
            # 有数值但没有合法 step：记录为坏行，不能静默消失。
            if saw_numeric:
                malformed += 1
            continue
        if chosen is None:
            continue
        by_step[step] = {"step": step, "value": chosen}
    pts = sorted(by_step.values(), key=lambda p: p["step"])
    return (pts, len(nonfinite_steps) + nonfinite_unknown, malformed,
            sorted(nonfinite_steps))


def _detect_series_spikes(points: List[Dict],
                          factor: float = LOSS_SPIKE_FACTOR) -> List[Dict]:
    """检出单点尖峰：值 ≥ factor × 两侧邻居中位数。

    只比较局部邻居（每侧最多 LOSS_SPIKE_WINDOW 个点）而非全局中位数/均值：
    损失整体单调下降时，全局统计会把早期高点误报为尖峰，而局部比较对
    正常下降免疫。窗口按实际可用点截断：末尾不足 W 个右侧邻居时仍用左侧
    邻居判定（否则训练最后一步的尖峰会被完全漏掉）；左侧邻居少于 2 个时
    无法区分“初始高损失”与尖峰，跳过该点。
    """
    n = len(points)
    w = LOSS_SPIKE_WINDOW
    spikes = []
    for i in range(2, n):
        left = [p["value"] for p in points[max(0, i - w):i]]
        if len(left) < 2:
            continue
        right = [p["value"] for p in points[i + 1:i + 1 + w]]
        neigh = left + right
        med = statistics.median(neigh)
        if med <= 0:
            continue
        ratio = points[i]["value"] / med
        clamped = not math.isfinite(ratio)
        if clamped and points[i]["value"] > med:
            # 合法但极端的尖峰（如 1e308 / 1e-300）比值会溢出成 inf；
            # JSON/汇总不能写 inf，用最大可表示浮点封顶并显式标记。
            ratio = float(np.finfo(np.float64).max)
        if ratio >= factor:
            rec = {"step": points[i]["step"], "value": points[i]["value"],
                   "local_median": med, "ratio": ratio}
            if clamped:
                rec["ratio_clamped"] = True
            spikes.append(rec)
    return spikes


def _detect_loss_spikes(train_pts: List[Dict]) -> List[Dict]:
    """兼容旧调用：对训练损失（区间均值）做局部单点尖峰检测。"""
    return _detect_series_spikes(train_pts, LOSS_SPIKE_FACTOR)


def _tail_change(points: List[Dict], min_points: int = 5):
    """末段趋势：最后 1/3 相对中 1/3 的均值变化。

    全程首/末分段对“早期大幅改善、后期才平台/回升”不敏感：只要早段足够
    低，末段即使已经恶化，首→末仍可能是下降。这里单独比较中段与末段，
    用于捕捉过拟合早期（验证先停步）与平台期回升。点太少（<min_points）
    时中/末段各可能只有 1 个点，变化完全由单点噪声决定，返回 None。
    """
    if len(points) < min_points:
        return None
    prev = _seg_mean(points, 1 / 3, 2 / 3)
    last = _seg_mean(points, 2 / 3, 1.0)
    return {"prev": prev, "last": last,
            "rel_change": _rel_change(prev, last)}


def _seg_mean(pts: List[Dict], lo: float, hi: float) -> Optional[float]:
    """按 step 区间 [lo, hi)（比例）取均值；空区间返回 None。"""
    if not pts:
        return None
    steps = [p["step"] for p in pts]
    lo_s = steps[0] + lo * (steps[-1] - steps[0])
    hi_s = steps[0] + hi * (steps[-1] - steps[0])
    vals = [p["value"] for p in pts if lo_s <= p["step"] < hi_s
            or (hi == 1.0 and p["step"] == steps[-1])]
    if not vals:
        return None
    # 与统计模块同口径按最大绝对值缩放：多个 1e308 的合法损失点在朴素
    # 求和时会溢出成 inf，使分段趋势/倒挂判断全部失真。
    scale = max(abs(v) for v in vals)
    if scale == 0.0:
        return 0.0
    return float(scale * (sum(v / scale for v in vals) / len(vals)))


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
        lo, hi, v = _finite_value(ci[0]), _finite_value(ci[1]), _finite_value(v)
        if lo is not None and hi is not None and v is not None and v > 0 \
                and hi >= lo:
            try:
                rel = (hi - lo) / 2.0 / v
            except (OverflowError, ZeroDivisionError):
                continue
            # 两端有限不代表差/比值可表示：hi−lo 或除以极小 v 仍可能溢出。
            # 非有限相对半宽不能进入趋势阈值/JSON。
            if math.isfinite(rel):
                vals.append(rel)
    return float(statistics.median(vals)) if vals else None


def _rel_change(first: Optional[float], last: Optional[float]) -> Optional[float]:
    """last 相对 first 的相对变化（正=上升）；first 接近 0 时返回 None
    （相对变化无定义，不回退成绝对差以免口径混淆）。"""
    if first is None or last is None:
        return None
    if abs(first) < 1e-12:
        return None
    try:
        out = (last - first) / abs(first)
    except (OverflowError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


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
        f = _finite_value(x)
        if f is not None:
            vals.append(f)
    if len(vals) < 2 or abs(vals[0]) < 1e-12:
        return None
    try:
        out = (vals[-1] - vals[0]) / abs(vals[0])
    except (OverflowError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


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
    best_f, val_f = _finite_value(best_val), _finite_value(val)
    if best_f is None or val_f is None or best_f == 0:
        return None
    gap = (val_f - best_f) / best_f
    if not math.isfinite(gap):
        return None
    # 验证抽样噪声：区间半宽相对当前 val。gap 小于该半宽时，"回升"与
    # "平台"都无法和有限窗口抽样误差区分。
    ci_rel = None
    if isinstance(val_ci95, (list, tuple)) and len(val_ci95) == 2:
        lo, hi = _finite_value(val_ci95[0]), _finite_value(val_ci95[1])
        if (lo is not None and hi is not None and hi >= lo and val_f > 0):
            try:
                ci_rel = (hi - lo) / 2.0 / val_f
            except (OverflowError, ZeroDivisionError):
                ci_rel = None
            if ci_rel is not None and not math.isfinite(ci_rel):
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


def _gate_points(records: List[Dict]) -> List[Dict]:
    """提取带门控统计的评测点；坏值/非法比例跳过，不让单条日志中断诊断。"""
    points = []
    for r in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(r, dict):
            continue
        step = _step_or_none(r.get("step"))
        mean = _finite_value(r.get("gate_mean"))
        if step is None or mean is None or not 0.0 <= mean <= 1.0:
            continue
        std = _finite_value(r.get("gate_std"))
        low = _finite_value(r.get("gate_frac_low"))
        high = _finite_value(r.get("gate_frac_high"))
        layer = r.get("gate_layer_mean")
        layer = [f for f in (_finite_value(x) for x in layer)
                 if f is not None and 0.0 <= f <= 1.0] \
            if isinstance(layer, (list, tuple)) else None
        points.append({
            "step": step, "mean": mean,
            "std": std if std is not None and std >= 0 else None,
            "frac_low": low if low is not None and 0.0 <= low <= 1.0 else None,
            "frac_high": high if high is not None and 0.0 <= high <= 1.0 else None,
            "layer_mean": layer,
        })
    points.sort(key=lambda p: p["step"])
    return points


def diagnose_gate(records: List[Dict]) -> Dict:
    """门控健康诊断：塌缩/饱和/常数门/剧烈漂移都是方法失效的独立信号。

    只看 PPL 无法发现“门控没在起作用”：g≡0 可能通过压制残差写入降低
    损失，g≡1 则让 LEGA 完全退化成标准注意力；两种情况下关于长度校准/
    集中度证据的因果主张都不成立。验证评测已记录 gate_std 与低/高饱和
    比例，这里把它们转成结构化结论，供曲线诊断与汇总页显式披露。
    """
    points = _gate_points(records)
    if not points:
        return {"available": False, "status": "不适用（无门控统计）",
                "n_points": 0, "evidence": [], "recommendations": []}
    first, last = points[0], points[-1]
    mean, std = last["mean"], last["std"]
    low, high = last["frac_low"], last["frac_high"]
    collapsed_low = ((low is not None and low >= GATE_SAT_FRAC)
                     or (mean <= 0.02 and (std is None or std <= 0.05)))
    collapsed_high = ((high is not None and high >= GATE_SAT_FRAC)
                      or (mean >= 0.98 and (std is None or std <= 0.05)))
    constant = (std is not None and std <= GATE_MIN_STD
                and not collapsed_low and not collapsed_high)
    drifts = []
    for a, b in zip(points, points[1:]):
        d = b["mean"] - a["mean"]
        if abs(d) >= GATE_DRIFT_TOL:
            drifts.append((a["step"], b["step"], d))
    status = "健康"
    evidence, recs = [], []
    if collapsed_low:
        status = "门控塌缩（几乎不写入）"
        evidence.append(
            f"门控均值 {mean:.3f}"
            + (f"、std={std:.3f}" if std is not None else "")
            + (f"、低于 {GATE_SAT_LOW:g} 的比例 {low:.0%}" if low is not None
               else "")
            + "：门几乎关闭，输出被整体压制；此时 PPL 变化不能归因于"
              "集中度/长度校准机制")
        recs.append(
            "门控塌缩：检查门控偏置 d、w/a/b 权重与学习率是否把 g 压到 0；"
            "若门控确实学成关闭，应报告“机制未生效”，不要用该 run 的 PPL "
            "给门控机制背书")
    elif collapsed_high:
        status = "门控饱和（退化为标准注意力）"
        evidence.append(
            f"门控均值 {mean:.3f}"
            + (f"、std={std:.3f}" if std is not None else "")
            + (f"、高于 {GATE_SAT_HIGH:g} 的比例 {high:.0%}" if high is not None
               else "")
            + "：g≈1 时 LEGA 退化为标准注意力，长度/集中度项没有实际作用")
        recs.append(
            "门控饱和：g≈1 时方法退化为标准注意力，不能声称门控机制有效；"
            "检查 gate_bias_init 是否过高、a/b 是否未学习，必要时降低初值"
            "或加门控正则后重跑")
    elif constant:
        status = "门控近似常数（无样本级区分）"
        evidence.append(
            f"门控 std={std:.4f} ≤ {GATE_MIN_STD:g}（均值 {mean:.3f}）："
            "所有 token 得到几乎相同的门值，门控只相当于全局缩放，"
            "并未按查询/集中度/长度做样本级决策")
        recs.append(
            "门控无样本级区分：检查 w/a/b 是否被 weight decay 压到 0、"
            "学习率是否过小；若门控参数未离开初始化，不要报告门控机制有效")
    if drifts:
        worst = max(drifts, key=lambda x: abs(x[2]))
        if status == "健康":
            status = "门控剧烈漂移"
        evidence.append(
            f"门控均值在评测间最大变化 {worst[2]:+.2f}"
            f"（step {worst[0]}→{worst[1]}，阈值 ±{GATE_DRIFT_TOL:g}）："
            "门控行为不稳定，不能只拿最终均值代表训练过程")
        recs.append(
            "门控剧烈漂移：优先检查学习率/梯度尖峰（见同页 grad_norm），"
            "并用多个评测点的门控曲线而不是最终 gate_mean 解释机制")
    if status == "健康" and std is None:
        # 只有均值无法证明“仍有样本级变化”，也无法排除常数门；中值
        # 既未塌缩也未饱和时只能判为数据不足，不能给机制背书。
        status = "数据不足（缺少门控分布统计）"
        evidence.append(
            f"门控末点均值 {mean:.3f}，但日志缺少 gate_std/低高饱和比例："
            "无法确认门控是否存在样本级变化，也不能排除常数门")
        recs.append(
            "门控只有均值：补充记录 gate_std、gate_frac_low、gate_frac_high "
            "后重跑诊断；在此之前不要用该 run 声称门控机制有效")
    elif status == "健康":
        evidence.append(
            f"门控末点均值 {mean:.3f}"
            + (f"、std={std:.3f}" if std is not None else "")
            + (f"、低饱和 {low:.0%}/高饱和 {high:.0%}"
               if low is not None and high is not None else "")
            + f"：门控在训练末段仍有样本级变化，未贴近 0/1 极值")
    return {"available": True, "status": status, "n_points": len(points),
            "mean_first": first["mean"], "mean_last": mean,
            "std_last": std, "frac_low_last": low, "frac_high_last": high,
            "layer_mean_last": last["layer_mean"],
            "drift_max": (max(abs(d[2]) for d in drifts) if drifts else 0.0),
            "evidence": evidence, "recommendations": recs}


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
    log_vocab = _finite_value(log_vocab)
    if log_vocab is not None and log_vocab <= 0:
        log_vocab = None
    epochs_seen = _finite_value(epochs_seen)
    if epochs_seen is not None and epochs_seen < 0:
        epochs_seen = None
    train_tokens = _step_or_none(train_tokens)
    if train_tokens is not None and train_tokens < 0:
        train_tokens = None
    train_pts, train_bad, train_malformed = _finite_points(records, "loss")
    val_pts, val_bad, val_malformed = _finite_points(records, "val_loss")
    # 早停终止标记：训练按设计提前结束（区别于日志缺失/训练中断），
    # 建议不再指向"延长训练"（验证已连续无改善）。
    # step 只采信可解析的有限值；缺失/损坏（外部或截断的日志）时回退到
    # 最后一个有限训练点，既保留早停标注，又不把 "step None" 写进证据。
    # 只认真正的布尔 true；外部日志里的 "false"/"0"/1 等真值会被
    # truthiness 误判成“早停终止”，从而伪造训练提前结束的证据。
    early_stop_recs = [r for r in records
                       if isinstance(r, dict) and r.get("early_stop") is True]
    early_stopped = bool(early_stop_recs)
    early_stop_steps = []
    for r in early_stop_recs:
        s = _step_or_none(r.get("step"))
        if s is None:
            continue
        early_stop_steps.append(s)
    es_step = (early_stop_steps[0] if early_stop_steps
               else (train_pts[-1]["step"] if train_pts else None))
    # 累计门控/统计点上的 NaN 只在对应序列里报；任何序列出现即整体报。
    # 单步峰值 loss_max 也属于训练损失序列：区间均值有限但单步峰值 NaN/Inf
    # 时训练已经失效，不能因为只看均值而漏报。
    malformed_count = train_malformed + val_malformed
    # 梯度范数体检：尖峰/非有限是发散与 NaN 的早期信号，损失曲线可能还没反映。
    # 优先用区间峰值 grad_norm_max：只记录 log 点最后一步会漏掉区间中段的爆炸。
    grad_pts, grad_bad, grad_malformed, grad_bad_steps = \
        _gradient_points(records)
    malformed_count += grad_malformed
    # 瞬态损失尖峰：分段均值会把它平均掉，必须按点单独体检（与损失状态独立）。
    # loss_max 是同一 log 区间内的单步峰值：一步异常批次被区间均值稀释时，
    # 均值序列可能看不到 8× 尖峰，峰值序列仍能看到。
    loss_spikes = _detect_series_spikes(train_pts, LOSS_SPIKE_FACTOR)
    loss_peak_pts, loss_peak_bad, loss_peak_malformed = \
        _finite_points(records, "loss_max")
    malformed_count += loss_peak_malformed
    nan_count = train_bad + val_bad + loss_peak_bad
    # 旧日志/新版日志峰值与均值相同（区间只有 1 步）时避免重复报告同一尖峰
    if [(p["step"], p["value"]) for p in loss_peak_pts] == \
            [(p["step"], p["value"]) for p in train_pts]:
        loss_peak_pts = []
    loss_peak_spikes = _detect_series_spikes(loss_peak_pts, LOSS_SPIKE_FACTOR)
    grad_vals = [p["value"] for p in grad_pts]
    grad_med = None
    grad_max = None
    grad_spike_ratio = None
    grad_spikes = []
    grad_max_step = None
    if grad_vals:
        grad_med = float(statistics.median(grad_vals))
        peak = max(grad_pts, key=lambda p: p["value"])
        grad_max, grad_max_step = float(peak["value"]), peak["step"]
        if grad_med > 0:
            grad_spike_ratio = grad_max / grad_med
            grad_spikes = [dict(p, ratio=p["value"] / grad_med)
                           for p in grad_pts
                           if p["value"] / grad_med >= GRAD_SPIKE_FACTOR]
    grad_spike = bool(grad_spikes)
    worst_grad_spike = (max(grad_spikes, key=lambda p: p["ratio"])
                        if grad_spikes else None)

    first_train = _seg_mean(train_pts, 0.0, 1 / 3)
    last_train = _seg_mean(train_pts, 2 / 3, 1.0)
    first_val = _seg_mean(val_pts, 0.0, 1 / 3)
    last_val = _seg_mean(val_pts, 2 / 3, 1.0)

    train_delta = _rel_change(first_train, last_train)
    val_delta = _rel_change(first_val, last_val)
    # 验证-训练损失倒挂：验证损失显著低于训练损失不是“收敛更好”，而是
    # 数据泄漏/划分错配/评估口径不一致（如 train 侧仍开 dropout、train loss
    # 用区间滑动均值而 val 用 eval 模式）的典型信号。只看两条曲线各自下降
    # 会完全漏掉这一点，必须把两条线的相对水平也纳入诊断。
    val_below_train_rel = None
    if (last_train is not None and last_val is not None
            and last_train > 0):
        val_below_train_rel = _rel_change(last_train, last_val)
    val_below_train = (val_below_train_rel is not None
                       and val_below_train_rel < -REL_TOL)
    train_down = train_delta is not None and train_delta < -REL_TOL
    train_up = train_delta is not None and train_delta > REL_TOL
    # 验证侧判定必须扣除评估抽样噪声：val_loss_ci95 存在时，变化幅度不超过
    # 该点级 CI 半宽就不算“实质回升/下降”。无 CI 的旧日志保持原口径。
    val_ci_rel = _val_ci_rel(records)
    val_tol = max(REL_TOL, val_ci_rel or 0.0)
    val_down = val_delta is not None and val_delta < -val_tol
    val_up = val_delta is not None and val_delta > val_tol
    val_unresolved = (val_delta is not None and abs(val_delta) <= val_tol)
    # 末段趋势（后 1/3 vs 中 1/3）：全程首/末分段对“早先大幅改善、后段
    # 平台/回升”不敏感。验证早段已降到很低时，末段回升后首→末仍可能为负，
    # 只看全程会漏掉过拟合；训练末段已停但全程仍在降也会被当成正常收敛。
    recent_train = _tail_change(train_pts)
    recent_val = _tail_change(val_pts)
    recent_train_delta = (recent_train or {}).get("rel_change")
    recent_val_delta = (recent_val or {}).get("rel_change")
    train_recent_down = (recent_train_delta is not None
                         and recent_train_delta < -REL_TOL)
    train_recent_flat = (recent_train_delta is not None
                         and abs(recent_train_delta) <= REL_TOL)
    train_recent_up = (recent_train_delta is not None
                       and recent_train_delta > REL_TOL)
    val_recent_down = (recent_val_delta is not None
                       and recent_val_delta < -val_tol)
    val_recent_up = (recent_val_delta is not None
                     and recent_val_delta > val_tol)
    val_recent_unresolved = (recent_val_delta is not None
                             and abs(recent_val_delta) <= val_tol)
    recent_available = recent_train is not None and recent_val is not None
    recent_overfit = bool(recent_available and val_recent_up
                          and (train_recent_down or train_recent_flat))
    recent_trend = bool(recent_available and train_recent_down
                        and val_recent_unresolved and not val_recent_up)

    best_val, best_step = None, None
    for p in val_pts:
        if best_val is None or p["value"] < best_val:
            best_val, best_step = p["value"], p["step"]
    final_val = val_pts[-1]["value"] if val_pts else None
    final_step = val_pts[-1]["step"] if val_pts else None
    early_stop_gain = _rel_change(best_val, final_val)  # >0 表示最终劣于最优
    # best checkpoint 选型可靠性：best 只是 val 点估计的最小值，未必与次优
    # 点有可区分的差异。若 best 与次优的差距落在该点 95% CI 内，选型可能
    # 由窗口抽样噪声决定（winner's curse）；此时不能只报告 best 的数字。
    val_ci_by_step = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        ci, v = r.get("val_loss_ci95"), _finite_value(r.get("val_loss"))
        s_ = _step_or_none(r.get("step"))
        if (s_ is None or v is None or not isinstance(ci, (list, tuple))
                or len(ci) != 2):
            continue
        lo, hi = _finite_value(ci[0]), _finite_value(ci[1])
        if lo is not None and hi is not None and hi >= lo and v > 0:
            val_ci_by_step.setdefault(s_, (lo, hi, v))
    best_ci_rel = None
    if best_step in val_ci_by_step:
        lo, hi, v = val_ci_by_step[best_step]
        try:
            best_ci_rel = (hi - lo) / 2.0 / v
        except (OverflowError, ZeroDivisionError):
            best_ci_rel = None
        if best_ci_rel is not None and not math.isfinite(best_ci_rel):
            best_ci_rel = None
    sorted_val = sorted(p["value"] for p in val_pts)
    second_best = sorted_val[1] if len(sorted_val) >= 2 else None
    best_margin_rel = (_rel_change(best_val, second_best)
                       if (best_val is not None and second_best is not None)
                       else None)
    best_selection_noise = bool(
        best_ci_rel is not None and best_margin_rel is not None
        and best_step != final_step and best_margin_rel <= best_ci_rel)
    # 早停空间同样要扣除评估噪声：final 比 best 差 2% 但单点 CI 就有 ±3% 时，
    # 不能据此断言“已经过拟合、早停能挽回”。
    gain_tol = max(EARLY_STOP_TOL, val_ci_rel or 0.0)
    gain_sig = (early_stop_gain or 0) > gain_tol

    status, evidence, recommendations = "数据不足", [], []
    n_train, n_val = len(train_pts), len(val_pts)

    if nan_count:
        status = "NaN/Inf"
        evidence.append(f"训练日志出现 {nan_count} 个非有限损失值"
                        f"（训练 {train_bad}、验证 {val_bad}、单步峰值 "
                        f"{loss_peak_bad}）")
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
                  and val_up and (gain_sig or recent_overfit))
              or recent_overfit):
            # 过拟合/平台期过拟合：训练不再下降（或总体仍在降但末段已停）
            # 而验证回升，同样需要早停/正则化。若按"训练没降"误判成欠拟合，
            # 会给出"延长训练/增大模型"的反向建议。
            status = "过拟合"
            if train_down:
                ev = [f"全程首/末分段：训练损失 "
                      f"{_fmt_num(first_train)}→{_fmt_num(last_train)}"
                      f"（{_fmt_rel(train_delta)}）而验证损失 "
                      f"{_fmt_num(first_val)}→{_fmt_num(last_val)}"
                      f"（{_fmt_rel(val_delta)}）"]
            else:
                ev = [f"全程首/末分段：训练损失 "
                      f"{_fmt_num(first_train)}→{_fmt_num(last_train)}"
                      f"（{_fmt_rel(train_delta)}）已停滞/不再实质下降，而验证损失 "
                      f"{_fmt_num(first_val)}→{_fmt_num(last_val)}"
                      f"（{_fmt_rel(val_delta)}）上升——泛化先于训练损失恶化，"
                      "按平台期过拟合处理"]
            # 末段趋势是独立证据：全程仍显示验证下降时，末段回升同样构成
            # 过拟合（早期改善不能抵消最近的泛化恶化）。
            if recent_available and (recent_overfit or val_recent_up):
                ev.append(
                    f"末段趋势（后 1/3 vs 中 1/3）：训练 "
                    f"{_fmt_num((recent_train or {}).get('prev'))}→"
                    f"{_fmt_num((recent_train or {}).get('last'))}"
                    f"（{_fmt_rel(recent_train_delta)}），验证 "
                    f"{_fmt_num((recent_val or {}).get('prev'))}→"
                    f"{_fmt_num((recent_val or {}).get('last'))}"
                    f"（{_fmt_rel(recent_val_delta)}）"
                    + ("；验证末段回升，按过拟合处理"
                       if val_recent_up else ""))
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
        elif recent_trend and not best_selection_noise:
            # 全程首→末验证仍在下降（早段改善占主导），但末段训练继续降、
            # 验证已停步：泛化改善停滞，按过拟合早期处理，不能因全程均值为
            # “下降”就判正常收敛/继续训练。
            status = "过拟合趋势"
            evidence.append(
                f"末段趋势（后 1/3 vs 中 1/3）：训练 "
                f"{_fmt_num((recent_train or {}).get('prev'))}→"
                f"{_fmt_num((recent_train or {}).get('last'))}"
                f"（{_fmt_rel(recent_train_delta)}）仍在下降，而验证 "
                f"{_fmt_num((recent_val or {}).get('prev'))}→"
                f"{_fmt_num((recent_val or {}).get('last'))}"
                f"（{_fmt_rel(recent_val_delta)}）已无实质变化"
                + (f"（评估噪声 ±{val_ci_rel:.1%}）" if val_ci_rel is not None
                   else "")
                + "：全程首/末分段仍显示验证下降（早段改善占主导），但末段"
                "验证不再随训练改善，泛化差距在扩大")
            recommendations += [
                "训练降而验证末段停滞：按过拟合早期信号处理——启用/加强正则化"
                "（dropout / weight decay），对比使用 val 最优的 model_best.pt",
                "不要仅因全程验证损失仍在下降就延长训练：末段新增收益只体现在"
                "训练集",
            ]
        elif not train_down and not train_recent_down:
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
            elif not (loss_spikes or loss_peak_spikes or grad_bad or grad_spike
                      or early_stopped or malformed_count):
                # 任一独立告警（损失尖峰/梯度尖峰/早停/坏记录）存在时都不写
                # “无需干预”，否则会与随后追加的诊断建议自相矛盾。
                recommendations.append("无需干预；对比实验使用 model_best.pt")
        if gain_sig and status not in ("过拟合",
                                       "验证损失倒挂（疑似泄漏/口径不一致）"):
            recommendations.append(
                f"早停提示：最终验证损失劣于 step {best_step} 最优值 "
                f"{early_stop_gain:.1%}")
        # 验证-训练倒挂是数据/评测口径的完整性问题，与曲线形状（过拟合/
        # 平台/欠拟合）独立：不能只在“正常收敛”分支里检查，否则一个仍在
        # 下降但验证停滞的 run 会一边写 val_below_train=True、一边建议加
        # 正则化。倒挂一旦成立，优先排查泄漏/口径，而不是调正则化。
        if (val_below_train
                and status in ("正常收敛", "过拟合", "过拟合趋势")):
            status = "验证损失倒挂（疑似泄漏/口径不一致）"
            evidence.append(
                f"验证损失末段 {last_val:.4f} 显著低于训练损失末段 "
                f"{last_train:.4f}（{val_below_train_rel:+.1%}）："
                "这与常规泛化差距方向相反，不能按“收敛更好”解读；"
                "优先排查数据划分/预处理泄漏（train 与 val 文档重叠、"
                "词表或归一化在 val 上拟合）、训练与验证的评估口径差异"
                "（dropout/eval 模式、滑动平均、不同 token 集合）")
            recommendations[:] = [
                r for r in recommendations
                if not any(key in r for key in
                           ("正则化", "早停", "无需干预", "过拟合"))]
            recommendations.append(
                "验证损失倒挂：先做数据泄漏/协议审计（datacheck 的跨 split "
                "重复与 n-gram 重叠、split 时序与文档边界），并核对训练/验证"
                "是否使用同一精度、同一 dropout 状态与同一损失口径；"
                "排除泄漏前不要报告该 run 的验证指标")
        elif val_below_train:
            # 发散/欠拟合等状态也不应把倒挂信号藏起来：保留曲线状态判定，
            # 但并列报告完整性告警，避免读者只看到“欠拟合→扩容量”等建议。
            evidence.append(
                f"注意：验证损失末段 {last_val:.4f} 也低于训练损失末段 "
                f"{last_train:.4f}（{val_below_train_rel:+.1%}），存在"
                "验证-训练倒挂；该信号独立于当前曲线状态，优先排查数据划分/"
                "预处理泄漏与训练/验证评估口径差异")
            recommendations.append(
                "同时存在验证-训练倒挂：在按上述建议调参/扩容量前，先完成"
                "跨 split 重复与 n-gram 重叠审计，并核对训练/验证的精度、"
                "dropout 状态与损失口径；排除泄漏前不要报告该 run 的验证指标")

    if val_ci_rel is not None:
        evidence.append(
            f"验证损失点级 95% CI 相对半宽中位数 ±{val_ci_rel:.1%}"
            "（窗口抽样误差；分段均值会进一步降低该项，按保守口径使用）："
            "小于该量级的验证变化按噪声处理，不作为过拟合/收敛证据")

    # best checkpoint 选型也可能是噪声：best 与次优点的差距小于该点 CI 时，
    # 谁被选中取决于有限验证窗口的抽样波动，而不是真实泛化差异。
    if best_selection_noise:
        evidence.append(
            f"best checkpoint 在 step {best_step}，但仅比次优验证损失 "
            f"{second_best:.4f} 好 {best_margin_rel:.1%}，小于该点 95% CI "
            f"相对半宽 ±{best_ci_rel:.1%}：最优与次优无法区分，"
            "选型可能由验证窗口抽样噪声决定（winner's curse）")
        recommendations.append(
            f"best checkpoint 选型落在评估噪声内（margin {best_margin_rel:.1%} "
            f"≤ CI ±{best_ci_rel:.1%}）：不要只报告 best 的指标；同时报告"
            "final/多种子均值，并增大 eval_iters 或验证集以缩小选型噪声；"
            "若结论依赖 best 与次优的排序，应视为未定")
        recommendations[:] = [r for r in recommendations
                              if "无需干预" not in r]

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

    # best 出现在最后一个评测点：没有观察到“开始变差”的拐点，训练预算
    # 本身可能就是性能上限。只降不升不是过拟合，但也不能声称已经收敛；
    # 若验证损失在末端仍显著下降，应把“延长训练”列为补充实验，而不是
    # 把当前截断点当成方法的收敛性能。
    best_at_final = bool(best_step is not None and final_step is not None
                         and best_step == final_step)
    still_improving = bool(
        best_at_final and status == "正常收敛"
        and val_delta is not None and val_delta < 0
        and (val_delta < -(val_ci_rel if val_ci_rel is not None
                           else REL_TOL)))
    if still_improving and epochs_seen is not None:
        recommendations[:] = [r for r in recommendations
                              if "无需干预" not in r]
        recommendations.append(
            f"最优验证损失出现在最后一个评测点（step {final_step}），"
            f"且验证损失仍下降 {abs(val_delta):.1%}：没有观察到收敛/过拟合"
            "拐点，当前训练预算可能就是性能上限；预算允许时应延长训练作为"
            "补充实验，不能把该 run 的最优值当作方法已收敛的结论")
        evidence.append(
            f"best 与 final 为同一评测点（step {final_step}），"
            f"验证损失首→末 {_fmt_rel(val_delta)} 且末端仍在下降："
            "训练尚未显示出收敛拐点（不是过拟合）")

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
        steps_txt = ("，step " + ", ".join(str(x) for x in grad_bad_steps[:8])
                     + ("…" if len(grad_bad_steps) > 8 else "")
                     if grad_bad_steps else "")
        evidence.append(f"pre-clip 梯度范数出现 {grad_bad} 个非有限值{steps_txt}"
                        "（梯度已爆炸，先于或伴随损失 NaN）")
        recommendations.append(
            "梯度爆炸：降低学习率、收紧 grad_clip（如 0.5）、延长 warmup，"
            "并确认 train_log.jsonl 的 grad_norm/grad_norm_max 不再出现非有限值")
    if grad_spike and worst_grad_spike is not None:
        evidence.append(
            f"pre-clip 梯度范数最大 {grad_max:.2e}（step {grad_max_step}）"
            f"为中位数 {grad_med:.2e} 的 {grad_spike_ratio:.1f} 倍"
            f"；共 {len(grad_spikes)} 个 ≥{GRAD_SPIKE_FACTOR:g}× 尖峰，"
            f"最严重在 step {worst_grad_spike['step']}（×{worst_grad_spike['ratio']:.1f}）"
            + ("；此处的 grad_norm 优先采用 log 区间内峰值 grad_norm_max"
               if any("grad_norm_max" in r for r in records
                      if isinstance(r, dict)) else ""))
        recommendations.append(
            f"检测到梯度范数尖峰（×{worst_grad_spike['ratio']:.0f} "
            f"@step {worst_grad_spike['step']}）：这是 NaN 之前的典型信号，"
            "应降低学习率/收紧 grad_clip 后再验证曲线")

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
    if loss_peak_spikes:
        worst_peak = max(loss_peak_spikes, key=lambda s: s["ratio"])
        evidence.append(
            f"log 区间内单步损失峰值出现 {len(loss_peak_spikes)} 个尖峰，最大在 "
            f"step {worst_peak['step']}: 单步峰值 {worst_peak['value']:.4f} 是邻近"
            f"峰值中位数 {worst_peak['local_median']:.4f} 的 "
            f"{worst_peak['ratio']:.1f} 倍——区间均值可能把单步异常批次平均掉，"
            "该信号需要单独核对批次")
        recommendations.append(
            f"单步损失峰值尖峰（×{worst_peak['ratio']:.0f} "
            f"@step {worst_peak['step']}）：优先复核该 log 区间内哪一 micro-step "
            "消费了异常批次/标签，并对照同一区间 grad_norm_max 是否尖峰")

    # 坏记录（值/step 无法解析）：结构化日志被中断或手工编辑的信号；明确
    # 报告已跳过多少点，避免剩余曲线被当作完整曲线解读。
    if malformed_count:
        evidence.append(
            f"训练日志有 {malformed_count} 条记录的 step/值无法解析（已跳过）："
            "日志可能被中断或损坏，上述结论只基于其余记录点")

    gap = None
    if last_train is not None and final_val is not None:
        gap = final_val - last_train

    gate = diagnose_gate(records)
    if gate["evidence"]:
        evidence += gate["evidence"]
    if gate["recommendations"]:
        recommendations += gate["recommendations"]
        if gate["status"] != "健康":
            recommendations[:] = [r for r in recommendations
                                  if "无需干预" not in r]

    return {
        "status": status,
        "evidence": evidence,
        "recommendations": recommendations,
        "n_train_points": n_train,
        "n_val_points": n_val,
        "nonfinite_values": int(nan_count),
        "loss_peak_nonfinite": int(loss_peak_bad),
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
        "val_second_best": second_best,
        "val_best_margin_rel": best_margin_rel,
        "val_best_ci_rel": best_ci_rel,
        "best_checkpoint_within_noise": best_selection_noise,
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
        "best_at_final_eval": best_at_final,
        "val_still_improving_at_end": still_improving,
        "train_budget_sufficient": (None if epochs_seen is None
                                    else not budget_insufficient),
        "generalization_gap": gap,
        # 末段验证损失相对训练损失的变化（负=验证低于训练）；倒挂是泄漏/
        # 口径不一致信号，而不是更好的泛化。
        "val_below_train_rel": val_below_train_rel,
        "val_below_train": bool(val_below_train),
        "grad_norm_points": len(grad_pts),
        "grad_norm_nonfinite": int(grad_bad),
        "grad_norm_nonfinite_steps": grad_bad_steps,
        "grad_norm_median": grad_med,
        "grad_norm_max": grad_max,
        "grad_norm_max_step": grad_max_step,
        "grad_spike_ratio": grad_spike_ratio,
        "grad_spikes": grad_spikes,
        "has_grad_spike": bool(grad_spike or grad_bad),
        "loss_spikes": loss_spikes,
        "loss_peak_spikes": loss_peak_spikes,
        "has_loss_spike": bool(loss_spikes or loss_peak_spikes),
        "has_single_step_loss_spike": bool(loss_peak_spikes),
        # 末段趋势：全程仍“下降”时，末段平台/回升是更早的过拟合信号。
        "train_recent_rel_change": recent_train_delta,
        "val_recent_rel_change": recent_val_delta,
        "train_recent_down": bool(train_recent_down),
        "train_recent_stalled": bool(train_recent_flat),
        "val_recent_up": bool(val_recent_up),
        "val_recent_unresolved": bool(val_recent_unresolved),
        "recent_overfit_signal": bool(recent_overfit),
        "recent_trend_signal": bool(recent_trend),
        "recent_trend_available": bool(recent_available),
        "recent_comparison": {"train": recent_train, "val": recent_val},
        "gate_health": gate,
        "thresholds": {"rel_tol": REL_TOL, "diverge_factor": DIVERGE_FACTOR,
                       "early_stop_tol": EARLY_STOP_TOL,
                       "grad_spike_factor": GRAD_SPIKE_FACTOR,
                       "loss_spike_factor": LOSS_SPIKE_FACTOR,
                       "loss_spike_window": LOSS_SPIKE_WINDOW,
                       "recent_window": "last 1/3 vs middle 1/3"},
    }


# 状态严重度排序（汇总时取"最差种子"的状态代表整个变体）
STATUS_ORDER = {
    "NaN/Inf": 6, "发散": 5, "过拟合": 4,
    "验证损失倒挂（疑似泄漏/口径不一致）": 3.5,
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
                # bool 是 int 子类，float(True)=1.0；直接 int(1.9)=1 也会
                # 伪造训练覆盖率。类型不符/负数/非有限值按缺失处理，避免
                # “训练不足”诊断被坏元数据静默改写。
                if (not isinstance(e, bool)
                        and isinstance(e, (int, float))):
                    ef = float(e)
                    if math.isfinite(ef) and ef >= 0:
                        epochs_seen = ef
                if (not isinstance(t, bool)
                        and isinstance(t, int) and t >= 0):
                    train_tokens = t
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
