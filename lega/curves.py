"""训练曲线诊断 —— 只看最终数值会掩盖训练动态，必须逐条检查曲线本身。

诊断口径（不依赖任何主观读图）：
  * NaN/Inf      训练或验证损失出现非有限值 —— 训练已失效，先走 NaN 排查路径；
  * 发散          训练损失后期显著高于前期（学习率过大/梯度爆炸的典型表现）；
  * 过拟合        训练损失下降而验证损失上升 —— 需要正则化、增大数据量或早停；
  * 欠拟合        训练损失也不下降（或停在 log(V) 随机猜测水平附近）——
                  学习率不当或数据管道本身有问题（标签错位/特征污染）；
  * 早停空间      最终验证损失劣于历史最优的程度——量化"早停能挽回多少"。

诊断输出为结构化 JSON（含建议清单），由 run_all 汇总进 summary.md；
也可独立运行：
  python -m lega.curves --run_dir out/tiny/lega [--out curve.json]
"""
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

# 相对变化阈值：低于该幅度视为"没有实质下降/上升"（覆盖常见噪声量级）
REL_TOL = 0.03
# 发散判定：后期均值超过前期均值的该倍数
DIVERGE_FACTOR = 1.5
# 早停提示阈值：最终 val 损失比历史最优差出该比例以上
EARLY_STOP_TOL = 0.02
# 欠拟合疑似数据管道问题：训练损失停在随机猜测水平的该比例以内
RANDOM_LEVEL_TOL = 0.9


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
                    recs.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return recs


def _finite_points(records: List[Dict], key: str) -> Tuple[List[Dict], int]:
    """取出 (step, value) 均有限并按 step 排序的记录点；非有限值单独计数。

    返回 (pts, bad)：pts 为有限点列表，bad 为非有限值个数。
    记录按 step 排序（train/val 两类点在 jsonl 中交错写入，不能假设有序）。
    """
    pts, bad = [], 0
    for r in records:
        v, s = r.get(key), r.get("step")
        if v is None or s is None:
            continue
        v, s = float(v), int(s)
        if math.isfinite(v):
            pts.append({"step": s, "value": v})
        else:
            bad += 1
    pts.sort(key=lambda p: p["step"])
    return pts, bad


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


def _rel_change(first: Optional[float], last: Optional[float]) -> Optional[float]:
    """last 相对 first 的相对变化（正=上升）；first 接近 0 时返回 None
    （相对变化无定义，不回退成绝对差以免口径混淆）。"""
    if first is None or last is None:
        return None
    if abs(first) < 1e-12:
        return None
    return (last - first) / abs(first)


def diagnose_curve(records: List[Dict], log_vocab: Optional[float] = None) -> Dict:
    """从单条 train_log 记录序列给出结构化诊断。

    records: load_train_log 的输出（含 'loss' 训练点与 'val_loss' 验证点）。
    返回 dict：status / 判定依据 / 关键数值 / 建议清单 recommendations。
    """
    train_pts, train_bad = _finite_points(records, "loss")
    val_pts, val_bad = _finite_points(records, "val_loss")
    # 累计门控/统计点上的 NaN 只在对应序列里报；任何序列出现即整体报
    nan_count = train_bad + val_bad

    first_train = _seg_mean(train_pts, 0.0, 1 / 3)
    last_train = _seg_mean(train_pts, 2 / 3, 1.0)
    first_val = _seg_mean(val_pts, 0.0, 1 / 3)
    last_val = _seg_mean(val_pts, 2 / 3, 1.0)

    train_delta = _rel_change(first_train, last_train)
    val_delta = _rel_change(first_val, last_val)
    train_down = train_delta is not None and train_delta < -REL_TOL
    train_up = train_delta is not None and train_delta > REL_TOL
    val_down = val_delta is not None and val_delta < -REL_TOL
    val_up = val_delta is not None and val_delta > REL_TOL

    best_val, best_step = None, None
    for p in val_pts:
        if best_val is None or p["value"] < best_val:
            best_val, best_step = p["value"], p["step"]
    final_val = val_pts[-1]["value"] if val_pts else None
    final_step = val_pts[-1]["step"] if val_pts else None
    early_stop_gain = _rel_change(best_val, final_val)  # >0 表示最终劣于最优

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
        elif train_down and (val_up or (early_stop_gain or 0) > EARLY_STOP_TOL):
            status = "过拟合"
            ev = [f"训练损失 {first_train:.4f}→{last_train:.4f}"
                  f"（{train_delta:+.1%}）而验证损失 "
                  f"{first_val:.4f}→{last_val:.4f}（{val_delta:+.1%}）"]
            if (early_stop_gain or 0) > EARLY_STOP_TOL:
                ev.append(f"最终验证损失劣于 step {best_step} 的最优值 "
                          f"{best_val:.4f} 达 {early_stop_gain:+.1%}")
            evidence += ev
            recommendations += [
                "加正则化（dropout / weight decay）或增大数据量",
                f"早停：val 最优在 step {best_step}，最终 step {final_step} 已过拟合"
                + (f"（早停可挽回 {early_stop_gain:.1%} 验证损失）"
                   if (early_stop_gain or 0) > 0 else ""),
                "下游对比应使用 val 上选出的 model_best.pt，而不是最终步模型"]
        elif not train_down:
            status = "欠拟合/学习不充分"
            evidence.append(
                f"训练损失 {first_train:.4f}→{last_train:.4f}（{train_delta:+.1%}），"
                "没有实质下降")
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
                    "检查数据管道（标签错位、特征污染）与 batch 组成",
                    "若容量不足：增大模型或延长训练"]
            if not val_down:
                recommendations.append("验证损失同样不降，与欠拟合判断一致")
        else:
            status = "正常收敛"
            evidence.append(
                f"训练损失 {first_train:.4f}→{last_train:.4f}（{train_delta:+.1%}），"
                f"验证损失 {first_val:.4f}→{last_val:.4f}（{val_delta:+.1%}）")
            if (early_stop_gain or 0) > EARLY_STOP_TOL:
                evidence.append(f"注意：最终验证损失仍比 step {best_step} 的最优值差 "
                                f"{early_stop_gain:.1%}，已按 best-checkpoint 口径评测")
            else:
                recommendations.append("无需干预；对比实验使用 model_best.pt")
        if (early_stop_gain or 0) > EARLY_STOP_TOL and status != "过拟合":
            recommendations.append(
                f"早停提示：最终验证损失劣于 step {best_step} 最优值 "
                f"{early_stop_gain:.1%}")

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
        "generalization_gap": gap,
        "thresholds": {"rel_tol": REL_TOL, "diverge_factor": DIVERGE_FACTOR,
                       "early_stop_tol": EARLY_STOP_TOL},
    }


# 状态严重度排序（汇总时取"最差种子"的状态代表整个变体）
STATUS_ORDER = {
    "NaN/Inf": 5, "发散": 4, "过拟合": 3,
    "欠拟合/学习不充分": 2, "数据不足": 1, "正常收敛": 0,
}


def worst_status(statuses) -> str:
    """多个种子的诊断状态里取最严重的（汇总表口径）。"""
    return max(statuses, key=lambda s: STATUS_ORDER.get(s, -1)) \
        if statuses else "数据不足"


def diagnose_run_dir(run_dir: str, log_vocab: Optional[float] = None) -> Dict:
    """诊断一个训练 run 目录（train_log.jsonl + 可选的 nan_report.json）。"""
    records = load_train_log(os.path.join(run_dir, "train_log.jsonl"))
    diag = diagnose_curve(records, log_vocab=log_vocab)
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
