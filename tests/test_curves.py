"""训练曲线诊断测试：过拟合/欠拟合/发散/NaN 的自动判定与早停空间分析。"""
import json
import math
import os
import shutil
import tempfile

from lega.curves import (diagnose_curve, diagnose_run_dir, load_train_log,
                         worst_status, STATUS_ORDER)


def _log(train, val, val_steps=None):
    """构造 train_log 记录：train=(首,末) 的指数衰减，val 同理。"""
    recs = []
    n = len(train)
    for i, v in enumerate(train):
        recs.append({"step": (i + 1) * 10, "loss": v})
    val_steps = val_steps or [(i + 1) * 50 for i in range(len(val))]
    for s, v in zip(val_steps, val):
        recs.append({"step": s, "val_loss": v, "val_ppl": math.exp(min(20, v))})
    return recs


def test_normal_convergence():
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛", d
    assert d["train_rel_change"] < -0.03 and d["val_rel_change"] < -0.03
    print("  [ok] 正常收敛判定")


def test_overfitting_detected():
    # 训练损失持续下降，验证损失先降后升 → 过拟合，且给出早停建议
    recs = _log([4.2, 3.6, 3.0, 2.4, 1.8, 1.2], [4.3, 3.6, 3.2, 3.4, 3.9])
    d = diagnose_curve(recs)
    assert d["status"] == "过拟合", d
    assert d["val_loss_best"] == 3.2 and d["val_best_step"] == 150
    assert d["early_stop_gain_rel"] > 0.02
    assert any("早停" in r for r in d["recommendations"]), d["recommendations"]
    assert any("正则化" in r for r in d["recommendations"])
    print("  [ok] 过拟合判定 + 早停/正则化建议")


def test_underfitting_and_random_level():
    # 训练损失不降且停在 log(V) 随机水平附近 → 欠拟合，指向数据管道/学习率
    lv = math.log(64)
    recs = _log([lv + 0.01, lv, lv - 0.005, lv + 0.004, lv - 0.002, lv],
                [lv, lv - 0.004, lv + 0.003, lv - 0.002, lv + 0.001])
    d = diagnose_curve(recs, log_vocab=lv)
    assert d["status"] == "欠拟合/学习不充分", d
    assert any("数据管道" in r for r in d["recommendations"])
    assert any("学习率" in r for r in d["recommendations"])
    # 远离随机水平的停滞同样是欠拟合（不触发随机水平证据行）
    d2 = diagnose_curve(_log([2.0, 1.98, 2.01, 1.99, 2.0, 1.98], [2.1, 2.05, 2.08, 2.02, 2.04]))
    assert d2["status"] == "欠拟合/学习不充分", d2
    print("  [ok] 欠拟合判定（含随机猜测水平识别）")


def test_divergence_and_nan():
    # 训练损失越训越大 → 发散
    d = diagnose_curve(_log([4.0, 4.5, 5.5, 7.0, 9.5, 13.0], [4.1, 4.6, 5.4, 6.8, 9.0]))
    assert d["status"] == "发散", d
    assert any("学习率" in r for r in d["recommendations"])
    # 任何序列出现 NaN/Inf → NaN/Inf 状态 + 排查路径建议
    recs = _log([4.2, 3.5, 3.0], [4.3, 3.2, 2.9])
    recs.append({"step": 40, "loss": float("nan")})
    recs.append({"step": 50, "val_loss": float("inf")})
    d = diagnose_curve(recs)
    assert d["status"] == "NaN/Inf", d
    assert d["nonfinite_values"] == 2
    assert any("nan_report" in r for r in d["recommendations"])
    print("  [ok] 发散与 NaN 判定")


def test_grad_norm_spike_and_nonfinite_reported():
    """梯度爆炸的早期信号（尖峰/非有限范数）必须报告，即使损失曲线看似正常。"""
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    for i, g in enumerate([1.0, 1.1, 1.0, 1.05, 100.0]):
        recs.append({"step": (i + 1) * 10, "grad_norm": g})
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛"
    assert d["has_grad_spike"] and d["grad_spike_ratio"] >= 10
    assert any("尖峰" in e for e in d["evidence"]), d["evidence"]
    assert any("梯度" in r for r in d["recommendations"]), d["recommendations"]
    # 非有限梯度范数（Infinity 写入 jsonl）：即使没有损失 NaN 也要判爆炸
    recs2 = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    recs2.append({"step": 10, "grad_norm": float("inf")})
    d2 = diagnose_curve(recs2)
    assert d2["status"] == "正常收敛" and d2["grad_norm_nonfinite"] == 1
    assert d2["has_grad_spike"]
    assert any("非有限" in e for e in d2["evidence"]), d2["evidence"]
    # 旧日志无 grad_norm：字段可缺省且不误报
    d3 = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                             [4.3, 3.8, 3.3, 2.9, 2.6]))
    assert d3["grad_norm_points"] == 0 and not d3["has_grad_spike"]
    print("  [ok] 梯度范数尖峰/非有限检测（与损失状态独立）")


def test_short_log_is_descriptive_only():
    d = diagnose_curve([{"step": 1, "loss": 4.2}])
    assert d["status"] == "数据不足"
    assert d["recommendations"]
    print("  [ok] 曲线点过少时只作描述统计")


def test_degenerate_segments_render_without_crash():
    """同一 step 的重复点会让区间均值为 None：证据行必须显示 '-' 而非崩溃。"""
    # 训练/验证点都挤在同一 step：区间均值与相对变化均为 None → 欠拟合分支
    recs = [{"step": 10, "loss": 4.0}, {"step": 10, "loss": 3.9},
            {"step": 10, "loss": 3.8},
            {"step": 100, "val_loss": 3.4}, {"step": 100, "val_loss": 3.6}]
    d = diagnose_curve(recs)
    assert d["status"] == "欠拟合/学习不充分", d
    assert d["train_rel_change"] is None
    assert any("-" in e for e in d["evidence"])
    # 训练下降 + val 区间均值为 None，但早停增益显著 → 过拟合分支同样不得崩
    recs2 = [{"step": 10, "loss": 4.0}, {"step": 20, "loss": 3.0},
             {"step": 30, "loss": 2.0},
             {"step": 100, "val_loss": 3.4}, {"step": 100, "val_loss": 3.6}]
    d2 = diagnose_curve(recs2)
    assert d2["status"] == "过拟合", d2
    print("  [ok] 退化区间（None 均值/变化率）诊断不再崩溃")


def test_worst_status_ordering():
    assert worst_status(["正常收敛", "过拟合"]) == "过拟合"
    assert worst_status(["过拟合", "NaN/Inf"]) == "NaN/Inf"
    assert worst_status(["欠拟合/学习不充分", "发散"]) == "发散"
    assert worst_status([]) == "数据不足"
    assert STATUS_ORDER["NaN/Inf"] > STATUS_ORDER["发散"] > STATUS_ORDER["过拟合"]
    print("  [ok] worst_status 严重度排序")


def test_diagnose_run_dir_reads_log_and_nan_report():
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "train_log.jsonl"), "w", encoding="utf-8") as f:
            for s, v in ((10, 4.2), (20, 3.9), (30, 3.5)):
                f.write(json.dumps({"step": s, "loss": v}) + "\n")
            f.write(json.dumps({"step": 30, "val_loss": 3.6}) + "\n")
            f.write(json.dumps({"step": 20, "val_loss": 3.8}) + "\n")
        with open(os.path.join(tmp, "nan_report.json"), "w", encoding="utf-8") as f:
            json.dump({"step": 30, "where": "train_loss"}, f)
        d = diagnose_run_dir(tmp)
        assert d["nan_report"]["where"] == "train_loss"
        assert len(load_train_log(os.path.join(tmp, "train_log.jsonl"))) == 5
        # 损坏行被容忍，不阻塞诊断
        with open(os.path.join(tmp, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write("{broken\n")
        assert len(load_train_log(os.path.join(tmp, "train_log.jsonl"))) == 5
        # 缺失目录不抛异常
        assert diagnose_run_dir(os.path.join(tmp, "nonexistent"))["status"] == "数据不足"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] diagnose_run_dir（日志/nan_report/容错）")


def test_rising_train_loss_not_advised_to_train_longer():
    """训练损失缓慢上升（未达发散阈值）时不能给出"延长训练/增大模型"
    这类反向建议：上升更可能来自学习率过大，延长训练会继续恶化。"""
    d = diagnose_curve(_log([3.0, 3.05, 3.1, 3.15, 3.2, 3.3],
                            [3.1, 3.2, 3.3, 3.4, 3.5]))
    assert d["status"] == "欠拟合/学习不充分", d
    assert any("不降反升" in e for e in d["evidence"]), d["evidence"]
    assert any("降低学习率" in r for r in d["recommendations"]), d["recommendations"]
    assert not any("延长训练" in r for r in d["recommendations"]), d["recommendations"]
    print("  [ok] 上升训练损失不误报「延长训练」建议")
