"""训练曲线诊断测试：过拟合/欠拟合/发散/NaN 的自动判定与早停空间分析。"""
import json
import math

import numpy as np
import os
import shutil
import tempfile

from lega.curves import (LIVE_WARN_EVALS, _finite_points, _step_or_none,
                         _val_ci_rel, curve_observability_warnings, diagnose_curve,
                         diagnose_run_dir, live_overfit_hint, load_train_log,
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


def test_nonfinite_values_with_bad_steps_still_fail_curve():
    """坏 step 上的 NaN 损失/Inf 梯度仍是训练失效证据，不能被当 malformed 吞掉。"""
    base = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                [4.3, 3.8, 3.3, 2.9, 2.6])
    d = diagnose_curve(base + [{"step": "bad", "loss": float("nan")}])
    assert d["status"] == "NaN/Inf", d
    assert d["nonfinite_values"] >= 1 and d["malformed_records"] >= 1, d
    d2 = diagnose_curve(base + [{"step": "bad", "grad_norm": float("inf")}])
    assert d2["grad_norm_nonfinite"] >= 1, d2
    assert d2["has_grad_spike"] is True and d2["malformed_records"] >= 1, d2
    print("  [ok] 坏 step 上的 NaN 损失/Inf 梯度仍触发失效诊断")


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


def test_curve_diagnosis_respects_validation_noise():
    """验证回升若落在 val_loss_ci95 内，不得判成过拟合：有限窗口的评估
    噪声不是泛化信号；无 CI 的旧日志保持原判定。"""
    recs = _log([4.2, 3.6, 3.0, 2.4, 1.8, 1.2], [3.0, 3.2, 3.5])
    # 每个验证点 ±20% 的 95% CI：+16.7% 的“回升”完全在噪声内
    for r in recs:
        if "val_loss" in r:
            v = r["val_loss"]
            r["val_loss_ci95"] = [v * 0.8, v * 1.2]
    d = diagnose_curve(recs)
    assert d["status"] != "过拟合", d
    assert d["val_ci_rel"] is not None and d["val_ci_rel"] > 0.15
    assert d["val_change_above_noise"] is False
    assert d["early_stop_gain_above_noise"] is False
    assert any("评估噪声" in e for e in d["evidence"]), d["evidence"]
    assert not any("早停可挽回" in r for r in d["recommendations"]), d["recommendations"]
    # 同样的曲线但没有 CI（旧日志）沿用原阈值，仍能识别过拟合
    plain = _log([4.2, 3.6, 3.0, 2.4, 1.8, 1.2], [3.0, 3.2, 3.5])
    d2 = diagnose_curve(plain)
    assert d2["status"] == "过拟合", d2
    print("  [ok] 曲线诊断扣除验证评估噪声（无 CI 时兼容旧判定）")


def test_overfitting_trend_train_down_val_stalled():
    """训练仍在实质下降而验证无实质变化：过拟合的早期形态，必须给出
    正则化/早停建议，且不得建议延长训练。"""
    d = diagnose_curve(_log([4.2, 3.6, 3.0, 2.4, 1.8, 1.2],
                            [4.3, 4.28, 4.31, 4.27, 4.3]))
    assert d["status"] == "过拟合趋势", d
    assert any("验证不再随训练改善" in e for e in d["evidence"]), d["evidence"]
    assert any("正则化" in r for r in d["recommendations"]), d["recommendations"]
    assert not any("延长训练" in r and "不要" not in r
                   for r in d["recommendations"]), d["recommendations"]
    # 训练与验证同步下降仍是正常收敛，不得误报趋势
    d2 = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                             [4.3, 3.8, 3.3, 2.9, 2.6]))
    assert d2["status"] == "正常收敛", d2
    # 训练停滞（不降）而验证也未降：仍按欠拟合处理，不是过拟合趋势
    d3 = diagnose_curve(_log([2.0, 1.98, 2.01, 1.99, 2.0, 1.98],
                             [2.0, 2.01, 2.005, 2.02, 2.01]))
    assert d3["status"] == "欠拟合/学习不充分", d3
    print("  [ok] 过拟合趋势（训练降/验证停滞）判定与建议")


def test_plateau_train_rising_val_is_overfitting():
    """训练停滞（平台期）而验证回升同样是过拟合，不能误判为欠拟合并建议延长训练。"""
    d = diagnose_curve(_log([2.0, 1.98, 2.01, 1.99, 2.0, 1.98],
                            [3.1, 3.3, 3.6, 4.0, 4.5]))
    assert d["status"] == "过拟合", d
    assert any("停滞" in e for e in d["evidence"]), d["evidence"]
    assert any("早停" in r for r in d["recommendations"])
    assert not any("延长训练" in r for r in d["recommendations"]), d["recommendations"]
    assert not any("增大模型" in r for r in d["recommendations"]), d["recommendations"]
    # 训练不降、验证仅轻微波动（未超噪声阈值）时仍按欠拟合描述，但不得
    # 出现"验证损失同样不降"的错误论断
    d2 = diagnose_curve(_log([2.0, 1.98, 2.01, 1.99, 2.0, 1.98],
                             [2.0, 2.01, 2.005, 2.02, 2.01]))
    assert d2["status"] == "欠拟合/学习不充分", d2
    assert not any("同样不降" in r for r in d2["recommendations"]), \
        d2["recommendations"]
    # 训练平台期而验证显著下降：更不能说"验证损失同样不降"，并应给出
    # 先自检训练链路的指引（而不是断言停滞或直接建议扩大容量）
    d3 = diagnose_curve(_log([2.0, 1.98, 2.01, 1.99, 2.0, 1.98],
                             [4.0, 3.5, 3.0, 2.6, 2.3]))
    assert d3["status"] == "欠拟合/学习不充分", d3
    assert not any("同样不降" in r for r in d3["recommendations"]), \
        d3["recommendations"]
    assert any("验证损失仍在下降" in r for r in d3["recommendations"]), \
        d3["recommendations"]
    print("  [ok] 平台期验证回升 → 过拟合（不误报欠拟合/延长训练）")


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
    # 发散状态下验证仍低于训练：不能把倒挂信号藏起来（状态保留“发散”，
    # 但证据/建议里要并列提示先排查泄漏/口径）。
    assert d["val_below_train"] is True and any(
        "倒挂" in e for e in d["evidence"]), d["evidence"]
    assert any("倒挂" in r for r in d["recommendations"]), d["recommendations"]
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
    assert not any("无需干预" in r for r in d["recommendations"]), \
        "已有梯度尖峰告警时不得同时建议无需干预"
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
    """同一 step 的重复点会让区间均值为 None：证据行必须显示 '-'。"""
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
    print("  [ok] 退化区间（None 均值/变化率）诊断可正常完成")


def test_transient_loss_spike_detected_even_when_trend_normal():
    """瞬态损失尖峰不能被分段均值掩盖：趋势正常也要单项报告，且不应再给
    “无需干预”的建议（尖峰常对应异常批次/一次梯度爆炸）。"""
    train = [5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 30.0, 3.4, 3.2,
             3.0, 2.8, 2.6, 2.4, 2.2]
    recs = _log(train, [5.0, 4.6, 4.2, 3.8, 3.4],
                val_steps=[90, 180, 270, 360, 450])
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛", d
    assert d["has_loss_spike"] and d["loss_spikes"], d
    assert d["loss_spikes"][0]["step"] == 80
    assert any("尖峰" in e for e in d["evidence"]), d["evidence"]
    assert any("尖峰" in r for r in d["recommendations"]), d["recommendations"]
    assert not any("无需干预" in r for r in d["recommendations"]), d["recommendations"]
    # 平滑下降（无尖峰）时不得误报，且保留“无需干预”
    d2 = diagnose_curve(_log(train[:7] + train[8:], [5.0, 4.6, 4.2, 3.8, 3.4],
                             val_steps=[90, 180, 270, 360, 450]))
    assert d2["status"] == "正常收敛" and not d2["has_loss_spike"], d2
    assert any("无需干预" in r for r in d2["recommendations"])
    print("  [ok] 瞬态损失尖峰检测（与趋势独立、平滑曲线不误报）")


def test_loss_spike_in_short_curve_and_final_point_detected():
    """短曲线（10 点）与训练最后一步的尖峰都不能因窗口边界被漏检。"""
    # 10 个训练点：旧实现 range(5, 5) 为空，任何尖峰都检测不到
    train10 = [5.0, 4.8, 4.6, 40.0, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2]
    d = diagnose_curve(_log(train10, [5.0, 4.6, 4.2, 3.8, 3.4]))
    assert d["has_loss_spike"] and any(
        s["step"] == 40 for s in d["loss_spikes"]), d["loss_spikes"]
    assert not any("无需干预" in r for r in d["recommendations"])
    # 20 个训练点 + 最后一个点尖峰：旧实现 range(5, 15) 永远扫不到末尾
    train20 = [5.0 - 0.15 * i for i in range(20)]
    train20[-1] = 30.0
    d2 = diagnose_curve(_log(train20, [5.0, 4.6, 4.2, 3.8, 3.4],
                             val_steps=[100, 200, 300, 400, 500]))
    assert d2["has_loss_spike"] and any(
        s["step"] == 200 for s in d2["loss_spikes"]), d2["loss_spikes"]
    print("  [ok] 短曲线/末尾训练损失尖峰均被检出")


def test_malformed_log_records_do_not_crash_diagnosis():
    """损坏记录（字符串值/非法 step）必须被跳过并计数，不能中断整个诊断。"""
    recs = [{"step": 10, "loss": 5.0}, {"step": 20, "loss": "oops"},
            {"step": "x", "loss": 4.0}, {"step": 30, "loss": 3.0},
            {"step": 40, "loss": 2.0}, {"step": 50, "loss": 1.5},
            {"step": "1_0", "loss": 4.0}, {"step": 60, "loss": "1_0"},
            {"step": 10, "val_loss": 5.0}, {"step": 30, "val_loss": 3.2},
            {"step": 50, "val_loss": 2.0}, {"step": 60, "val_loss": "bad"},
            {"step": 70, "val_loss": "1_0"}]
    d = diagnose_curve(recs)
    assert d["status"] in ("正常收敛", "过拟合趋势", "欠拟合/学习不充分"), d
    # "1_0" 若经 int()/float() 会变成第 10 步/损失 10.0，与真实值混淆。
    assert d["malformed_records"] == 6, d
    assert any("无法解析" in e for e in d["evidence"]), d["evidence"]
    print("  [ok] 损坏日志记录跳过并计数（诊断正常完成）")


def test_infinite_step_records_do_not_crash_diagnosis():
    """step 为 JSON Infinity（json.loads 接受该非标准字面量）时必须按坏记录
    跳过：int(inf) 抛 OverflowError，不能中断整个诊断。"""
    recs = [json.loads('{"step": 10, "loss": 5.0}'),
            json.loads('{"step": Infinity, "loss": 4.0}'),
            json.loads('{"step": 20, "loss": 3.5}'),
            json.loads('{"step": 30, "loss": 3.0}'),
            json.loads('{"step": 40, "loss": 2.0}'),
            json.loads('{"step": 10, "val_loss": 5.0}'),
            json.loads('{"step": 20, "val_loss": 4.0}'),
            json.loads('{"step": 30, "val_loss": 3.0}')]
    d = diagnose_curve(recs)
    assert d["malformed_records"] == 1, d
    assert d["status"] == "正常收敛", d
    print("  [ok] step=Infinity 的记录跳过并计数（OverflowError 不中断诊断）")


def test_huge_step_records_do_not_crash_diagnosis():
    """超出 float 范围的巨大整数 step 也必须按坏记录跳过：否则分段均值
    的 step 算术会在 float 转换时 OverflowError。"""
    recs = [{"step": 10, "loss": 5.0}, {"step": 20, "loss": 3.5},
            {"step": 30, "loss": 3.0}, {"step": 40, "loss": 2.0},
            {"step": 10 ** 400, "loss": 1.0},
            {"step": 10, "val_loss": 5.0}, {"step": 20, "val_loss": 4.0},
            {"step": 30, "val_loss": 3.0},
            {"step": 10 ** 400, "val_loss": 1.0}]
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛", d
    assert d["malformed_records"] == 2, d
    print("  [ok] 超出 float 范围的巨大 step 跳过并计数（诊断不中断）")


def test_huge_early_stop_step_does_not_crash_diagnosis():
    """early_stop 记录的 step 是合法超大整数时 float(step) 会 OverflowError；
    该标记应被忽略并回退到最后一个有限训练点。"""
    recs = [{"step": 10, "loss": 5.0}, {"step": 20, "loss": 3.5},
            {"step": 30, "loss": 3.0}, {"step": 40, "loss": 2.0},
            {"step": 10, "val_loss": 5.0}, {"step": 20, "val_loss": 4.0},
            {"step": 30, "val_loss": 3.0},
            {"step": 10 ** 400, "early_stop": True}]
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛", d
    assert d["early_stop_step"] == 40, d
    print("  [ok] early_stop 的超大整数 step 不中断诊断")


def test_fractional_and_boolean_steps_are_malformed():
    """小数/布尔 step 不能被 int() 截断后参与诊断：1.5→1、True→1 会把
    不同记录折叠到同一步，曲线与早停标记随之错位；按坏记录跳过并计数。"""
    recs = [{"step": 1, "loss": 4.0},
            {"step": 1.5, "loss": 3.9},
            {"step": True, "loss": 3.8},
            {"step": 2.0, "loss": 3.5},      # 整数值浮点仍是合法 step
            {"step": "3", "loss": 3.0},
            {"step": 10 ** 400, "loss": 2.5}]
    pts, bad, malformed = _finite_points(recs, "loss")
    assert [p["step"] for p in pts] == [1, 2, 3], pts
    assert bad == 0 and malformed == 3, (bad, malformed)

    # 小数的 early_stop step 不可采信：回退到最后一个有限训练点，而不是截断
    recs2 = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    recs2.append({"step": 45.5, "early_stop": True})
    d = diagnose_curve(recs2)
    assert d["early_stopped"] is True
    assert d["early_stop_step"] == 60, d

    # 非布尔真值不能伪造早停；只有 JSON 布尔 true 才算。
    for bad in ("false", "0", 1, np.bool_(True)):
        recs3 = _log([4.2, 3.9, 3.5], [4.3, 3.8, 3.3])
        recs3.append({"step": 40, "early_stop": bad})
        d = diagnose_curve(recs3)
        assert d["early_stopped"] is False, (bad, d["early_stopped"])
    print("  [ok] 小数/布尔 step 按坏记录跳过（不截断）；非布尔早停标记不采信")


def test_boolean_log_values_are_malformed_not_numeric():
    """loss/grad_norm/val_loss/CI 里的 bool 是坏数据，不能当成 1.0/0.0。"""
    recs = [{"step": 10, "loss": True}, {"step": 20, "loss": 2.0}]
    pts, bad, malformed = _finite_points(recs, "loss")
    assert pts == [{"step": 20, "value": 2.0}]
    assert bad == 0 and malformed == 1, (bad, malformed)
    assert _val_ci_rel([{"val_loss": 3.0,
                         "val_loss_ci95": [True, 4.0]}]) is None
    assert abs(_val_ci_rel([{"val_loss": 3.0,
                             "val_loss_ci95": [2.0, 4.0]}]) - 1 / 3) < 1e-12
    assert live_overfit_hint(5, True, 4.0) is None
    d = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                            [4.3, 3.8, 3.3, 2.9, 2.6])
                       + [{"step": 55, "loss": True}])
    assert d["malformed_records"] == 1, d
    assert d["status"] != "发散", d
    print("  [ok] bool 曲线值按坏记录跳过（不伪造损失/区间/过拟合提示）")


def test_insufficient_train_coverage_is_not_normal_convergence():
    """训练/验证都在降但只见过 <1 epoch：必须判“训练不足”，不得写无需干预。"""
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    d = diagnose_curve(recs, epochs_seen=0.03, train_tokens=1000)
    assert d["status"] == "训练不足（数据覆盖不足）", d
    assert any("epoch" in e for e in d["evidence"]), d["evidence"]
    assert not any("无需干预" in r for r in d["recommendations"]), d
    assert any("继续训练" in r or "覆盖不足" in r for r in d["recommendations"])
    assert d["epochs_seen"] == 0.03 and d["train_budget_sufficient"] is False
    # 覆盖足够时仍是正常收敛；缺少 coverage 信息（旧结果）时也不误报
    d2 = diagnose_curve(recs, epochs_seen=2.0)
    assert d2["status"] == "正常收敛" and d2["train_budget_sufficient"] is True
    d3 = diagnose_curve(recs)
    assert d3["status"] == "正常收敛"
    assert d3["epochs_seen"] is None and d3["train_budget_sufficient"] is None
    # 更严重的状态（过拟合）不被覆盖率信息覆盖
    d4 = diagnose_curve(_log([4.2, 3.6, 3.0, 2.4, 1.8, 1.2],
                             [4.3, 3.6, 3.2, 3.4, 3.9]), epochs_seen=0.1)
    assert d4["status"] == "过拟合"
    assert any("epoch" in e for e in d4["evidence"]), d4["evidence"]
    print("  [ok] 训练覆盖 <1 epoch → 训练不足（不误写无需干预）")


def test_diagnose_curve_tolerates_bad_coverage_metadata():
    """外部 eval.json 的坏覆盖字段不能中断诊断或伪造训练不足。"""
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    for bad in ("bad", True, float("nan"), float("inf"), -0.5):
        d = diagnose_curve(recs, epochs_seen=bad, train_tokens=bad,
                           log_vocab=bad)
        assert d["status"] == "正常收敛", (bad, d)
        assert d["epochs_seen"] is None and d["train_tokens"] is None, (bad, d)
        assert d["train_budget_sufficient"] is None, (bad, d)
    d = diagnose_curve(recs, epochs_seen=0.5, train_tokens=3.5)
    assert d["status"] == "训练不足（数据覆盖不足）", d
    assert d["train_tokens"] is None and d["epochs_seen"] == 0.5, d
    print("  [ok] 曲线诊断拒绝 bool/非法覆盖元数据，不崩溃也不误报")


def test_huge_finite_losses_keep_curve_stats_finite():
    """合法但接近 float 上限的损失点不能让分段均值/趋势变成 inf。"""
    recs = []
    for i in range(6):
        recs.append({"step": (i + 1) * 10, "loss": 1e308})
    for i in range(3):
        recs.append({"step": (i + 1) * 50, "val_loss": 1e308})
    d = diagnose_curve(recs)
    assert d["train_loss_first"] == 1e308, d
    assert d["train_loss_last"] == 1e308, d
    assert d["val_loss_first"] == 1e308, d
    assert d["val_loss_last"] == 1e308, d
    for key in ("train_rel_change", "val_rel_change",
                "train_recent_rel_change", "val_recent_rel_change"):
        assert d[key] is None or math.isfinite(d[key]), (key, d[key])
    print("  [ok] 巨大有限损失：分段均值/趋势保持有限或显式缺失")


def test_worst_status_ordering():
    assert worst_status(["正常收敛", "过拟合"]) == "过拟合"
    assert worst_status(["过拟合", "NaN/Inf"]) == "NaN/Inf"
    assert worst_status(["欠拟合/学习不充分", "发散"]) == "发散"
    assert worst_status(["正常收敛", "过拟合趋势"]) == "过拟合趋势"
    assert worst_status(["过拟合趋势", "过拟合"]) == "过拟合"
    assert worst_status([]) == "数据不足"
    assert worst_status(["训练不足（数据覆盖不足）", "正常收敛"]) \
        == "训练不足（数据覆盖不足）"
    assert (STATUS_ORDER["NaN/Inf"] > STATUS_ORDER["发散"]
            > STATUS_ORDER["过拟合"] > STATUS_ORDER["欠拟合/学习不充分"]
            > STATUS_ORDER["训练不足（数据覆盖不足）"]
            > STATUS_ORDER["过拟合趋势"] > STATUS_ORDER["数据不足"]
            > STATUS_ORDER["正常收敛"])
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
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"epochs_seen": 0.05, "train_tokens": 1000,
                       "val_ppl_ci95": [1.0, 2.0],
                       "curve_observability_warnings": ["验证点预计仅 2 个"]}, f)
        d = diagnose_run_dir(tmp)
        assert d["curve_observability_warnings"] == ["验证点预计仅 2 个"]
        assert d["nan_report"]["where"] == "train_loss"
        # eval.json 的训练数据覆盖率必须被曲线诊断读取并用于“训练不足”判定
        assert d["epochs_seen"] == 0.05 and d["train_tokens"] == 1000
        assert d["train_budget_sufficient"] is False
        assert d["status"] == "训练不足（数据覆盖不足）", d
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


def test_load_train_log_skips_valid_json_non_object_lines():
    """合法 JSON 但非对象的行（数组/标量/null）必须按损坏行跳过。"""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "train_log.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]\n")          # 合法 JSON 数组（无 .get）
            f.write("42\n")                  # 合法 JSON 标量
            f.write("null\n")
            f.write('{"step": 1, "loss": 2.0}\n')
            f.write("{broken\n")
        assert load_train_log(p) == [{"step": 1, "loss": 2.0}]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] load_train_log：非对象 JSON 行按损坏行跳过")


def test_load_train_log_tolerates_invalid_utf8():
    """日志被中断留下非法 UTF-8 字节时跳过坏行，不能让诊断崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "train_log.jsonl")
        with open(p, "wb") as f:
            f.write(b'{"step": 1, "loss": 2.0}\n')
            f.write(b'{"step": 2, "loss": \xff}\n')
            f.write(b'{"step": 3, "loss": 1.5}\n')
        assert load_train_log(p) == [
            {"step": 1, "loss": 2.0}, {"step": 3, "loss": 1.5}]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] load_train_log：非法 UTF-8 字节按坏行跳过")


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


def test_live_overfit_hint():
    """训练中途预警：连续 N 次无改善才提示；早停开启时不重复提示。"""
    assert live_overfit_hint(0, 2.0, 2.1, patience=0) is None
    assert live_overfit_hint(LIVE_WARN_EVALS - 1, 2.0, 2.5, patience=0) is None
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.5, patience=0)
    assert msg and "回升" in msg and "过拟合" in msg
    assert "2 次" not in msg and f"{LIVE_WARN_EVALS} 次" in msg
    plateau = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.0, patience=0)
    assert plateau and "平台" in plateau
    # 早停开启 → 预警交给早停逻辑，不重复提示
    assert live_overfit_hint(5, 2.0, 2.5, patience=2) is None
    # 非有限/异常输入不炸
    assert live_overfit_hint(LIVE_WARN_EVALS, 2.0, float("nan"), patience=0) is None
    assert live_overfit_hint(LIVE_WARN_EVALS, None, 2.0, patience=0) is None
    assert live_overfit_hint("bad", 2.0, 2.0, patience=0) is None

    # 结合训练侧趋势：验证回升但训练仍在降 = 典型过拟合
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.2,
                            train_recent=[3.0, 2.9, 2.8])
    assert "训练损失仍在下降" in msg and "典型过拟合" in msg, msg
    # 验证回升且训练也上升：更像优化不稳定/学习率过大，不能建议加正则化
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.2,
                            train_recent=[2.8, 2.9, 3.0])
    assert "学习率" in msg and "不是典型过拟合" in msg, msg
    assert "加正则化" not in msg.split("：")[-1] or "不要盲目加正则化" in msg, msg
    # 验证平台 + 训练仍在降：早期过拟合，而不是欠拟合
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.0,
                            train_recent=[3.0, 2.9, 2.8])
    assert "早期过拟合" in msg, msg
    # 两边都停滞：更像欠拟合/优化不足，建议检查学习率而不是正则化
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.0,
                            train_recent=[2.8, 2.8, 2.8])
    assert "欠拟合" in msg and "不要盲目加正则化" in msg, msg
    # 回升幅度落在验证抽样噪声内：只要求继续观察，不判过拟合
    msg = live_overfit_hint(LIVE_WARN_EVALS, 2.0, 2.01,
                            train_recent=[3.0, 2.9, 2.8],
                            val_ci95=[1.99, 2.03])
    assert "抽样噪声" in msg and "不要据此加正则化" in msg, msg
    print("  [ok] live_overfit_hint（训练趋势区分过拟合/欠拟合 + 验证噪声）")


def test_diagnose_curve_tolerates_non_dict_records():
    """日志含非对象行（旧格式/手改）时按坏记录跳过，不抛属性错误。"""
    d = diagnose_curve([1, None, "x", {"step": 1, "loss": 2.0}])
    assert d["status"] == "数据不足"
    assert diagnose_curve(None)["status"] == "数据不足"
    print("  [ok] diagnose_curve：非对象记录安全跳过")


def test_curve_observability_warnings():
    """验证/日志点过少时启动即告警，并给出可直接套用的间隔建议。"""
    # 充足：20 步、每 4 步验证（5 个点）、每 2 步日志（11 个点）
    assert curve_observability_warnings(20, 4, 2) == []
    # 验证点：steps=100/eval_every=50 -> 2 个；100/20 -> 5 个（刚好下限）
    w = curve_observability_warnings(100, 50, 10)
    assert len(w) == 1 and "验证点" in w[0] and "eval_every<=20" in w[0], w
    # 末尾补一次最终评测：101 步时 50 的间隔得到 3 个点（仍不足）
    assert curve_observability_warnings(100, 20, 10) == []
    w = curve_observability_warnings(101, 50, 10)
    assert any("3 个" in x for x in w), w
    # 日志点：steps=20/log_every=10 -> 1+2=3 个（验证密度另算）
    w = curve_observability_warnings(20, 4, 10)
    assert any("日志" in x and "log_every<=2" in x for x in w), w
    # 非法间隔/步数不抛异常
    assert curve_observability_warnings(0, 5, 5)
    assert curve_observability_warnings("bad", 5, 5)
    w = curve_observability_warnings(100, 0, 10)
    assert any("eval_every=0" in x for x in w), w
    # bool/小数不能被截断成另一个采样间隔（True→1、7.5→7）
    for bad in ((True, 4, 2), (20, 7.5, 2), (20, 4, 2.5),
                (20, False, 2), (20, 4, float("nan")),
                ("1_0", 4, 2)):
        w = curve_observability_warnings(*bad)
        assert len(w) == 1 and "不是整数" in w[0], (bad, w)
    assert curve_observability_warnings(20.0, 4.0, 2.0) == []
    print("  [ok] 曲线可观测性：稀疏验证/日志点告警，bool/小数拒绝")


def test_gate_health_distinguishes_collapse_saturation_and_constant():
    """只看 gate_mean 不够：g→0（关闭）与 g→1（透明）都让机制失效，
    std≈0 则门控没有样本级区分；这些必须作为独立信号进入曲线诊断。"""
    base = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    low = base + [{"step": 50, "gate_mean": 0.01, "gate_std": 0.003,
                   "gate_frac_low": 0.99, "gate_frac_high": 0.0,
                   "gate_layer_mean": [0.01, 0.012]}]
    d = diagnose_curve(low)
    gh = d["gate_health"]
    assert gh["available"] and "塌缩" in gh["status"], gh
    assert any("机制未生效" in r or "不能归因" in r
               for r in gh["recommendations"]), gh
    assert not any("无需干预" in r for r in d["recommendations"]), d

    high = base + [{"step": 50, "gate_mean": 0.995, "gate_std": 0.001,
                    "gate_frac_low": 0.0, "gate_frac_high": 0.99}]
    d = diagnose_curve(high)
    assert "饱和" in d["gate_health"]["status"], d["gate_health"]
    assert any("退化为标准注意力" in e for e in d["evidence"]), d["evidence"]

    const = base + [{"step": 50, "gate_mean": 0.5, "gate_std": 0.001,
                     "gate_frac_low": 0.0, "gate_frac_high": 0.0}]
    d = diagnose_curve(const)
    assert "常数" in d["gate_health"]["status"], d["gate_health"]
    assert any("样本级" in e for e in d["evidence"]), d["evidence"]

    healthy = base + [{"step": 50, "gate_mean": 0.7, "gate_std": 0.15,
                       "gate_frac_low": 0.02, "gate_frac_high": 0.03}]
    d = diagnose_curve(healthy)
    assert d["gate_health"]["status"] == "健康", d["gate_health"]
    assert not d["gate_health"]["recommendations"]
    assert "无需干预" in " ".join(d["recommendations"])

    # 只有 gate_mean 无法证明样本级变化，不能判健康。
    mean_only = base + [{"step": 50, "gate_mean": 0.5}]
    d = diagnose_curve(mean_only)
    gh = d["gate_health"]
    assert gh["status"] != "健康" and "数据不足" in gh["status"], gh
    assert not any("样本级变化，未贴近" in e for e in gh["evidence"]), gh
    print("  [ok] 门控健康：塌缩/饱和/常数门与健康门控区分；仅均值判数据不足")


def test_gate_health_flags_drift_and_ignores_malformed_stats():
    """门控均值在评测间剧烈变化时，不能只用最终均值解释机制；坏统计值
    必须被跳过，不能让曲线诊断崩溃或伪造门控状态。"""
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    recs += [{"step": 20, "gate_mean": 0.8, "gate_std": 0.1},
             {"step": 40, "gate_mean": 0.35, "gate_std": 0.1},
             {"step": 60, "gate_mean": "bad", "gate_std": True,
              "gate_frac_low": float("nan")}]
    d = diagnose_curve(recs)
    gh = d["gate_health"]
    assert gh["available"] and "漂移" in gh["status"], gh
    assert gh["n_points"] == 2, gh
    assert gh["drift_max"] >= 0.25, gh
    assert any("最终" in e or "过程" in e for e in gh["evidence"]), gh
    # 无门控统计（standard/null 或旧日志）时明确标为不适用，且不误报
    d2 = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                             [4.3, 3.8, 3.3, 2.9, 2.6]))
    assert not d2["gate_health"]["available"], d2["gate_health"]
    assert not d2["gate_health"]["recommendations"]
    print("  [ok] 门控漂移检测与坏统计值跳过")


def test_validation_below_train_flags_leakage_or_mismatch():
    """验证损失显著低于训练损失（倒挂）不是收敛，而是泄漏/口径不一致信号。

    只看两条曲线各自下降会漏掉相对水平；倒挂必须进入状态与建议，且不得
    再写“无需干预”。
    """
    d = diagnose_curve(_log([4.2, 3.9, 3.6, 3.3, 3.0, 2.8],
                            [3.5, 3.2, 2.9, 2.6, 2.4]))
    assert d["status"] == "验证损失倒挂（疑似泄漏/口径不一致）", d
    assert d["val_below_train"] is True and d["val_below_train_rel"] < -0.03, d
    assert any("泄漏" in e for e in d["evidence"]), d["evidence"]
    assert any("泄漏" in r or "协议" in r for r in d["recommendations"]), d
    assert not any("无需干预" in r for r in d["recommendations"]), d

    # 正常的“验证略高”与小幅波动不得误报
    d2 = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                             [4.3, 3.8, 3.3, 2.9, 2.6]))
    assert d2["status"] == "正常收敛" and d2["val_below_train"] is False, d2
    assert any("无需干预" in r for r in d2["recommendations"]), d2
    # 倒挂信号与曲线形状独立：训练仍下降、验证停滞（本会被判为过拟合趋势）
    # 时也必须升级为泄漏/口径排查，不能给出“加正则化”的错误建议。
    d3 = diagnose_curve(_log(
        [4.2, 3.9, 3.6, 3.3, 3.0, 2.7, 2.5, 2.35, 2.2, 2.1],
        [2.0, 2.01, 1.99, 2.0, 2.02, 2.0, 2.01, 2.0, 1.99, 2.0]))
    assert d3["val_below_train"] is True, d3
    assert d3["status"] == "验证损失倒挂（疑似泄漏/口径不一致）", d3
    assert any("泄漏" in e for e in d3["evidence"]), d3["evidence"]
    assert not any("正则化" in r or "早停" in r
                   for r in d3["recommendations"]), d3["recommendations"]
    print("  [ok] 验证-训练损失倒挂 → 泄漏/口径排查（正常曲线不误报）")


def test_best_checkpoint_selection_noise_is_reported():
    """best 与次优验证损失的差距落在 95% CI 内时，选型不可靠。

    这是 winner's curse 的曲线诊断口径：报告 best 的数字会把它当成真实
    最优，实际可能只是有限窗口抽样噪声。
    """
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                [6.0, 4.0, 3.9, 3.95, 4.0])
    for r in recs:
        if "val_loss" in r:
            v = r["val_loss"]
            r["val_loss_ci95"] = [v * 0.8, v * 1.2]  # ±20%
    d = diagnose_curve(recs)
    assert d["status"] == "正常收敛", d
    assert d["best_checkpoint_within_noise"] is True, d
    assert d["val_best_margin_rel"] < d["val_best_ci_rel"], d
    assert any("winner" in e for e in d["evidence"]), d["evidence"]
    assert any("选型" in r for r in d["recommendations"]), d
    assert not any("无需干预" in r for r in d["recommendations"]), d

    # best 就是最终点时不存在“挑早期幸运点”的问题；无 CI 的旧日志不伪造标记
    recs2 = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                 [4.3, 4.0, 3.8, 3.6, 3.4])
    d2 = diagnose_curve(recs2)
    assert d2["best_checkpoint_within_noise"] is False, d2
    print("  [ok] best checkpoint 选型噪声（winner's curse）诊断")


def test_best_on_final_eval_flags_truncated_training():
    """best=最后一个评测点且验证仍降：没有收敛拐点，不能声称已收敛。

    只有知道训练覆盖（epochs_seen）时才提示延长训练；旧日志缺覆盖率信息
    时保持原“正常收敛/无需干预”口径，避免无依据地要求加训。
    """
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    d = diagnose_curve(recs, epochs_seen=2.0)
    assert d["status"] == "正常收敛", d
    assert d["best_at_final_eval"] is True
    assert d["val_still_improving_at_end"] is True
    assert any("延长训练" in r for r in d["recommendations"]), d
    assert not any("无需干预" in r for r in d["recommendations"]), d

    d2 = diagnose_curve(recs)
    assert d2["status"] == "正常收敛"
    assert d2["val_still_improving_at_end"] is True
    assert any("无需干预" in r for r in d2["recommendations"]), d2
    print("  [ok] best 在末次评测且仍下降 → 提示训练预算可能是上限")


def test_recent_tail_detects_late_rebound_and_plateau():
    """全程首/末分段仍显示验证下降时，末段回升/停滞必须单独识别。

    场景：验证早先大幅下降、后 1/3 才回升（或停步）。只看首→末会把这种
    过拟合误判为正常收敛/欠拟合；末段（后 1/3 vs 中 1/3）必须给出信号。
    """
    # 训练末段已平台，验证末段回升：即使全程验证仍在下降，也按过拟合处理
    d = diagnose_curve(_log([3.0, 2.0, 1.9, 1.9, 1.9, 1.9],
                            [3.2, 2.4, 2.0, 1.95, 2.1, 2.4]))
    assert d["status"] == "过拟合", d
    assert d["recent_trend_available"] is True
    assert d["val_recent_up"] is True and d["recent_overfit_signal"] is True
    assert any("末段趋势" in e for e in d["evidence"]), d["evidence"]
    assert any("早停" in r for r in d["recommendations"]), d["recommendations"]

    # 训练末段仍在下降、验证末段停步：全程验证仍下降，但已进入过拟合趋势
    d2 = diagnose_curve(_log([3.0, 2.7, 2.4, 2.1, 1.8, 1.5],
                             [3.2, 2.4, 2.0, 1.96, 1.97, 1.98]))
    assert d2["status"] == "过拟合趋势", d2
    assert d2["train_recent_down"] is True
    assert d2["val_recent_unresolved"] is True
    assert any("末段" in e and "无实质变化" in e for e in d2["evidence"]), d2["evidence"]

    # 正常同步下降不得误报末段信号
    d3 = diagnose_curve(_log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5],
                             [4.3, 3.8, 3.3, 2.9, 2.6]))
    assert d3["status"] == "正常收敛", d3
    assert not d3["recent_overfit_signal"] and not d3["recent_trend_signal"]
    print("  [ok] 末段趋势：晚期回升/停步不被全程均值掩盖")


def test_gradient_peak_uses_interval_max_and_reports_step():
    """新版日志的 grad_norm_max 是区间峰值：不能只看区间最后一步。"""
    recs = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    # 末步梯度正常（1.0），区间内峰值 80.0：旧口径会漏掉尖峰
    recs.append({"step": 10, "grad_norm": 1.0, "grad_norm_max": 80.0})
    recs.append({"step": 20, "grad_norm": 1.1, "grad_norm_max": 1.1})
    recs.append({"step": 30, "grad_norm": 1.0, "grad_norm_max": 1.0})
    recs.append({"step": 40, "grad_norm": 1.05, "grad_norm_max": 1.05})
    d = diagnose_curve(recs)
    assert d["has_grad_spike"], d
    assert d["grad_norm_max"] == 80.0 and d["grad_norm_max_step"] == 10
    assert d["grad_spikes"] and d["grad_spikes"][0]["step"] == 10
    assert any("step 10" in e for e in d["evidence"]), d["evidence"]

    # 区间峰值非有限：即使损失曲线正常，也要报非有限梯度并带步号
    recs2 = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    recs2.append({"step": 10, "grad_norm": 1.0,
                  "grad_norm_max": float("inf")})
    d2 = diagnose_curve(recs2)
    assert d2["grad_norm_nonfinite"] == 1
    assert d2["grad_norm_nonfinite_steps"] == [10]
    assert any("step 10" in e for e in d2["evidence"])

    # 峰值有限但末步非有限（或反之）：两个键都要检查，不能因优先键可解析
    # 就漏掉另一个键里的非有限证据。
    recs3 = _log([4.2, 3.9, 3.5, 3.1, 2.8, 2.5], [4.3, 3.8, 3.3, 2.9, 2.6])
    recs3.append({"step": 10, "grad_norm": float("nan"),
                  "grad_norm_max": 1.2})
    d3 = diagnose_curve(recs3)
    assert d3["grad_norm_nonfinite_steps"] == [10]
    assert d3["grad_norm_max"] == 1.2
    print("  [ok] pre-clip 梯度峰值/非有限：区间峰值 + 发生步")


def test_curve_step_beyond_float_exact_range_is_skipped():
    """巨大 step 不能进入分段均值：int→float 溢出会让整份诊断崩溃。"""
    assert _step_or_none(10 ** 400) is None
    assert _step_or_none(1e308) is None
    assert _step_or_none(-1e308) is None
    assert _step_or_none(2 ** 53 + 1) is None
    assert _step_or_none(10) == 10
    # 一段正常曲线混入巨大 step 时，坏记录被跳过，诊断仍能完成。
    recs = [{"step": 10, "loss": 4.0}, {"step": 20, "loss": 3.0},
            {"step": 30, "loss": 2.5},
            {"step": 50, "val_loss": 4.2}, {"step": 100, "val_loss": 3.0},
            {"step": 150, "val_loss": 2.4},
            {"step": 1e308, "loss": 1.0},
            {"step": -1e308, "val_loss": 1.0}]
    d = diagnose_curve(recs, log_vocab=10.0)
    assert isinstance(d, dict) and d.get("status")
    print("  [ok] 曲线诊断跳过超出 float 精确范围的巨大 step")
