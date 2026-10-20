"""坏案例分析测试：错误归因（标注错误/误抄干扰/幻觉/无回答/fwe）、语言建模
最差窗口归因、跨变体错误对照与报告生成。"""
import json
import os
import shutil
import tempfile

from lega.badcase import (CATEGORY_CN, analyze_badcases, analyze_worst_windows,
                           case_axis, case_key, categorize_case, categorize_window,
                           cross_variant_analysis, format_case, format_window_case,
                           process_eval_json, process_ruler_json,
                           write_badcase_md, write_worstcase_md)


def _win(pred_top5=None, gold="green", tail="The sky is blue. The sky is",
         **kw):
    rec = {"window": 7, "split": "val", "loss": 8.0, "context_tail": tail,
           "gold_next": gold,
           "pred_top5": ([[" blue", 0.7], [" red", 0.1]]
                         if pred_top5 is None else pred_top5)}
    rec.update(kw)
    return rec


def _case(task="niah_multikey", answer="7021853", pred="", ctx_nums=None,
          in_prompt=True, **kw):
    rec = {"task": task, "ctx": 256, "answer": answer, "answer_words": None,
           "pred": pred, "em": 0.0, "recall": None,
           "question": "What is the special magic number for lyre?",
           "needle_excerpt": "One of the special magic numbers for lyre is: 7021853.",
           "depth": 0.5, "meta": {}, "context_numbers": ctx_nums or [],
           "answer_in_prompt": in_prompt if task != "fwe" else None}
    rec.update(kw)
    return rec


def test_categorize_numeric_causes():
    # 误抄干扰值：输出的错误数字来自上下文（multikey/vt 的典型失败模式）
    c = categorize_case(_case(pred="The number is 1234567.", ctx_nums=["1234567", "9988776"]))
    assert c["category"] == "copy_distractor" and "干扰" in c["explanation"]
    # 幻觉：输出上下文中不存在的数字
    c = categorize_case(_case(pred="The number is 5550001.", ctx_nums=["1234567"]))
    assert c["category"] == "hallucinated"
    # 无数值回答（模型复读问题/提前停止 → 格式盲区）
    c = categorize_case(_case(pred="The special magic number for lyre mentioned in..."))
    assert c["category"] == "no_answer"
    print("  [ok] 数值任务归因（误抄干扰/幻觉/无回答）")


def test_categorize_tolerates_non_string_pred():
    """外部 JSON 的 pred 可能是数字：按字符串归一化后归因。"""
    c = categorize_case(_case(pred=1234567, ctx_nums=["1234567"]))
    assert c["category"] == "copy_distractor", c
    c = categorize_case(_case(pred=1234567.0, ctx_nums=["7654321"]))
    assert c["category"] == "hallucinated", c
    # 整份报告级入口同样按此口径归因
    ana = analyze_badcases([_case(pred=1234567, ctx_nums=["1234567"])])
    assert ana["category_counts"] == {"copy_distractor": 1}
    print("  [ok] 数值型 pred 容错（归因正确）")


def test_categorize_label_error():
    # gold 不在 prompt 中 → 标注/构造错误，不计为模型失败
    c = categorize_case(_case(pred="1234567", ctx_nums=["1234567"], in_prompt=False))
    assert c["category"] == "label_error"
    assert "不应计入模型失败" in c["explanation"]
    print("  [ok] 标注/构造错误识别（gold 缺失于 prompt）")


def test_categorize_fwe():
    c = categorize_case(_case(task="fwe", answer="ka ro mi", pred="ka ro extra",
                              answer_words=["ka", "ro", "mi"]))
    assert c["category"] == "fwe_missing"
    assert c["missing_words"] == ["mi"] and c["spurious_words"] == ["extra"]
    c = categorize_case(_case(task="fwe", answer="ka ro", pred="ka ro",
                              answer_words=["ka", "ro"]))
    assert c["category"] == "fwe_spurious" or c["category"] == "fwe_missing"
    c = categorize_case(_case(task="fwe", answer="ka", pred="123",
                              answer_words=["ka"]))
    assert c["category"] == "fwe_no_parse"
    print("  [ok] fwe 归因（漏答/多答/无法解析）")


def test_case_axis_matches_synthesized_advice_axes():
    """评测/导出类错误的 axis 分布与改进建议的 axis 必须一致。

    fwe_no_parse 与 unparsed/fwe_no_ref 同属解析/导出环节；若 case_axis 漏掉
    它，同一批案例会被同时算到"模型/方法"（分布表）与"评测/导出"（建议），
    报告自相矛盾。
    """
    cases = [_case(task="fwe", answer="ka", pred="123",
                   answer_words=["ka"])] * 3
    ana = analyze_badcases(cases)
    assert ana["category_counts"] == {"fwe_no_parse": 3}
    assert ana["axis_counts"] == {"评测/导出": 3}, ana["axis_counts"]
    assert {a["axis"] for a in ana["recommendations"]} == {"评测/导出"}
    # 最差窗口的 unparsed（旧格式/导出失败）同样归评测/导出
    c = categorize_window({"window": 1, "loss": 1.0, "gold_next": "x",
                           "pred_top5": None})
    assert c["category"] == "unparsed"
    assert case_axis(c) == "评测/导出"
    # top1 正确但低置信属校准问题：若归入"模型/方法"，会与建议里的
    # "校准"自相矛盾；深位也不例外（排位已对，不是检索失败）
    for deep in (0.1, 0.9):
        w = categorize_window({"window": 2, "loss": 1.0, "gold_next": "x",
                               "pred_top5": ["x", "y"],
                               "gold_in_context": True, "depth": deep})
        w["hard_sample"] = deep > 0.75   # 即使深位，排位已对 → 校准优先
        assert w["category"] == "top1_correct"
        assert case_axis(w) == "校准", (deep, case_axis(w))
    print("  [ok] 评测/导出/校准类错误的 axis 与改进建议一致")


def test_hard_sample_flag():
    c = categorize_case(_case(pred="1111111", ctx_nums=[], depth=0.9))
    assert c["hard_sample"] is True
    c = categorize_case(_case(pred="1111111", ctx_nums=[], depth=0.1))
    assert c["hard_sample"] is False
    # vt 的 depth 是链深度（整数）而非 needle 位置比例，不得按位置判难度
    c = categorize_case(_case(task="vt", pred="1111111", ctx_nums=[], depth=4))
    assert c["hard_sample"] is False
    # 提前返回的分支（no_answer 等）也必须带 hard_sample：深位失败仍是难度边界
    c = categorize_case(_case(pred="no digits here at all", depth=0.9))
    assert c["category"] == "no_answer" and c["hard_sample"] is True
    print("  [ok] 难度边界样本标注（needle 深位；vt 链深度不误判）")


def test_non_numeric_depth_tolerated():
    """外部 JSON 的字符串 depth 按非数值归一化处理。"""
    c = categorize_case(_case(pred="1111111", ctx_nums=[], depth="0.9"))
    assert c["depth"] == 0.9 and c["hard_sample"] is True
    c = categorize_case(_case(task="vt", pred="1111111", ctx_nums=[], depth="deep"))
    assert c["depth"] is None and c["hard_sample"] is False
    # 混合进集合分析：坏 depth 只被忽略，不中断报告
    ana = analyze_badcases([
        _case(task="niah_multikey", pred="5550001", ctx_nums=[], depth="0.9"),
        _case(task="vt", pred="5550001", ctx_nums=[], depth="bad"),
    ])
    assert ana["deep_position_fail_ratio"] == 1.0
    print("  [ok] 非数值 depth 安全归一化（不误判难度）")


def test_deep_ratio_excludes_vt_chain_depth():
    """deep_position_fail_ratio 只能统计 niah 的 needle 位置，不含 vt 链深度。"""
    cases = [
        _case(task="niah_multikey", pred="5550001", ctx_nums=[], depth=0.2),
        _case(task="vt", pred="5550001", ctx_nums=[], depth=4),
    ]
    ana = analyze_badcases(cases)
    assert ana["deep_position_fail_ratio"] == 0.0
    assert ana["hard_sample_ratio"] == 0.0
    cases = [
        _case(task="niah_multikey", pred="5550001", ctx_nums=[], depth=0.9),
        _case(task="vt", pred="5550001", ctx_nums=[], depth=4),
    ]
    ana = analyze_badcases(cases)
    assert ana["deep_position_fail_ratio"] == 1.0
    print("  [ok] 深位失败占比仅计 niah 位置 depth")


def test_analyze_and_md_report():
    cases = [
        _case(pred="1234567", ctx_nums=["1234567"]),
        _case(pred="5550001", ctx_nums=["1234567"]),
        _case(pred="no numbers here"),
        _case(task="vt", answer="48213", pred="99321", ctx_nums=["99321", "13579"]),
        _case(task="fwe", answer="ka ro", pred="ka", answer_words=["ka", "ro"]),
        _case(pred="0000000", ctx_nums=[], in_prompt=False),
    ]
    ana = analyze_badcases(cases, variant="lega")
    assert ana["n_badcases"] == 6
    assert ana["category_counts"]["copy_distractor"] == 2
    assert ana["category_counts"]["hallucinated"] == 1
    assert ana["category_counts"]["no_answer"] == 1
    assert ana["category_counts"]["label_error"] == 1
    assert ana["label_errors"] == 1
    tmp = tempfile.mkdtemp()
    try:
        write_badcase_md(ana, os.path.join(tmp, "badcases_lega.md"))
        text = open(os.path.join(tmp, "badcases_lega.md"), encoding="utf-8").read()
        assert "误抄干扰值" in text and "标注/构造错误" in text
        assert "gold 证据摘录" in text and "模型输出" in text
        # 空案例也须能写报告
        write_badcase_md(analyze_badcases([], "standard"),
                          os.path.join(tmp, "badcases_std.md"))
        assert "无预测错误样本" in open(os.path.join(tmp, "badcases_std.md"),
                                         encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 归因汇总与 Markdown 报告")


def test_categorize_worst_window():
    # 复读吸引子：低置信下首选续写退化为抄上下文已有串
    c = categorize_window(_win(pred_top5=[[" blue", 0.3], [" green", 0.02]]))
    assert c["category"] == "repetition_attractor", c
    assert c["repeats_context"] is True and c["gold_in_context"] is False
    # 高置信错答（系统性盲区）优先于复读机制：概率高、不是 gold
    c = categorize_window(_win(pred_top5=[[" blue", 0.9], [" green", 0.02]]))
    assert c["category"] == "confident_bias", c
    assert c["repeats_context"] is True and "高置信复读" in c["explanation"]
    # 高置信错答且 gold 在上下文出现过但没被复制
    c = categorize_window(_win(gold=" blue", tail="a blue car",
                               pred_top5=[[" red", 0.85], [" rope", 0.05]]))
    assert c["category"] == "confident_bias", c
    assert c["gold_in_context"] is True and "未被复制" in c["explanation"]
    # 高置信错答：gold 从未出现在上下文中（真正的预测失败）
    c = categorize_window(_win(gold=" xyzzy", tail="a blue car",
                               pred_top5=[[" red", 0.85]]))
    assert c["category"] == "confident_bias"
    assert "未出现" in c["explanation"]
    # 低置信：模型自己在多个候选间平均（样本多解/噪声或训练不足）
    c = categorize_window(_win(gold=" xyzzy", tail="a blue car",
                               pred_top5=[[" red", 0.2], [" rope", 0.18]]))
    assert c["category"] == "low_confidence", c
    # top1 正确但概率低：校准问题，不是检索错误
    c = categorize_window(_win(gold=" blue", tail="a blue car",
                               pred_top5=[[" blue", 0.2]]))
    assert c["category"] == "top1_correct"
    # 无预测分布（旧格式）→ unparsed
    c = categorize_window(_win(pred_top5=[]))
    assert c["category"] == "unparsed" and c["top1"] is None
    print("  [ok] 最差窗口归因（复读/高置信盲区/低置信/校准/旧格式）")


def test_categorize_window_prefers_full_context_flag():
    """采集端按完整窗口判定的 gold_in_context 优先于尾部兜底猜测。"""
    # 旧格式（无该字段）：沿用 context_tail 检查
    c = categorize_window(_win(pred_top5=[[" red", 0.85]]))
    assert c["gold_in_context"] is False and "未出现" in c["explanation"]
    # 新格式：gold 在窗口前部出现过（尾部没有），布尔为 True 时必须采信，
    # 否则会把“复制失败”误报成“gold 从未出现”的预测失败
    c2 = categorize_window(_win(pred_top5=[[" red", 0.85]],
                                gold_in_context=True))
    assert c2["gold_in_context"] is True
    assert "未被复制" in c2["explanation"]
    print("  [ok] 最差窗口 gold_in_context：完整窗口布尔优先/旧格式尾部兜底")


def test_format_window_case_shows_gold_location_and_context():
    """gold 出现在早段时，报告要给出位置/距离/附近原文，便于区分检索失败
    与复制失败；旧记录缺失这些字段时安全降级。"""
    c = categorize_window({
        "window": 1, "loss": 3.0, "gold_next": " blue",
        "gold_in_context": True, "gold_occurrences": 2,
        "gold_last_pos": 10, "gold_distance_tokens": 22,
        "gold_context_excerpt": "a blue car parked",
        "context_tail": "near the road", "pred_top5": [[" red", 0.9]]})
    text = "\n".join(format_window_case(1, c))
    assert "距预测位 22 tokens" in text, text
    assert "共出现 2 次" in text and "a blue car parked" in text, text
    # 旧格式没有位置字段：只保留原有的出现过/未出现提示，不伪造距离
    old = categorize_window(_win(gold=" blue", tail="a blue car",
                                 pred_top5=[[" red", 0.85]]))
    assert "距预测位" not in "\n".join(format_window_case(2, old))
    print("  [ok] 最差窗口报告展示 gold 位置/距离/附近上下文")


def test_format_window_case_tolerates_null_loss():
    """loss 缺失/为 null 的窗口记录也必须能出报告行（外部/旧格式 JSON 容错）。"""
    c = categorize_window(_win(loss=None, pred_top5=[[" blue", 0.7]]))
    assert "loss=-" in format_window_case(1, c)[0]
    c2 = categorize_window(_win(pred_top5=[[" blue", 0.7]]))
    c2.pop("loss")
    assert "loss=-" in format_window_case(1, c2)[0]
    print("  [ok] 最差窗口报告行：loss 缺失/null 安全")


def test_analyze_worst_windows_and_md():
    wins = [
        _win(pred_top5=[[" blue", 0.3]]),
        _win(gold=" xyzzy", tail="a blue car", window=8,
             pred_top5=[[" red", 0.85]]),
        _win(gold=" xyzzy", tail="a blue car", window=9,
             pred_top5=[[" rope", 0.1], [" red", 0.09]]),
    ]
    ana = analyze_worst_windows(wins, variant="lega")
    assert ana["n_windows"] == 3
    assert ana["category_counts"]["repetition_attractor"] == 1
    assert ana["category_counts"]["confident_bias"] == 1
    assert ana["category_counts"]["low_confidence"] == 1
    assert 0.0 <= ana["gold_in_context_rate"] <= 1.0
    assert ana["repeats_context_rate"] == 1 / 3
    assert abs(ana["mean_top1_prob"] - (0.3 + 0.85 + 0.1) / 3) < 1e-9
    tmp = tempfile.mkdtemp()
    try:
        write_worstcase_md(ana, os.path.join(tmp, "worstcases_lega.md"))
        text = open(os.path.join(tmp, "worstcases_lega.md"), encoding="utf-8").read()
        assert "复读吸引子" in text and "真实续写(gold)" in text
        assert "案例 1" in text
        write_worstcase_md(analyze_worst_windows([], "standard"),
                           os.path.join(tmp, "worstcases_std.md"))
        assert "无最差窗口记录" in open(os.path.join(tmp, "worstcases_std.md"),
                                        encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 最差窗口汇总与 Markdown 报告")


def test_process_eval_json_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        payload = {"variant": "lega", "seed": 1337, "split": "test",
                   "val_ppl": 10.0,
                   "worst_windows": [_win(pred_top5=[[" blue", 0.3]])]}
        path = os.path.join(tmp, "eval_lega_seed1337.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        ana = process_eval_json(path)
        assert ana["n_windows"] == 1 and ana["seed"] == 1337
        md = os.path.join(tmp, "worstcases_lega_seed1337.md")
        assert os.path.exists(md), "worstcase md 应与 eval json 同目录同名换后缀"
        assert "复读吸引子" in open(md, encoding="utf-8").read()
        # 旧格式（无 worst_windows）→ n_windows=0
        old = os.path.join(tmp, "eval_old.json")
        with open(old, "w", encoding="utf-8") as f:
            json.dump({"variant": "standard"}, f)
        assert process_eval_json(old)["n_windows"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] process_eval_json（含旧格式兼容）")


def _case_sid(sid, task="niah_synth", answer="1234567", pred="", **kw):
    rec = _case(task=task, answer=answer, pred=pred, **kw)
    rec["sample_id"] = sid
    return rec


def test_cross_variant_analysis_shared_vs_specific():
    a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
    b = _case_sid("niah_synth@256#1", pred="7654321", ctx_nums=["7654321"])
    cross = cross_variant_analysis(
        {"standard": [a], "lega": [a, b], "qgate": [b]}, reference="standard")
    assert cross["n_unique_failed_items"] == 2
    assert cross["n_shared_all_failures"] == 0
    lv = cross["per_variant"]["lega"]
    assert lv["n_failed"] == 2 and lv["n_new_vs_reference"] == 1
    assert lv["new_vs_reference"][0]["key"] == "niah_synth@256#1"
    assert cross["per_variant"]["standard"]["n_new_vs_reference"] == 0
    # 三个变体全错的样本 → 共识失败（更可能是数据/构造/内在难度）
    cross = cross_variant_analysis(
        {"standard": [a], "lega": [a], "qgate": [a]}, reference="standard")
    assert cross["n_shared_all_failures"] == 1
    assert cross["shared_failures"][0]["key"] == "niah_synth@256#0"
    # 参照变体无错误时不得把“全部失败”误报为“大量新增错误”
    cross = cross_variant_analysis({"lega": [a], "qgate": [b]},
                                   reference="standard")
    assert cross["reference_available"] is False
    assert cross["per_variant"]["lega"]["n_new_vs_reference"] is None
    # 少于 2 个变体：空对照
    cross = cross_variant_analysis({"lega": [a]})
    assert cross["per_variant"] == {}
    # 旧格式无 sample_id：回退键仍能对齐同一题的多个变体
    legacy = _case(pred="1234567", ctx_nums=["1234567"])
    k1 = case_key(legacy)
    k2 = case_key(dict(legacy))
    assert k1 == k2 and "niah" in k1
    # 同一问题、不同 gold 的旧格式样本必须分开，否则会被误合并成共识失败
    other = dict(legacy, answer="7654321")
    assert case_key(legacy) != case_key(other)
    print("  [ok] 跨变体错误对照（共识失败/特异性新增/参照缺失/旧格式对齐）")


def test_process_ruler_json_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        payload = {"variant": "lega", "seed": 1337,
                   "results": {"niah_synth": {"256": {"em": 0.0, "n": 1}}},
                   "badcases": [_case(pred="1234567", ctx_nums=["1234567"])]}
        ruler_path = os.path.join(tmp, "ruler_lega_seed1337.json")
        with open(ruler_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        ana = process_ruler_json(ruler_path)
        assert ana["n_badcases"] == 1 and ana["seed"] == 1337
        md_path = os.path.join(tmp, "badcases_lega_seed1337.md")
        assert os.path.exists(md_path), "badcase md 应与 ruler json 同目录同名换后缀"
        assert CATEGORY_CN["copy_distractor"] in open(md_path, encoding="utf-8").read()
        # 无 badcases 键的旧格式文件 → n_badcases=0
        old = os.path.join(tmp, "ruler_old.json")
        with open(old, "w", encoding="utf-8") as f:
            json.dump({"variant": "standard", "results": {}}, f)
        assert process_ruler_json(old)["n_badcases"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] process_ruler_json（含旧格式兼容）")


def test_nonfinite_loss_excluded_from_mean():
    """发散 run 的窗口 loss 可能是 NaN/Inf：均值必须剔除（与 utils.mean_std
    同口径），否则 NaN 会污染报告并被写进非法 JSON。"""
    ana = analyze_worst_windows([
        _win(loss=float("nan"), pred_top5=[[" blue", 0.3]]),
        _win(loss=float("inf"), pred_top5=[[" blue", 0.3]]),
        _win(loss=2.0, pred_top5=[[" blue", 0.3]]),
    ], variant="lega")
    assert ana["mean_loss"] == 2.0
    # 报告聚合字段不得含 NaN/Inf（NaN 字面量不是合法 JSON）
    aggregates = {k: v for k, v in ana.items() if k != "cases"}
    json.dumps(aggregates, allow_nan=False)
    print("  [ok] 最差窗口均值剔除 NaN/Inf（聚合字段 JSON 合法）")


def test_mean_tolerates_non_numeric_values():
    """旧格式/坏数据里的 "n/a" 等非数值必须跳过，不能让整份报告崩溃。"""
    from lega.badcase import _mean
    assert abs(_mean([1.0, "n/a", 2.0]) - 1.5) < 1e-12
    assert _mean(["n/a", None, float("nan")]) is None
    ana = analyze_worst_windows([
        _win(loss="n/a", pred_top5=[[" blue", 0.3]]),
        _win(loss=2.0, pred_top5=[[" blue", 0.3]]),
    ], variant="lega")
    assert ana["mean_loss"] == 2.0
    print("  [ok] 非数值 loss 容错（\"n/a\" 跳过且不中断报告）")


def test_huge_integers_tolerated_in_badcase():
    """合法 JSON 里的超大整数转 float 会 OverflowError，坏案例报告不能崩。"""
    from lega.badcase import _mean
    huge = 10 ** 400
    assert _mean([huge]) is None
    c = categorize_case(_case(pred="1234567", ctx_nums=[], depth=huge))
    assert c["depth"] is None and c["hard_sample"] is False
    format_case(1, c)
    text = "\n".join(format_window_case(1, {
        "loss": huge, "pred_top5": [[" blue", 0.5]],
        "gold_next": " green", "context_tail": "The sky is"}))
    assert "loss=-" in text
    ana = analyze_worst_windows([
        _win(loss=huge, pred_top5=[[" blue", 0.3]]),
        _win(loss=2.0, pred_top5=[[" blue", 0.3]]),
    ], variant="lega")
    assert ana["mean_loss"] == 2.0
    print("  [ok] 超大整数（OverflowError）容错且不中断报告")


def test_process_json_non_dict_payload_raises_value_error():
    """顶层非对象（[] / 字符串）的坏文件：抛 ValueError 供调用方跳过，
    而不是 AttributeError 中断整条流水线。"""
    tmp = tempfile.mkdtemp()
    try:
        for name, fn in (("eval_bad.json", process_eval_json),
                         ("ruler_bad.json", process_ruler_json)):
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            try:
                fn(path, write_md=False)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} 对非对象 payload 未抛 ValueError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 非对象 JSON payload 抛 ValueError（按坏文件跳过）")


def test_main_skips_non_dict_ruler_and_keeps_cross_variant():
    """main 遇到顶层非对象的 ruler 文件必须跳过，且该坏 payload 不能进入
    跨变体对照导致二次崩溃。"""
    import sys
    from lega.badcase import main as badcase_main
    tmp = tempfile.mkdtemp()
    try:
        a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
        for v, cases in (("standard", []), ("lega", [a])):
            with open(os.path.join(tmp, f"ruler_{v}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"variant": v, "seed": 1, "badcases": cases}, f,
                          ensure_ascii=False)
        with open(os.path.join(tmp, "ruler_bad.json"), "w",
                  encoding="utf-8") as f:
            json.dump([], f)
        old_argv = sys.argv
        sys.argv = ["badcase", "--results_dir", tmp, "--no_md"]
        try:
            badcase_main()
        finally:
            sys.argv = old_argv
        with open(os.path.join(tmp, "badcase_summary.json"),
                  encoding="utf-8") as f:
            merged = json.load(f)
        assert merged["n_files"] == 2
        assert merged["cross_variant"]["per_variant"]["standard"]["n_failed"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] main：非对象 JSON 跳过且不污染跨变体对照")


def test_empty_gold_not_counted_as_top1_correct():
    """gold 缺失/空串时，空的 top1 不能因空串相等被判“top1 正确”。"""
    c = categorize_window(_win(gold=None, pred_top5=[["", 0.1], ["x", 0.2]]))
    assert c["category"] != "top1_correct", c
    c2 = categorize_window(_win(gold="", pred_top5=[["", 0.1]]))
    assert c2["category"] != "top1_correct", c2
    print("  [ok] 空 gold 不计为 top1_correct")


def test_cross_variant_zero_error_variant_blocks_consensus():
    """某变体已评测且零错误（空列表）时，绝不能再把其余变体的共同失败
    标成“全体变体共识失败”——那是方法特异性失败，不是数据/难度问题。"""
    a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
    cross = cross_variant_analysis(
        {"standard": [a], "lega": [a], "qgate": []}, reference="standard")
    assert cross["variants"] == ["lega", "qgate", "standard"]
    assert cross["n_shared_all_failures"] == 0
    assert cross["per_variant"]["qgate"]["n_failed"] == 0
    # 三个变体都错时仍正确计为共识失败
    cross = cross_variant_analysis(
        {"standard": [a], "lega": [a], "qgate": [a]}, reference="standard")
    assert cross["n_shared_all_failures"] == 1
    # 参照变体已评测且零错误：其余变体的每个失败都确实是“参照对而该方法错”，
    # 全部计入新增是正确的（不能因为参照没有错误样本就放弃计数）。
    cross = cross_variant_analysis(
        {"standard": [], "lega": [a], "qgate": [a]}, reference="standard")
    assert cross["reference_available"] is True
    assert cross["per_variant"]["lega"]["n_new_vs_reference"] == 1
    assert cross["per_variant"]["qgate"]["n_fixed_vs_reference"] == 0
    print("  [ok] 零错误变体阻止“全体共识失败”误报")


def test_malformed_records_do_not_crash_attribution():
    """外部/旧格式 JSON 的字段可能是数字/缺失/None：单条坏样本必须被容错
    处理（pred=123 的 fwe、缺 task、非字符串 needle/depth），不能中断整份报告。"""
    bad = [
        {"task": "fwe", "pred": 123, "answer_words": None},
        {"task": "fwe", "pred": None, "answer_words": ["a"]},
        {"task": "fwe", "pred": "ka ro", "answer_words": 123},
        {"task": "fwe", "pred": "ka", "answer_words": [1, None]},
        {"task": "niah_synth", "depth": {"a": 1}, "pred": None,
         "answer": 123, "context_numbers": None, "needle_excerpt": 42},
        {"task": "niah_synth", "pred": "1234567", "context_numbers": 5},
        {},  # 缺 task
    ]
    ana = analyze_badcases(bad, variant="lega")
    assert ana["n_badcases"] == len(bad)
    for i, c in enumerate(ana["cases"], 1):
        format_case(i, c)  # 渲染输出
    win = analyze_worst_windows(
        [{"pred_top5": ["a"]}, {"pred_top5": [{"t": "a", "p": 0.9}],
                                "gold_next": None, "context_tail": None}],
        variant="lega")
    for i, c in enumerate(win["cases"], 1):
        format_window_case(i, c)
    tmp = tempfile.mkdtemp()
    try:
        write_badcase_md(ana, os.path.join(tmp, "b.md"))
        write_worstcase_md(win, os.path.join(tmp, "w.md"))
        cross = cross_variant_analysis({"a": bad, "b": bad}, reference="a")
        json.dumps(cross, ensure_ascii=False)  # 结果必须可序列化
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 坏样本字段类型异常的归因（fwe/缺 task/非字符串字段）")


def test_huge_and_nonfinite_window_fields_tolerated():
    """窗口距离/位置可能是超大整数或 NaN/Inf：中位数与 Markdown 渲染必须
    跳过不可用值，不能因 float 转换溢出中断最差窗口报告。"""
    win = analyze_worst_windows([
        {"pred_top5": ["a"], "gold_next": "b",
         "gold_last_pos": 10, "gold_distance_tokens": 100,
         "gold_occurrences": 2},
        {"pred_top5": ["a"], "gold_next": "b",
         "gold_last_pos": 20, "gold_distance_tokens": 200,
         "gold_occurrences": 3},
        {"pred_top5": ["a"], "gold_next": "b",
         "gold_last_pos": float("inf"), "gold_distance_tokens": 10 ** 400,
         "gold_occurrences": float("inf")},
        {"pred_top5": ["a"], "gold_next": "b",
         "gold_last_pos": float("nan"), "gold_distance_tokens": 10 ** 400,
         "gold_occurrences": 10 ** 400},
        {"pred_top5": ["a"], "gold_next": "b",
         "gold_distance_tokens": float("inf")},
    ], variant="lega")
    assert win["median_gold_distance_tokens"] == 150.0
    for i, c in enumerate(win["cases"], 1):
        format_window_case(i, c)
    tmp = tempfile.mkdtemp()
    try:
        write_worstcase_md(win, os.path.join(tmp, "w.md"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 最差窗口的超大整数/非有限数值容错")


def test_cross_variant_skips_non_dict_records():
    """跨变体列表混入 null/标量时必须跳过坏记录，不能让整份对照崩溃。"""
    a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
    cross = cross_variant_analysis(
        {"standard": [None, a, "bad"], "lega": [5, a, None]},
        reference="standard")
    assert cross["n_unique_failed_items"] == 1
    assert cross["per_variant"]["lega"]["n_failed"] == 1
    json.dumps(cross, ensure_ascii=False, allow_nan=False)
    print("  [ok] 跨变体对照：非对象样本跳过")


def test_main_keeps_zero_error_variants_for_cross_variant():
    """badcase.main 汇总时必须把 "badcases": []（已评测且零错误）传给跨变体
    对照，使零错误变体参与共识判定。"""
    import sys
    from lega.badcase import main as badcase_main
    tmp = tempfile.mkdtemp()
    try:
        a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
        for v, cases in (("standard", []), ("lega", [a]), ("qgate", [a])):
            with open(os.path.join(tmp, f"ruler_{v}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"variant": v, "seed": 1, "badcases": cases}, f,
                          ensure_ascii=False)
        old_argv = sys.argv
        sys.argv = ["badcase", "--results_dir", tmp, "--no_md"]
        try:
            badcase_main()
        finally:
            sys.argv = old_argv
        with open(os.path.join(tmp, "badcase_summary.json"),
                  encoding="utf-8") as f:
            merged = json.load(f)
        cross = merged["cross_variant"]
        assert cross["variants"] == ["lega", "qgate", "standard"], cross
        assert cross["reference_available"] is True
        assert cross["n_shared_all_failures"] == 0
        assert cross["per_variant"]["lega"]["n_new_vs_reference"] == 1
        # 汇总必须把改进方向一并落盘，且标注来源（分析不止于分类）
        assert merged["recommendations"], merged.get("recommendations")
        assert {r["source"] for r in merged["recommendations"]} >= {
            "ruler:standard", "ruler:lega", "ruler:qgate", "cross_variant"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] badcase.main 保留零错误变体进入跨变体对照")


def test_main_picks_smallest_seed_per_variant_not_lexicographic_first():
    """跨变体对照必须按数值最小种子对齐；文件名字典序会把 seed1000 排在
    seed999 前面，从而把一次“零错误”的运行当成该变体的代表结果。"""
    import sys
    from lega.badcase import main as badcase_main
    tmp = tempfile.mkdtemp()
    try:
        a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
        payloads = {
            "ruler_standard_seed999.json": {"variant": "standard", "seed": 999,
                                            "badcases": []},
            "ruler_lega_seed999.json": {"variant": "lega", "seed": 999,
                                        "badcases": [a]},
            "ruler_lega_seed1000.json": {"variant": "lega", "seed": 1000,
                                         "badcases": []},
        }
        for name, obj in payloads.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
        old_argv = sys.argv
        sys.argv = ["badcase", "--results_dir", tmp, "--no_md"]
        try:
            badcase_main()
        finally:
            sys.argv = old_argv
        with open(os.path.join(tmp, "badcase_summary.json"),
                  encoding="utf-8") as f:
            merged = json.load(f)
        assert merged["n_files"] == 3
        pv = merged["cross_variant"]["per_variant"]["lega"]
        assert pv["n_failed"] == 1, pv      # seed999，而非字典序在前的 seed1000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] badcase.main 按数值最小种子对齐跨变体对照")


def test_label_errors_excluded_from_model_advice_denominator():
    """标注/构造错误不是模型失败：深位比例与类别主导度都不能被它污染。"""
    cases = ([_case(pred="1234567", ctx_nums=[], in_prompt=False, depth=0.9)] * 10
             + [_case(pred="1234567", ctx_nums=["1234567"], depth=0.2)] * 3
             + [_case(pred="5550001", ctx_nums=[], depth=0.2)] * 3)
    ana = analyze_badcases(cases, variant="lega")
    assert ana["label_errors"] == 10
    # 非标注错误的 6 条都在 depth=0.2：深位比例必须是 0，而不是 10/16
    assert ana["deep_position_fail_ratio"] == 0.0
    # 误抄 3/6 达到主导阈值；用 3/16 会被稀释而漏报
    assert any("误抄干扰值占模型错误 3/6" in r["trigger"]
               for r in ana["recommendations"]), ana["recommendations"]
    assert not any("位于 needle 深位" in r["trigger"]
                   for r in ana["recommendations"])
    print("  [ok] 标注错误从模型改进方向的分母中剔除")


def test_ruler_improvement_synthesis():
    """归因不止于分类：主导错误模式必须转成带触发证据的改进方向。"""
    keys = {"axis", "priority", "trigger", "suggestion", "next_experiment"}
    # 误抄干扰主导 → 检索定位方向（先机制分析再改结构）
    ana = analyze_badcases([_case(pred="1234567", ctx_nums=["1234567"])] * 3,
                           variant="lega")
    assert ana["recommendations"] and all(
        set(r) == keys for r in ana["recommendations"])
    assert any(r["axis"] == "模型/方法" and "检索" in r["trigger"]
               and "注意力" in r["suggestion"] for r in ana["recommendations"])
    # 标注/构造错误 → 数据轴，优先级最高
    ana2 = analyze_badcases(
        [_case(pred="1234567", ctx_nums=[], in_prompt=False)] * 3, variant="lega")
    assert any(r["axis"] == "数据/构造" and r["priority"] == "高"
               for r in ana2["recommendations"])
    assert ana2["axis_counts"].get("数据/构造") == 3
    # 深位失败主导 → 难度边界（轴计数直接回答“标注/难度/盲区”三问）
    ana3 = analyze_badcases([_case(pred="5550001", ctx_nums=[], depth=0.9)] * 3)
    assert any(r["axis"] == "难度边界" for r in ana3["recommendations"])
    assert ana3["axis_counts"].get("难度边界") == 3
    # 零错误也给出解释边界（零错误 ≠ 方法更强）
    ana4 = analyze_badcases([])
    assert ana4["recommendations"][0]["axis"] == "评测"
    json.dumps(ana, ensure_ascii=False, allow_nan=False)
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "badcases_lega.md")
        write_badcase_md(ana, path)
        text = open(path, encoding="utf-8").read()
        assert "改进方向（坏案例驱动）" in text and "下一步实验" in text
        assert "粗归因轴" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] RULER 坏案例 → 改进方向合成（误抄/标注/深位/零错误）")


def test_window_improvement_synthesis():
    """PPL 最差窗口：复制失败与复读必须给出不同的改进方向。"""
    # gold 在上下文中却没抄对（检索/复制失败）
    wins = [_win(gold=" blue", tail="a blue car",
                 pred_top5=[[" red", 0.85]]) for _ in range(3)]
    ana = analyze_worst_windows(wins, variant="lega")
    assert ana["gold_in_context_rate"] == 1.0
    assert any("抄对" in r["suggestion"] or "复制" in r["suggestion"]
               for r in ana["recommendations"]), ana["recommendations"]
    # 复读吸引子主导
    wins2 = [_win(pred_top5=[[" blue", 0.3]]) for _ in range(3)]
    ana2 = analyze_worst_windows(wins2, variant="lega")
    assert ana2["category_counts"].get("repetition_attractor") == 3
    assert any("复读" in r["suggestion"] or "重复" in r["suggestion"]
               for r in ana2["recommendations"])
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "worstcases_lega.md")
        write_worstcase_md(ana, path)
        assert "改进方向" in open(path, encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 最差窗口 → 改进方向合成（复制失败/复读）")


def test_window_advice_uses_confident_bias_subset_rate():
    """confident_bias 的“看得到却没抄对”判定必须只看该类窗口。

    其它类别（如 gold 在上下文中的复读窗口）会把全局可见率抬到 0.5 以上；
    若沿用全局率，gold 从未出现的 confident_bias 窗口会被错误地建议去查
    “注意力是否落在 gold 跨度”，把长尾失败误归因为检索失败。
    """
    wins = [
        # confident_bias：高置信错答，gold 从未出现在窗口里
        _win(gold="zzz", tail="no match here",
             pred_top5=[[" red", 0.85]]),
        # repetition_attractor：gold 在上下文尾出现，抬高全局可见率
        _win(gold=" blue", tail="a blue car",
             pred_top5=[[" car", 0.2]]),
    ]
    ana = analyze_worst_windows(wins, variant="lega")
    assert ana["category_counts"]["confident_bias"] == 1
    assert ana["gold_in_context_rate"] == 0.5
    cb_advs = [r for r in ana["recommendations"] if "高置信" in r["trigger"]]
    assert cb_advs and "大多不在上下文中" in cb_advs[0]["trigger"], cb_advs
    print("  [ok] 高置信错答建议按同类别子集的 gold 可见率判定")


def test_cross_improvement_reports_fixed_only_variant():
    """仅改对错误、没有新增错误的变体也必须得到净收益提示。"""
    bad = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
    cross = cross_variant_analysis({"standard": [bad], "lega": []},
                                   reference="standard")
    pv = cross["per_variant"]["lega"]
    assert pv["n_new_vs_reference"] == 0 and pv["n_fixed_vs_reference"] == 1
    assert any("改对 1" in r["trigger"] for r in cross["recommendations"]),         cross["recommendations"]
    print("  [ok] 跨变体：仅改对（无新增）也报告净收益")


def test_cross_improvement_synthesis():
    """跨变体：特异性新增错误按类别统计并给出方法侧方向；共识失败走数据轴。"""
    a = _case_sid("niah_synth@256#0", pred="1234567", ctx_nums=["1234567"])
    b = _case_sid("niah_synth@256#1", pred="7654321", ctx_nums=["7654321"])
    cross = cross_variant_analysis(
        {"standard": [a], "lega": [a, b], "qgate": [b]}, reference="standard")
    lv = cross["per_variant"]["lega"]
    assert lv["new_category_counts"].get("copy_distractor") == 1
    assert lv["new_vs_reference"][0]["category"] == "copy_distractor"
    assert any(r["axis"] == "模型/方法" and "lega" in r["trigger"]
               for r in cross["recommendations"])
    # 全体共识失败主导 → 先怀疑数据/难度而不是方法
    cross2 = cross_variant_analysis(
        {"standard": [a], "lega": [a], "qgate": [a]}, reference="standard")
    assert any(r["axis"] == "数据/难度" for r in cross2["recommendations"])
    json.dumps(cross, ensure_ascii=False, allow_nan=False)
    # 少于 2 个变体：空对照也有 recommendations 键（消费方无需分叉）
    assert cross_variant_analysis({"lega": [a]})["recommendations"] == []
    print("  [ok] 跨变体对照 → 改进方向合成（特异性新增/共识失败）")


def test_analyze_skips_non_dict_records_in_lists():
    """badcases/worst_windows 列表混入 null/标量时跳过坏样本而非崩溃。"""
    ana = analyze_badcases([None, 5, _case(pred="1234567")])
    assert ana["n_badcases"] == 1
    win = analyze_worst_windows([None, "oops", _win()])
    assert win["n_windows"] == 1
    ana2 = process_ruler_json("ignored.json", write_md=False, payload={
        "variant": "lega", "badcases": [None, {"task": "niah_synth",
                                               "pred": "1", "answer": "1"}]})
    assert ana2["n_badcases"] == 1
    # 列表字段本身是非可迭代标量/对象时按空处理，而不是 TypeError。
    assert analyze_badcases(5)["n_badcases"] == 0
    assert analyze_worst_windows({"bad": 1})["n_windows"] == 0
    assert process_eval_json("ignored.json", write_md=False, payload={
        "variant": "lega", "worst_windows": 5})["n_windows"] == 0
    assert cross_variant_analysis([1])["variants"] == []
    assert categorize_case([1])["category"] == "no_answer"
    assert categorize_window("bad")["category"] == "unparsed"
    print("  [ok] 归因列表：非对象样本/标量字段跳过")


def test_categorize_window_tolerates_malformed_top5_and_tail():
    """pred_top5/context_tail 为对象/标量时按缺失处理，不能崩溃。"""
    assert categorize_window({"context_tail": [1, 2],
                              "pred_top5": {"x": 1}})["category"] == "unparsed"
    assert categorize_window({"context_tail": 123,
                              "pred_top5": [[" x", 0.4]]})["category"] \
        == "low_confidence"
    assert categorize_window({"context_tail": None,
                              "pred_top5": "bad"})["category"] == "unparsed"
    # 词项本身缺失/为对象时是“无可用预测分布”，不能被 str() 成
    # "{'token': ...}" / "None" 后当成高置信错答。
    assert categorize_window({"gold_next": "x",
                              "pred_top5": [{"token": "x", "p": 0.9}]}
                             )["category"] == "unparsed"
    assert categorize_window({"gold_next": "x",
                              "pred_top5": [[None, 0.9]]}
                             )["category"] == "unparsed"
    print("  [ok] 窗口归因：畸形 pred_top5/context_tail 安全降级")


def test_write_reports_tolerate_malformed_analysis():
    """手改/旧版 analysis dict 缺字段或类型错误时，报告仍能写出。"""
    tmp = tempfile.mkdtemp()
    try:
        p1 = os.path.join(tmp, "badcase.md")
        p2 = os.path.join(tmp, "worst.md")
        for ana in (None, 5, [], {"n_badcases": "bad", "category_counts": [1],
                                  "axis_counts": {"data": "bad"},
                                  "cases": [1, None,
                                            {"category": [], "depth": "bad"}],
                                  "recommendations": [1, {"axis": "x"}]}):
            write_badcase_md(ana, p1)
            assert os.path.exists(p1)
        for ana in (None, 5, [], {"n_windows": "bad", "category_counts": 5,
                                  "mean_top1_prob": "bad",
                                  "gold_in_context_rate": [1],
                                  "cases": [1, None,
                                            {"category": [],
                                             "top1_prob": "bad"}],
                                  "recommendations": "bad"}):
            write_worstcase_md(ana, p2)
            assert os.path.exists(p2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 坏案例报告：畸形 analysis 字段安全降级")
