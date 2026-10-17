"""坏案例分析测试：错误归因（标注错误/误抄干扰/幻觉/无回答/fwe）、语言建模
最差窗口归因、跨变体错误对照与报告生成。"""
import json
import os
import shutil
import tempfile

from lega.badcase import (CATEGORY_CN, analyze_badcases, analyze_worst_windows,
                           case_key, categorize_case, categorize_window,
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
    """外部/旧格式 JSON 的字符串 depth 不能让整个归因流程崩溃。"""
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
    print("  [ok] 非数值 depth 安全归一化（不崩溃、不误判难度）")


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
    # 无预测分布（旧格式）不崩溃
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
        # 旧格式（无 worst_windows）不崩溃
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
        # 无 badcases 键的旧格式文件不崩溃
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
        {"task": "niah_synth", "depth": {"a": 1}, "pred": None,
         "answer": 123, "context_numbers": None, "needle_excerpt": 42},
        {},  # 缺 task
    ]
    ana = analyze_badcases(bad, variant="lega")
    assert ana["n_badcases"] == len(bad)
    for i, c in enumerate(ana["cases"], 1):
        format_case(i, c)  # 渲染不得崩溃
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
    print("  [ok] 坏样本字段类型异常不崩溃（fwe/缺 task/非字符串字段）")


def test_main_keeps_zero_error_variants_for_cross_variant():
    """badcase.main 汇总时必须把 "badcases": []（已评测且零错误）传给跨变体
    对照：否则全对变体被静默排除，方法特异性失败会被误报为共识失败。"""
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] badcase.main 保留零错误变体进入跨变体对照")
