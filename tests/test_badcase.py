"""坏案例分析测试：错误归因（标注错误/误抄干扰/幻觉/无回答/fwe）与报告生成。"""
import json
import os
import shutil
import tempfile

from lega.badcase import (analyze_badcases, categorize_case, process_ruler_json,
                           write_badcase_md, CATEGORY_CN)


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
    print("  [ok] 难度边界样本标注（needle 深位）")


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
