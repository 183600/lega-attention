"""防泄漏数据协议测试：时序划分、去污染、token 硬校验、重叠筛查、访问账本。

覆盖用户协议中的三条硬要求：
  1) 时序划分不打乱、文档不跨 split（train 最早 → test 最新；含标题文档
     优先切分与 documents_cross_cut 自检）；
  2) 群组/重复样本不跨侧：与 val/test 重复（精确或空白归一化）的长行只从
     train 移除；train 内部重复同样只改 train（保留首次出现）；val/test
     之间的重复只统计告警、不改写；test split 默认拒绝读取（尘封），需显式
     allow_test=True；
  3) 测试集访问可审计：账本记录每次 test 评测与重复评测的 checkpoint。

另覆盖 bin 缓存完整性：源文本/诊断报告哈希变化时强制重建，防止旧协议
或过期缓存的 bin 被静默训练；以及数据自检告警（alerts）：阈值触发项与
划分协议违规（群组跨 split / 字符切分 / 打乱）都必须显式上报。
"""
import json
import os
import tempfile

import numpy as np

from lega.data import (CACHE_SCHEMA, CharTokenizer, get_tokenizer, load_bin,
                       load_meta, load_tokenizer, prepare_bins)
from lega.datacheck import (TEST_ACCESS_LOG, append_test_access,
                            build_data_report, check_vocab_capacity,
                            critical_split_alerts, decontaminate_documents,
                            decontaminate_texts,
                            distribution_shift, ngram_overlap,
                            leakage_alerts, read_test_access_log, sanitize_text,
                            split_protocol_alerts, temporal_split_text,
                            text_length_stats, token_stream_stats,
                            unknown_char_rate)


def _doc(i):
    return f"Document {i}: " + " ".join(f"word{j}" for j in range(20)) + "."


def test_temporal_split_keeps_documents_whole():
    docs = [_doc(i) for i in range(40)]
    text = "\n\n".join(docs) + "\n"
    train, val, test, info = temporal_split_text(text)
    assert info["order"].startswith("temporal")
    assert info["shuffled"] is False
    assert info["cut_offsets"][0] < info["cut_offsets"][1]
    assert train + val + test == text
    parts = {"train": train, "val": val, "test": test}
    for i, d in enumerate(docs):
        where = [k for k, p in parts.items() if d in p]
        assert len(where) == 1, f"文档 {i} 出现在 {where}"
    ids = {k: [i for i, d in enumerate(docs) if d in p] for k, p in parts.items()}
    assert max(ids["train"]) < min(ids["val"])
    assert max(ids["val"]) < min(ids["test"])
    print("  [ok] 时序划分：文档整体不跨 split，train<val<test，无打乱")


def test_temporal_split_handles_crlf_headings():
    """CRLF 语料的标题/空行边界必须被识别，否则文档会跨 split。"""
    docs = [f"= Doc {i} =\r\n" + " ".join(f"word{j}" for j in range(20)) + "."
            for i in range(40)]
    text = "\r\n\r\n".join(docs) + "\r\n"
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["n_headings"] == 40
    assert info["documents_cross_cut"] is False
    parts = {"train": train, "val": val, "test": test}
    for i, doc in enumerate(docs):
        where = [k for k, p in parts.items() if doc in p]
        assert len(where) == 1, f"CRLF 文档 {i} 出现在 {where}"
    print("  [ok] 时序划分：CRLF 标题/空行边界可识别，文档不跨 split")


def test_temporal_split_single_line_fallback():
    text = "x" * 3000
    train, val, test, info = temporal_split_text(text, window_frac=0.01)
    assert info["fallback_char_cut"] is True
    assert (len(train), len(val), len(test)) == (2400, 300, 300)
    assert train + val + test == text
    print("  [ok] 无换行的单行文本退化为字符切分并显式标注")


def test_temporal_split_preserves_content_with_newlines():
    text = "\n".join(f"line {i} content" for i in range(500))
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["fallback_char_cut"] is False
    assert info["cut_kinds"] == ["newline", "newline"]
    print("  [ok] 换行边界切分不丢内容")


def test_temporal_split_short_text_boundary_after_t2():
    # 仅有的换行落在 t2 之后（末尾换行的小文本很常见）：退化为字符切分
    # 且三段非空。
    for text in ("x" * 67 + "\n" + "x", "y" * 500 + "\n" + "z"):
        train, val, test, info = temporal_split_text(text)
        assert train and val and test
        assert train + val + test == text
        assert info["fallback_char_cut"] is True
        assert 0 < info["cut_offsets"][0] < info["cut_offsets"][1] < len(text)
    print("  [ok] 时序划分：唯一边界晚于 t2 的短文本不产生空 split")


def test_temporal_split_no_phantom_heading_at_slice_start():
    """切片起点落在行中间时，串首 '^' 不是真实行首：不得当作标题切点。

    坏切点会产出 "= abc =" 标题标签但 n_headings=0 的自相矛盾报告，
    同时把换行切点误升为 heading。
    """
    text = "m" * 1343 + "x" + "= abc =\n" + "y" * 648
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["n_headings"] == 0
    assert "heading" not in info["cut_kinds"], info
    # 第二切点附近无换行 → 字符切分并显式标注；首切点用真实换行
    assert info["cut_kinds"][0] == "newline"
    assert text[info["cut_offsets"][0] - 1] == "\n"
    print("  [ok] 时序划分：切片首位的假标题被剔除（不产生幻影 heading）")


def test_temporal_split_heading_guard_uses_usable_headings():
    """窗口内命中的标题若不可用（越过 t2 / 离 p1 太近），必须换全局标题。

    只在“窗口内没标题”时才做全局搜索是不够的：两切点会落在错误的
    窗口内标题上、val 被压空或切序倒置。
    """
    # 触发 A：窗口内 t1 命中 >= t2 的标题（950），p2 命中被压空的 800
    text = ("\n" + "= H =\n" + "m" * 492 + "\n" + "= K =\n"
            + "m" * 443 + "\n" + "= Q =\n" + "m" * 44)
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["cut_offsets"] == [500, 950], info
    assert info["cut_kinds"] == ["heading", "heading"], info
    assert info["fallback_char_cut"] is False
    assert info["documents_cross_cut"] is False
    # 触发 B：窗口内 t2 命中的标题（810）离 p1 太近
    text2 = "m" * 809 + "\n" + "= A =\n" + "m" * 178 + "\n" + "= B =\n"
    train, val, test, info = temporal_split_text(text2)
    assert train + val + test == text2
    assert info["cut_offsets"] == [810, 995], info
    assert info["cut_kinds"] == ["heading", "heading"], info
    assert info["fallback_char_cut"] is False
    print("  [ok] 标题切点必须可用：越过 t2 / 离 p1 太近时换全局最近标题")


def test_temporal_split_prefers_heading_documents_whole():
    """标题文档是群组单位：即使最近标题在窗口外，也必须切在标题上。

    12 个等长文档使 80%/90% 切点附近存在文档边界（1/12≈8.3% 间隔）：
    才能同时满足“两切点都在标题上”与“三 split 非空”，6 个文档时
    90% 切点落在最后一个文档内部，任何算法都无法既不过约又非空。
    """
    docs = []
    for i in range(12):
        body = " ".join(f"w{i}x{j}" for j in range(400))
        docs.append(f"= Doc {i} =\n{body}")
    text = "\n\n".join(docs) + "\n"
    train, val, test, info = temporal_split_text(text)
    assert info["n_headings"] == 12
    assert info["documents_cross_cut"] is False
    assert info["cut_kinds"] == ["heading", "heading"]
    assert train + val + test == text
    parts = {"train": train, "val": val, "test": test}
    idx = {}
    for i, d in enumerate(docs):
        where = [k for k, p in parts.items() if d in p]
        assert len(where) == 1, f"文档 {i} 出现在 {where}"
        idx[where[0]] = idx.get(where[0], []) + [i]
    assert max(idx["train"]) < min(idx["val"])
    assert max(idx["val"]) < min(idx["test"])
    print("  [ok] 标题优先切分：窗口内外均落在文档边界，文档不跨 split")


def test_temporal_split_reports_document_crossing():
    """无法在标题处切分时必须如实报告 documents_cross_cut=True（可审计）。"""
    # 唯一有效标题在 t2 之后：p1 只能退回换行切分 → 有文档跨 split
    body = "\n".join(f"line {i} padding" for i in range(400))
    text = body + "\n= Late =\nfin\n"
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["n_headings"] == 1
    assert info["cut_kinds"][0] != "heading"
    assert info["documents_cross_cut"] is True
    print("  [ok] 标题无法覆盖切点时 documents_cross_cut 显式上报")


def test_heading_regex_matches_multilevel_wikitext_headings():
    from lega.datacheck import _HEADING_RE
    for line in ("= Title =", "== Section ==", "=== Sub ===",
                 "= = Section = =", "= = = Sub = = =", "==NoSpace==",
                 "= Title =\r", "== Section ==\r"):
        assert _HEADING_RE.match(line), f"应视为标题: {line!r}"
    for line in ("x = y = z", "plain text", "= = ", "= = =", ""):
        assert not _HEADING_RE.match(line), f"不应视为标题: {line!r}"
    print("  [ok] 标题正则：一级/多级（含 WikiText-103 分隔写法/CRLF）均可识别")


def test_decontaminate_drops_train_duplicates_only():
    shared = ("This is a long duplicated paragraph that appears in both splits "
              "for testing decontamination behavior.")
    train_only = "A train-only paragraph with sufficient length to be tracked by checks."
    val_only = "A val-only paragraph with sufficient length to be tracked by checks."
    test_only = "A test-only paragraph with sufficient length to be tracked by checks."
    texts = {"train": f"{shared}\n{train_only}", "val": f"{shared}\n{val_only}",
             "test": f"{shared}\n{test_only}"}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert shared not in out["train"]
    assert shared in out["val"] and shared in out["test"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 1
    print("  [ok] 去污染：只从 train 移除与 val/test 重复的长行，val/test 不改写")


def test_decontaminate_ignores_short_lines():
    short = "= References ="
    long_train = "A long training paragraph repeated twice but not in eval sets at all."
    texts = {"train": f"{short}\n{long_train}\n{long_train}",
             "val": f"{short}\nA long validation paragraph not present in train.",
             "test": ""}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert short in out["train"]
    assert out["train"].count(long_train) == 1
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    assert stats["train_intra_split_duplicate_lines"] == 1
    print("  [ok] 短行不参与去污染；train 内部长行重复被删除且仍计数")


def test_decontaminate_counts_short_line_duplicates():
    """短行不参与去污染，但同一 split 内重复的短行仍要计数供审查
    （docstring 承诺；否则重复标题行在报告里完全不可见）。"""
    short = "= References ="
    train_long = "A train-only paragraph long enough to satisfy the length threshold."
    val_long = "A validation paragraph that is distinct and sufficiently long too."
    test_long = "A test paragraph that is distinct from the others and long enough."
    texts = {"train": f"{short}\n{short}\n{train_long}",
             "val": val_long, "test": test_long}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert short in out["train"]
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    # 重复的是短行，同样必须计入 intra-split 重复统计
    assert stats["train_intra_split_duplicate_lines"] == 1
    assert stats["val_intra_split_duplicate_lines"] == 0
    assert stats["test_intra_split_duplicate_lines"] == 0
    print("  [ok] 短行不参与去污染，但重复短行仍被计数供审查")


def test_decontaminate_catches_whitespace_normalized_duplicates():
    """仅空白差异的跨 split 复制同样是泄露，必须只从 train 移除。"""
    val_line = ("A long paragraph repeated with a whitespace variant that must "
                "still be caught by decontamination.")
    train_variant = ("A  long paragraph repeated with a whitespace   variant "
                     "that must still be caught by decontamination.")
    train_only = ("A train-only paragraph with sufficient length to be tracked "
                  "by checks.")
    texts = {"train": f"{train_variant}\n{train_only}", "val": val_line,
             "test": ""}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert train_variant not in out["train"]
    assert train_only in out["train"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    assert stats["train_lines_dropped_as_normalized_duplicates_of_eval"] == 1
    assert stats["normalization"] == "collapse_whitespace"
    print("  [ok] 去污染：空白归一化重复只从 train 移除，val/test 不改写")


def test_decontaminate_catches_nfkc_compatibility_duplicates():
    """全角/表意空格等 Unicode 兼容变体不能绕过重复检测。"""
    val_line = ("Ａ long paragraph with fullwidth A and\u3000an ideographic "
                "space that must still be caught by decontamination.")
    train_variant = ("A long paragraph with fullwidth A and an ideographic "
                     "space that must still be caught by decontamination.")
    train_only = ("A train-only paragraph with sufficient length to be tracked "
                  "by checks.")
    texts = {"train": f"{train_variant}\n{train_only}", "val": val_line,
             "test": ""}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert train_variant not in out["train"]
    assert train_only in out["train"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    assert stats["train_lines_dropped_as_normalized_duplicates_of_eval"] == 1
    assert stats["unicode_normalization"] == "NFKC"
    print("  [ok] 去污染：NFKC 兼容字符变体同样被识别，val/test 不改写")


def test_decontaminate_removes_train_internal_duplicates_only():
    """train 内部重复样本会放大权重/记忆：保留首次出现，删除后续重复。"""
    dup = "A long training paragraph that appears more than once inside train."
    train_only = "Another long training paragraph that is unique in the corpus."
    val_only = "A long validation paragraph that must remain completely untouched."
    test_only = "A long test paragraph that must remain completely untouched."
    texts = {"train": f"{dup}\n{train_only}\n{dup}\n{dup}",
             "val": val_only, "test": test_only}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert out["train"].count(dup) == 1
    assert train_only in out["train"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["train_intra_split_duplicate_lines_removed"] == 2
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    print("  [ok] 去重：train 内部重复只改 train（保留首次出现），val/test 不改写")


def test_decontaminate_documents_catches_multiline_short_line_copies():
    """行级去污染只看 >=40 字符的单行；整段复制由块级逻辑补齐。"""
    block = "Chapter One\nshort a\nshort b"  # 每行都低于行级阈值
    texts = {
        "train": (f"{block}\n\ntrain-only long line with enough characters to stay.\n\n"
                  "second train-only long line with enough characters to stay."),
        "val": (f"{block}\n\nval-only long line with enough characters to stay."),
        "test": "test-only long line with enough characters to stay.",
    }
    out, stats = decontaminate_documents(texts)
    assert block not in out["train"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["train_blocks_dropped_as_duplicates_of_eval"] == 1
    assert stats["train_lines_dropped"] == 3
    assert stats["train_chars_dropped"] > 0
    assert stats["val_blocks_scanned"] == 1
    print("  [ok] 块级去污染：多行短行副本只从 train 删除，val/test 不改写")


def test_decontaminate_documents_handles_heading_blocks_and_eval_overlap():
    section_a = "= A =\nshort a1\nshort a2"
    section_b = "= B =\nshort b1\nshort b2"
    # 两个标题之间没有空行：若标题不开启新块，会被并成一个 6 行块而漏检。
    body = f"{section_a}\n{section_b}"
    texts = {
        "train": f"{body}\n\nunique train long line with enough characters.",
        "val": f"{body}\n\nunique val long line with enough characters.",
        "test": f"{section_a}\n\nunique test long line with enough characters.",
    }
    out, stats = decontaminate_documents(texts)
    assert stats["train_blocks_scanned"] == 2
    assert stats["train_blocks_dropped_as_duplicates_of_eval"] == 2
    assert section_a not in out["train"] and section_b not in out["train"]
    # val/test 之间的块重复只统计不改写
    assert stats["val_test_overlap_blocks"] == 1
    assert section_a in out["val"] and section_a in out["test"]
    print("  [ok] 块级去污染：标题分块、跨 split 删除、val↔test 重叠仅告警")


def test_decontaminate_documents_removes_newly_merged_eval_blocks():
    """删除块后剩余行可能重新拼成与 eval 相同的块，必须继续删到不动点。

    若只扫一遍原始分块，删掉中间标题后前后行会合并成 val 的整段并留在
    train 中；函数也不幂等，二次调用还能删出新内容。
    """
    train = "= A =\n= X =\nx1\nx2\n\nb\n"
    val = "= A =\nb\n\n= X =\nx1\nx2\n"
    out, stats = decontaminate_documents(
        {"train": train, "val": val, "test": ""})
    assert out["train"] == "", repr(out["train"])
    assert stats["train_blocks_dropped_as_duplicates_of_eval"] == 2
    again, _ = decontaminate_documents(out)
    assert again["train"] == out["train"]
    print("  [ok] 块级去污染：删除后新拼出的 eval 副本同样被清除（幂等）")


def test_decontaminate_documents_tolerates_missing_or_bad_input():
    for texts in (None, [], {"val": "x"}, {"train": 123, "val": ["x"]}):
        out, stats = decontaminate_documents(texts)
        assert out["train"] == ""
        assert stats["train_blocks_dropped_as_duplicates_of_eval"] == 0
        assert stats["val_test_overlap_blocks"] == 0
    print("  [ok] 块级去污染：容忍非对象/缺失 train/非字符串输入")


def test_decontaminate_diagnoses_val_test_overlap_without_rewriting():
    """val/test 重叠会削弱 final test 的独立性，但两侧尘封，只统计不改写。"""
    shared = "A long paragraph copied into both evaluation splits for diagnosis."
    texts = {"train": "A long independent training paragraph with enough length.",
             "val": f"{shared}\nA long val-only paragraph with enough length.",
             "test": f"{shared}\nA long test-only paragraph with enough length."}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["val_test_overlap_long_lines"] == 1
    assert stats["val_test_overlap_normalized_long_lines"] == 1
    assert 0 < stats["val_test_overlap_rate"] <= 1
    print("  [ok] val↔test 长行重叠被统计/告警，但两侧数据不被改写")


def test_leakage_alerts_flags_normalized_only_val_test_overlap():
    """仅空白归一化后才相等的 val/test 长行也必须告警，不能只统计。"""
    out = leakage_alerts(
        None, None,
        {"val_test_overlap_long_lines": 0,
         "val_test_overlap_normalized_long_lines": 1},
        None, None, None)
    hits = [a for a in out if a["code"] == "eval_splits_overlap"]
    assert len(hits) == 1
    assert hits[0]["level"] == "warning"
    assert "精确 0 条" in hits[0]["message"]
    assert "共 1 条" in hits[0]["message"]
    print("  [ok] val↔test 仅归一化重复时也触发告警")


def test_critical_split_alerts_filters_actionable_violations():
    report = {"alerts": [
        {"level": "warning", "code": "split_shuffled", "message": "x"},
        {"level": "warning", "code": "split_char_cut", "message": "y"},
        {"level": "warning", "code": "split_documents_cross_cut", "message": "z"},
        {"level": "warning", "code": "ngram_overlap_train_val", "message": "w"},
        {"level": "warning", "code": "ngram_overlap_train_test", "message": "w"},
        {"level": "warning", "code": "ngram_overlap_val_test", "message": "w"},
        {"level": "warning", "code": "eval_splits_overlap", "message": "o"},
        {"level": "warning", "code": "eval_splits_overlap_documents", "message": "o"},
        # 分布漂移警告不属“必须先处理的泄露”：temporal split 下正常存在，
        # 不能因此拦截训练（否则合法时序划分永远无法开跑）。
        {"level": "warning", "code": "distribution_tv_test_vs_train", "message": "d"},
        {"level": "info", "code": "decontamination_applied", "message": "v"},
    ]}
    picked = critical_split_alerts(report)
    assert [a["code"] for a in picked] == [
        "split_shuffled", "split_char_cut", "split_documents_cross_cut",
        "ngram_overlap_train_val", "ngram_overlap_train_test",
        "ngram_overlap_val_test",
        "eval_splits_overlap", "eval_splits_overlap_documents"]
    assert critical_split_alerts({}) == []
    assert critical_split_alerts(None) == []
    assert critical_split_alerts({"alerts": 5}) == []
    assert split_protocol_alerts("not-a-dict") == []
    print("  [ok] critical_split_alerts 只筛出必须先处理的泄露（含跨 split 复制），"
          "分布漂移不拦截")


def test_decontaminate_tolerates_non_dict_and_missing_train():
    """旧缓存/手改 JSON 缺 train 或顶层非对象时按空文本处理，而非崩溃。"""
    for texts in (None, [], {"val": "x"}, {"train": 123, "val": ["x"]}):
        out, stats = decontaminate_texts(texts)
        assert out["train"] == ""
        assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    print("  [ok] decontaminate_texts 容忍非对象/缺失 train")


def test_ngram_overlap_detects_and_clears():
    rng = np.random.default_rng(0)
    leak = rng.integers(0, 1000, size=64).astype(np.uint16)
    train = np.concatenate(
        [rng.integers(0, 1000, size=5000).astype(np.uint16), leak])
    test = np.concatenate(
        [rng.integers(0, 1000, size=3000).astype(np.uint16), leak])
    res = ngram_overlap(train, test, n=64, n_samples=20000, seed=0)
    assert res["windows_found_in_train_sample"] >= 1
    assert res["overlap_rate"] > 0
    # 独立随机流不应检出 64-gram 重叠
    a = rng.integers(10 ** 6, 2 * 10 ** 6, size=5000).astype(np.uint16)
    b = rng.integers(0, 10 ** 6, size=4000).astype(np.uint16)
    res2 = ngram_overlap(a, b, n=64, n_samples=20000, seed=0)
    assert res2["windows_found_in_train_sample"] == 0
    print("  [ok] 64-gram 采样筛查：植入的跨 split 复制被检出，独立流为 0")


def test_token_stream_hard_checks():
    stats = token_stream_stats(np.array([0, 1, 5], dtype=np.uint16), 10, "train")
    assert stats["max_id"] == 5 and stats["nan"] == 0 and stats["inf"] == 0
    # 整数值浮点流合法（如从 JSON/缓存读入），非整数必须显式拒绝：
    # 1.5 截断会静默改 token；-0.5 截断为 0 会绕过越界检查。
    ok = token_stream_stats(np.array([0.0, 3.0]), 10, "train")
    assert ok["min_id"] == 0 and ok["max_id"] == 3
    bad_cases = [
        (np.array([0, 10], dtype=np.uint16), "越界"),
        (np.array([1.0, np.nan]), "NaN"),
        (np.array([1.0, np.inf]), "Inf"),
        (np.array([], dtype=np.uint16), "空流"),
        (np.array([1.5, 2.0]), "非整数"),
        (np.array([-0.5, 1.0]), "非整数"),
    ]
    for bad, name in bad_cases:
        try:
            token_stream_stats(bad, 10, "val")
            raise AssertionError(f"{name} 应抛 ValueError")
        except ValueError:
            pass
    check_vocab_capacity(65536)
    try:
        check_vocab_capacity(65537)
        raise AssertionError("超 uint16 容量应抛 ValueError")
    except ValueError:
        pass
    # 即使 vocab_size 更大（如误传），>65535 的 id 也必须显式拒绝：
    # 写入 uint16 会被静默截断成别的 token。
    try:
        token_stream_stats(np.array([70000], dtype=np.int64), 100000, "train")
        raise AssertionError("超 uint16 的 id 应抛 ValueError")
    except ValueError as e:
        assert "65535" in str(e)
    assert token_stream_stats(np.array([0, 65535], dtype=np.int64),
                              65536, "train")["max_id"] == 65535
    print("  [ok] token 流硬校验：越界/NaN/Inf/空流/非整数/uint16 容量")


def test_char_vocab_fit_on_train_only():
    tok = get_tokenizer("char", train_text="abc def", val_text="!!!zzz")
    assert isinstance(tok, CharTokenizer)
    assert "!" not in tok.stoi and "z" not in tok.stoi
    ids = tok.encode("az")
    assert ids[0] == tok.stoi["a"]
    assert ids[1] == tok.unk_id
    assert tok.itos[tok.unk_id] == "<unk>"
    rate = unknown_char_rate(tok, "a!z")
    assert abs(rate - 2.0 / 3.0) < 1e-12
    # 旧缓存（无 <unk> 标记）保持未知→0，保证与旧 bin 编码一致
    legacy = CharTokenizer(["a", "b"], add_unk=False)
    assert legacy.encode("z") == [0]
    print("  [ok] 字符词表只在 train 拟合；未知字符 → <unk>；旧缓存兼容")


def test_char_tokenizer_keeps_single_unk_at_id_zero():
    """add_unk=True 时旧词表里的 <unk> 不能重复保留到非零 id。

    stoi 后写覆盖前写；若只检查 itos[0] 就前置一个 <unk>，输入中已有的
    <unk> 会让 unk_id 指向后面的位置，未知字符不再映射到 0。
    """
    tok = CharTokenizer(["a", "<unk>", "b"], add_unk=True)
    assert tok.itos == ["<unk>", "a", "b"]
    assert tok.unk_id == 0
    assert tok.encode("az") == [tok.stoi["a"], 0]
    dup = CharTokenizer(["<unk>", "a", "<unk>"], add_unk=True)
    assert dup.itos == ["<unk>", "a"]
    assert dup.unk_id == 0
    print("  [ok] 字符词表 <unk> 唯一且固定在 id 0")


def test_sanitize_text_stats():
    raw = "ok\x00bad\x07line\ufffdend"
    clean, st = sanitize_text(raw)
    assert clean == "okbadline end"
    assert "\x00" not in clean and "\x07" not in clean and "\ufffd" not in clean
    assert st["nul_deleted"] == 1
    assert st["control_chars_deleted"] == 2
    assert st["replacement_char_replaced_with_space"] == 1
    print("  [ok] 文本清洗：NUL/控制符删除、U+FFFD 替换为空格并可计数")


def test_sanitize_text_line_level_label_noise_stats():
    """行级坏字符统计：标签噪声按行比例报告，而不是只看总字符数。"""
    raw = ("clean line\nbad \ufffd line\ncontrol \x07 line\n"
           "both \ufffd and \x01 here\nclean again")
    clean, st = sanitize_text(raw)
    assert "\ufffd" not in clean and "\x07" not in clean and "\x01" not in clean
    assert st["lines"] == 5
    assert st["lines_with_replacement_char"] == 2
    assert st["lines_with_control_chars"] == 2
    assert st["lines_with_bad_chars"] == 3
    assert abs(st["bad_char_line_rate"] - 3.0 / 5.0) < 1e-12
    print("  [ok] 文本清洗：行级标签噪声统计（含两类坏字符的行只计一次）")


def test_text_length_stats_outlier_detection():
    text = "short\n" + "x" * 200001 + "\nmedium line"
    st = text_length_stats(text, huge_line_chars=100000)
    assert st["lines"] == 3
    assert st["line_chars_max"] == 200001
    assert st["lines_gt_huge_threshold"] == 1
    print("  [ok] 行长度统计：超长行作为异常值被单独计数")


def test_write_clean_lines_preserves_blank_line_boundaries():
    """空行是段落/文档边界：清洗后至多保留一个，不能被全部丢弃。"""
    import io
    from scripts.prepare_data import write_clean_lines

    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "clean.txt")
        n = write_clean_lines(
            io.StringIO("first\n\n\nsecond\n  \nthird\n"), dst, 0)
        with open(dst, "r", encoding="utf-8") as f:
            text = f.read()
        assert text == "first\n\nsecond\n\nthird"
        assert n == len(text)
        # 截断仍只发生在完整行边界
        dst2 = os.path.join(d, "truncated.txt")
        write_clean_lines(io.StringIO("aaaa\n\nbbbb\n\ncccc\n"), dst2, 1)
        with open(dst2, "r", encoding="utf-8") as f:
            assert f.read() == "aaaa"
    print("  [ok] 数据准备：空行块边界被保留，截断仍只发生在完整行")


def test_verify_splits_disjoint_binds_claim_to_content():
    """官方 split 的 document_disjoint 声明必须与实际文本绑定核对。"""
    from scripts.prepare_data import verify_splits_disjoint

    shared = "A shared line with enough characters to count as duplicate."
    with tempfile.TemporaryDirectory() as d:
        paths = {s: os.path.join(d, f"{s}.txt") for s in
                 ("train", "val", "test")}
        # 整洁：三份互不相同
        for s, p in paths.items():
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"unique {s} line with plenty of characters here.\n")
        r = verify_splits_disjoint(paths)
        assert r["verified_disjoint"] is True
        # 注入 val/test 共享长行 → 声明被证伪
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write(f"{shared}\nval unique with plenty of characters.\n")
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write(f"{shared}\ntest unique with plenty of characters.\n")
        r = verify_splits_disjoint(paths)
        assert r["verified_disjoint"] is False
        assert r["overlap_long_lines"]["val_test"]["exact_lines"] == 1
        # 短行（标题）不计入，避免误报
        with open(paths["train"], "w", encoding="utf-8") as f:
            f.write("= References =\ntrain unique plenty of characters here.\n")
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write("= References =\nval unique plenty of characters here.\n")
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write("test unique plenty of characters here.\n")
        assert verify_splits_disjoint(paths)["verified_disjoint"] is True
    print("  [ok] verify_splits_disjoint：声明与文本核对，短行不误报")


def test_distribution_shift_metric():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=100000).astype(np.uint16)
    d_same = distribution_shift(a, a.copy(), 100)
    assert d_same["tv"] < 1e-12 and d_same["js_bits"] < 1e-12
    b = rng.integers(100, 200, size=100000).astype(np.uint16)
    d_diff = distribution_shift(a, b, 200)
    assert d_diff["tv"] > 0.9
    assert d_diff["eval_oov_token_rate"] > 0.9
    # 默认全量精确计数：train 中只出现一次的 token 也必须算已见，不能因
    # 采样漏掉而误报 OOV。
    rare = np.ones(10000, dtype=np.uint16)
    rare[0] = 7
    d_rare = distribution_shift(rare, np.array([7, 1], dtype=np.uint16), 10)
    assert d_rare["count_mode"] == "exact"
    assert d_rare["tokens_train"] == rare.size
    assert d_rare["eval_oov_token_rate"] == 0
    # 显式 max_samples 仍保留有放回采样路径，并标记 count_mode 供审计。
    d_sample = distribution_shift(rare, np.array([7, 1], dtype=np.uint16), 10,
                                  max_samples=1000)
    assert d_sample["count_mode"] == "sampled"
    assert d_sample["tokens_train"] == 1000
    print("  [ok] 分布漂移：精确计数避免 OOV 假阳性；显式采样可审计")


def test_test_access_ledger():
    with tempfile.TemporaryDirectory() as d:
        assert read_test_access_log(d) == []
        s1 = append_test_access(d, {"ckpt_sha256": "aaa", "ckpt": "/x/a.pt"})
        assert s1["evaluations"] == 1 and s1["distinct_checkpoints"] == 1
        s2 = append_test_access(d, {"ckpt_sha256": "bbb", "ckpt": "/x/b.pt"})
        assert s2["evaluations"] == 2 and s2["distinct_checkpoints"] == 2
        assert s2["checkpoints_evaluated_multiple_times"] == []
        s3 = append_test_access(d, {"ckpt_sha256": "aaa", "ckpt": "/x/a.pt"})
        assert s3["evaluations"] == 3 and s3["distinct_checkpoints"] == 2
        assert s3["checkpoints_evaluated_multiple_times"] == ["aaa"[:16]]
        assert len(read_test_access_log(d)) == 3
    print("  [ok] 测试集访问账本：追加、去重统计、重复评测告警")


def test_prepare_bins_writes_report_and_decontaminates():
    shared = ("A shared paragraph with enough characters to count as a long "
              "duplicate line.")
    train_unique = ("A train-only line that provides enough characters for "
                    "tokenization checks.")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(f"{shared}\n{train_unique}\n{train_unique}\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(f"{shared}\nval unique text § with enough characters here.\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("test unique text with enough characters for held out use.\n")
        meta = prepare_bins(d, kind="char", force=True)
        assert meta["fit_on"] == "train"
        assert meta["char_unk"] is True
        with open(os.path.join(d, "data_report.json"), "r",
                  encoding="utf-8") as f:
            rep = json.load(f)
        assert rep["duplicates"]["train_lines_dropped_as_duplicates_of_eval"] == 1
        assert rep["tokenizer"]["fit_on"] == "train"
        # 协议声明的拟合来源必须与分词器实际来源一致：字符级=train、
        # BPE=gpt2_pretrained；写死 "train_only" 会与 tokenizer.fit_on 矛盾，
        # 使“无全量拟合”的审计声明失真。
        assert rep["protocol"]["fit_on"] == rep["tokenizer"]["fit_on"]
        assert rep["tokenizer"]["unk_rate"]["val"] > 0  # 独有的 § 不在 train 词表
        # 建模前读 test 仅用于编码/诊断，必须显式记录且标注不用于选择
        assert rep["test_reads_at_prep"]["encoded"] is True
        assert rep["test_reads_at_prep"]["used_for_selection"] is False
        tok = load_tokenizer(d)
        assert "§" not in tok.stoi
        # 缓存命中：第二次调用不重建（token 数一致），且报告仍存在
        meta2 = prepare_bins(d, kind="char")
        assert meta2["train_tokens"] == meta["train_tokens"]
        assert load_bin(d, "train").size == meta["train_tokens"]
    print("  [ok] prepare_bins：去污染 + train-only 词表 + data_report + 缓存命中")


def test_prepare_bins_applies_document_level_decontamination():
    """多行短行副本必须由 prepare_bins 真正删除并写入报告/告警。"""
    block = "Doc block line one\nDoc block line two"
    train_unique = ("A train-only paragraph with enough characters to survive "
                    "line-level decontamination.")
    val_unique = ("A val-only paragraph with enough characters to survive "
                  "line-level decontamination.")
    test_unique = ("A test-only paragraph with enough characters for held-out "
                   "evaluation.")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(f"{block}\n\n{train_unique}\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(f"{block}\n\n{val_unique}\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write(f"{test_unique}\n")
        meta = prepare_bins(d, kind="char", force=True)
        with open(os.path.join(d, "data_report.json"), "r",
                  encoding="utf-8") as f:
            rep = json.load(f)
        doc = rep["duplicates"]["document_level"]
        assert doc["train_blocks_dropped_as_duplicates_of_eval"] == 1
        assert doc["train_lines_dropped"] == 2
        assert doc["val_test_overlap_blocks"] == 0
        codes = {a["code"] for a in rep["alerts"]}
        assert "document_decontamination_applied" in codes
        decoded = load_tokenizer(d).decode(load_bin(d, "train"))
        assert "Doc block line one" not in decoded
        assert "train-only paragraph" in decoded
        assert meta["cache_schema"] == CACHE_SCHEMA
    print("  [ok] prepare_bins：块级去污染删除多行短行副本并写入报告/告警")


def _write_min_splits(d: str) -> None:
    with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
        f.write("alpha beta gamma delta epsilon zeta eta theta iota kappa.\n")
    with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
        f.write("val text with enough characters to survive sanitization.\n")
    with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
        f.write("test text with enough characters to survive sanitization.\n")


def test_prepare_bins_cache_invalidated_on_source_or_report_change():
    """源 txt 或诊断报告被改动后，旧 bin 缓存必须失效重建（不得静默沿用）。"""
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        meta1 = prepare_bins(d, kind="char", force=True)
        assert meta1["cache_schema"] == CACHE_SCHEMA
        assert set(meta1["source_sha256"]) == {"train", "val", "test"}
        assert isinstance(meta1["report_sha256"], str)
        # 未改动：缓存命中
        meta2 = prepare_bins(d, kind="char")
        assert meta2["train_tokens"] == meta1["train_tokens"]
        assert meta2["report_sha256"] == meta1["report_sha256"]
        # 改 train.txt（追加内容）→ 源文本哈希变化，必须重建
        with open(os.path.join(d, "train.txt"), "a", encoding="utf-8") as f:
            f.write("additional training content appended.\n")
        meta3 = prepare_bins(d, kind="char")
        assert meta3["source_sha256"]["train"] != meta1["source_sha256"]["train"]
        # 篡改诊断报告 → 报告哈希不匹配，重建并恢复完整报告
        with open(os.path.join(d, "data_report.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        prepare_bins(d, kind="char")
        with open(os.path.join(d, "data_report.json"), "r",
                  encoding="utf-8") as f:
            rep = json.load(f)
        assert "protocol" in rep and "alerts" in rep
    print("  [ok] bin 缓存完整性：源文本/诊断报告哈希变化即失效重建")


def test_prepare_bins_bins_only_cache_hit_without_sources():
    """纯 bin 数据目录（无任何源 txt）必须按不可判定放行缓存命中，
    而不是在重建路径上因缺少 train.txt 直接 FileNotFoundError。"""
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        meta1 = prepare_bins(d, kind="char", force=True)
        for name in ("train.txt", "val.txt", "test.txt"):
            os.remove(os.path.join(d, name))
        meta2 = prepare_bins(d, kind="char")
        assert meta2["train_tokens"] == meta1["train_tokens"]
        assert meta2["val_tokens"] == meta1["val_tokens"]
        assert meta2["test_tokens"] == meta1["test_tokens"]
    print("  [ok] prepare_bins：无源 txt 的纯 bin 目录缓存命中（不误报缺失）")


def test_make_dummy_data_removes_stale_test_when_disabled():
    """--test_words<=0 必须清除残留 test.txt/test.bin，否则旧 test split
    会被 force 重建，与“评测回退 val”的口径矛盾。"""
    import importlib.util
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "make_dummy_data.py")
    spec = importlib.util.spec_from_file_location("_make_dummy_data", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as d:
        def run(test_words):
            argv = sys.argv
            sys.argv = ["make_dummy_data", "--out_dir", d, "--train_words",
                        "60", "--val_words", "30", "--test_words",
                        str(test_words)]
            try:
                mod.main()
            finally:
                sys.argv = argv
        run(20)
        assert os.path.exists(os.path.join(d, "test.txt"))
        run(0)
        assert not os.path.exists(os.path.join(d, "test.txt"))
        assert not os.path.exists(os.path.join(d, "test.bin"))
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            assert "test_tokens" not in json.load(f)
    print("  [ok] make_dummy_data：--test_words<=0 清除残留 test.txt/test.bin")


def test_load_bin_test_split_is_sealed_by_default():
    """test split 默认拒绝读取；唯一出口 allow_test=True（evaluate 专用）。"""
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        prepare_bins(d, kind="char", force=True)
        try:
            load_bin(d, "test")
            raise AssertionError("未显式放行时读取 test 必须抛 PermissionError")
        except PermissionError:
            pass
        assert load_bin(d, "test", allow_test=True).size > 0
        # train/val 不受影响
        assert load_bin(d, "train").size > 0
        assert load_bin(d, "val").size > 0
    print("  [ok] 测试集尘封：load_bin('test') 默认拒绝，allow_test=True 显式放行")


def test_build_data_report_emits_threshold_alerts():
    """防泄漏自检必须把超阈值的诊断转成显式告警，而不是只躺在报告里。"""
    tokenizer = {"kind": "char", "fit_on": "train", "vocab_size": 50,
                 "unk_rate": {"val": 0.2, "test": 0.0}}
    sanitization = {"train": {"bad_char_line_rate": 0.0},
                    "val": {"bad_char_line_rate": 0.05}}
    duplicates = {"train_lines_dropped_as_duplicates_of_eval": 1,
                  "train_lines_dropped_as_normalized_duplicates_of_eval": 2}
    contamination = {"train_val": {"n": 64, "overlap_rate": 0.01},
                     "train_test": {"n": 64, "overlap_rate": 0.0}}
    distribution = {
        "val_vs_train": {"tv": 0.2, "js_bits": 0.01,
                         "eval_oov_token_rate": 0.0},
        "test_vs_train": {"tv": 0.01, "js_bits": 0.2,
                          "eval_oov_token_rate": 0.5}}
    text_stats = {"train": {"lines_gt_huge_threshold": 2,
                            "huge_line_threshold_chars": 100000}}
    rep = build_data_report(tokenizer, {}, sanitization, duplicates,
                            contamination, distribution, text_stats)
    codes = {a["code"] for a in rep["alerts"]}
    for expected in ("ngram_overlap_train_val", "distribution_tv_val_vs_train",
                     "distribution_js_test_vs_train", "eval_oov_test_vs_train",
                     "unk_char_rate_val", "bad_char_lines_val",
                     "decontamination_applied", "huge_lines_train"):
        assert expected in codes, f"缺少告警 {expected}: {sorted(codes)}"
    # 未超阈值的方向不产生告警
    assert "ngram_overlap_train_test" not in codes
    assert "distribution_tv_test_vs_train" not in codes
    assert all(a["level"] in ("warning", "info") for a in rep["alerts"])
    # 阈值本身随报告落盘，供审计确认告警口径
    assert "ngram_overlap_rate" in rep["alert_thresholds"]
    print("  [ok] 数据自检告警：重叠/漂移/OOV/坏字符行/去污染/超长行均可审计")


def test_build_data_report_flags_normalized_and_short_intra_duplicates():
    """val/test 内部重复告警必须覆盖归一化重复，且不能把短行重复漏掉。

    decontaminate_texts 的 intra-split 统计覆盖所有非空行，归一化计数也包含
    精确重复；报告若只读精确计数会把“仅空白差异”的重复静默掉，文案写
    “长行”还会与统计口径矛盾。
    """
    duplicates = {
        "val_intra_split_duplicate_lines": 0,
        "val_intra_split_normalized_duplicate_lines": 2,
        "test_intra_split_duplicate_lines": 1,
        "test_intra_split_normalized_duplicate_lines": 3,
    }
    rep = build_data_report(
        {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}},
        {}, {}, duplicates, {}, {}, {})
    by_code = {a["code"]: a for a in rep["alerts"]}
    assert "val_intra_duplicates" in by_code
    assert "test_intra_duplicates" in by_code
    val_msg = by_code["val_intra_duplicates"]["message"]
    test_msg = by_code["test_intra_duplicates"]["message"]
    assert "2" in val_msg and "精确 0" in val_msg
    assert "3" in test_msg and "精确 1" in test_msg
    assert "长行" not in val_msg and "长行" not in test_msg
    print("  [ok] 内部重复告警：归一化差异与短行重复均可见，文案口径一致")


def test_split_protocol_alerts_flag_group_crossing():
    """群组跨 split / 字符切分 / 打乱必须在 alerts 中显式告警，不能只留在元数据。"""
    assert split_protocol_alerts(None) == []
    assert split_protocol_alerts({}) == []
    clean = {"mode": "temporal", "shuffled": False,
             "fallback_char_cut": False, "documents_cross_cut": False,
             "group_unit": "heading_document", "n_headings": 3}
    assert split_protocol_alerts(clean) == []
    alerts = split_protocol_alerts({
        "shuffled": True, "fallback_char_cut": True,
        "documents_cross_cut": True, "group_unit": "heading_document",
        "n_headings": 1, "cut_kinds": ["newline", "newline"]})
    codes = {a["code"] for a in alerts}
    assert codes == {"split_shuffled", "split_char_cut",
                     "split_documents_cross_cut"}
    assert all(a["level"] == "warning" for a in alerts)
    cross = next(a for a in alerts if a["code"] == "split_documents_cross_cut")
    assert "heading_document" in cross["message"]
    print("  [ok] 划分协议告警：打开 shuffled / 字符切分 / 文档跨切点均显式上报")


def test_build_data_report_surfaces_group_crossing_alerts():
    """带 split_protocol 的报告：协议告警置顶，避免被阈值告警挤出汇总页。"""
    rep = build_data_report(
        {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}},
        {"train": {"tokens": 10}}, {}, {}, {}, {},
        extra={"split_protocol": {
            "mode": "temporal", "shuffled": False, "fallback_char_cut": False,
            "group_unit": "heading_document", "n_headings": 1,
            "documents_cross_cut": True}})
    assert rep["alerts"][0]["code"] == "split_documents_cross_cut"
    assert rep["split_protocol"]["documents_cross_cut"] is True
    # 官方/独立生成 split 无违规字段时不产生协议告警
    rep2 = build_data_report(
        {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}},
        {"train": {"tokens": 10}}, {}, {}, {}, {},
        extra={"split_protocol": {"mode": "official", "shuffled": False,
                                  "documents_cross_cut": False}})
    assert all(not a["code"].startswith("split_") for a in rep2["alerts"])
    print("  [ok] data_report：协议违规告警置顶；无违规协议不误报")


def test_test_access_ledger_skips_non_dict_lines():
    """账本中的 null/数组/坏 JSON 行不能进入统计或让追加崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        append_test_access(d, {"ckpt_sha256": "aaa"})
        log_path = os.path.join(d, TEST_ACCESS_LOG)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("null\n[1, 2]\n不是 JSON\n")
        records = read_test_access_log(d)
        assert len(records) == 1 and records[0]["ckpt_sha256"] == "aaa"
        summary = append_test_access(d, {"ckpt_sha256": "bbb"})
        assert summary["evaluations"] == 2
        assert len(read_test_access_log(d)) == 2
    print("  [ok] 测试集访问账本：非对象行按坏行跳过")


def test_prepare_bins_rebuilds_on_non_object_meta_cache():
    """meta.json 语法合法但非对象时按无缓存处理并重建，而非崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write("train text with enough characters for tokenization checks.\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write("val text with enough characters for tokenization checks.\n")
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(b"\x00" * 16)
        with open(os.path.join(d, "val.bin"), "wb") as f:
            f.write(b"\x00" * 16)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        meta = prepare_bins(d, kind="char", force=False)
        assert meta["tokenizer"] == "char" and meta["train_tokens"] > 0
        saved = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        assert isinstance(saved, dict) and saved["train_tokens"] > 0
    print("  [ok] 非对象 meta 缓存：按缺失处理并重建")


def test_prepare_bins_rebuilds_on_malformed_cache_schema():
    """meta.json 结构合法但 cache_schema 类型异常时按旧缓存重建，而非崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        prepare_bins(d, kind="char", force=True)
        for bad in (None, [3], {"v": 3}, "not-a-version"):
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                saved = json.load(f)
            saved["cache_schema"] = bad
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(saved, f)
            meta = prepare_bins(d, kind="char", force=False)
            assert meta["cache_schema"] == CACHE_SCHEMA and meta["train_tokens"] > 0
    print("  [ok] 异常 cache_schema：按旧缓存重建而非崩溃")


def test_load_meta_rejects_broken_json_with_clear_error():
    """meta.json 损坏或顶层非对象时必须抛 ValueError，而非深层 TypeError。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        try:
            load_meta(d)
        except ValueError as e:
            assert "meta.json" in str(e)
        else:
            raise AssertionError("损坏 JSON 未抛 ValueError")
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        try:
            load_meta(d)
        except ValueError as e:
            assert "JSON 对象" in str(e)
        else:
            raise AssertionError("非对象 meta 未抛 ValueError")
    print("  [ok] load_meta：坏 JSON/非对象给出明确 ValueError")


def test_leakage_alerts_tolerates_malformed_nested_sections():
    """data_report 的嵌套诊断被手改成标量/列表时仍返回告警列表。"""
    assert leakage_alerts(None, None, None, None, None, None, None) == []
    out = leakage_alerts(
        tokenizer={"unk_rate": {"val": "bad"}},
        sanitization={"val": [1]},
        duplicates={"train_lines_dropped_as_duplicates_of_eval": "bad"},
        contamination={"train_val": [1]},
        distribution={"val_vs_train": {"tv": "bad", "js_bits": [1]}},
        text_stats={"train": {"lines_gt_huge_threshold": "bad"}},
        thresholds=[1])
    assert out == []
    print("  [ok] leakage_alerts：畸形嵌套诊断不崩溃")
