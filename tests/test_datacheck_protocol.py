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
import shutil
import sys
import tempfile

import numpy as np

from lega.data import (CACHE_SCHEMA, CharTokenizer, get_tokenizer, load_bin,
                       load_meta, load_tokenizer, prepare_bins)
from lega.datacheck import (TEST_ACCESS_LOG, append_test_access,
                            build_data_report, check_vocab_capacity,
                            critical_split_alerts, decontaminate_documents,
                            decontaminate_near_duplicates,
                            decontaminate_texts,
                            distribution_shift, ngram_overlap,
                            CRITICAL_SPLIT_ALERT_CODES,
                            leakage_alerts, post_cleanup_residual,
                            read_test_access_log, sanitize_text,
                            short_ngram_overlap, short_overlap_alerts,
                            split_protocol_alerts, temporal_split_text,
                            text_length_stats, token_stream_stats,
                            unknown_char_rate,
                            NEAR_DUP_ROWS, _near_dup_band_key)


from tests.datacheck_helpers import _doc, _write_min_splits

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


def test_temporal_split_keeps_blank_line_paragraphs_whole():
    """无标题语料的空行段落是群组单位：即使最近段落边界在窗口外也要切在段落上。"""
    text = "A" * 90000 + "\n\n" + "B" * 5000 + "\n\n" + "C" * 5000
    train, val, test, info = temporal_split_text(text, window_frac=0.01)
    assert train + val + test == text
    assert info["group_unit"] == "blank_line_paragraph"
    assert info["n_paragraph_boundaries"] == 2, info
    assert info["cut_offsets"] == [90002, 95004], info
    assert info["cut_kinds"] == ["blank", "blank"], info
    assert info["paragraphs_cross_cut"] is False, info
    assert info["documents_cross_cut"] is False, info
    assert "B" * 5000 in val and "C" * 5000 in test
    print("  [ok] 无标题语料：空行段落整体不跨 split")


def test_temporal_split_reports_paragraph_crossing():
    """段落长于可用划分区间时必须如实报告 paragraphs_cross_cut=True（可审计）。"""
    text = "A" * 110000 + "\n\n" + "B" * 10000 + "\n"
    train, val, test, info = temporal_split_text(text, window_frac=0.02)
    assert train + val + test == text
    assert info["group_unit"] == "blank_line_paragraph"
    assert info["paragraphs_cross_cut"] is True, info
    assert info["fallback_char_cut"] is True, info
    print("  [ok] 段落无法覆盖切点时 paragraphs_cross_cut 显式上报")


def test_heading_regex_matches_multilevel_wikitext_headings():
    from lega.datacheck import _HEADING_RE
    for line in ("= Title =", "== Section ==", "=== Sub ===",
                 "= = Section = =", "= = = Sub = = =", "==NoSpace==",
                 "= Title =\r", "== Section ==\r"):
        assert _HEADING_RE.match(line), f"应视为标题: {line!r}"
    for line in ("x = y = z", "plain text", "= = ", "= = =", ""):
        assert not _HEADING_RE.match(line), f"不应视为标题: {line!r}"
    print("  [ok] 标题正则：一级/多级（含 WikiText-103 分隔写法/CRLF）均可识别")


def test_temporal_split_rejects_invalid_fractions():
    """比例非法时必须在切分前报错，不能被 max/min 夹成"看似成功"的空 split。"""
    text = "x" * 1000
    bad_kwargs = [
        {"train_frac": 0}, {"train_frac": -0.1}, {"train_frac": 1.0},
        {"train_frac": 1.2}, {"train_frac": float("nan")},
        {"val_frac": 0}, {"val_frac": -0.1}, {"val_frac": 1.0},
        {"train_frac": 0.95, "val_frac": 0.1},
        {"window_frac": 0}, {"window_frac": -0.1},
        {"window_frac": 1.1}, {"window_frac": float("nan")},
        {"window_frac": float("inf")}, {"train_frac": True},
        {"train_frac": "0_5"},
    ]
    for kwargs in bad_kwargs:
        try:
            temporal_split_text(text, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法比例未被拒绝: {kwargs}")
    # 合法字符串比例仍可用
    train, val, test, _ = temporal_split_text(
        text, train_frac="0.8", val_frac="0.1", window_frac="0.02")
    assert train and val and test
    # min_gap 是切片算术参数：bool/小数/非数值/负数必须显式拒绝，
    # 不能 int() 截断后继续切分。
    for bad in (True, 1.5, "bad", None, -1):
        try:
            temporal_split_text(text, min_gap=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 min_gap 未被拒绝: {bad!r}")
    train, val, test, _ = temporal_split_text(text, min_gap="2")
    assert train and val and test
    print("  [ok] 时序划分：非法比例/min_gap 在切分前显式拒绝")


def test_prepare_data_refuses_group_crossing_local_split(monkeypatch):
    """本地时序划分检测到标题文档跨 split 时必须拒绝写出，而非只告警。"""
    from scripts.prepare_data import main as prepare_main
    body = "\n".join(f"line {i} padding" for i in range(400))
    text = body + "\n= Late =\nfin\n"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out_dir = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--local_txt", src,
            "--source", "local"])
        try:
            prepare_main()
            raise AssertionError("群组跨 split 的本地划分应被拒绝")
        except SystemExit as e:
            assert "拒绝写出泄露划分" in str(e)
            assert "--allow_leaky_split" in str(e)
        assert not os.path.exists(os.path.join(out_dir, "train.txt"))
    print("  [ok] prepare_data：群组跨 split 的本地划分 fail closed")


def test_prepare_data_refuses_paragraph_crossing_local_split(monkeypatch):
    """本地时序划分检测到空行段落跨 split 时同样必须拒绝写出。"""
    from scripts.prepare_data import main as prepare_main
    text = "A" * 110000 + "\n\n" + "B" * 10000 + "\n"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out_dir = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--local_txt", src,
            "--source", "local"])
        try:
            prepare_main()
            raise AssertionError("段落跨 split 的本地划分应被拒绝")
        except SystemExit as e:
            assert "拒绝写出泄露划分" in str(e)
            assert "段落" in str(e)
        assert not os.path.exists(os.path.join(out_dir, "train.txt"))
    print("  [ok] prepare_data：段落跨 split 的本地划分 fail closed")


def test_prepare_data_refuses_val_test_overlap(monkeypatch):
    """val/test 共享长行无法靠只改 train 修复，准备阶段必须拒绝写出。"""
    import lega.datacheck as dc
    from scripts.prepare_data import main as prepare_main
    shared = "shared held-out line " + "x" * 80

    def fake_split(text, *args, **kwargs):
        train = "train line " + "a" * 120 + "\n"
        val = shared + "\n"
        test = shared + "\n"
        return train, val, test, {
            "order": "temporal:train(earliest)->val->test(latest)",
            "cut_offsets": [len(train), len(train) + len(val)],
            "cut_kinds": ["newline", "newline"], "window_chars": 1,
            "fallback_char_cut": False, "shuffled": False, "n_headings": 0,
            "documents_cross_cut": False, "group_unit": "line_segment",
            "lengths_chars": {"train": len(train), "val": len(val),
                              "test": len(test)}}

    monkeypatch.setattr(dc, "temporal_split_text", fake_split)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out_dir = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write("ignored by fake split\n")
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--local_txt", src,
            "--source", "local"])
        try:
            prepare_main()
            raise AssertionError("val/test 重叠的划分应被拒绝")
        except SystemExit as e:
            assert "拒绝写出泄露划分" in str(e)
            assert "val/test 共享长行" in str(e)
    print("  [ok] prepare_data：val/test 共享长行 fail closed")


def test_prepare_data_rejects_negative_max_tokens(monkeypatch):
    """负数 --max_tokens 会让首行即被截断，必须在写文件前报错。"""
    import io
    from scripts.prepare_data import main as prepare_main, write_clean_lines

    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "clean.txt")
        try:
            write_clean_lines(io.StringIO("a line\n"), dst, -1)
        except ValueError:
            pass
        else:
            raise AssertionError("write_clean_lines 未拒绝负 max_tokens")
        out_dir = os.path.join(d, "out")
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--max_tokens", "-5"])
        try:
            prepare_main()
        except SystemExit as e:
            assert "不能为负" in str(e), e
        else:
            raise AssertionError("prepare_data 未拒绝负 max_tokens")
        assert not os.path.exists(out_dir)
    print("  [ok] prepare_data/write_clean_lines：负 max_tokens 显式拒绝")


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
    # 阈值是计数身份：bool/小数/非数值不能被 int() 截断成另一个长度口径
    for bad in (True, 40.5, "bad", None, -1):
        try:
            decontaminate_texts(texts, min_chars=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 min_chars 未被拒绝: {bad!r}")
    _, stats_str = decontaminate_texts(texts, min_chars="40")
    assert stats_str["min_chars"] == 40
    print("  [ok] 短行不参与去污染；train 内部长行重复被删除且仍计数；"
          "min_chars 严格整数化")


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


def test_decontaminate_reports_short_line_cross_split_overlap():
    """短行跨 split 复制必须显式计数：不自动删除，但不能成为静默盲区。"""
    short = "shared short heading"
    texts = {
        "train": f"{short}\ntrain long line with enough characters to survive.",
        "val": f"{short}\nval long line with enough characters to survive.",
        "test": "test long line with enough characters to survive.",
    }
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert short in out["train"]
    assert stats["train_val_overlap_short_lines"] == 1
    assert stats["train_test_overlap_short_lines"] == 0
    assert stats["val_test_overlap_short_lines"] == 0
    assert stats["train_val_short_overlap_rate"] > 0
    # NFKC/空白归一化后的短行同样计入（全角 Ａ + 多空格 vs 普通 A 单空格）
    texts2 = {"train": "Ａ  shared   short\n" + "x" * 50,
              "val": "A shared short\n" + "y" * 50, "test": ""}
    _, stats2 = decontaminate_texts(texts2, min_chars=40)
    assert stats2["train_val_overlap_short_lines"] == 0
    assert stats2["train_val_overlap_normalized_short_lines"] == 1
    print("  [ok] 短行跨 split 复制：精确与归一化计数均可见（不静默）")


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


def test_decontaminate_classifies_long_lines_after_normalization():
    """NFKC 展开后的长行不能被原始长度误判为短行而逃过去污染。"""
    # 20 个连字原始长度 20，NFKC 后变成 40 个字符；对照行是 40 个普通字符。
    ligature_line = "ﬀ" * 20
    plain_line = "f" * 40
    train_only = "A train-only paragraph with enough characters to survive."
    texts = {"train": f"{ligature_line}\n{train_only}",
             "val": plain_line, "test": "A test-only paragraph with enough chars."}
    out, stats = decontaminate_texts(texts, min_chars=40)
    assert ligature_line not in out["train"]
    assert train_only in out["train"]
    assert stats["train_lines_dropped_as_normalized_duplicates_of_eval"] == 1
    assert stats["train_val_overlap_normalized_short_lines"] == 0
    print("  [ok] 去污染：长短行按归一化后长度分类，NFKC 展开重复不漏删")


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
    # 清洗后仍残留 train↔eval 精确重复是训练前必须拦截的直接证据
    assert "post_cleanup_residual_overlap" in CRITICAL_SPLIT_ALERT_CODES
    assert split_protocol_alerts("not-a-dict") == []
    print("  [ok] critical_split_alerts 只筛出必须先处理的泄露（含跨 split 复制），"
          "分布漂移不拦截")


def test_split_protocol_alerts_report_unverified_overlap_and_missing_files():
    """上游声明“已核对”但实际重叠/缺文件时，报告必须有可见告警。"""
    overlapping = {
        "verified_disjoint": False,
        "overlap_long_lines": {
            "train_val": {"exact_lines": 3, "normalized_lines": 4},
            "val_test": {"exact_lines": 0, "normalized_lines": 0},
        },
    }
    alerts = split_protocol_alerts(overlapping)
    codes = [a["code"] for a in alerts]
    assert "split_overlap_long_lines" in codes
    msg = next(a["message"] for a in alerts
               if a["code"] == "split_overlap_long_lines")
    assert "train_val" in msg and "val/test" in msg
    missing = {"verified_disjoint": False, "missing_files": ["test"]}
    alerts = split_protocol_alerts(missing)
    assert [a["code"] for a in alerts] == ["split_protocol_unverified"]
    assert "test" in alerts[0]["message"]
    # 声明已核对且无缺失时不产生额外告警。
    assert split_protocol_alerts({"verified_disjoint": True}) == []
    # 声明“已验证互斥”但实际重叠计数非零：矛盾证据必须告警，不能只看声明。
    contradictory = {
        "verified_disjoint": True,
        "overlap_long_lines": {"train_val": {"exact_lines": 1,
                                             "normalized_lines": 0}},
    }
    alerts = split_protocol_alerts(contradictory)
    assert [a["code"] for a in alerts] == ["split_overlap_long_lines"]
    assert "verified_disjoint=True" in alerts[0]["message"]
    assert "矛盾" in alerts[0]["message"]
    missing_true = {"verified_disjoint": True, "missing_files": ["test"]}
    alerts = split_protocol_alerts(missing_true)
    assert [a["code"] for a in alerts] == ["split_protocol_unverified"]
    assert "verified_disjoint=True" in alerts[0]["message"]
    print("  [ok] 划分协议：实际长行重叠/缺失文件不再静默躺在元数据里")


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


def test_ngram_overlap_ignores_unk_collisions():
    """不同未见字符都映射 <unk>，不能因此被当成跨 split 复制。"""
    a = np.array([0] * 50 + [1] * 50, dtype=np.uint16)
    b = np.array([0] * 50 + [2] * 50, dtype=np.uint16)
    raw = ngram_overlap(a, b, n=8, n_samples=200)
    assert raw["windows_found_in_train_sample"] > 0
    res = ngram_overlap(a, b, n=8, n_samples=200, ignore_ids={0})
    assert res["windows_found_in_train_sample"] == 0
    assert res["overlap_rate"] == 0.0
    print("  [ok] 64-gram 筛查：<unk> 碰撞不当作跨 split 复制证据")


def test_prepare_bins_does_not_flag_unk_collision(monkeypatch):
    """字符级 OOV 长串（val/test 各用不同未见字符）不得触发 hard 重叠告警。"""
    import lega.data as data_mod
    monkeypatch.setattr(data_mod, "get_gpt2_tokenizer",
                        lambda: (_ for _ in ()).throw(RuntimeError("no tiktoken")))
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 50)
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write("あ" * 300 + "\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("Ω" * 300 + "\n")
        meta = prepare_bins(d, kind="auto")
        assert meta["tokenizer"] == "char"
        with open(os.path.join(d, "data_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        assert rep["contamination"]["val_test"][
            "windows_found_in_train_sample"] == 0
        bad = {a["code"] for a in rep["alerts"]
               if a["code"].startswith("ngram_overlap")}
        assert not bad, bad
    print("  [ok] prepare_bins：字符级 OOV 碰撞不产生跨 split 重叠告警")


def test_ngram_overlap_rejects_non_token_ids_and_bad_params():
    """越界/非整数 id 不得被 astype(uint16) 静默环绕，n/n_samples 也须合法。"""
    ok = np.array([0, 1, 2, 3], dtype=np.int64)
    res = ngram_overlap(ok, ok.copy(), n=2, n_samples=8, seed=0)
    assert res["overlap_rate"] == 1.0
    for bad in (np.array([70000], dtype=np.int64),
                np.array([-1], dtype=np.int64),
                np.array([1.5], dtype=np.float64),
                np.array([True, False])):
        try:
            ngram_overlap(bad, ok, n=2)
            raise AssertionError(f"非法 token 流未被拒绝: {bad!r}")
        except ValueError:
            pass
    for kwargs in ({"n": 0}, {"n": -3}, {"n": True},
                   {"n": 2, "n_samples": 0}, {"n": 2, "n_samples": True}):
        try:
            ngram_overlap(ok, ok, **kwargs)
            raise AssertionError(f"非法参数未被拒绝: {kwargs}")
        except ValueError:
            pass
    # ignore_ids 也必须逐个校验：1.5/True 经 int() 会变成 1，越界值经
    # uint16 转换会变成另一个合法 id，从而静默排除错误窗口。
    for bad_ignore in ({1.5}, {True}, {-1}, {65536}, {float("nan")}, 5):
        try:
            ngram_overlap(ok, ok, n=2, ignore_ids=bad_ignore)
            raise AssertionError(f"非法 ignore_ids 未被拒绝: {bad_ignore!r}")
        except ValueError:
            pass
    # 整数值浮点/NumPy 整数仍是合法 token id（与 token_stream_stats 同口径）。
    res = ngram_overlap(np.array([0, 1, 0, 1], dtype=np.int64),
                        np.array([0, 1, 0, 1], dtype=np.int64),
                        n=2, ignore_ids={0.0, np.int64(1)})
    assert res["eval_windows_sampled"] == 0
    print("  [ok] 64-gram 筛查：越界/非整数 id、非法 n/n_samples 与 "
          "ignore_ids 显式拒绝")


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


def test_prepare_bins_rejects_token_id_overflow_before_uint16_cast(monkeypatch):
    """分词器返回 >65535 的 id 必须在 astype(uint16) 前拒绝，不能被环绕。"""
    import lega.data as data_mod

    class BadTok:
        name = "char"
        n_vocab = 65536

        def encode(self, text):
            return [70000] if text else []

    monkeypatch.setattr(data_mod, "get_tokenizer", lambda *a, **k: BadTok())
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        try:
            prepare_bins(d, kind="char", force=True)
            raise AssertionError("越界 token id 应抛 ValueError")
        except ValueError as e:
            assert "65535" in str(e) or "越界" in str(e)
        assert not os.path.exists(os.path.join(d, "train.bin"))
    print("  [ok] encode_stream：越界 id 在 uint16 转换前显式拒绝")


def test_char_vocab_fit_on_train_only():
    tok = get_tokenizer("char", train_text="abc def", val_text="!!!zzz")
    assert isinstance(tok, CharTokenizer)
    # 未知/拼写错误的 kind 不能静默退化成字符级分词。
    for bad in ("gpt2 ", "", "GPT2", None, 1, True):
        try:
            get_tokenizer(bad, train_text="abc")
            raise AssertionError(f"非法 tokenizer kind 未被拒绝: {bad!r}")
        except ValueError:
            pass
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
    # 阈值是审计身份：bool/小数/下划线串不能被 int() 截断成另一个口径。
    assert text_length_stats(text, huge_line_chars="100000")[
        "lines_gt_huge_threshold"] == 1
    for bad in (True, 40.5, "1_0", None, -1):
        try:
            text_length_stats(text, huge_line_chars=bad)
            raise AssertionError(f"非法 huge_line_chars 应报错: {bad!r}")
        except ValueError:
            pass
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
        # max_tokens 是数据规模身份：bool/小数/下划线串不能被截断成另一个
        # 预算后仍返回“写入成功”。
        for bad in (True, 1.5, "1_0"):
            try:
                write_clean_lines(io.StringIO("a\nb\n"), dst2, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 max_tokens 未被拒绝: {bad!r}")
        # 合法但极大的整数预算不应因 int→float 溢出而崩溃：语义上等同于
        # “不截断”，两行必须完整写入。
        huge = os.path.join(d, "huge_budget.txt")
        n_huge = write_clean_lines(io.StringIO("a\nb\n"), huge, 10 ** 400)
        with open(huge, "r", encoding="utf-8") as f:
            assert f.read() == "a\nb"
        assert n_huge == 3
    print("  [ok] 数据准备：空行块边界被保留，截断仍只发生在完整行")


def test_verify_splits_disjoint_binds_claim_to_content():
    """官方 split 的 document_disjoint 声明必须与实际文本绑定核对。"""
    from lega.datacheck import verify_splits_disjoint

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
        assert r["missing_files"] == []
        # min_chars=True 会把所有行当短行，长行重叠被跳过 → 伪造 verified。
        for bad in (True, 40.5, "bad", None, -1):
            try:
                verify_splits_disjoint(paths, min_chars=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 min_chars 未被拒绝: {bad!r}")
        assert verify_splits_disjoint(paths, min_chars="40")["min_chars"] == 40
        # 缺失文件不是“无可核对内容”→ 空集合交集不能伪造成核对通过
        missing = dict(paths)
        missing["train"] = os.path.join(d, "does-not-exist.txt")
        r = verify_splits_disjoint(missing)
        assert r["verified_disjoint"] is False
        assert r["missing_files"] == ["train"]
        # 空路径/空输入同样不能声称已验证
        assert verify_splits_disjoint({"train": "", "val": paths["val"]})[
            "verified_disjoint"] is False
        assert verify_splits_disjoint({})["verified_disjoint"] is False
        # 注入 val/test 共享长行 → 声明被证伪
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write(f"{shared}\nval unique with plenty of characters.\n")
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write(f"{shared}\ntest unique with plenty of characters.\n")
        r = verify_splits_disjoint(paths)
        assert r["verified_disjoint"] is False
        assert r["overlap_long_lines"]["val_test"]["exact_lines"] == 1
        # 短行（标题）不计入 verified_disjoint，避免误报；但短行重叠必须
        # 单独报告，不能成为“既不删除也不报告”的盲区。
        with open(paths["train"], "w", encoding="utf-8") as f:
            f.write("= References =\ntrain unique plenty of characters here.\n")
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write("= References =\nval unique plenty of characters here.\n")
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write("test unique plenty of characters here.\n")
        r = verify_splits_disjoint(paths)
        assert r["verified_disjoint"] is True
        assert r["overlap_short_lines"]["train_val"]["exact_lines"] == 1
        assert r["overlap_long_lines"]["train_val"]["exact_lines"] == 0
        # NFKC 展开造成的长短差异不能让归一化重复逃过互斥核对。
        with open(paths["train"], "w", encoding="utf-8") as f:
            f.write("ﬀ" * 20 + "\ntrain unique plenty of characters here.\n")
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write("f" * 40 + "\nval unique plenty of characters here.\n")
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write("test unique plenty of characters here.\n")
        r = verify_splits_disjoint(paths)
        assert r["verified_disjoint"] is False
        assert r["overlap_long_lines"]["train_val"]["normalized_lines"] == 1
        # 只提供部分 split 时不能把“碰巧比较过的文件”当成三个 split 已验证。
        partial = {k: paths[k] for k in ("train", "val")}
        r = verify_splits_disjoint(partial)
        assert r["verified_disjoint"] is False
        assert "test" in r["missing_files"]
    print("  [ok] verify_splits_disjoint：声明与文本核对，短行不误报，"
          "缺失文件不伪装成通过")


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
    # 显式采样预算非法时不能静默退回全量或截断成 1：报告会声称采样
    # 却实际使用另一口径，审计字段失去意义。
    for bad in (True, 0, -1, 1.5, "bad", float("nan"), float("inf")):
        try:
            distribution_shift(rare, np.array([7, 1], dtype=np.uint16), 10,
                               max_samples=bad)
            raise AssertionError(f"非法 max_samples 应报错: {bad!r}")
        except ValueError:
            pass
    # uint16 词表上限：超出后密集直方图会按声明容量做巨额分配（10**12 即
    # TB 级），必须像 prepare_bins 一样显式拒绝，而不是在计数阶段 OOM。
    for bad_vocab in (0, -1, True, 1.5, "bad", 65537, 10 ** 12):
        try:
            distribution_shift(rare, np.array([7, 1], dtype=np.uint16),
                               bad_vocab)
            raise AssertionError(f"非法 vocab_size 应报错: {bad_vocab!r}")
        except ValueError:
            pass
    # 越界/非整数/hash 伪装的 id 不能改变直方图长度或静默成为合法 token：
    # 两边同样越界时 TV/JS 会看似正常却统计了词表外的类别。
    for bad_ids in (np.array([0, 1, 3]), np.array([-1, 0]),
                    np.array([0.5, 1.0]), np.array([np.nan, 1.0]),
                    np.array([True, False])):
        try:
            distribution_shift(bad_ids, np.array([0, 1]), 3)
            raise AssertionError(f"非法 token id 应报错: {bad_ids!r}")
        except ValueError:
            pass
    # 整数值浮点流仍是合法 token id，不能因硬化误伤。
    d_float = distribution_shift(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                                 3)
    assert d_float["tokens_train"] == 2 and d_float["tv"] == 0.0, d_float
    print("  [ok] 分布漂移：精确计数避免 OOV 假阳性；显式采样可审计，"
          "非法预算/词表显式拒绝")


def test_test_access_ledger():
    with tempfile.TemporaryDirectory() as d:
        assert read_test_access_log(d) == []
        s1 = append_test_access(d, {"ckpt_sha256": "aaa", "ckpt": "/x/a.pt",
                                    "test_data_fingerprint": "v1"})
        assert s1["evaluations"] == 1 and s1["distinct_checkpoints"] == 1
        s2 = append_test_access(d, {"ckpt_sha256": "bbb", "ckpt": "/x/b.pt",
                                    "test_data_fingerprint": "v1"})
        assert s2["evaluations"] == 2 and s2["distinct_checkpoints"] == 2
        assert s2["checkpoints_evaluated_multiple_times"] == []
        s3 = append_test_access(d, {"ckpt_sha256": "aaa", "ckpt": "/x/a.pt",
                                    "test_data_fingerprint": "v1"})
        assert s3["evaluations"] == 3 and s3["distinct_checkpoints"] == 2
        assert s3["checkpoints_evaluated_multiple_times"] == ["aaa"[:16]]
        # 重建 test.bin 后，同一 checkpoint 在 v2 上是新版本评测，不算重复；
        # 但旧账本仍保留 v1 的重复记录供审计。
        s4 = append_test_access(d, {"ckpt_sha256": "aaa", "ckpt": "/x/a.pt",
                                    "test_data_fingerprint": "v2"})
        assert s4["distinct_test_data_versions"] == 2
        assert s4["checkpoints_evaluated_multiple_times"] == ["aaa"[:16]]
        assert s4["repeated_evaluations"] == [
            {"ckpt_prefix": "aaa"[:16], "test_data_fingerprint": "v1",
             "evaluations": 2}]
        assert len(read_test_access_log(d)) == 4
    print("  [ok] 测试集访问账本：追加、去重统计、按 test 数据版本判定重复")


def test_prepare_bins_rejects_bool_or_fractional_token_counts():
    """缓存元数据的 bool/小数/下划线 token 数不能伪造尺寸一致而绕过重建。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write("abcdefghij")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write("klmnopqrst")
        meta = prepare_bins(d, kind="char", force=True)
        assert meta["train_tokens"] == 10 and meta["val_tokens"] == 10
        meta_path = os.path.join(d, "meta.json")
        for bad in (True, 1.5, float("nan"), "bad", "1_0"):
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            m["train_tokens"] = bad
            m["val_tokens"] = bad
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(m, f)
            rebuilt = prepare_bins(d, kind="char")
            assert rebuilt["train_tokens"] == 10, (bad, rebuilt)
            assert rebuilt["val_tokens"] == 10, (bad, rebuilt)
    print("  [ok] bin 缓存拒绝 bool/小数/下划线 token 数，强制重建")


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

