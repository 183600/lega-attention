"""防泄漏数据协议测试：时序划分、去污染、token 硬校验、重叠筛查、访问账本。

覆盖用户协议中的三条硬要求：
  1) 时序划分不打乱、文档不跨 split（train 最早 → test 最新）；
  2) 群组/重复样本不跨侧：与 val/test 重复的长行只从 train 移除；
  3) 测试集访问可审计：账本记录每次 test 评测与重复评测的 checkpoint。
"""
import json
import os
import tempfile

import numpy as np

from lega.data import (CharTokenizer, get_tokenizer, load_bin, load_meta,
                       load_tokenizer, prepare_bins)
from lega.datacheck import (append_test_access, check_vocab_capacity,
                            decontaminate_texts, distribution_shift,
                            ngram_overlap, read_test_access_log, sanitize_text,
                            temporal_split_text, text_length_stats,
                            token_stream_stats, unknown_char_rate)


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


def test_heading_regex_matches_multilevel_wikitext_headings():
    from lega.datacheck import _HEADING_RE
    for line in ("= Title =", "== Section ==", "=== Sub ===",
                 "= = Section = =", "= = = Sub = = =", "==NoSpace=="):
        assert _HEADING_RE.match(line), f"应视为标题: {line!r}"
    for line in ("x = y = z", "plain text", "= = ", "= = =", ""):
        assert not _HEADING_RE.match(line), f"不应视为标题: {line!r}"
    print("  [ok] 标题正则：一级/多级（含 WikiText-103 分隔写法）标题行均可识别")


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
    assert stats["train_lines_dropped_as_duplicates_of_eval"] == 0
    assert stats["train_intra_split_duplicate_lines"] == 1
    print("  [ok] 短行（如标题）不参与去污染；train 内部重复仅计数")


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


def test_sanitize_text_stats():
    raw = "ok\x00bad\x07line\ufffdend"
    clean, st = sanitize_text(raw)
    assert clean == "okbadline end"
    assert "\x00" not in clean and "\x07" not in clean and "\ufffd" not in clean
    assert st["nul_deleted"] == 1
    assert st["control_chars_deleted"] == 2
    assert st["replacement_char_replaced_with_space"] == 1
    print("  [ok] 文本清洗：NUL/控制符删除、U+FFFD 替换为空格并可计数")


def test_text_length_stats_outlier_detection():
    text = "short\n" + "x" * 200001 + "\nmedium line"
    st = text_length_stats(text, huge_line_chars=100000)
    assert st["lines"] == 3
    assert st["line_chars_max"] == 200001
    assert st["lines_gt_huge_threshold"] == 1
    print("  [ok] 行长度统计：超长行作为异常值被单独计数")


def test_distribution_shift_metric():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=100000).astype(np.uint16)
    d_same = distribution_shift(a, a.copy(), 100)
    assert d_same["tv"] < 1e-12 and d_same["js_bits"] < 1e-12
    b = rng.integers(100, 200, size=100000).astype(np.uint16)
    d_diff = distribution_shift(a, b, 200)
    assert d_diff["tv"] > 0.9
    assert d_diff["eval_oov_token_rate"] > 0.9
    print("  [ok] 分布漂移：同分布 TV≈0，完全错位 TV≈1 且 OOV 率≈1")


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
        assert rep["tokenizer"]["unk_rate"]["val"] > 0  # 独有的 § 不在 train 词表
        tok = load_tokenizer(d)
        assert "§" not in tok.stoi
        # 缓存命中：第二次调用不重建（token 数一致），且报告仍存在
        meta2 = prepare_bins(d, kind="char")
        assert meta2["train_tokens"] == meta["train_tokens"]
        assert load_bin(d, "train").size == meta["train_tokens"]
    print("  [ok] prepare_bins：去污染 + train-only 词表 + data_report + 缓存命中")
