"""防泄漏数据协议测试：时序划分、去污染、token 硬校验、重叠筛查、访问账本。

覆盖用户协议中的三条硬要求：
  1) 时序划分不打乱、文档不跨 split（train 最早 → test 最新；含标题文档
     优先切分与 documents_cross_cut 自检）；
  2) 群组/重复样本不跨侧：与 val/test 重复（精确或空白归一化）的长行只从
     train 移除；test split 默认拒绝读取（尘封），需显式 allow_test=True；
  3) 测试集访问可审计：账本记录每次 test 评测与重复评测的 checkpoint。

另覆盖 bin 缓存完整性：源文本/诊断报告哈希变化时强制重建，防止旧协议
或过期缓存的 bin 被静默训练；以及数据自检告警（alerts）：阈值触发项与
划分协议违规（群组跨 split / 字符切分 / 打乱）都必须显式上报。
"""
import json
import os
import tempfile

import numpy as np

from lega.data import (CharTokenizer, get_tokenizer, load_bin, load_meta,
                       load_tokenizer, prepare_bins)
from lega.datacheck import (TEST_ACCESS_LOG, append_test_access,
                            build_data_report,
                            check_vocab_capacity, decontaminate_texts,
                            distribution_shift, ngram_overlap,
                            read_test_access_log, sanitize_text,
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
        assert meta1["cache_schema"] == 2
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
