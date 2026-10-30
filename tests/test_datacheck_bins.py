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


def test_prepare_bins_detects_same_size_bin_tampering():
    """同尺寸替换 bin 必须被内容哈希识破：缓存失效重建，load_bin 拒绝读取。"""
    from lega.data import _sha256_file
    with tempfile.TemporaryDirectory() as d:
        _write_min_splits(d)
        meta1 = prepare_bins(d, kind="char", force=True)
        assert set(meta1["bin_sha256"]) == {"train", "val", "test"}
        assert (meta1["bin_sha256"]["train"]
                == _sha256_file(os.path.join(d, "train.bin")))
        # 同尺寸篡改 train.bin 的首个 token（uint16）
        with open(os.path.join(d, "train.bin"), "r+b") as f:
            first = f.read(2)
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF, first[1]]))
        tampered = _sha256_file(os.path.join(d, "train.bin"))
        assert tampered != meta1["bin_sha256"]["train"]
        try:
            load_bin(d, "train")
            raise AssertionError("被篡改的 bin 不应通过 load_bin 哈希核对")
        except ValueError as e:
            assert "哈希不一致" in str(e)
        # 有源 txt：缓存失效并按源文本重建，token 流恢复
        meta2 = prepare_bins(d, kind="char")
        assert meta2["bin_sha256"]["train"] == meta1["bin_sha256"]["train"]
        assert load_bin(d, "train").size == meta1["train_tokens"]
        assert _sha256_file(os.path.join(d, "train.bin")) \
            == meta1["bin_sha256"]["train"]
        # 纯 bin 目录（无源 txt）：篡改后不能静默命中缓存，必须报错
        bin_only = os.path.join(d, "bin_only")
        os.makedirs(bin_only)
        for name in ("train.bin", "val.bin", "test.bin", "meta.json",
                     "data_report.json"):
            shutil.move(os.path.join(d, name), os.path.join(bin_only, name))
        with open(os.path.join(bin_only, "train.bin"), "r+b") as f:
            first = f.read(2)
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF, first[1]]))
        try:
            load_bin(bin_only, "train")
            raise AssertionError("纯 bin 目录的篡改 bin 不应通过 load_bin")
        except ValueError as e:
            assert "哈希不一致" in str(e)
        try:
            prepare_bins(bin_only, kind="char")
            raise AssertionError("纯 bin 目录的篡改缓存不应静默命中")
        except FileNotFoundError:
            pass
    print("  [ok] bin 缓存完整性：同尺寸替换被内容哈希识破（含纯 bin 目录）")


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
        for bad in (False, None, 1, "false", "1"):
            try:
                load_bin(d, "test", allow_test=bad)
                raise AssertionError(
                    f"allow_test={bad!r} 不是显式 bool True，必须拒绝")
            except PermissionError:
                pass
        assert load_bin(d, "test", allow_test=True).size > 0
        # train/val 不受影响
        assert load_bin(d, "train").size > 0
        assert load_bin(d, "val").size > 0
        # split 别名/路径不能绕过尘封闸门："./test" 与绝对路径都会被
        # os.path.join 解析到 test.bin，必须与字面 "test" 同口径拒绝。
        for alias in ("./test", os.path.join(d, "test")):
            try:
                load_bin(d, alias)
                raise AssertionError(f"split={alias!r} 绕过了 test 尘封闸门")
            except ValueError:
                pass
        for bad in ("", "Test", "../test", "train/../../test", None, 1):
            try:
                load_bin(d, bad, allow_test=True)
                raise AssertionError(f"非法 split={bad!r} 必须被拒绝")
            except ValueError:
                pass
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


def test_leakage_alerts_cover_short_line_cross_split_overlap():
    """短行跨 split 复制必须进入 alerts（不自动删除，但须显式告警）。"""
    duplicates = {
        "min_chars": 40,
        "train_val_overlap_short_lines": 2,
        "train_val_overlap_normalized_short_lines": 3,
        "train_test_overlap_short_lines": 1,
        "train_test_overlap_normalized_short_lines": 1,
        "val_test_overlap_short_lines": 4,
        "val_test_overlap_normalized_short_lines": 5,
    }
    rep = build_data_report(
        {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}},
        {}, {}, duplicates, {}, {}, {})
    by_code = {a["code"]: a for a in rep["alerts"]}
    assert "short_line_cross_split_overlap" in by_code
    assert "eval_short_lines_overlap" in by_code
    assert "train/val 精确 2" in by_code["short_line_cross_split_overlap"]["message"]
    assert "精确 4" in by_code["eval_short_lines_overlap"]["message"]
    print("  [ok] 短行跨 split 复制告警：train/eval 与 val/test 均显式上报")


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


def test_split_protocol_alerts_flag_paragraph_crossing():
    """空行段落跨 split 与标题文档跨 split 同级进入告警与训练前硬拦截。"""
    assert split_protocol_alerts({"paragraphs_cross_cut": False}) == []
    alerts = split_protocol_alerts({
        "paragraphs_cross_cut": True, "group_unit": "blank_line_paragraph",
        "n_paragraph_boundaries": 1, "cut_kinds": ["char", "blank"]})
    assert [a["code"] for a in alerts] == ["split_paragraphs_cross_cut"]
    assert alerts[0]["level"] == "warning"
    assert "blank_line_paragraph" in alerts[0]["message"]
    assert "split_paragraphs_cross_cut" in CRITICAL_SPLIT_ALERT_CODES
    critical = critical_split_alerts({"alerts": alerts})
    assert [a["code"] for a in critical] == ["split_paragraphs_cross_cut"]
    print("  [ok] 划分协议告警：段落跨切点显式上报并进入硬拦截")


def test_split_protocol_alerts_flag_short_line_overlap():
    """实测短行跨 split 重叠必须进入协议告警（不进入硬闸门，但不可静默）。"""
    alerts = split_protocol_alerts({
        "overlap_short_lines": {"train_val": {"exact_lines": 2,
                                              "normalized_lines": 3}},
        "min_chars": 40})
    assert [a["code"] for a in alerts] == ["split_overlap_short_lines"]
    assert "train_val" in alerts[0]["message"]
    assert "2" in alerts[0]["message"]
    clean = split_protocol_alerts({
        "overlap_short_lines": {"train_val": {"exact_lines": 0,
                                              "normalized_lines": 0}}})
    assert clean == []
    print("  [ok] 划分协议告警：短行跨 split 重叠显式上报")


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


def test_build_data_report_does_not_attest_split_first_without_protocol():
    """缺少 split_protocol 证据时不能写死“先划分后预处理=True”。"""
    tok = {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}}
    rep = build_data_report(tok, {"train": {"tokens": 10}}, {}, {}, {}, {}, {})
    assert rep["protocol"]["split_first_then_preprocess"] is None
    assert rep["protocol"]["split_first_then_preprocess_attested"] is False
    assert rep["protocol"]["shuffle_at_split_time"] is None
    assert "split_first_unattested" in {a["code"] for a in rep["alerts"]}
    rep2 = build_data_report(
        tok, {"train": {"tokens": 10}}, {}, {}, {}, {}, {},
        extra={"split_protocol": {"mode": "official", "shuffled": False}})
    assert rep2["protocol"]["split_first_then_preprocess"] is True
    assert rep2["protocol"]["split_first_then_preprocess_attested"] is True
    assert rep2["protocol"]["shuffle_at_split_time"] is False
    assert "split_first_unattested" not in {a["code"] for a in rep2["alerts"]}
    # 上游声明打乱时必须如实照写（并由 split_protocol 告警拦截）
    rep3 = build_data_report(
        tok, {"train": {"tokens": 10}}, {}, {}, {}, {}, {},
        extra={"split_protocol": {"mode": "temporal", "shuffled": True}})
    assert rep3["protocol"]["shuffle_at_split_time"] is True
    assert "split_shuffled" in {a["code"] for a in rep3["alerts"]}
    print("  [ok] data_report：无划分协议证据时“先划分”声明为未知并告警")


def test_build_data_report_extra_cannot_clobber_audit_fields():
    """extra 只能扩充报告，不能覆盖 alerts/protocol/splits 等自检结论。

    否则 prepare_data 的上游元数据（甚至手改调用）传入 "alerts": [] 就能
    把跨 split 重叠告警清空，报告的哈希绑定也拦不住——内容确实是被合法
    写出的那一份，只是审计结论已被替换。
    """
    rep = build_data_report(
        {"kind": "char", "fit_on": "train", "vocab_size": 5, "unk_rate": {}},
        {"train": {"tokens": 10}}, {}, {},
        {"train_val": {"n": 64, "overlap_rate": 0.5}}, {},
        extra={"alerts": [], "protocol": {"test_sealed": False},
               "splits": {}, "source": "local",
               "split_protocol": {"mode": "official", "shuffled": False}})
    assert any(a["code"] == "ngram_overlap_train_val" for a in rep["alerts"])
    assert rep["protocol"]["test_sealed"] is True
    assert rep["protocol"]["split_first_then_preprocess"] is True
    assert rep["splits"] == {"train": {"tokens": 10}}
    assert rep["source"] == "local"
    assert rep["split_protocol"]["mode"] == "official"
    assert set(rep["ignored_extra_keys"]) == {"alerts", "protocol", "splits"}
    print("  [ok] data_report：extra 不能覆盖审计字段，冲突键显式记录")


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
        for bad in (None, [3], {"v": 3}, "not-a-version",
                    CACHE_SCHEMA + 0.9):
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                saved = json.load(f)
            saved["cache_schema"] = bad
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(saved, f)
            meta = prepare_bins(d, kind="char", force=False)
            assert meta["cache_schema"] == CACHE_SCHEMA and meta["train_tokens"] > 0
    print("  [ok] 异常 cache_schema：按旧缓存重建而非崩溃")


def test_load_tokenizer_rejects_malformed_char_meta():
    """缺少/畸形词表元数据须给出 ValueError，而不是 KeyError/TypeError。"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "meta.json")
        cases = ({}, {"tokenizer": True}, {"tokenizer": "char"},
                 {"tokenizer": "char", "itos": []},
                 {"tokenizer": "char", "itos": ["a", 1]},
                 {"tokenizer": "char_typo", "itos": ["a", "b"]})
        for meta in cases:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            try:
                load_tokenizer(d)
            except ValueError:
                pass
            else:
                raise AssertionError(f"应拒绝畸形 meta: {meta}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "itos": ["a", "b"]}, f)
        assert load_tokenizer(d).decode([0, 1]) == "ab"
    print("  [ok] load_tokenizer：畸形词表元数据显式拒绝")


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


def test_leakage_alerts_reject_bool_and_fractional_counts():
    """bool/小数计数不能截断成行数，也不能让布尔比率触发重叠告警。"""
    out = leakage_alerts(
        None, None,
        {"train_lines_dropped_as_duplicates_of_eval": True,
         "val_test_overlap_long_lines": 1.9},
        None,
        {"val_vs_train": {"tv": True, "js_bits": False}},
        None, None)
    assert out == [], out
    # "0_9" 经 float() 会变成 0.9 并伪造分布漂移告警，必须按坏数据忽略
    out = leakage_alerts(
        None, None, None, None,
        {"val_vs_train": {"tv": "0_9"}}, None, None)
    assert out == [], out
    # 真正的整数计数与比率仍然工作
    out = leakage_alerts(
        None, None, {"val_test_overlap_long_lines": 1}, None,
        {"val_vs_train": {"tv": 0.9}}, None, None)
    assert [a["code"] for a in out] == ["distribution_tv_val_vs_train",
                                        "eval_splits_overlap"], out
    print("  [ok] leakage_alerts：bool/小数计数与布尔比率按坏数据忽略")


def test_short_ngram_overlap_and_alerts_are_diagnostic_only():
    """<64 token 的整句复用要被短 n 筛查抓到，但只告警、不进入硬闸门。"""
    rng = np.random.default_rng(1)
    # 一段 32 长的短句在 train 与 test 中逐 token 复制（64-gram 测不到）
    shared = rng.integers(0, 1000, size=32).astype(np.uint16)
    train = np.concatenate(
        [rng.integers(0, 1000, size=4000).astype(np.uint16), shared])
    test = np.concatenate(
        [rng.integers(0, 1000, size=3000).astype(np.uint16), shared])
    res = short_ngram_overlap(train, test, n=32, n_samples=20000, seed=0)
    assert res["windows_found_in_train_sample"] >= 1
    assert res["diagnostic_only"] is True
    alerts = short_overlap_alerts({"train_test": res})
    assert alerts and alerts[0]["code"] == "ngram_overlap_short_train_test"
    assert alerts[0]["level"] == "warning"
    # 短 n-gram 码不得进入训练前硬闸门（否则常见搭配会阻断训练）
    assert critical_split_alerts({"alerts": alerts}) == []
    # rate=0 / 缺失 / 畸形 不产生告警
    assert short_overlap_alerts({"train_test": {"overlap_rate": 0}}) == []
    assert short_overlap_alerts({"train_test": {"overlap_rate": "bad"}}) == []
    # bool/越界比率不是重叠率证据，不能触发 100%/150% 假告警
    for bad in (True, False, -0.1, 1.5, float("nan"), float("inf"),
                "0_5", [0.1]):
        assert short_overlap_alerts({"train_test": {"overlap_rate": bad}}) == [], bad
    assert short_overlap_alerts(None) == []
    print("  [ok] 短 n-gram 重叠：检出整句复用且仅告警、不进硬闸门")


def test_verify_splits_disjoint_compares_custom_required_splits():
    """required_splits 含 dev 等非标准名称时，也必须与其它 split 逐一比对。"""
    from lega.datacheck import verify_splits_disjoint
    shared = "T" * 60
    with tempfile.TemporaryDirectory() as d:
        paths = {s: os.path.join(d, f"{s}.txt")
                 for s in ("train", "val", "dev")}
        with open(paths["train"], "w", encoding="utf-8") as f:
            f.write(shared + "\n")
        with open(paths["val"], "w", encoding="utf-8") as f:
            f.write("V" * 60 + "\n")
        with open(paths["dev"], "w", encoding="utf-8") as f:
            f.write(shared + "\n")
        r = verify_splits_disjoint(paths,
                                   required_splits=("train", "val", "dev"))
        assert r["missing_files"] == []
        assert r["verified_disjoint"] is False
        assert r["overlap_long_lines"]["train_dev"]["exact_lines"] == 1
        assert "val_dev" in r["overlap_long_lines"]


def test_tokenizer_fit_source_must_match_tokenizer_kind():
    """char 必须 fit_on=train，GPT-2 BPE 必须 fit_on=gpt2_pretrained。"""
    from lega.data import CACHE_SCHEMA, tokenizer_fit_source_ok
    from lega.preflight import data_cache_schema_issue
    assert tokenizer_fit_source_ok("char", "train") is True
    assert tokenizer_fit_source_ok("gpt2_bpe", "gpt2_pretrained") is True
    assert tokenizer_fit_source_ok("char", "gpt2_pretrained") is False
    assert tokenizer_fit_source_ok("gpt2_bpe", "train") is False
    # 旧 meta 没有 tokenizer 字段：保持兼容，仅按 fit_on 判断。
    assert tokenizer_fit_source_ok(None, "train") is True
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "meta.json")

        def _write(tok, fit_on):
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({"cache_schema": CACHE_SCHEMA,
                           "tokenizer": tok, "fit_on": fit_on}, f)
        _write("char", "train")
        assert data_cache_schema_issue(d) is None
        _write("char", "gpt2_pretrained")
        assert data_cache_schema_issue(d), "char+pretrained 不能通过"
        _write("gpt2_bpe", "gpt2_pretrained")
        assert data_cache_schema_issue(d) is None
        _write("gpt2_bpe", "train")
        assert data_cache_schema_issue(d), "bpe+train 不能通过"


def test_data_report_without_fit_on_does_not_claim_safe_provenance():
    """tokenizer 段缺少 fit_on 时，报告不能默认声明安全的 train_only。"""
    from lega.preflight import data_report_protocol_issues
    rep = build_data_report({"kind": "char", "vocab_size": 5},
                            {}, {}, {}, {}, {})
    assert rep["protocol"]["fit_on"] is None
    issues = data_report_protocol_issues(rep)
    assert any("fit_on" in msg for msg in issues), issues


def test_summarize_test_access_tolerates_non_iterable_ledger():
    """损坏的访问账本（bool/数字）不能让评测入口在汇总时崩溃。"""
    from lega.datacheck import summarize_test_access
    for bad in (None, True, False, 0, 1.5, {"a": 1}):
        out = summarize_test_access(bad)
        assert out["evaluations"] == 0 and out["distinct_checkpoints"] == 0
    out = summarize_test_access([
        {"ckpt_sha256": "a" * 64, "time": "t1"},
        {"ckpt": "b", "test_data_fingerprint": "v1", "time": "t2"},
        "bad", None, 3])
    assert out["evaluations"] == 2
    assert out["distinct_checkpoints"] == 2
    assert out["first_time"] == "t1" and out["last_time"] == "t2"


def test_prepare_data_rows_api_uses_validated_integer_budget(monkeypatch):
    """rows 兜底路径必须用已严格解析的整数预算，而非 argparse 原始字符串。"""
    import sys
    import lega.data as data_mod
    import scripts.prepare_data as prep

    seen = {}

    def fake_rows(max_rows, retries=4):
        seen["max_rows"] = max_rows
        return {
            "train": ["train line " + "a" * 80],
            "val": ["val line " + "b" * 80],
            "test": ["test line " + "c" * 80],
        }

    def fake_prepare_bins(data_dir, **kwargs):
        seen["prepare_dir"] = data_dir
        return {}

    monkeypatch.setattr(prep, "from_rows_api", fake_rows)
    monkeypatch.setattr(data_mod, "prepare_bins", fake_prepare_bins)
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, "out")
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--source", "rows",
            "--max_tokens", "600"])
        prep.main()
        # 600 // 30 = 20，受下限 500 约束；关键是不能对字符串做 //。
        assert seen["max_rows"] == 500, seen
        assert seen["prepare_dir"] == out_dir, seen
        for name in ("train.txt", "val.txt", "test.txt"):
            assert os.path.exists(os.path.join(out_dir, name)), name
    print("  [ok] prepare_data：rows 兜底路径使用校验后的整数预算")


def _near_dup_base():
    return (
        "The history of the town begins in the early nineteenth century, when "
        "a small settlement grew around the river crossing. Merchants and "
        "farmers brought goods to the weekly market, and the population "
        "increased steadily over several decades. By the end of the century "
        "the railway had arrived, connecting the town to larger cities and "
        "changing the local economy in lasting ways. Schools, churches, and a "
        "public library were established during this period of growth.")


def _near_dup_unrelated():
    return (
        "Quantum chromodynamics describes the strong interaction between "
        "quarks and gluons. Lattice calculations provide numerical predictions "
        "for hadron masses and coupling constants, while effective field "
        "theories organize the low energy behavior. Experiments at colliders "
        "test these predictions with increasing precision, and the agreement "
        "between theory and data constrains possible new physics scenarios.")


def test_near_duplicate_removes_train_copies_only():
    """改了几个词的 train 副本也必须删除；val/test 逐字节保持原样。"""
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth").replace(
        "several decades", "many decades")
    train_unique = "A train-only paragraph that is unrelated to the evaluation."
    texts = {"train": f"{edited}\n\n{train_unique}",
             "val": base, "test": base}
    out, stats = decontaminate_near_duplicates(texts)
    assert "history of the town" not in out["train"]
    assert train_unique in out["train"]
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    near = stats
    assert near["train_blocks_dropped_as_near_duplicates_of_eval"] == 1
    assert near["train_blocks_dropped_total"] == 1
    assert near["threshold_sensitivity"]["0.8"]["train_val"] >= 1
    assert 0.0 < near["lsh_recall_at_threshold"] <= 1.0
    print("  [ok] 近似重复：只从 train 删除跨 split 副本，val/test 不改写")


def test_near_duplicate_containment_catches_longer_train_block():
    """val 的小段被整段抄进更长 train 块：Jaccard 被长度稀释，
    containment 仍应识别（跨 split 用 max(Jaccard, containment)）。"""
    base = _near_dup_base()
    val_block = base[:260]
    prefix = "A long train document that embeds an evaluation passage: "
    suffix = " " + ("additional train-only material " * 30)
    texts = {"train": f"{prefix}{val_block}{suffix}",
             "val": val_block, "test": ""}
    out, stats = decontaminate_near_duplicates(texts)
    assert out["train"].strip() == ""
    assert out["val"] == texts["val"]
    assert stats["train_blocks_dropped_as_near_duplicates_of_eval"] == 1
    # 报告里保留两种相似度，而不是只给一个无法复核的分数
    assert stats["top_similarities"]
    assert "containment" in stats["top_similarities"][0]
    print("  [ok] 近似重复：containment 捕捉“短 eval 段被长 train 文档包含”")


def test_near_duplicate_containment_without_lsh_collision():
    """containment=1 但 Jaccard 很低时也必须进入候选并删除。

    短 val 段被抄进极长 train 文档时，MinHash/LSH 的 Jaccard band 不会碰撞；
    若候选阶段只看 Jaccard，副本会留在 train 中（数据泄漏）。
    """
    base = _near_dup_base()
    val_block = base[:260]
    texts = {"train": ("A long train document that embeds an evaluation "
                       "passage: " + val_block + " " + "x" * 2000),
             "val": val_block, "test": ""}
    out, stats = decontaminate_near_duplicates(texts)
    assert val_block not in out["train"]
    assert stats["containment_candidate_pairs"] >= 1
    assert stats["train_blocks_dropped_as_near_duplicates_of_eval"] == 1
    print("  [ok] 近似重复：低 Jaccard/high containment 也走候选并删除")


def test_near_duplicate_val_test_diagnostic_does_not_depend_on_train():
    """train 没有可扫描块时，val/test 近似重复诊断仍必须运行。"""
    base = _near_dup_base()
    texts = {"train": "short train", "val": base, "test": base}
    _, stats = decontaminate_near_duplicates(texts)
    assert stats["val_test_near_duplicate_pairs"] >= 1
    assert stats["candidate_pairs"] >= 1
    print("  [ok] 近似重复：val/test 诊断不依赖 train 是否有候选块")


def test_near_duplicate_intra_train_keeps_first_and_short_blocks_ignored():
    """train 内部近似重复保留首次出现；短块不参与，避免误删标题/固定语。"""
    base = _near_dup_base()
    edited = base.replace("river crossing", "river ford")
    train_unique = "Another train-only paragraph that is not a duplicate."
    texts = {"train": f"{base}\n\n{edited}\n\n{train_unique}",
             "val": _near_dup_unrelated(), "test": ""}
    out, stats = decontaminate_near_duplicates(texts)
    assert out["train"].count(base) == 1
    assert "river ford" not in out["train"]
    assert train_unique in out["train"]
    assert stats["train_intra_split_near_duplicates_removed"] == 1
    # 短于 min_chars 的块即使内容相同也不进入扫描（避免误删标题）
    short = "= References ="
    out2, stats2 = decontaminate_near_duplicates(
        {"train": short, "val": short, "test": ""})
    assert out2["train"] == short
    assert stats2["train_blocks_skipped_short"] == 1
    print("  [ok] 近似重复：train 内部保留首次出现；短块显式跳过")


def test_near_duplicate_lsh_band_uses_declared_rows():
    """LSH 桶键必须用完一个 band 的全部行；少用一行会让实际召回率与报告不符。"""
    sig = np.arange(NEAR_DUP_ROWS * 2, dtype=np.uint64)
    key = _near_dup_band_key(sig, 0)
    assert len(key) == NEAR_DUP_ROWS
    # 只改第 3 行（前两行完全一致）也必须改变桶键；旧实现只用前两行，
    # 会把这一对送进候选集，实际退化为 rows=2 的 LSH。
    changed = sig.copy()
    changed[NEAR_DUP_ROWS - 1] += 1
    assert _near_dup_band_key(changed, 0) != key
    # 不同 band 的不相交行必须各自参与键。
    assert _near_dup_band_key(sig, 1) != key
    print("  [ok] 近似重复：LSH band 使用声明的全部行")


def test_near_duplicate_val_test_is_diagnostic_only_and_threshold_sensitivity():
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth")
    texts = {"train": _near_dup_unrelated(), "val": base, "test": edited}
    out, stats = decontaminate_near_duplicates(texts)
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    assert stats["val_test_near_duplicate_pairs"] >= 1
    assert stats["val_test_near_duplicate_pairs_by_threshold"]["0.7"] >= 1
    # 更严格的阈值会减少判定数：删除量对协议阈值有依赖，不能只报一个点。
    _, strict = decontaminate_near_duplicates(texts, threshold=0.99)
    assert strict["val_test_near_duplicate_pairs"] == 0
    assert strict["threshold_sensitivity"]["0.7"]["val_test"] >= 1
    # 阈值恰好等于显示键（0.7）时不能重复计数
    _, at70 = decontaminate_near_duplicates(texts, threshold=0.7)
    assert (at70["val_test_near_duplicate_pairs"]
            == at70["threshold_sensitivity"]["0.7"]["val_test"])
    print("  [ok] 近似重复：val/test 只诊断不改写；三档阈值敏感性可见")


def test_near_duplicate_eval_internal_clusters_do_not_rewrite_eval():
    """val/test 内部近重复必须被诊断，并给出块级有效样本量；两侧不改写。"""
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth").replace(
        "public library", "public museum")
    other = _near_dup_unrelated()
    texts = {
        "train": ("A train-only paragraph that is unrelated to the evaluation "
                  "and long enough to remain after near-duplicate scanning."),
        "val": f"{base}\n\n{edited}\n\n{other}",
        "test": f"{base}\n\n{other}",
    }
    out, stats = decontaminate_near_duplicates(texts)
    assert out["val"] == texts["val"] and out["test"] == texts["test"]
    eint = stats["eval_internal_near_duplicates"]
    assert eint["scan_possible"] is True
    assert eint["candidate_pairs"] >= 1
    assert eint["max_similarity"] >= 0.8
    assert any(row["pair"] == "val_val"
               for row in eint["top_similarities"])
    val = eint["val"]
    assert val["eligible_blocks"] == 3
    assert val["near_duplicate_clusters"] >= 1
    assert val["blocks_in_near_duplicate_clusters"] >= 2
    assert 0.0 < val["duplicate_fraction_chars"] <= 1.0
    assert val["effective_blocks"] < val["eligible_blocks"]
    assert val["eligible_coverage_fraction"] == 1.0
    # test 里 base/other 不是近重复；val 的簇只属于 val，不能把跨 split 对混入
    assert eint["test"]["near_duplicate_clusters"] == 0
    # 三档阈值都保留簇，且更严格阈值不会增加被聚类的字符
    by_t = eint["by_threshold"]
    assert by_t["0.7"]["val"]["clustered_chars"] >= (
        by_t["0.9"]["val"]["clustered_chars"])
    print("  [ok] 近似重复：val/test 内部近重复聚簇与有效块数可见，eval 不改写")


def test_near_duplicate_tolerates_bad_input_and_validates_params():
    for texts in (None, [], {}, {"train": 123, "val": ["x"]}):
        out, stats = decontaminate_near_duplicates(texts)
        assert isinstance(out, dict)
        assert stats["train_blocks_dropped_total"] == 0
    for bad in (True, 0, 1.2, float("nan"), "bad"):
        try:
            decontaminate_near_duplicates({"train": "x" * 300}, threshold=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 threshold 未被拒绝: {bad!r}")
    for bad in (True, 0, -1, 1.5, "bad"):
        try:
            decontaminate_near_duplicates({"train": "x" * 300}, min_chars=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 min_chars 未被拒绝: {bad!r}")
    print("  [ok] 近似重复：容忍损坏输入，threshold/min_chars 严格校验")


def test_eval_internal_near_duplicates_surface_alerts():
    """评测集内部近重复必须进入 alerts；超过协议比例时升级为 warning。"""
    near = {
        "threshold": 0.8,
        "eval_internal_near_duplicates": {
            "warn_fraction_chars": 0.05,
            "candidate_pairs_truncated": False,
            "val": {
                "near_duplicate_clusters": 1,
                "blocks_in_near_duplicate_clusters": 2,
                "duplicate_fraction_chars": 0.10,
                "effective_blocks": 2,
                "eligible_blocks": 3,
            },
            "test": {
                "near_duplicate_clusters": 1,
                "blocks_in_near_duplicate_clusters": 2,
                "duplicate_fraction_chars": 0.01,
                "effective_blocks": 5,
                "eligible_blocks": 6,
            },
        },
    }
    codes = {a["code"]: a["level"] for a in leakage_alerts(
        {}, {}, {"near_duplicate": near}, {}, {})}
    assert codes["val_internal_near_duplicates"] == "warning"
    assert codes["test_internal_near_duplicates"] == "info"
    near["eval_internal_near_duplicates"]["candidate_pairs_truncated"] = True
    codes2 = {a["code"] for a in leakage_alerts(
        {}, {}, {"near_duplicate": near}, {}, {})}
    assert "eval_internal_near_duplicate_scan_truncated" in codes2
    print("  [ok] 近似重复：val/test 内部近重复告警与截断标记可见")


def test_prepare_bins_applies_near_duplicate_decontamination():
    """近似去污染必须真正进入 bin/报告/告警，而不只是一个独立工具函数。"""
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth").replace(
        "public library", "public museum")
    train_unique = ("A train-only paragraph with enough characters to survive "
                    "tokenization and remain after decontamination.")
    val_unique = ("A val-only paragraph with enough characters for a held-out "
                  "selection split.")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(f"{edited}\n\n{train_unique}\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(f"{base}\n\n{val_unique}\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write(_near_dup_unrelated() + "\n")
        prepare_bins(d, kind="char", force=True)
        with open(os.path.join(d, "data_report.json"), "r",
                  encoding="utf-8") as f:
            rep = json.load(f)
        near = rep["duplicates"]["near_duplicate"]
        assert near["train_blocks_dropped_as_near_duplicates_of_eval"] == 1
        assert near["threshold"] == 0.8
        assert near["threshold_sensitivity"]["0.8"]["train_val"] >= 1
        eint = near["eval_internal_near_duplicates"]
        assert eint["scan_possible"] is True
        assert eint["val"]["eligible_blocks"] >= 1
        assert eint["test"]["eligible_blocks"] >= 1
        codes = {a["code"] for a in rep["alerts"]}
        assert "near_duplicate_decontamination_applied" in codes
        decoded = load_tokenizer(d).decode(load_bin(d, "train"))
        assert "history of the town" not in decoded
        assert "train-only paragraph" in decoded
        # val 必须逐字节保留：尘封 split 不被近似去污染改写
        with open(os.path.join(d, "val.txt"), "r", encoding="utf-8") as f:
            assert f.read().startswith(base)
    print("  [ok] prepare_bins：近似去污染进入 bin/报告/告警，val 不改写")

def test_leakage_alerts_surface_near_duplicate_evidence():
    """近似去污染的删除/val-test 重叠/高删除比例都必须进入 alerts。"""
    dup = {"near_duplicate": {
        "threshold": 0.8,
        "train_blocks_dropped_as_near_duplicates_of_eval": 2,
        "train_intra_split_near_duplicates_removed": 1,
        "train_lines_dropped": 3,
        "val_test_near_duplicate_pairs": 4,
        "removal_fraction_chars": 0.4,
        "candidate_pairs_truncated": False,
        "oversized_groups_truncated": 0,
    }}
    alerts = leakage_alerts({}, {}, dup, {}, {})
    codes = {a["code"] for a in alerts}
    assert "near_duplicate_decontamination_applied" in codes
    assert "eval_splits_near_duplicates" in codes
    assert "near_duplicate_removal_high_fraction" in codes
    # 截断时额外告警；未截断时不出现
    dup["near_duplicate"]["candidate_pairs_truncated"] = True
    codes2 = {a["code"] for a in leakage_alerts({}, {}, dup, {}, {})}
    assert "near_duplicate_scan_truncated" in codes2
    print("  [ok] 近似重复：删除/val-test 重叠/高删除比例/截断均进入告警")

def test_prepare_bins_rejects_catastrophic_near_duplicate_removal():
    """近似去污染若会删空/删除 >90% train，必须失败而不是静默训练空集。"""
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth")
    val = base.replace("public library", "public museum")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(edited + "\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(val + "\n")
        try:
            prepare_bins(d, kind="char", force=True)
        except ValueError as e:
            assert "近似重复去污染" in str(e)
        else:
            raise AssertionError("近似去污染删空 train 时未失败")
    print("  [ok] prepare_bins：近似去污染删空 train 时 fail closed")

def test_prepare_data_passes_near_duplicate_protocol(monkeypatch):
    """近似去污染的协议阈值必须能从数据准备入口显式设置并原样传递。"""
    import sys
    import lega.data as data_mod
    import scripts.prepare_data as prep

    seen = {}

    def fake_rows(max_rows, retries=4):
        return {"train": ["train line " + "a" * 80],
                "val": ["val line " + "b" * 80],
                "test": ["test line " + "c" * 80]}

    def fake_prepare_bins(data_dir, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(prep, "from_rows_api", fake_rows)
    monkeypatch.setattr(data_mod, "prepare_bins", fake_prepare_bins)
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, "out")
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out_dir, "--source", "rows",
            "--max_tokens", "600", "--near_dup_threshold", "0.7",
            "--near_dup_min_chars", "250"])
        prep.main()
        assert seen["near_dup_threshold"] == 0.7
        assert seen["near_dup_min_chars"] == 250
    print("  [ok] prepare_data：近似去污染协议参数可显式设置并传递")

def test_prepare_bins_cache_invalidated_on_near_duplicate_params():
    """同一份 txt 用不同近似去污染阈值构建的 bin 不同，缓存必须失效重建。"""
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth")
    train_unique = ("A legitimate train-only paragraph that is unrelated to the "
                    "evaluation split and long enough to keep the removal fraction "
                    "below the catastrophic guard threshold.")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(f"{edited}\n\n{train_unique}\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(base + "\n")
        meta1 = prepare_bins(d, kind="char", force=True)
        assert meta1["near_dup"]["threshold"] == 0.8
        n1 = meta1["train_tokens"]
        decoded1 = load_tokenizer(d).decode(load_bin(d, "train"))
        assert "history of the town" not in decoded1
        # 阈值提高到 1.0：编辑副本不再算近重复，必须重建并保留该块
        meta2 = prepare_bins(d, kind="char", near_dup_threshold=1.0)
        assert meta2["near_dup"]["threshold"] == 1.0
        assert meta2["train_tokens"] > n1
        decoded2 = load_tokenizer(d).decode(load_bin(d, "train"))
        assert "history of the town" in decoded2
        # 参数相同则继续命中缓存
        meta3 = prepare_bins(d, kind="char", near_dup_threshold=1.0)
        assert meta3["train_tokens"] == meta2["train_tokens"]
    print("  [ok] prepare_bins：近似去污染阈值进入缓存身份，变化即重建")


def test_prepare_bins_near_duplicate_params_strict_even_on_cache_hit():
    """近似去污染参数必须严格校验，缓存命中不能绕过 bool/小数校验。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write("A train-only paragraph with enough distinct characters "
                    "to survive near-duplicate scanning and keep a valid bin.\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write("A validation paragraph that is unrelated to the training "
                    "text and long enough for the tokenizer.\n")
        prepare_bins(d, kind="char", force=True)
        # 非法请求参数即使已有可复用缓存也必须拒绝，不能被缓存命中吞掉。
        for bad in (True, 0, 1.2, float("nan"), "bad", "0_8"):
            try:
                prepare_bins(d, kind="char", near_dup_threshold=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 near_dup_threshold 未被拒绝: {bad!r}")
        for bad in (True, 0, -1, 1.5, "bad", "2_00"):
            try:
                prepare_bins(d, kind="char", near_dup_min_chars=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 near_dup_min_chars 未被拒绝: {bad!r}")
        meta_path = os.path.join(d, "meta.json")
        # 小数 min_chars 不能被 int() 截断成 200 后继续命中缓存。
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
        m["near_dup"]["min_chars"] = 200.9
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(m, f)
        rebuilt = prepare_bins(d, kind="char")
        assert rebuilt["near_dup"]["min_chars"] == 200, rebuilt["near_dup"]
        # bool threshold 不能被 float(True)==1.0 伪装成请求的 1.0。
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
        m["near_dup"]["threshold"] = True
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(m, f)
        rebuilt = prepare_bins(d, kind="char", near_dup_threshold=1.0)
        assert rebuilt["near_dup"]["threshold"] == 1.0, rebuilt["near_dup"]
    print("  [ok] prepare_bins：近似去污染参数严格校验，缓存不能绕过")


def test_near_duplicate_threshold_sensitivity_reports_actual_removal():
    """三档阈值必须给出真实删除量，而不只是判定对数。

    阈值从 0.7 收紧到 0.9 时，删除的块/字符数只能不增；且选定阈值那一档
    必须与函数实际执行的删除量一致，否则汇总页会展示一套、bin 里是另一套。
    """
    base = _near_dup_base()
    edited = base.replace("early nineteenth", "early twentieth").replace(
        "several decades", "many decades").replace(
        "public library", "public museum")
    train_unique = ("A train-only paragraph with enough distinct material to "
                    "remain after decontamination and keep the corpus large.")
    texts = {"train": f"{edited}\n\n{train_unique}", "val": base, "test": ""}
    out, stats = decontaminate_near_duplicates(texts)
    assert "history of the town" not in out["train"]
    sens = stats["threshold_sensitivity_removal"]
    assert set(sens) == {"0.7", "0.8", "0.9"}
    # 删除量随阈值单调不增（更严格 → 删得更少）
    assert (sens["0.7"]["train_chars_removed"]
            >= sens["0.8"]["train_chars_removed"]
            >= sens["0.9"]["train_chars_removed"])
    chosen = sens[str(round(stats["threshold"], 4))]
    assert chosen["train_blocks_removed_total"] == stats[
        "train_blocks_dropped_total"]
    assert chosen["train_chars_removed"] == stats["train_chars_dropped"]
    assert chosen["removal_fraction_chars"] == stats["removal_fraction_chars"]
    # 判定对数与删除量是两个口径：都必须在报告里，不可互相替代
    assert stats["threshold_sensitivity"]["0.8"]["train_val"] >= 1
    print("  [ok] 近似重复：三档阈值报告真实删除块/字符量并与执行结果一致")


def test_near_duplicate_reports_scan_char_coverage():
    """短块/少 shingle 被跳过时必须报告未覆盖字符，不能暗示全语料已扫描。"""
    short = "= References ="
    long_block = ("A long train paragraph that is long enough to enter the "
                  "near-duplicate scanner and provide a denominator. " * 6)
    _, stats = decontaminate_near_duplicates(
        {"train": f"{short}\n\n{long_block}", "val": "", "test": ""})
    near = stats
    assert near["train_blocks_skipped_short"] >= 1
    assert near["train_chars_skipped_short"] > 0
    assert 0.0 < near["train_chars_coverage_fraction"] < 1.0
    assert near["train_chars_coverage_fraction"] == (
        near["train_chars_eligible"] / near["train_chars_total"])
    assert near["near_duplicate_min_coverage"] > 0
    print("  [ok] 近似重复：扫描字符覆盖率与被跳过字符数显式落盘")


def test_post_cleanup_residual_verifies_exact_removal():
    """清洗后复核：删除后必须证明剩余为零；未清洗时必须判为未闭合。"""
    shared = ("This is a long shared paragraph that appears verbatim on both "
              "sides of the split and must be removed from train only.")
    clean_texts, _ = decontaminate_texts(
        {"train": shared + "\n" + "train-only " * 20,
         "val": shared, "test": ""}, min_chars=40)
    audit = post_cleanup_residual(clean_texts, min_chars=40, min_lines=2)
    assert audit["checked_pairs"]
    assert audit["train_eval_clean"] is True
    assert audit["overlap_long_lines"]["train_val"]["normalized_lines"] == 0
    # 未清洗时同一内容必须被复核抓到（否则复核只是形式）
    dirty = {"train": shared, "val": shared, "test": ""}
    bad = post_cleanup_residual(dirty, min_chars=40, min_lines=2)
    assert bad["train_eval_clean"] is False
    assert bad["overlap_long_lines"]["train_val"]["exact_lines"] >= 1
    # 短行重复单独计数，不因“精确长行已干净”而消失
    short_dirty = {"train": "= Same Title =", "val": "= Same Title =",
                   "test": ""}
    short_audit = post_cleanup_residual(short_dirty, min_chars=40)
    assert short_audit["train_eval_clean"] is True
    assert short_audit["overlap_short_lines"]["train_val"]["exact_lines"] == 1
    print("  [ok] 清洗后复核：train↔eval 精确/归一化重复闭合，短行单独计数")


def test_post_cleanup_residual_detects_multiline_block_copy():
    """多行块复制即使每行都短于 min_chars，也必须被块级复核抓到。"""
    block = "\n".join(["short line one", "short line two", "short line three"])
    audit = post_cleanup_residual(
        {"train": block + "\n", "val": "other\n\n" + block, "test": ""},
        min_chars=40, min_lines=2)
    assert audit["overlap_blocks"]["train_val"] >= 1
    assert audit["train_eval_clean"] is False
    print("  [ok] 清洗后复核：多行短行块的跨 split 复制可被识别")


def test_temporal_split_reports_actual_fraction_skew():
    """切点吸附会让实际比例偏离声明目标；必须落盘偏差并生成告警。"""
    # 文档边界在 70% 与 95% 附近，最近的可用切点会把 80/10/10 拉偏
    text = ("= A =\n" + ("alpha " * 120) + "\n\n"
            "= B =\n" + ("beta " * 40) + "\n\n"
            "= C =\n" + ("gamma " * 12) + "\n")
    train, val, test, info = temporal_split_text(text)
    assert train + val + test == text
    assert info["target_fractions"]["train"] == 0.8
    assert abs(sum(info["actual_fractions"].values()) - 1.0) < 1e-12
    assert info["max_abs_fraction_deviation"] > 0.05
    assert len(info["cut_deviation_chars"]) == 2
    codes = {a["code"] for a in split_protocol_alerts(info)}
    assert "split_fraction_skew" in codes
    # 未超过容差时不应误报
    codes_ok = {a["code"] for a in split_protocol_alerts(
        {"actual_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
         "target_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
         "max_abs_fraction_deviation": 0.0,
         "fraction_tolerance": 0.05})}
    assert "split_fraction_skew" not in codes_ok
    print("  [ok] 时序划分：目标/实际比例、切点偏移与偏斜告警可审计")


def test_text_and_token_stats_report_data_accounting():
    """字符/字节/词表覆盖度是数据规模与退化的可复核分母。"""
    st = text_length_stats("a\n中文")
    assert st["chars"] == 4
    assert st["utf8_bytes"] == 8
    assert text_length_stats("")["utf8_bytes"] == 0
    tok = token_stream_stats(np.array([0, 0, 1, 2], dtype=np.uint16), 10,
                             "train")
    assert tok["tokens"] == 4 and tok["distinct_ids"] == 3
    assert abs(tok["type_token_ratio"] - 0.75) < 1e-12
    assert abs(tok["most_common_token_share"] - 0.5) < 1e-12
    assert abs(tok["singleton_token_mass"] - 0.5) < 1e-12
    assert abs(tok["token_entropy_bits"] - 1.5) < 1e-12
    assert abs(tok["vocab_coverage"] - 0.3) < 1e-12
    print("  [ok] 数据核算：UTF-8 字节、词表覆盖度与 token 熵可复核")


def test_prepare_bins_report_includes_post_cleanup_and_accounting():
    """新增证据必须进入 data_report，而不是只存在于独立工具函数。"""
    shared = ("A long shared paragraph present in both train and validation "
              "that the decontamination pass must remove from train.")
    unique = ("A distinct train-only paragraph long enough to remain after "
              "decontamination and give the report a real denominator.")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write(f"{shared}\n\n{unique}\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(shared + "\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("held-out test text with enough characters to encode.\n")
        prepare_bins(d, kind="char", force=True)
        with open(os.path.join(d, "data_report.json"), "r",
                  encoding="utf-8") as f:
            rep = json.load(f)
        post = rep["duplicates"]["post_cleanup"]
        assert post["train_eval_clean"] is True
        assert post["overlap_long_lines"]["train_val"]["exact_lines"] == 0
        acct = rep["accounting"]
        assert acct["train"]["tokens"] > 0
        assert acct["train"]["utf8_bytes"] > 0
        assert acct["train"]["bytes_per_token"] is not None
        near = rep["duplicates"]["near_duplicate"]
        assert 0.0 <= near["train_chars_coverage_fraction"] <= 1.0
        codes = {a["code"] for a in rep["alerts"]}
        assert "post_cleanup_verified" in codes
    print("  [ok] prepare_bins：清洗后复核与字符/字节核算进入 data_report")
