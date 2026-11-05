"""文档结构证据与划分集合级指纹测试。

覆盖三件事：
1. 群组级/段落级/行前缀级 split 互斥性审计能把“文档被复制/被切开”查出来；
2. 跨文档窗口比例与文档内窗口覆盖量的口径正确；
3. 划分集合级指纹（split_set）能检出“新 train/val + 旧 test”的混合版本。
"""

import json
import os
import tempfile

from lega.datacheck import (
    build_split_set,
    document_structure_report,
    group_disjointness_audit,
    load_split_set,
    sha256_file,
    split_set_digest,
    split_documents,
    verify_split_set,
    write_split_set,
)


def _doc(i: int, tag: str = "Doc") -> str:
    return (
        f"= {tag} {i} =\n"
        + " ".join(f"{tag}token{i}_{j}" for j in range(30))
        + f"\nunique tail marker {tag} {i}\n"
    )


def _corpus(n: int, tag: str) -> str:
    """每个 split 用不同 tag：标题与正文都互不相同（否则会被判为共享行）。"""
    return "\n\n".join(_doc(i, tag) for i in range(n)) + "\n"


def test_group_disjointness_audit_detects_shared_document():
    shared = _doc(0, "Shared")
    texts = {
        "train": _corpus(3, "Train") + "\n" + shared,
        "val": _corpus(2, "Val") + "\n" + shared,
        "test": _corpus(2, "Test"),
    }
    audit = group_disjointness_audit(texts)
    assert audit["groups_disjoint"] is False
    assert audit["verdict"].startswith("shared:")
    pair = audit["pairs"]["train|val"]
    assert pair["shared_groups"] >= 1
    assert pair["shared_groups_examples"]
    assert "shared_groups" in audit["problems"]

    clean = group_disjointness_audit(
        {
            "train": _corpus(3, "Train"),
            "val": _corpus(2, "Val"),
            "test": _corpus(2, "Test"),
        }
    )
    assert clean["groups_disjoint"] is True
    assert clean["paragraphs_disjoint"] is True
    assert clean["line_prefixes_disjoint"] is True
    assert clean["verdict"] == "disjoint"
    print("  [ok] 结构审计：共享文档被群组级检出，互斥语料判为 disjoint")


def test_group_disjointness_audit_detects_split_document():
    """文档被从中间切开时，群组/逐行比较看不到，但切点两侧会留下词片段。"""
    full = _doc(0, "Split")
    cut = len(full) // 2
    head = full[:cut]
    tail = full[cut:]
    texts = {"train": head, "val": tail, "test": _corpus(2, "Test")}
    audit = group_disjointness_audit(texts)
    pair = audit["pairs"]["train|val"]
    assert pair["shared_groups"] == 0
    # 切点两侧的词片段被记录下来供人工复核
    continuity = audit["boundary_continuity"]
    boundary = {p["pair"]: p for p in continuity["pairs"]}["train|val"]
    assert boundary["tail_word"]
    assert boundary["head_word"]
    print("  [ok] 结构审计：切点词片段被记录（疑似行中切开由精确自检覆盖）")


def test_temporal_split_reports_line_level_cut():
    """本地时序划分的精确自检：切点必须落在换行处。"""
    from lega.datacheck import temporal_split_text

    text = _corpus(12, "Doc")
    _, _, _, info = temporal_split_text(text)
    assert info["lines_cross_cut"] is False
    assert info["documents_cross_cut"] is False
    assert info["paragraphs_cross_cut"] is False
    assert len(info["cut_offsets"]) == 2
    print("  [ok] 划分自检：切点落在行边界上")


def test_document_structure_report_counts_cross_document_windows():
    text = _corpus(4, "Doc")
    info = split_documents(text)
    counts = [100, 100, 100, 100]
    report = document_structure_report(
        text, info["segments"], info["group_unit"], counts, ctx_grid=[64]
    )
    cross = report["cross_document_windows"]["64"]
    aligned = report["doc_aligned_windows"]["64"]
    assert report["n_documents"] == 4
    assert report["tokens_total"] == 400
    assert cross["windows_total"] > 0
    # 4 个等长文档（各 100 token）、ctx=64：起点空间 400-64=336，
    # 每个文档边界对应 64 个跨文档起点，3 个边界共 192 → 4/7
    assert abs(cross["crossing_window_fraction"] - 192 / 336) < 1e-9
    # 每个文档 100 token，ctx=64：文档内窗口数 = floor((100-1)/64) = 1
    assert aligned["n_windows"] == 4
    assert aligned["n_documents_eligible"] == 4
    assert abs(aligned["token_coverage_fraction"] - (4 * 64 / 400)) < 1e-9
    print("  [ok] 结构报告：跨文档窗口期望比例与文档内覆盖量口径一致")


def test_split_set_digest_detects_mixed_version_splits():
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for split, body in (("train", "a\n"), ("val", "b\n"), ("test", "c\n")):
            path = os.path.join(d, f"{split}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            paths[split] = path
        hashes = {s: sha256_file(p) for s, p in paths.items()}

        split_set = build_split_set(
            hashes, {s: os.path.getsize(p) for s, p in paths.items()}
        )
        assert split_set["digest"] == split_set_digest(hashes)
        assert verify_split_set(split_set, hashes) == []
        write_split_set(d, split_set)
        assert load_split_set(d)["digest"] == split_set["digest"]

        # 只替换 test：逐文件哈希各自都没问题，但集合级指纹变了
        with open(paths["test"], "w", encoding="utf-8") as f:
            f.write("stale-test\n")
        new_hashes = {s: sha256_file(p) for s, p in paths.items()}
        issues = verify_split_set(split_set, new_hashes)
        assert issues, issues
        assert any("test.txt" in x for x in issues)
        assert any("混合版本" in x for x in issues)
        assert split_set_digest(new_hashes) != split_set["digest"]
    print("  [ok] 划分集合指纹：混合版本 split 被集合级指纹检出")


def test_preflight_detects_mixed_version_split_set():
    import lega.data as data_mod
    from lega.preflight import data_split_set_issue

    with tempfile.TemporaryDirectory() as d:
        for split, body in (
            ("train", "alpha\n"),
            ("val", "beta\n"),
            ("test", "gamma\n"),
        ):
            with open(os.path.join(d, f"{split}.txt"), "w", encoding="utf-8") as f:
                f.write(body)
        hashes = {
            s: sha256_file(os.path.join(d, f"{s}.txt"))
            for s in ("train", "val", "test")
        }
        split_set = build_split_set(hashes)
        write_split_set(d, split_set)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_sha256": hashes,
                    "split_set_digest": split_set["digest"],
                    "cache_schema": data_mod.CACHE_SCHEMA,
                },
                f,
            )
        assert data_split_set_issue(d) is None

        # test.txt 换成旧版本：集合指纹不再匹配
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("old-test\n")
        issue = data_split_set_issue(d)
        assert issue and "混合版本" in issue, issue
    print("  [ok] preflight：混合版本划分在训练前被拦截")


def test_prepare_data_publishes_splits_as_one_set():
    import lega.data as data_mod
    import scripts.prepare_data as prep

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(_corpus(12, "Doc"))
        old_get_tokenizer = data_mod.get_gpt2_tokenizer
        data_mod.get_gpt2_tokenizer = lambda: (_ for _ in ()).throw(
            RuntimeError("offline")
        )
        try:
            import sys

            argv = sys.argv
            sys.argv = [
                "prepare_data",
                "--out_dir",
                out,
                "--source",
                "local",
                "--local_txt",
                src,
                "--tokenizer",
                "auto",
            ]
            try:
                prep.main()
            finally:
                sys.argv = argv
        finally:
            data_mod.get_gpt2_tokenizer = old_get_tokenizer

        # 三个 split 一次性发布，集合级指纹落盘
        split_set = load_split_set(out)
        assert split_set is not None
        assert set(split_set["splits"]) == {"train", "val", "test"}
        assert split_set["digest"] == split_set_digest(
            {
                s: sha256_file(os.path.join(out, f"{s}.txt"))
                for s in ("train", "val", "test")
            }
        )
        # 没有遗留暂存文件
        assert not os.path.exists(os.path.join(out, "train.txt.part"))
        with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["split_set_digest"] == split_set["digest"]
    print("  [ok] 数据准备：三个 split 作为一组发布并落盘集合指纹")


def test_staged_write_keeps_previous_splits_on_failure():
    """准备中途失败不应把上一份一致的划分替换成混合版本。"""
    from scripts.prepare_data import cleanup_staged, staged_path, write_clean_lines

    with tempfile.TemporaryDirectory() as d:
        train = os.path.join(d, "train.txt")
        val = os.path.join(d, "val.txt")
        test = os.path.join(d, "test.txt")
        for path, body in (
            (train, "old train\n"),
            (val, "old val\n"),
            (test, "old test\n"),
        ):
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
        before = {p: open(p, encoding="utf-8").read() for p in (train, val, test)}

        # 只写 train 的暂存文件，随后“失败”：val/test 的暂存文件缺失
        write_clean_lines(["new train\n"], train, 1000, publish=False)
        assert os.path.exists(staged_path(train))
        assert open(train, encoding="utf-8").read() == before[train]

        cleanup_staged({"train": train, "val": val, "test": test})
        assert not os.path.exists(staged_path(train))
        for path in (train, val, test):
            assert open(path, encoding="utf-8").read() == before[path]
    print("  [ok] 数据准备：暂存失败不破坏上一份一致划分")
