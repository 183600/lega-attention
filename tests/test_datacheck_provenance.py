"""原始数据来源溯源测试：manifest 哈希、fail-closed、preflight 绑定。"""
import hashlib
import json
import os
import sys
import tempfile

from lega.datacheck import (SOURCE_MANIFEST_NAME,
                            build_source_provenance, file_entry,
                            load_source_manifest, manifest_digest,
                            provenance_text_hashes, sha256_file,
                            verify_source_manifest_file,
                            verify_source_provenance,
                            write_source_manifest)
from lega.preflight import data_source_provenance_issue


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local_text(n_docs: int = 40) -> str:
    docs = [f"= Doc {i} =\n" + " ".join(f"word{j}_d{i}" for j in range(20)) + "."
            for i in range(n_docs)]
    return "\n\n".join(docs) + "\n"


def test_source_provenance_detects_file_and_digest_tampering():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw.txt")
        with open(raw, "wb") as f:
            f.write(b"original source bytes")
        entry = file_entry(raw, "raw_source", data_dir=d)
        assert entry["path"] == "raw.txt"
        assert entry["location"] == "data_dir"
        assert entry["sha256"] == _sha(b"original source bytes")
        prov = build_source_provenance(
            "local", [entry], source_path=raw, api={"kind": "test"})
        assert prov["digest"] == manifest_digest(prov)
        assert verify_source_provenance(prov, d) == []
        path = write_source_manifest(d, prov)
        assert os.path.exists(path)
        assert load_source_manifest(d)["digest"] == prov["digest"]

        with open(raw, "wb") as f:
            f.write(b"tampered source bytes")
        issues = verify_source_provenance(prov, d)
        assert any("SHA-256" in x for x in issues), issues

        bad_digest = dict(prov)
        bad_digest["digest"] = "0" * 64
        issues = verify_source_provenance(bad_digest, d)
        assert any("digest" in x for x in issues), issues

        unsafe = {
            "role": "raw_source", "path": "../escape.txt",
            "location": "data_dir", "size": 1, "sha256": "0" * 64}
        try:
            build_source_provenance("local", [unsafe])
        except ValueError as e:
            assert "不安全" in str(e)
        else:
            raise AssertionError("路径穿越条目未被 build 拒绝")
    print("  [ok] provenance：文件哈希/digest/路径穿越均能检出")


def test_download_refuses_changed_source_without_explicit_allowance():
    from scripts.prepare_data import SourceChangedError, _download

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.bin")
        dst = os.path.join(d, "dst.bin")
        with open(src, "wb") as f:
            f.write(b"version one")
        url = "file://" + src
        entry = _download(url, dst, "test file")
        assert entry["sha256"] == _sha(b"version one")

        same = _download(url, dst, "test file",
                         expected_sha256=entry["sha256"])
        assert not same.get("changed")

        with open(dst, "wb") as f:
            f.write(b"version two")
        try:
            _download(url, dst, "test file",
                      expected_sha256=entry["sha256"])
        except SourceChangedError as e:
            assert "--allow_source_change" in str(e)
        else:
            raise AssertionError("来源变化未被 fail closed")

        allowed = _download(url, dst, "test file",
                            expected_sha256=entry["sha256"],
                            allow_source_change=True)
        assert allowed["changed"] is True
        assert allowed["previous_sha256"] == entry["sha256"]
        assert allowed["sha256"] == _sha(b"version two")
    print("  [ok] provenance：已下载原始文件变化默认拒绝，显式放行才记录变更")


def test_prepare_data_records_local_source_provenance(monkeypatch):
    import lega.data as data_mod
    import scripts.prepare_data as prep

    monkeypatch.setattr(
        data_mod, "get_gpt2_tokenizer",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(_local_text())
        monkeypatch.setattr(sys, "argv", [
            "prepare_data", "--out_dir", out, "--source", "local",
            "--local_txt", src, "--tokenizer", "auto"])
        prep.main()

        with open(os.path.join(out, "data_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        prov = report["source_provenance"]
        assert prov["source"] == "local"
        assert prov["source_path"] == os.path.abspath(src)
        assert prov["digest"] == meta["source_provenance_digest"]
        assert os.path.exists(os.path.join(out, SOURCE_MANIFEST_NAME))
        assert data_source_provenance_issue(out) is None

        text_hashes = provenance_text_hashes(prov)
        for split in ("train", "val", "test"):
            assert text_hashes[split] == sha256_file(
                os.path.join(out, f"{split}.txt"))
            assert meta["source_sha256"][split] == text_hashes[split]
        roles = {e["role"] for e in prov["files"]}
        assert "local_source" in roles
        assert "text_split" in roles

        with open(os.path.join(out, "train.txt"), "a", encoding="utf-8") as f:
            f.write("tampered\n")
        issue = data_source_provenance_issue(out)
        assert issue and "train.txt" in issue, issue
    print("  [ok] provenance：local 准备写入三方一致清单，篡改 txt 被 preflight 拦截")


def test_local_source_change_requires_explicit_allowance(monkeypatch):
    import lega.data as data_mod
    import scripts.prepare_data as prep
    from scripts.prepare_data import SourceChangedError

    monkeypatch.setattr(
        data_mod, "get_gpt2_tokenizer",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "local.txt")
        out = os.path.join(d, "out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(_local_text())
        base_argv = ["prepare_data", "--out_dir", out, "--source", "local",
                     "--local_txt", src, "--tokenizer", "auto"]
        monkeypatch.setattr(sys, "argv", base_argv)
        prep.main()
        with open(os.path.join(out, "source_manifest.json"),
                  encoding="utf-8") as f:
            old_digest = json.load(f)["digest"]

        with open(src, "w", encoding="utf-8") as f:
            f.write(_local_text(45))
        try:
            prep.main()
        except SourceChangedError as e:
            assert "--allow_source_change" in str(e)
        else:
            raise AssertionError("本地源文件变化未被 fail closed")

        monkeypatch.setattr(sys, "argv", base_argv + ["--allow_source_change"])
        prep.main()
        with open(os.path.join(out, "source_manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["digest"] != old_digest
        assert manifest["previous_digest"] == old_digest
        assert len(manifest["changes"]) == 1
        assert manifest["changes"][0]["role"] == "local_source"
        assert data_source_provenance_issue(out) is None
    print("  [ok] provenance：本地源变化默认拒绝，显式放行后记录 previous/changes")


def test_source_manifest_must_match_report_digest():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "train.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("train text\n")
        entry = file_entry(path, "text_split", data_dir=d)
        prov = build_source_provenance("local", [entry])
        write_source_manifest(d, prov)
        assert verify_source_manifest_file(d, prov) == []
        other = build_source_provenance("local", [entry], api={"v": 2})
        issues = verify_source_manifest_file(d, other)
        assert issues and "不一致" in issues[0], issues
    print("  [ok] provenance：独立 source_manifest 与 report 不同版会被拒绝")
