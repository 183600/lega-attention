"""原始数据来源溯源：URL/revision/文件大小/SHA-256 清单与训练前复核。

为什么需要这一层
================
`data_report.json` 已记录“处理后的 train/val/test 文本与 bin”的哈希，但
“这些文本究竟来自哪一份原始下载/本地文件”此前只有一个 source 名称。HF
`resolve/main`、镜像 zip 与本地文件都可能被替换或更新；仅凭相同文件名与
相同 split 协议，无法证明两次实验消费的是同一份字节。

本模块把原始来源与处理结果绑定为可复核的 manifest：

* 逐文件记录 role/path/location/size/sha256；
* 记录 HF revision、下载 URL 或 rows API 参数、local 源路径；
* 用规范化 JSON 计算 `digest`，供 `data_report.json`、`source_manifest.json`
  与 `meta.json` 三方交叉核对；
* 复核时只对仍存在的文件重算哈希；文件已被删除时跳过并如实视为“未核对”，
  不把缺失当成通过。

边界：SHA-256 只能证明“拿到的是同一份字节”，不能证明上游数据本身没有
污染、偏见或许可问题；`main` 之类的可变 revision 必须由正式实验显式固定。
"""
from __future__ import annotations

import hashlib
import json
import os
import re

SOURCE_PROVENANCE_SCHEMA = "lega-source-provenance-v1"
SOURCE_MANIFEST_NAME = "source_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCATIONS = ("data_dir", "external")
_ENTRY_KEYS = ("role", "path", "location", "size", "sha256")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _posix(path: str) -> str:
    return str(path).replace(os.sep, "/")


def _data_dir_relpath(path: str, data_dir: str):
    """path 在 data_dir 内时返回安全的 POSIX 相对路径，否则 None。"""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(data_dir))
    except (OSError, ValueError):
        return None
    rel = _posix(rel)
    if rel in ("", ".") or rel == ".." or rel.startswith("../"):
        return None
    return rel


def file_entry(path: str, role: str, data_dir: str = None) -> dict:
    """构造一个原始文件条目；data_dir 内文件用相对路径，外部文件用文件名。"""
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"role 必须是非空字符串，收到 {role!r}")
    if not isinstance(path, str) or not path:
        raise ValueError(f"path 必须是非空字符串，收到 {path!r}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"无法溯源：文件不存在或不是普通文件 {path!r}")
    entry = {
        "role": role.strip(),
        "size": int(os.path.getsize(path)),
        "sha256": sha256_file(path),
    }
    rel = _data_dir_relpath(path, data_dir) if data_dir else None
    if rel is not None:
        entry["path"] = rel
        entry["location"] = "data_dir"
    else:
        entry["path"] = os.path.basename(os.path.abspath(path))
        entry["location"] = "external"
        entry["external_path"] = os.path.abspath(path)
    return entry


def _validate_entry(entry, where: str) -> list:
    issues = []
    if not isinstance(entry, dict):
        return [f"{where} 不是对象"]
    role = entry.get("role")
    path = entry.get("path")
    location = entry.get("location")
    size = entry.get("size")
    sha = entry.get("sha256")
    if not isinstance(role, str) or not role.strip():
        issues.append(f"{where} 的 role 不是非空字符串")
    if not isinstance(path, str) or not path.strip():
        issues.append(f"{where} 的 path 不是非空字符串")
    if location not in _LOCATIONS:
        issues.append(f"{where} 的 location={location!r} 不是 {_LOCATIONS}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        issues.append(f"{where} 的 size={size!r} 不是非负整数")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        issues.append(f"{where} 的 sha256 不是 64 位小写十六进制")
    if location == "data_dir" and isinstance(path, str):
        p = _posix(path)
        if p.startswith("/") or p.startswith("../") or "/../" in p or p in ("", "."):
            issues.append(f"{where} 的 data_dir 路径不安全: {path!r}")
    return issues


def _entry_key(entry: dict):
    return (str(entry.get("role")), str(entry.get("path")),
            str(entry.get("location")), str(entry.get("sha256")))


def _canonical_payload(provenance: dict) -> dict:
    files = []
    for entry in (provenance.get("files") or []):
        if isinstance(entry, dict):
            files.append({k: entry.get(k) for k in _ENTRY_KEYS})
    files.sort(key=lambda e: (str(e.get("role")), str(e.get("path")),
                              str(e.get("location")), str(e.get("sha256"))))
    urls = provenance.get("urls")
    if isinstance(urls, list):
        urls = sorted(
            [u for u in urls if isinstance(u, dict)],
            key=lambda u: (str(u.get("name")), str(u.get("url"))))
    payload = {
        "schema": provenance.get("schema"),
        "source": provenance.get("source"),
        "revision": provenance.get("revision"),
        "source_path": provenance.get("source_path"),
        "urls": urls,
        "api": provenance.get("api"),
        "files": files,
    }
    return payload


def manifest_digest(provenance: dict) -> str:
    """规范化 manifest 的内容摘要（不含 digest/时间戳/变更历史本身）。"""
    raw = json.dumps(_canonical_payload(provenance), sort_keys=True,
                     ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_source_provenance(source: str, files, revision: str = None,
                            source_path: str = None, urls=None, api=None,
                            changes=None, previous_digest: str = None) -> dict:
    """组装来源 manifest；files 必须已包含原始文件与处理后的 split 文本。"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source 必须是非空字符串，收到 {source!r}")
    if revision is not None and (not isinstance(revision, str)
                                 or not revision.strip()):
        raise ValueError(f"revision 必须是非空字符串或 None，收到 {revision!r}")
    if source_path is not None and not isinstance(source_path, str):
        raise ValueError(f"source_path 必须是字符串或 None，收到 {source_path!r}")
    entries = list(files or [])
    if not entries:
        raise ValueError("source_provenance 至少需要一个文件条目")
    issues = []
    for i, entry in enumerate(entries):
        issues.extend(_validate_entry(entry, f"files[{i}]"))
    if issues:
        raise ValueError("；".join(issues))
    entries.sort(key=_entry_key)
    prov = {
        "schema": SOURCE_PROVENANCE_SCHEMA,
        "source": source.strip(),
        "revision": revision.strip() if isinstance(revision, str) else None,
        "source_path": source_path,
        "urls": list(urls) if isinstance(urls, list) else urls,
        "api": api if isinstance(api, dict) else None,
        "files": entries,
        "changes": list(changes) if isinstance(changes, list) else [],
        "previous_digest": previous_digest if isinstance(previous_digest, str)
        else None,
    }
    prov["digest"] = manifest_digest(prov)
    return prov


def write_source_manifest(data_dir: str, provenance: dict) -> str:
    """原子写入 data_dir/source_manifest.json，返回路径。"""
    from ..utils import save_json
    path = os.path.join(data_dir, SOURCE_MANIFEST_NAME)
    save_json(provenance, path)
    return path


def load_source_manifest(data_dir: str):
    """读取 source_manifest.json；缺失/损坏/非对象返回 None。"""
    path = os.path.join(data_dir, SOURCE_MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def verify_source_provenance(provenance, data_dir: str,
                             check_external: bool = False) -> list:
    """复核 manifest 自洽性，并对仍存在的文件重算 size/SHA-256。

    返回问题列表；缺失文件不计为问题（原始下载可能已清理、纯 bin 目录
    分发时更不可能有源文件），但也不会被当成“已核对通过”。
    """
    issues = []
    if not isinstance(provenance, dict):
        return ["source_provenance 不是对象"]
    if provenance.get("schema") != SOURCE_PROVENANCE_SCHEMA:
        issues.append("source_provenance.schema="
                      f"{provenance.get('schema')!r} 与当前协议 "
                      f"{SOURCE_PROVENANCE_SCHEMA} 不一致")
    files = provenance.get("files")
    if not isinstance(files, list) or not files:
        return issues + ["source_provenance.files 不是非空列表"]
    for i, entry in enumerate(files):
        issues.extend(_validate_entry(entry, f"files[{i}]"))
    if issues:
        return issues
    seen = set()
    for i, entry in enumerate(files):
        key = _entry_key(entry)
        if key in seen:
            issues.append(f"files[{i}] 与其他条目重复（role/path/location/sha256）")
        seen.add(key)
    recorded = provenance.get("digest")
    if not isinstance(recorded, str) or not _SHA256_RE.match(recorded):
        issues.append("source_provenance.digest 缺失或不是 64 位十六进制")
    elif manifest_digest(provenance) != recorded:
        issues.append("source_provenance.digest 与清单内容不一致（被改写/损坏）")
    for i, entry in enumerate(files):
        location = entry.get("location")
        if location == "data_dir":
            path = os.path.join(data_dir, *str(entry["path"]).split("/"))
        elif location == "external":
            if not check_external:
                continue
            path = entry.get("external_path")
            if not isinstance(path, str) or not path:
                continue
        else:
            continue
        if not os.path.isfile(path):
            continue
        try:
            actual_size = int(os.path.getsize(path))
            actual_sha = sha256_file(path)
        except OSError as e:
            issues.append(f"files[{i}] 无法读取核对: {e}")
            continue
        if actual_size != entry.get("size"):
            issues.append(f"files[{i}] 大小与 manifest 不一致: "
                          f"{actual_size} != {entry.get('size')}")
        if actual_sha != entry.get("sha256"):
            issues.append(f"files[{i}] SHA-256 与 manifest 不一致: "
                          f"{str(entry.get('path'))}")
    return issues


def verify_source_manifest_file(data_dir: str, provenance: dict) -> list:
    """核对独立的 source_manifest.json 与 report 内 provenance 是否同版。"""
    path = os.path.join(data_dir, SOURCE_MANIFEST_NAME)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{SOURCE_MANIFEST_NAME} 无法解析: {e}"]
    if not isinstance(manifest, dict):
        return [f"{SOURCE_MANIFEST_NAME} 顶层不是对象"]
    if manifest.get("digest") != provenance.get("digest"):
        return [f"{SOURCE_MANIFEST_NAME} 的 digest 与 data_report 内的 "
                "source_provenance 不一致（两份来源证据不同版）"]
    return []


def provenance_text_hashes(provenance) -> dict:
    """从 manifest 取清洗后 split 文本哈希：{'train': sha, ...}。"""
    out = {}
    if not isinstance(provenance, dict):
        return out
    for entry in provenance.get("files") or []:
        if not isinstance(entry, dict) or entry.get("role") != "text_split":
            continue
        name = os.path.basename(str(entry.get("path") or ""))
        if name in ("train.txt", "val.txt", "test.txt"):
            sha = entry.get("sha256")
            if isinstance(sha, str):
                out[name[:-4]] = sha
    return out
