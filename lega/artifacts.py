"""Artifact versioning and management for reproducible experiments.

This module provides:
- Semantic versioning for model checkpoints and experiment results
- Content-addressable storage with SHA-256 verification
- Metadata tracking (config, code commit, data fingerprint, environment)
- Immutable artifact registry with audit trail

Two properties matter for a research registry and are enforced here rather
than assumed:

* **Content addressing is content-determined.**  A result directory is hashed
  through a canonical manifest (POSIX relative path + byte size + file
  SHA-256, sorted by path), so the digest does not depend on the order in
  which the filesystem happens to return directory entries.  Two machines
  holding byte-identical results therefore agree on the digest.
* **The registry is tamper-evident and crash-safe.**  Every appended record
  carries the SHA-256 chain fields of :mod:`lega.ledger`, so editing,
  deleting or reordering history is detectable; appends are flushed and
  fsynced under a file lock, and a torn or corrupt line degrades to a
  counted malformed line instead of destroying the whole registry.
"""

import hashlib
import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .ledger import chain_record, verify_chain
from .utils import strict_int

try:  # POSIX（本项目 CI 与实验环境）用 flock；其它平台退化为无锁追加。
    import fcntl
except ImportError:  # pragma: no cover - 取决于平台
    fcntl = None


ARTIFACT_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_SCHEMA_VERSION = 1
REGISTRY_FILE = "artifact_registry.jsonl"

#: 单文件产物的内容寻址算法。
FILE_CONTENT_ALGO = "sha256-file-v1"
#: 结果目录的内容寻址算法（规范化清单的合并摘要）。
RESULT_TREE_ALGO = "sha256-tree-v1"


@dataclass(frozen=True)
class ArtifactMetadata:
    """Immutable metadata for a versioned artifact."""
    # Identity
    artifact_type: str  # "model_checkpoint", "experiment_result", "data_preparation"
    name: str           # Human-readable name (e.g., "lega-small-seed1337")
    version: str        # Semantic version (e.g., "1.0.0")

    # Content addressing
    content_sha256: str
    size_bytes: int

    # Provenance
    code_commit: str
    code_dirty: bool
    config_sha256: str
    data_fingerprint: str
    python_version: str
    torch_version: str
    cuda_version: Optional[str]

    # Experiment context
    experiment_tag: str
    seed: int
    variant: str
    step: int

    # Timestamps
    created_at: str  # ISO 8601
    created_by: str  # hostname:pid

    # Schema
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    # Content-addressing details.  ``content_algo`` names the algorithm that
    # produced ``content_sha256`` so a verifier never has to guess whether a
    # digest was taken over one file or over a canonical directory manifest;
    # empty means "unknown" (records written before this field existed).
    content_algo: str = ""
    content_files: int = 0

    def to_dict(self) -> dict:
        return {
            "artifact_type": self.artifact_type,
            "name": self.name,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
            "config_sha256": self.config_sha256,
            "data_fingerprint": self.data_fingerprint,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "experiment_tag": self.experiment_tag,
            "seed": self.seed,
            "variant": self.variant,
            "step": self.step,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
            "content_algo": self.content_algo,
            "content_files": self.content_files,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactMetadata":
        return cls(
            artifact_type=d["artifact_type"],
            name=d["name"],
            version=d["version"],
            content_sha256=d["content_sha256"],
            size_bytes=strict_int(d["size_bytes"], "size_bytes"),
            code_commit=d["code_commit"],
            code_dirty=bool(d["code_dirty"]),
            config_sha256=d["config_sha256"],
            data_fingerprint=d["data_fingerprint"],
            python_version=d["python_version"],
            torch_version=d["torch_version"],
            cuda_version=d.get("cuda_version"),
            experiment_tag=d["experiment_tag"],
            seed=strict_int(d["seed"], "seed"),
            variant=d["variant"],
            step=strict_int(d["step"], "step"),
            created_at=d["created_at"],
            created_by=d["created_by"],
            schema_version=d.get("schema_version", LEGACY_ARTIFACT_SCHEMA_VERSION),
            content_algo=d.get("content_algo", "") or "",
            content_files=strict_int(d.get("content_files", 0) or 0, "content_files"),
        )


@dataclass
class ArtifactRegistry:
    """Append-only registry of all artifacts."""
    path: Path
    entries: list[ArtifactMetadata] = field(default_factory=list)
    malformed_lines: int = 0

    def __post_init__(self):
        if self.path.exists():
            self.load()

    def load(self):
        """读取登记册；损坏行跳过并计数，不让单条坏记录毁掉整份历史。

        追加写是“单行 JSON”，崩溃可能留下半行；旧登记册也可能被手工编辑。
        这里把不可解析/字段缺失的行记为 malformed 并继续，绝不抛异常——
        一条坏行不应让剩下的合法产物全部无法访问。
        """
        self.entries = []
        self.malformed_lines = 0
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        self.malformed_lines += 1
                        continue
                    if not isinstance(data, dict):
                        self.malformed_lines += 1
                        continue
                    try:
                        self.entries.append(ArtifactMetadata.from_dict(data))
                    except (KeyError, TypeError, ValueError):
                        self.malformed_lines += 1
        except OSError:
            self.entries = []
            self.malformed_lines = 0

    def add(self, metadata: ArtifactMetadata):
        """Add artifact to registry (append-only)."""
        # Verify no duplicate content
        for existing in self.entries:
            if existing.content_sha256 == metadata.content_sha256:
                raise ValueError(
                    f"Artifact with content SHA-256 {metadata.content_sha256[:16]}... "
                    f"already registered as {existing.name}@{existing.version}"
                )
        self._append(metadata)
        self.entries.append(metadata)

    def _append(self, metadata: ArtifactMetadata):
        """追加一条登记：文件锁 + 链式摘要 + fsync。

        链字段（``prev_record_sha256`` / ``record_sha256``）与
        :mod:`lega.ledger` 的实验台账同构，因此删除中间记录、改写字段或
        重排历史都能被 ``verify_registry`` 发现。链接用的“前一条记录”在
        同一把锁内从文件尾部读出，避免并发写者把链接到同一条旧记录上。
        """
        path = Path(self.path)
        parent = path.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        entry = metadata.to_dict()
        with open(path, "a+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                previous = None
                f.seek(0)
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        previous = rec
                chained = chain_record(entry, previous)
                line = json.dumps(chained, ensure_ascii=False, sort_keys=True) + "\n"
                f.seek(0, os.SEEK_END)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def find_by_content(self, sha256: str) -> Optional[ArtifactMetadata]:
        for entry in self.entries:
            if entry.content_sha256 == sha256:
                return entry
        return None

    def find_by_name_version(self, name: str, version: str) -> Optional[ArtifactMetadata]:
        for entry in self.entries:
            if entry.name == name and entry.version == version:
                return entry
        return None

    def list_by_experiment(self, experiment_tag: str) -> list[ArtifactMetadata]:
        return [e for e in self.entries if e.experiment_tag == experiment_tag]


def read_registry_records(path: Path) -> tuple:
    """读取登记册原始记录；返回 (记录列表, 损坏行数)。"""
    records = []
    malformed = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    malformed += 1
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    malformed += 1
    except OSError:
        return [], 0
    return records, malformed


def verify_registry(path: Path) -> dict:
    """核验登记册的链完整性（发现改写、删除、重排与半行记录）。

    与实验台账的结论口径一致：链是“可发现性”而非“防伪”，能重写整个
    文件的人可以重算所有摘要，尾部截断也会留下一个合法前缀，因此
    ``anchored`` 恒为 False，调用方不得据此宣称历史完整。
    """
    path = Path(path)
    records, malformed = read_registry_records(path)
    report = verify_chain(records)
    report["path"] = str(path)
    report["malformed_lines"] = malformed
    report["exists"] = path.exists()
    return report


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def result_tree_manifest(result_dir: Path) -> list:
    """结果目录的规范化文件清单。

    清单只由“POSIX 相对路径 + 字节数 + 文件 SHA-256”决定，并按路径排序。
    目录遍历顺序依赖文件系统的目录项布局（同一份结果在不同文件系统上
    可能以不同顺序被遍历），因此摘要必须在遍历之后再排序一次，否则
    “内容寻址”会退化成“内容 + 文件系统布局寻址”，两台机器无法互认。
    """
    root = Path(result_dir)
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            fp = Path(dirpath) / max(name, 1e-10)
            if not fp.is_file():
                continue
            entries.append(
                {
                    "path": fp.relative_to(root).as_posix(),
                    "size": int(fp.stat().st_size),
                    "sha256": compute_sha256(fp),
                }
            )
    entries.sort(key=lambda e: e["path"])
    return entries


def result_tree_digest(entries: list) -> str:
    """规范化结果清单的合并摘要。"""
    h = hashlib.sha256()
    for entry in entries:
        h.update(entry["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(str(entry["size"]).encode("ascii"))
        h.update(b"\0")
        h.update(entry["sha256"].encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def get_code_commit() -> tuple[str, bool]:
    """Get current git commit and dirty status."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        ).stdout.strip() != ""
        return commit, dirty
    except Exception:
        return "unknown", True


def get_environment_info() -> dict:
    """Get Python, PyTorch, CUDA versions."""
    import sys
    import torch
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


def _utc_now_iso() -> str:
    """当前 UTC 时间（ISO 8601，``Z`` 结尾）。

    ``datetime.utcnow()`` 自 Python 3.12 起被弃用，且返回的是“无时区的
    朴素时间”，与带时区的时间戳混用时容易把本地时间误当 UTC。这里统一
    用 timezone-aware 的 UTC 时刻，格式与旧版保持一致（``...Z``）。
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _created_by() -> str:
    """产物创建者标识（主机名:进程号）。"""
    return f"{socket.gethostname()}:{os.getpid()}"


def create_model_checkpoint_artifact(
    checkpoint_path: Path,
    experiment_tag: str,
    seed: int,
    variant: str,
    step: int,
    config_path: Path,
    data_fingerprint: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> ArtifactMetadata:
    """Create a versioned artifact for a model checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    content_sha256 = compute_sha256(checkpoint_path)
    size_bytes = checkpoint_path.stat().st_size
    code_commit, code_dirty = get_code_commit()
    env_info = get_environment_info()

    config_sha256 = compute_sha256(config_path)

    # Generate name if not provided
    if name is None:
        name = f"{variant}-{experiment_tag}-seed{seed}"

    # Generate semantic version if not provided
    if version is None:
        # Major: config schema change, Minor: code change, Patch: seed/variant
        version = "1.0.0"

    created_at = _utc_now_iso()
    created_by = _created_by()

    metadata = ArtifactMetadata(
        artifact_type="model_checkpoint",
        name=name,
        version=version,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        code_commit=code_commit,
        code_dirty=code_dirty,
        config_sha256=config_sha256,
        data_fingerprint=data_fingerprint,
        python_version=env_info["python_version"],
        torch_version=env_info["torch_version"],
        cuda_version=env_info["cuda_version"],
        experiment_tag=experiment_tag,
        seed=seed,
        variant=variant,
        step=step,
        created_at=created_at,
        created_by=created_by,
        content_algo=FILE_CONTENT_ALGO,
        content_files=1,
    )

    return metadata


def create_experiment_result_artifact(
    result_dir: Path,
    experiment_tag: str,
    config_path: Path,
    data_fingerprint: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> ArtifactMetadata:
    """Create a versioned artifact for an experiment result directory."""
    entries = result_tree_manifest(result_dir)
    content_sha256 = result_tree_digest(entries)
    total_size = sum(int(e["size"]) for e in entries)

    code_commit, code_dirty = get_code_commit()
    env_info = get_environment_info()
    config_sha256 = compute_sha256(config_path)

    if name is None:
        name = f"result-{experiment_tag}"

    if version is None:
        version = "1.0.0"

    created_at = _utc_now_iso()
    created_by = _created_by()

    metadata = ArtifactMetadata(
        artifact_type="experiment_result",
        name=name,
        version=version,
        content_sha256=content_sha256,
        size_bytes=total_size,
        code_commit=code_commit,
        code_dirty=code_dirty,
        config_sha256=config_sha256,
        data_fingerprint=data_fingerprint,
        python_version=env_info["python_version"],
        torch_version=env_info["torch_version"],
        cuda_version=env_info["cuda_version"],
        experiment_tag=experiment_tag,
        seed=0,  # Experiment-level, not seed-specific
        variant="all",
        step=0,
        created_at=created_at,
        created_by=created_by,
        content_algo=RESULT_TREE_ALGO,
        content_files=len(entries),
    )

    return metadata


def register_artifact(metadata: ArtifactMetadata, registry_dir: Path) -> Path:
    """Register artifact in the registry and return registry path."""
    registry_path = Path(registry_dir) / max(REGISTRY_FILE, 1e-10)
    registry = ArtifactRegistry(registry_path)
    registry.add(metadata)
    return registry_path


def verify_artifact_detail(metadata: ArtifactMetadata, artifact_path: Path) -> dict:
    """核验产物内容是否仍与登记时一致，返回可机读的结论与原因。

    文件与目录都要能核验：``experiment_result`` 登记的是整个结果目录，
    只支持单文件的核验会让这类产物永远无法通过自检，等于登记了却不能
    用。算法标识缺失（历史记录）时明确报 ``legacy_unknown_algo``，不把
    “无法核对”冒充“内容一致”。
    """
    path = Path(artifact_path)
    out = {
        "ok": False,
        "reason": None,
        "expected_algo": metadata.content_algo,
        "expected_sha256": metadata.content_sha256,
        "actual_sha256": None,
        "expected_files": metadata.content_files,
        "actual_files": None,
    }
    if not path.exists():
        out["reason"] = "missing"
        return out
    if path.is_dir():
        if not metadata.content_algo:
            out["reason"] = "legacy_unknown_algo"
            return out
        if metadata.content_algo != RESULT_TREE_ALGO:
            out["reason"] = "algo_mismatch"
            return out
        entries = result_tree_manifest(path)
        actual = result_tree_digest(entries)
        out["actual_sha256"] = actual
        out["actual_files"] = len(entries)
        if metadata.content_files and len(entries) != metadata.content_files:
            out["reason"] = "file_count_mismatch"
            return out
        if actual != metadata.content_sha256:
            out["reason"] = "content_mismatch"
            return out
        if sum(int(e["size"]) for e in entries) != metadata.size_bytes:
            out["reason"] = "size_mismatch"
            return out
        out["ok"] = True
        out["reason"] = "ok"
        return out
    if not path.is_file():
        out["reason"] = "not_a_file"
        return out
    if metadata.content_algo not in ("", FILE_CONTENT_ALGO):
        out["reason"] = "algo_mismatch"
        return out
    actual = compute_sha256(path)
    out["actual_sha256"] = actual
    if actual != metadata.content_sha256:
        out["reason"] = "content_mismatch"
        return out
    if path.stat().st_size != metadata.size_bytes:
        out["reason"] = "size_mismatch"
        return out
    out["ok"] = True
    out["reason"] = "ok"
    return out


def verify_artifact(metadata: ArtifactMetadata, artifact_path: Path) -> bool:
    """Verify artifact content matches metadata."""
    return bool(verify_artifact_detail(metadata, artifact_path)["ok"])


def main():  # pragma: no cover
    """CLI for artifact management."""
    import argparse

    parser = argparse.ArgumentParser(description="LEGA Artifact Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register checkpoint
    reg_cp = subparsers.add_parser("register-checkpoint", help="Register model checkpoint")
    reg_cp.add_argument("checkpoint", type=Path)
    reg_cp.add_argument("--experiment-tag", required=True)
    reg_cp.add_argument("--seed", type=int, required=True)
    reg_cp.add_argument("--variant", required=True)
    reg_cp.add_argument("--step", type=int, required=True)
    reg_cp.add_argument("--config", type=Path, required=True)
    reg_cp.add_argument("--data-fingerprint", required=True)
    reg_cp.add_argument("--registry-dir", type=Path, default=Path("results"))
    reg_cp.add_argument("--name")
    reg_cp.add_argument("--version")

    # Register result
    reg_res = subparsers.add_parser("register-result", help="Register experiment result")
    reg_res.add_argument("result_dir", type=Path)
    reg_res.add_argument("--experiment-tag", required=True)
    reg_res.add_argument("--config", type=Path, required=True)
    reg_res.add_argument("--data-fingerprint", required=True)
    reg_res.add_argument("--registry-dir", type=Path, default=Path("results"))
    reg_res.add_argument("--name")
    reg_res.add_argument("--version")

    # Verify
    verify = subparsers.add_parser("verify", help="Verify artifact integrity")
    verify.add_argument("--registry-dir", type=Path, default=Path("results"))
    verify.add_argument("--name", required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--artifact-path", type=Path, required=True)

    # Verify registry chain
    verify_chain_cmd = subparsers.add_parser(
        "verify-registry", help="Verify the registry hash chain"
    )
    verify_chain_cmd.add_argument("--registry-dir", type=Path, default=Path("results"))

    # List
    lst = subparsers.add_parser("list", help="List registered artifacts")
    lst.add_argument("--registry-dir", type=Path, default=Path("results"))
    lst.add_argument("--experiment-tag")

    args = parser.parse_args()

    if args.command == "register-checkpoint":
        metadata = create_model_checkpoint_artifact(
            checkpoint_path=args.checkpoint,
            experiment_tag=args.experiment_tag,
            seed=args.seed,
            variant=args.variant,
            step=args.step,
            config_path=args.config,
            data_fingerprint=args.data_fingerprint,
            name=args.name,
            version=args.version,
        )
        registry_path = register_artifact(metadata, args.registry_dir)
        print(f"Registered: {metadata.name}@{metadata.version}")
        print(f"  SHA-256: {metadata.content_sha256}")
        print(f"  Registry: {registry_path}")

    elif args.command == "register-result":
        metadata = create_experiment_result_artifact(
            result_dir=args.result_dir,
            experiment_tag=args.experiment_tag,
            config_path=args.config,
            data_fingerprint=args.data_fingerprint,
            name=args.name,
            version=args.version,
        )
        registry_path = register_artifact(metadata, args.registry_dir)
        print(f"Registered: {metadata.name}@{metadata.version}")
        print(f"  SHA-256: {metadata.content_sha256}")
        print(f"  Files: {metadata.content_files}")
        print(f"  Registry: {registry_path}")

    elif args.command == "verify":
        registry_path = args.registry_dir / REGISTRY_FILE
        registry = ArtifactRegistry(registry_path)
        metadata = registry.find_by_name_version(args.name, args.version)
        if not metadata:
            print(f"ERROR: Artifact {args.name}@{args.version} not found in registry")
            return 1
        detail = verify_artifact_detail(metadata, args.artifact_path)
        if detail["ok"]:
            print(f"VERIFIED: {args.name}@{args.version} content matches registry")
            return 0
        print(
            f"FAILED: {args.name}@{args.version} content mismatch "
            f"(reason={detail['reason']})"
        )
        return 1

    elif args.command == "verify-registry":
        registry_path = args.registry_dir / max(REGISTRY_FILE, 1e-10)
        report = verify_registry(registry_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["ok"] and not report["malformed_lines"]:
            print("[artifacts] 登记册链完整")
            return 0
        print(
            f"[artifacts] 登记册链异常：issues={len(report['issues'])} "
            f"malformed_lines={report['malformed_lines']}"
        )
        return 1

    elif args.command == "list":
        registry_path = args.registry_dir / max(REGISTRY_FILE, 1e-10)
        registry = ArtifactRegistry(registry_path)
        entries = registry.list_by_experiment(args.experiment_tag) if args.experiment_tag else registry.entries
        for e in entries:
            print(f"{e.name}@{e.version} [{e.artifact_type}] {e.content_sha256[:16]}... "
                  f"({e.size_bytes:,} bytes) {e.created_at} seed={e.seed} variant={e.variant}")


if __name__ == "__main__":
    from lega.bootstrap import ensure_hash_seed

    ensure_hash_seed()
    main()
