"""Artifact versioning and management for reproducible experiments.

This module provides:
- Semantic versioning for model checkpoints and experiment results
- Content-addressable storage with SHA-256 verification
- Metadata tracking (config, code commit, data fingerprint, environment)
- Immutable artifact registry with audit trail
"""
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .utils import save_json, load_json, strict_int, config_finite_float


ARTIFACT_SCHEMA_VERSION = 1
REGISTRY_FILE = "artifact_registry.jsonl"


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
            schema_version=d.get("schema_version", ARTIFACT_SCHEMA_VERSION),
        )


@dataclass
class ArtifactRegistry:
    """Append-only registry of all artifacts."""
    path: Path
    entries: list[ArtifactMetadata] = field(default_factory=list)
    
    def __post_init__(self):
        if self.path.exists():
            self.load()
    
    def load(self):
        self.entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(ArtifactMetadata.from_dict(json.loads(line)))
    
    def add(self, metadata: ArtifactMetadata):
        """Add artifact to registry (append-only)."""
        # Verify no duplicate content
        for existing in self.entries:
            if existing.content_sha256 == metadata.content_sha256:
                raise ValueError(
                    f"Artifact with content SHA-256 {metadata.content_sha256[:16]}... "
                    f"already registered as {existing.name}@{existing.version}"
                )
        self.entries.append(metadata)
        self._append(metadata)
    
    def _append(self, metadata: ArtifactMetadata):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + "\n")
    
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


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
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
    
    created_at = datetime.utcnow().isoformat() + "Z"
    created_by = f"{os.uname().nodename}:{os.getpid()}"
    
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
    # Compute combined hash of all files in result directory
    hashes = []
    total_size = 0
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files):
            fp = Path(root) / f
            if fp.is_file():
                h = compute_sha256(fp)
                hashes.append(f"{fp.relative_to(result_dir)}:{h}")
                total_size += fp.stat().st_size
    
    # Combined content hash
    content_sha256 = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    
    code_commit, code_dirty = get_code_commit()
    env_info = get_environment_info()
    config_sha256 = compute_sha256(config_path)
    
    if name is None:
        name = f"result-{experiment_tag}"
    
    if version is None:
        version = "1.0.0"
    
    created_at = datetime.utcnow().isoformat() + "Z"
    created_by = f"{os.uname().nodename}:{os.getpid()}"
    
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
    )
    
    return metadata


def register_artifact(metadata: ArtifactMetadata, registry_dir: Path) -> Path:
    """Register artifact in the registry and return registry path."""
    registry_path = registry_dir / REGISTRY_FILE
    registry = ArtifactRegistry(registry_path)
    registry.add(metadata)
    return registry_path


def verify_artifact(metadata: ArtifactMetadata, artifact_path: Path) -> bool:
    """Verify artifact content matches metadata."""
    if not artifact_path.exists() or not artifact_path.is_file():
        return False
    actual_sha256 = compute_sha256(artifact_path)
    actual_size = artifact_path.stat().st_size
    return actual_sha256 == metadata.content_sha256 and actual_size == metadata.size_bytes


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
        print(f"  Registry: {registry_path}")
    
    elif args.command == "verify":
        registry_path = args.registry_dir / REGISTRY_FILE
        registry = ArtifactRegistry(registry_path)
        metadata = registry.find_by_name_version(args.name, args.version)
        if not metadata:
            print(f"ERROR: Artifact {args.name}@{args.version} not found in registry")
            return 1
        ok = verify_artifact(metadata, args.artifact_path)
        if ok:
            print(f"VERIFIED: {args.name}@{args.version} content matches registry")
        else:
            print(f"FAILED: {args.name}@{args.version} content mismatch")
            return 1
    
    elif args.command == "list":
        registry_path = args.registry_dir / REGISTRY_FILE
        registry = ArtifactRegistry(registry_path)
        entries = registry.list_by_experiment(args.experiment_tag) if args.experiment_tag else registry.entries
        for e in entries:
            print(f"{e.name}@{e.version} [{e.artifact_type}] {e.content_sha256[:16]}... "
                  f"({e.size_bytes:,} bytes) {e.created_at} seed={e.seed} variant={e.variant}")


if __name__ == "__main__":
    from lega.bootstrap import ensure_hash_seed

    ensure_hash_seed()
    main()