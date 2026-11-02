"""Tests for lega.artifacts module."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from lega.artifacts import (
    ArtifactMetadata,
    ArtifactRegistry,
    compute_sha256,
    create_model_checkpoint_artifact,
    create_experiment_result_artifact,
    register_artifact,
    verify_artifact,
    get_code_commit,
    get_environment_info,
)


class TestComputeSHA256:
    def test_compute_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        try:
            sha = compute_sha256(path)
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert sha == expected
        finally:
            path.unlink()

    def test_compute_sha256_large_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 2MB
            f.write(b"x" * (2 * 1024 * 1024))
            path = Path(f.name)
        try:
            sha = compute_sha256(path, chunk_size=1 << 15)
            expected = hashlib.sha256(b"x" * (2 * 1024 * 1024)).hexdigest()
            assert sha == expected
        finally:
            path.unlink()


class TestArtifactMetadata:
    def test_roundtrip(self):
        meta = ArtifactMetadata(
            artifact_type="model_checkpoint",
            name="test-model",
            version="1.0.0",
            content_sha256="a" * 64,
            size_bytes=1024,
            code_commit="abc123",
            code_dirty=False,
            config_sha256="b" * 64,
            data_fingerprint="c" * 64,
            python_version="3.12.0",
            torch_version="2.14.1",
            cuda_version="12.1",
            experiment_tag="test-exp",
            seed=1337,
            variant="lega",
            step=8000,
            created_at="2026-01-01T00:00:00Z",
            created_by="host:1234",
        )
        d = meta.to_dict()
        restored = ArtifactMetadata.from_dict(d)
        assert restored == meta

    def test_from_dict_requires_all_fields(self):
        d = {"artifact_type": "model_checkpoint"}
        with pytest.raises(KeyError):
            ArtifactMetadata.from_dict(d)


class TestArtifactRegistry:
    def test_add_and_find(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="a" * 64,
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            registry.add(meta)
            
            # Find by content
            found = registry.find_by_content("a" * 64)
            assert found is not None
            assert found.name == "test-model"
            
            # Find by name/version
            found = registry.find_by_name_version("test-model", "1.0.0")
            assert found is not None
            
            # List by experiment
            entries = registry.list_by_experiment("test-exp")
            assert len(entries) == 1

    def test_duplicate_content_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            meta1 = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="model-1",
                version="1.0.0",
                content_sha256="a" * 64,
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            meta2 = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="model-2",
                version="1.0.0",
                content_sha256="a" * 64,  # Same content!
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            registry.add(meta1)
            with pytest.raises(ValueError, match="already registered"):
                registry.add(meta2)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            
            # Create and add
            registry1 = ArtifactRegistry(registry_path)
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="a" * 64,
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            registry1.add(meta)
            
            # Reload
            registry2 = ArtifactRegistry(registry_path)
            assert len(registry2.entries) == 1
            assert registry2.entries[0].name == "test-model"


class TestVerifyArtifact:
    def test_verify_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "model.pt"
            content = b"model weights"
            artifact_path.write_bytes(content)
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            assert verify_artifact(meta, artifact_path) is True

    def test_verify_content_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "model.pt"
            artifact_path.write_bytes(b"model weights")
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="0" * 64,  # Wrong hash
                size_bytes=14,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            assert verify_artifact(meta, artifact_path) is False

    def test_verify_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "model.pt"
            artifact_path.write_bytes(b"model weights")
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256=hashlib.sha256(b"model weights").hexdigest(),
                size_bytes=999,  # Wrong size
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            assert verify_artifact(meta, artifact_path) is False

    def test_verify_missing_file(self):
        meta = ArtifactMetadata(
            artifact_type="model_checkpoint",
            name="test-model",
            version="1.0.0",
            content_sha256="a" * 64,
            size_bytes=1024,
            code_commit="abc123",
            code_dirty=False,
            config_sha256="b" * 64,
            data_fingerprint="c" * 64,
            python_version="3.12.0",
            torch_version="2.14.1",
            cuda_version=None,
            experiment_tag="test-exp",
            seed=1337,
            variant="lega",
            step=8000,
            created_at="2026-01-01T00:00:00Z",
            created_by="host:1234",
        )
        assert verify_artifact(meta, Path("/nonexistent/path")) is False


class TestGetCodeCommit:
    def test_get_code_commit(self):
        commit, dirty = get_code_commit()
        assert isinstance(commit, str)
        assert isinstance(dirty, bool)
        # In a git repo, commit should be a valid SHA
        if commit != "unknown":
            assert len(commit) == 40  # Full SHA-1
            assert all(c in "0123456789abcdef" for c in commit)


class TestGetEnvironmentInfo:
    def test_get_environment_info(self):
        info = get_environment_info()
        assert "python_version" in info
        assert "torch_version" in info
        assert "cuda_version" in info
        assert info["python_version"].startswith("3.")
        # torch_version should be parseable
        assert "." in info["torch_version"]


class TestCreateArtifacts:
    def test_create_model_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"d_model": 384}))
            
            meta = create_model_checkpoint_artifact(
                checkpoint_path=checkpoint_path,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                config_path=config_path,
                data_fingerprint="d" * 64,
            )
            
            assert meta.artifact_type == "model_checkpoint"
            assert meta.name == "lega-test-exp-seed1337"
            assert meta.version == "1.0.0"
            assert meta.seed == 1337
            assert meta.variant == "lega"
            assert meta.step == 8000
            assert meta.content_sha256 == hashlib.sha256(b"fake checkpoint").hexdigest()
            assert meta.size_bytes == len(b"fake checkpoint")

    def test_create_experiment_result_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "results"
            result_dir.mkdir()
            (result_dir / "summary.md").write_text("# Summary\n\nOK")
            (result_dir / "metrics.json").write_text('{"ppl": 10.5}')
            
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"d_model": 384}))
            
            meta = create_experiment_result_artifact(
                result_dir=result_dir,
                experiment_tag="test-exp",
                config_path=config_path,
                data_fingerprint="d" * 64,
            )
            
            assert meta.artifact_type == "experiment_result"
            assert meta.name == "result-test-exp"
            assert meta.version == "1.0.0"
            assert meta.experiment_tag == "test-exp"
            assert meta.seed == 0
            assert meta.variant == "all"
            assert meta.step == 0
            # Content hash should combine both files
            assert meta.content_sha256 is not None
            assert meta.size_bytes > 0


class TestArtifactRegistryExtended:
    def test_find_by_content_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            found = registry.find_by_content("b" * 64)
            assert found is None

    def test_find_by_name_version_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            found = registry.find_by_name_version("nonexistent", "1.0.0")
            assert found is None
            
            found = registry.find_by_name_version("test-model", "2.0.0")
            assert found is None

    def test_list_by_experiment_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            entries = registry.list_by_experiment("nonexistent-exp")
            assert len(entries) == 0

    def test_list_by_experiment_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            for i in range(3):
                meta = ArtifactMetadata(
                    artifact_type="model_checkpoint",
                    name=f"model-{i}",
                    version="1.0.0",
                    content_sha256=f"{i}" * 64,
                    size_bytes=1024,
                    code_commit="abc123",
                    code_dirty=False,
                    config_sha256="b" * 64,
                    data_fingerprint="c" * 64,
                    python_version="3.12.0",
                    torch_version="2.14.1",
                    cuda_version=None,
                    experiment_tag="test-exp",
                    seed=1337 + i,
                    variant="lega",
                    step=8000,
                    created_at="2026-01-01T00:00:00Z",
                    created_by="host:1234",
                )
                registry.add(meta)
            
            entries = registry.list_by_experiment("test-exp")
            assert len(entries) == 3
            
            # Different experiment tag
            entries2 = registry.list_by_experiment("other-exp")
            assert len(entries2) == 0


class TestVerifyArtifactExtended:
    def test_verify_missing_file_path(self):
        meta = ArtifactMetadata(
            artifact_type="model_checkpoint",
            name="test-model",
            version="1.0.0",
            content_sha256="a" * 64,
            size_bytes=1024,
            code_commit="abc123",
            code_dirty=False,
            config_sha256="b" * 64,
            data_fingerprint="c" * 64,
            python_version="3.12.0",
            torch_version="2.14.1",
            cuda_version=None,
            experiment_tag="test-exp",
            seed=1337,
            variant="lega",
            step=8000,
            created_at="2026-01-01T00:00:00Z",
            created_by="host:1234",
        )
        # Test with a non-existent path
        assert verify_artifact(meta, Path("/nonexistent/path/that/does/not/exist")) is False
        
        # Test with a directory (should return False, not crash)
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            assert verify_artifact(meta, dir_path) is False


class TestRegisterArtifactExtended:
    def test_register_artifact_overwrite_prevention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir)
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="a" * 64,
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            
            # Register first time
            registry_path1 = register_artifact(meta, registry_dir)
            assert registry_path1.exists()
            
            # Register again with different content but same name/version
            meta2 = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="b" * 64,  # Different content
                size_bytes=2048,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            registry_path2 = register_artifact(meta2, registry_dir)
            assert registry_path2.exists()
            
            # Both should be in the registry
            with open(registry_path2) as f:
                lines = f.readlines()
                assert len(lines) == 2


class TestCreateArtifactsExtended:
    def test_create_model_checkpoint_artifact_custom_name_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"d_model": 384}))
            
            meta = create_model_checkpoint_artifact(
                checkpoint_path=checkpoint_path,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                config_path=config_path,
                data_fingerprint="d" * 64,
                name="custom-name",
                version="2.0.0",
            )
            
            assert meta.name == "custom-name"
            assert meta.version == "2.0.0"

    def test_create_experiment_result_artifact_custom_name_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "results"
            result_dir.mkdir()
            (result_dir / "summary.md").write_text("# Summary\n\nOK")
            
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"d_model": 384}))
            
            meta = create_experiment_result_artifact(
                result_dir=result_dir,
                experiment_tag="test-exp",
                config_path=config_path,
                data_fingerprint="d" * 64,
                name="custom-result",
                version="3.0.0",
            )
            
            assert meta.name == "custom-result"
            assert meta.version == "3.0.0"


class TestArtifactRegistryInternals:
    def test_append_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.jsonl"
            registry = ArtifactRegistry(registry_path)
            
            meta = ArtifactMetadata(
                artifact_type="model_checkpoint",
                name="test-model",
                version="1.0.0",
                content_sha256="a" * 64,
                size_bytes=1024,
                code_commit="abc123",
                code_dirty=False,
                config_sha256="b" * 64,
                data_fingerprint="c" * 64,
                python_version="3.12.0",
                torch_version="2.14.1",
                cuda_version=None,
                experiment_tag="test-exp",
                seed=1337,
                variant="lega",
                step=8000,
                created_at="2026-01-01T00:00:00Z",
                created_by="host:1234",
            )
            # Test the private _append method directly
            registry._append(meta)
            
            # Verify it was written
            with open(registry_path) as f:
                line = f.readline()
                entry = json.loads(line)
                assert entry["name"] == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])