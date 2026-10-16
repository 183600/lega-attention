"""可复现性与溯源测试：种子覆盖、数据批采样确定、文件/数据指纹稳定。"""
import json
import os
import random
import tempfile

import numpy as np
import torch

from lega.data import get_train_batch
from lega.repro import (build_run_manifest, data_fingerprint, git_info,
                        runtime_info, sha256_file)
from lega.utils import set_seed, seed_report


def test_set_seed_covers_python_numpy_torch():
    set_seed(1234)
    a = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    set_seed(1234)
    b = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    assert a == b
    # 数据顺序同样由 seed 唯一确定（本项目“dataloader”= 显式 rng 的批采样）
    data = np.arange(4096, dtype=np.uint16)
    x1, y1 = get_train_batch(data, 16, 4, "cpu", np.random.default_rng(7))
    x2, y2 = get_train_batch(data, 16, 4, "cpu", np.random.default_rng(7))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])  # y 为 x 右移一位
    print("  [ok] set_seed 覆盖 random/np/torch，批采样顺序由 seed 唯一确定")


def test_sha256_file_deterministic_and_sensitive():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.bin")
        with open(p, "wb") as f:
            f.write(b"lega-attention")
        h1, h2 = sha256_file(p), sha256_file(p)
        assert h1 == h2 and len(h1) == 64
        with open(p, "wb") as f:
            f.write(b"lega-attention!")
        assert sha256_file(p) != h1
    print("  [ok] sha256_file 稳定且对内容变化敏感")


def test_data_fingerprint_cache_and_invalidation():
    with tempfile.TemporaryDirectory() as d:
        for name, n in (("train.bin", 64), ("val.bin", 16)):
            with open(os.path.join(d, name), "wb") as f:
                f.write(bytes(range(n)))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "vocab_size": 8}, f)
        fp1 = data_fingerprint(d)
        assert sorted(e["name"] for e in fp1["files"]) == \
            ["meta.json", "train.bin", "val.bin"]
        assert data_fingerprint(d)["digest"] == fp1["digest"]  # 缓存命中
        with open(os.path.join(d, "train.bin"), "ab") as f:
            f.write(b"\xff")                                   # 大小/mtime 变化
        fp2 = data_fingerprint(d)
        assert fp2["digest"] != fp1["digest"]
    print("  [ok] 数据指纹缓存命中/内容变化失效")


def test_git_and_runtime_info_json_safe():
    git = git_info()
    assert set(git) == {"commit", "commit_short", "branch", "dirty"}
    if git["commit"] is not None:  # 非 git 环境（如源码包）容忍为 None
        assert len(git["commit"]) == 40
    rt = runtime_info("cpu")
    assert rt["torch"] and rt["python"] and rt["numpy"]
    assert rt["hostname"] and rt["python_executable"] and rt["cpu_count"]
    json.dumps({"git": git, "runtime": rt})  # 必须是 JSON 原生类型
    print(f"  [ok] git/runtime 信息可序列化 (commit={git['commit_short']})")


def test_seed_report_covers_all_random_sources():
    """种子审计：每个随机源以传入 seed 记录在案，且信息可 JSON 序列化。"""
    set_seed(2024)
    rep = seed_report(2024)
    assert rep["seed"] == 2024
    for key in ("python_random", "numpy_global", "torch_cpu"):
        assert rep[key] == 2024, (key, rep)
    assert rep["torch_cuda"] in (None, 2024)
    assert rep["cudnn_deterministic"] is True
    assert rep["cudnn_benchmark"] is False
    assert rep["deterministic_algorithms"] is True
    assert rep["data_order"]
    json.dumps(rep)
    print("  [ok] seed_report 覆盖 Python/NumPy/Torch(CPU/CUDA) 与确定性开关")


def test_run_manifest_embeds_seed_report_and_runtime():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(bytes(8))
        cfg = os.path.join(d, "cfg.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"seed": 5}, f)
        manifest = build_run_manifest(tag="t", variant="lega", seed=5,
                                      config_path=cfg, data_dir=d, device="cpu")
        assert manifest["seed_report"]["python_random"] == 5
        assert manifest["runtime"]["hostname"]
        json.dumps(manifest)
    print("  [ok] run_manifest 内嵌 seed_report 与机器级 runtime 信息")
