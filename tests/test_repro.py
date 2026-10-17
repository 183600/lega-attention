"""可复现性与溯源测试：种子覆盖、数据批采样确定、文件/数据指纹稳定。"""
import json
import os
import random
import subprocess
import sys
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
    # 影响产物的可选依赖显式记录（未安装为 None，不假装已记录环境）
    assert "tiktoken" in rt["packages"] and "matplotlib" in rt["packages"]
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


def test_seed_report_rng_fingerprints_are_seed_evidence():
    """种子生效的证据而非声称：指纹来自 RNG 真实状态，可被第三方复算。

    同 seed 的 set_seed 必然得到相同指纹；不同 seed 得到不同指纹；当前状态
    属于别的 seed 时，请求旧 seed 的证据必须为 None（不能张冠李戴）。
    """
    set_seed(11)
    fp_a = seed_report(11)["rng_fingerprints"]
    assert fp_a and {"python_random", "numpy_global", "torch_cpu"} <= set(fp_a)
    assert all(isinstance(v, str) and len(v) == 16 for v in fp_a.values())
    set_seed(11)
    assert seed_report(11)["rng_fingerprints"] == fp_a     # 同 seed 可复算
    set_seed(12)
    fp_b = seed_report(12)["rng_fingerprints"]
    for key in ("python_random", "numpy_global", "torch_cpu"):
        assert fp_b[key] != fp_a[key], key               # 异 seed 状态不同
    assert seed_report(11)["rng_fingerprints"] is None   # 不冒充旧 seed
    json.dumps(fp_a)
    print("  [ok] seed_report 内嵌 RNG 状态指纹（同 seed 可复算/异 seed 不同/"
          "不冒充）")


def test_seed_report_distinguishes_process_hashseed_from_child_env():
    """PYTHONHASHSEED 审计不得自证：当前进程与子进程要分开记录。

    set_seed 会写 os.environ（只影响之后启动的子进程）；若审计直接读当前
    env，字段会永远等于 seed，“当前进程哈希随机化其实没固定”会被掩盖。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import json\n"
            "from lega.utils import set_seed, seed_report\n"
            "set_seed(7)\n"
            "print(json.dumps(seed_report(7)))\n")
    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep_unset = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep_unset["pythonhashseed_at_start"] is None
    assert rep_unset["pythonhashseed_env"] == "7"
    assert rep_unset["pythonhashseed_effective"] is False

    env["PYTHONHASHSEED"] = "7"
    r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep_set = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep_set["pythonhashseed_at_start"] == "7"
    assert rep_set["pythonhashseed_effective"] is True
    print("  [ok] PYTHONHASHSEED 审计区分解释器启动值/子进程 env（不自证）")


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


def test_same_seed_training_is_bitwise_reproducible():
    """同一台机器上，同 seed 的完整训练必须逐位可复现（可复现性门槛）。

    两次独立子进程运行 `lega.train`，覆盖模型初始化（含门控参数）、
    数据批采样 rng、优化器/反向、checkpoint 序列化整条链路；训练曲线
    （去掉时间戳）与权重必须逐位一致。跨硬件架构不保证一致属预期
    （见 set_seed 文档），但本测试固定在本机运行，加回该保证。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        rng = np.random.default_rng(0)
        rng.integers(0, 64, size=4096, dtype=np.uint16).tofile(
            os.path.join(data_dir, "train.bin"))
        rng.integers(0, 64, size=512, dtype=np.uint16).tofile(
            os.path.join(data_dir, "val.bin"))
        with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "vocab_size": 64}, f)
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_mult": 4,
                       "block_size": 32, "batch_size": 4, "steps": 2,
                       "lr": 1e-3, "warmup": 1, "weight_decay": 0.0,
                       "grad_clip": 1.0, "train_ctx": 32, "amp": False,
                       "seed": 5, "log_every": 1, "eval_every": 2,
                       "eval_iters": 1, "dropout": 0.0, "attn_dropout": 0.0}, f)
        outs = []
        for name in ("run_a", "run_b"):
            out_dir = os.path.join(d, name)
            r = subprocess.run(
                [sys.executable, "-m", "lega.train", "--config", cfg_path,
                 "--variant", "lega", "--data_dir", data_dir,
                 "--out_dir", out_dir, "--device", "cpu", "--seed", "5"],
                cwd=root, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            outs.append(out_dir)

        def curves(p):
            with open(os.path.join(p, "train_log.jsonl"), encoding="utf-8") as f:
                recs = [json.loads(ln) for ln in f if ln.strip()]
            for rec in recs:
                rec.pop("time", None)      # 时间戳不参与数值比对
                # 步时是墙钟测量，天然不可逐位复现（与 time 同口径剔除）；
                # 模型权重与损失数值仍逐位比对
                rec.pop("step_time_ms", None)
            return recs

        assert curves(outs[0]) == curves(outs[1])
        w_a = torch.load(os.path.join(outs[0], "model.pt"), map_location="cpu",
                         weights_only=True)["state_dict"]
        w_b = torch.load(os.path.join(outs[1], "model.pt"), map_location="cpu",
                         weights_only=True)["state_dict"]
        assert w_a.keys() == w_b.keys()
        diff = [k for k in w_a if not torch.equal(w_a[k], w_b[k])]
        assert diff == [], diff
    print("  [ok] 同 seed 两次完整训练：曲线与权重逐位一致")
