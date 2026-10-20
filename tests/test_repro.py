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
                        load_provenance, runtime_info, sanity_identity_check,
                        sha256_file, worktree_state, write_worktree_patch)
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


def test_data_fingerprint_survives_malformed_cache():
    """缓存文件语法合法但结构畸形（非对象/files 非列表/元素非对象）时
    必须视为未命中并重建，而不是让指纹计算崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(bytes(32))
        cache = os.path.join(d, ".lega_fingerprint.json")
        ref = data_fingerprint(d)
        for bad in ([1, 2], "oops", 5, {"algo": "sha256-16", "files": None},
                    {"algo": "sha256-16", "files": [1, "x"]}):
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(bad, f)
            assert data_fingerprint(d)["digest"] == ref["digest"]
    print("  [ok] 畸形指纹缓存被视为未命中并重建")


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
    assert rep["data_order_seed"] == 2024
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


def test_load_provenance_falls_back_for_partial_or_malformed_manifest():
    """清单缺字段/结构畸形时逐字段补算，不能把 None 写进下游溯源。"""
    with tempfile.TemporaryDirectory() as d:
        cfg = os.path.join(d, "config_used.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"seed": 5}, f)
        mpath = os.path.join(d, "run_manifest.json")

        def _manifest(obj):
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(obj, f)

        base = load_provenance(d)          # 无清单：按当前环境/配置补算
        assert base["config_sha256"] == sha256_file(cfg)
        # 语法合法但只是空壳/缺字段：缺失项补算，而不是返回一整排 None
        _manifest({"tag": "legacy"})
        p1 = load_provenance(d)
        assert p1["config_sha256"] == sha256_file(cfg)
        assert p1["code_commit"] == base["code_commit"]
        # 已有字段优先于当前环境，数据摘要缺失时保持 None 而不是乱填
        _manifest({"git": {"commit": "abc", "branch": "main"},
                   "config_sha256": "deadbeef"})
        p2 = load_provenance(d)
        assert p2["code_commit"] == "abc" and p2["code_branch"] == "main"
        assert p2["config_sha256"] == "deadbeef"
        assert p2["data_fingerprint"] is None
        # git/data 不是对象（结构畸形）：不崩，回退到当前环境
        _manifest({"git": "oops", "data": [1, 2], "config_sha256": "x"})
        p3 = load_provenance(d)
        assert p3["config_sha256"] == "x"
        assert p3["code_commit"] == base["code_commit"]
    print("  [ok] load_provenance：缺字段/畸形清单逐字段回退补算")


def test_load_provenance_rejects_stale_sanity_evidence():
    """下游评测不得把身份不一致的旧 sanity PASS 继承成自己的闸门结论。"""
    with tempfile.TemporaryDirectory() as d:
        mpath = os.path.join(d, "run_manifest.json")

        def _write(sanity):
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump({"sanity": sanity}, f)
            return load_provenance(d)

        stale = _write({
            "present": True, "passed": True, "variant": "lega",
            "identity": {"hard_mismatches": ["config_sha256"],
                         "unknown_fields": [], "verified": False}})
        assert stale["sanity_passed"] is None
        assert stale["sanity_verified"] is False
        # 身份核对通过：PASS 与 verified 都继承
        fresh = _write({
            "present": True, "passed": True, "variant": "lega",
            "identity": {"hard_mismatches": [], "unknown_fields": [],
                         "verified": True}})
        assert fresh["sanity_passed"] is True
        assert fresh["sanity_verified"] is True
        # 旧清单/无法核对：保留 PASS 但不冒充已验证
        legacy = _write({"present": True, "passed": True, "variant": "lega"})
        assert legacy["sanity_passed"] is True
        assert legacy["sanity_verified"] is None
        # 结构畸形的 identity 也不能让闸门结论崩溃或冒充已验证
        malformed = _write({"present": True, "passed": True,
                            "identity": "oops"})
        assert malformed["sanity_passed"] is True
        assert malformed["sanity_verified"] is None
    print("  [ok] load_provenance：旧 sanity PASS 的身份核对口径与 train 一致")


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


def _init_git_repo(d):
    def git(*args):
        r = subprocess.run(["git"] + list(args), cwd=d, capture_output=True,
                           text=True)
        assert r.returncode == 0, (args, r.stderr)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    return git


def test_worktree_state_and_patch_capture_dirty_code():
    """dirty 工作区必须留下 diff 指纹与可重放补丁，而不是只有 dirty=True。"""
    with tempfile.TemporaryDirectory() as d:
        git = _init_git_repo(d)
        src = os.path.join(d, "code.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        git("add", "code.py")
        git("commit", "-q", "-m", "init")
        clean = worktree_state(d)
        assert clean["available"] and clean["dirty"] is False
        assert clean["diff_sha256"] is None and clean["diff_bytes"] == 0
        # 修改 tracked 文件 + 新增 untracked：两者都要可见
        with open(src, "a", encoding="utf-8") as f:
            f.write("y = 2\n")
        with open(os.path.join(d, "new.py"), "w", encoding="utf-8") as f:
            f.write("z = 3\n")
        wt = worktree_state(d)
        assert wt["dirty"] is True
        assert "code.py" in wt["changed_files"]
        assert "new.py" in wt["untracked_files"]
        assert len(wt["diff_sha256"]) == 64 and wt["diff_bytes"] > 0
        patch_path = os.path.join(d, "worktree.patch")
        patch = write_worktree_patch(patch_path, root=d)
        assert patch["patch_sha256"] == wt["diff_sha256"]  # 同一字节口径
        with open(patch_path, "rb") as f:
            raw = f.read()
        assert raw and sha256_file(patch_path) == patch["patch_sha256"]
        # 未跟踪的新代码文件也必须进入补丁，否则 commit+tracked diff 仍无法重放
        assert b"new.py" in raw and b"new file mode" in raw
        # 再做一次修改：指纹变化（不能拿旧补丁冒充当前状态）
        with open(src, "a", encoding="utf-8") as f:
            f.write("w = 4\n")
        assert worktree_state(d)["diff_sha256"] != wt["diff_sha256"]
    print("  [ok] worktree_state/write_worktree_patch：diff 指纹与补丁可重放")


def test_worktree_state_degrades_outside_git():
    with tempfile.TemporaryDirectory() as d:
        st = worktree_state(d)
        assert st["available"] is False and st["dirty"] is None
        assert write_worktree_patch(
            os.path.join(d, "p.patch"), root=d)["patch_file"] is None
    print("  [ok] 非 git 目录：工作区快照显式不可用，不阻塞实验")


def test_sanity_identity_check_detects_stale_gate_evidence():
    """旧配置/旧数据/旧代码的 sanity PASS 不得绑定到当前 run。"""
    run = {"variant": "lega", "config_sha256": "c" * 64, "seed": 5,
           "git": {"commit": "a" * 40, "dirty": False},
           "data": {"digest": "d" * 16},
           "worktree": {"available": True, "diff_sha256": None}}
    fresh = {"variant": "lega", "config_sha256": "c" * 64, "seed": 5,
             "git_commit": "a" * 40, "data_fingerprint": "d" * 16,
             "worktree_recorded": True, "worktree_diff_sha256": None}
    ok = sanity_identity_check(fresh, run)
    assert ok["hard_mismatches"] == [] and ok["verified"] is True
    assert ok["unknown_fields"] == []
    # seed 不同是设计行为（run_all 只用首种子跑 sanity）：不算硬不一致
    deeper = dict(fresh, seed=9)
    assert sanity_identity_check(deeper, run)["hard_mismatches"] == []
    assert sanity_identity_check(deeper, run)["seed_mismatch"] is True
    for field, bad in (("config_sha256", "e" * 64),
                       ("data_fingerprint", "f" * 16),
                       ("git_commit", "b" * 40),
                       ("variant", "standard")):
        stale = dict(fresh, **{field: bad})
        res = sanity_identity_check(stale, run)
        assert field in res["hard_mismatches"], (field, res)
        assert res["verified"] is False
    # 工作区补丁不同（同 commit 但代码不同）同样判硬不一致
    dirty_run = dict(run, worktree={"available": True, "diff_sha256": "1" * 64})
    dirty_ok = dict(fresh, worktree_diff_sha256="1" * 64)
    assert sanity_identity_check(dirty_ok, dirty_run)["verified"] is True
    assert ("worktree_diff_sha256"
            in sanity_identity_check(fresh, dirty_run)["hard_mismatches"])
    # 旧版 sanity 缺身份字段：记为 unknown，不冒充已验证
    old = {"variant": "lega", "seed": 5}
    res_old = sanity_identity_check(old, run)
    assert res_old["verified"] is False and res_old["hard_mismatches"] == []
    assert "config_sha256" in res_old["unknown_fields"]
    # 非 git 目录：两侧工作区都不可用 → unknown，而不是把无法核对当失败
    nogit_run = dict(run, worktree={"available": False})
    nogit_sanity = dict(fresh, worktree_available=False)
    res_nogit = sanity_identity_check(nogit_sanity, nogit_run)
    assert res_nogit["hard_mismatches"] == []
    assert "worktree_diff_sha256" in res_nogit["unknown_fields"]
    print("  [ok] sanity_identity_check：旧配置/数据/代码/工作区不一致被拦下")


def test_data_fingerprint_tolerates_non_object_meta():
    """meta.json 是数组/标量时视为无 tokenizer，而不是崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(bytes(32))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        fp = data_fingerprint(d)
        assert fp["tokenizer"] is None and fp["digest"]
    print("  [ok] 数据指纹：非对象 meta.json 不崩溃")


def test_seed_report_and_runtime_record_warn_only_mode_honestly():
    """确定性算法只记 True 会夸大保证：必须如实标注 warn_only 而非严格模式。"""
    from lega.utils import deterministic_algorithms_warn_only
    set_seed(31)
    rep = seed_report(31)
    mode = rep["deterministic_algorithms_warn_only"]
    assert mode in (True, False), rep
    assert mode == deterministic_algorithms_warn_only()
    # 本项目为兼容个别无确定实现的算子显式采用 warn_only；若未来改严格模式
    # 该断言会失败，提醒同步更新文档与审计口径。
    assert mode is True, "set_seed 当前应运行在 warn_only 模式且如实记录"
    rt = runtime_info("cpu")
    assert rt["deterministic_algorithms_warn_only"] == mode
    json.dumps(rep) and json.dumps(rt)
    print("  [ok] seed_report/runtime 如实记录确定性算法 warn_only 模式")


def test_runtime_info_records_precision_threads_and_device_visibility():
    """影响数值复现的精度开关/线程数/CUDA 可见性必须进入环境快照。"""
    rt = runtime_info("cpu")
    for key in ("tf32_matmul", "tf32_cudnn", "cudnn_enabled",
                "torch_threads", "torch_interop_threads",
                "cuda_visible_devices"):
        assert key in rt, key
    assert isinstance(rt["tf32_matmul"], bool)
    assert isinstance(rt["tf32_cudnn"], bool)
    assert rt["torch_threads"] >= 1 and rt["torch_interop_threads"] >= 1
    assert rt["cuda_visible_devices"] in (None, "") or \
        isinstance(rt["cuda_visible_devices"], str)
    json.dumps(rt)
    # 精度开关记录的是“当前实际状态”，而不是硬编码默认值：
    # enable_tf32 之后 runtime_info 必须如实反映
    from lega.utils import enable_tf32
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        enable_tf32("cuda")
        rt2 = runtime_info("cuda")
        assert rt2["tf32_matmul"] is True and rt2["tf32_cudnn"] is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    print("  [ok] runtime_info 记录 TF32/线程数/CUDA_VISIBLE_DEVICES")


def test_sanity_summary_binds_gate_evidence_to_manifest():
    """sanity 结果摘要与 manifest 绑定：缺失/畸形不得冒充“通过”。"""
    from lega.repro import sanity_summary
    with tempfile.TemporaryDirectory() as d:
        assert sanity_summary(os.path.join(d, "missing.json"))["present"] is False
        bad = os.path.join(d, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("[1, 2]")
        assert sanity_summary(bad)["present"] is True
        assert sanity_summary(bad)["passed"] is None
        good = os.path.join(d, "sanity_lega.json")
        with open(good, "w", encoding="utf-8") as f:
            json.dump({"passed": True, "variant": "lega", "seed": 3,
                       "checks": {"a": True, "b": False},
                       "initial": {"loss": 4.1}, "final": {"loss": 0.01,
                                                          "token_acc": 0.99},
                       "manifest": {
                           "config_sha256": "c" * 64,
                           "git": {"commit": "a" * 40, "dirty": False},
                           "data": {"digest": "d" * 16},
                           "worktree": {"available": True,
                                        "diff_sha256": None}}}, f)
        s = sanity_summary(good)
        assert s["passed"] is True and s["variant"] == "lega" and s["seed"] == 3
        assert s["failed_checks"] == ["b"]
        assert s["final_loss"] == 0.01 and s["token_acc"] == 0.99
        assert len(s["sha256"]) == 64
        # 身份字段提取：train 据此核对闸门是否属于本 run
        assert s["config_sha256"] == "c" * 64
        assert s["data_fingerprint"] == "d" * 16
        assert s["git_commit"] == "a" * 40 and s["git_dirty"] is False
        assert s["worktree_recorded"] is True
        assert s["worktree_available"] is True
        assert s["worktree_diff_sha256"] is None
        # 清单内嵌 + 下游 provenance 继承
        cfg = os.path.join(d, "cfg.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"seed": 3}, f)
        data = os.path.join(d, "data")
        os.makedirs(data)
        with open(os.path.join(data, "train.bin"), "wb") as f:
            f.write(bytes(8))
        m = build_run_manifest(tag="t", variant="lega", seed=3,
                               config_path=cfg, data_dir=data, device="cpu",
                               sanity=s)
        assert m["sanity"]["sha256"] == s["sha256"]
        with open(os.path.join(d, "run_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(m, f)
        prov = load_provenance(d)
        assert prov["sanity_passed"] is True
        assert prov["sanity_variant"] == "lega"
    print("  [ok] sanity 摘要绑定 manifest，load_provenance 继承 sanity_passed")


def test_train_rejects_stale_sanity_evidence_identity():
    """端到端：sanity 文件声明的 config 与本 run 不一致时不得冒充 PASS。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        rng = np.random.default_rng(0)
        rng.integers(0, 64, size=1024, dtype=np.uint16).tofile(
            os.path.join(data_dir, "train.bin"))
        rng.integers(0, 64, size=128, dtype=np.uint16).tofile(
            os.path.join(data_dir, "val.bin"))
        with open(os.path.join(data_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "vocab_size": 64}, f)
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_mult": 4,
                       "block_size": 32, "batch_size": 2, "steps": 1,
                       "lr": 1e-3, "warmup": 1, "weight_decay": 0.0,
                       "grad_clip": 1.0, "train_ctx": 32, "amp": False,
                       "seed": 5, "log_every": 1, "eval_every": 1,
                       "eval_iters": 1, "dropout": 0.0, "attn_dropout": 0.0}, f)
        sanity_path = os.path.join(d, "sanity_lega.json")
        with open(sanity_path, "w", encoding="utf-8") as f:
            json.dump({"passed": True, "variant": "lega", "seed": 5,
                       "checks": {}, "initial": {}, "final": {},
                       "manifest": {
                           "config_sha256": "f" * 64,       # 旧配置
                           "git": {"commit": "a" * 40, "dirty": False},
                           "data": {"digest": "d" * 16},
                           "worktree": {"available": True,
                                        "diff_sha256": None}}}, f)
        out_dir = os.path.join(d, "out")
        r = subprocess.run(
            [sys.executable, "-m", "lega.train", "--config", cfg_path,
             "--variant", "lega", "--data_dir", data_dir,
             "--out_dir", out_dir, "--device", "cpu", "--seed", "5",
             "--sanity_file", sanity_path],
            cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(os.path.join(out_dir, "run_manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        ident = manifest["sanity"]["identity"]
        assert "config_sha256" in ident["hard_mismatches"]
        assert ident["verified"] is False
        with open(os.path.join(out_dir, "eval.json"), encoding="utf-8") as f:
            ev = json.load(f)
        assert ev["sanity_passed"] is None and ev["sanity_verified"] is False
    print("  [ok] 旧配置的 sanity PASS 被身份校验拒绝绑定到本 run")


def test_train_manifest_binds_sanity_file_and_actual_tf32_state():
    """端到端：train --sanity_file 把闸门证据与训练时精度开关写进 manifest。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        rng = np.random.default_rng(0)
        rng.integers(0, 64, size=2048, dtype=np.uint16).tofile(
            os.path.join(data_dir, "train.bin"))
        rng.integers(0, 64, size=256, dtype=np.uint16).tofile(
            os.path.join(data_dir, "val.bin"))
        with open(os.path.join(data_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "vocab_size": 64}, f)
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_mult": 4,
                       "block_size": 32, "batch_size": 2, "steps": 1,
                       "lr": 1e-3, "warmup": 1, "weight_decay": 0.0,
                       "grad_clip": 1.0, "train_ctx": 32, "amp": False,
                       "seed": 5, "log_every": 1, "eval_every": 1,
                       "eval_iters": 1, "dropout": 0.0, "attn_dropout": 0.0}, f)
        sanity_path = os.path.join(d, "sanity_lega.json")
        with open(sanity_path, "w", encoding="utf-8") as f:
            json.dump({"passed": True, "variant": "lega", "seed": 5,
                       "checks": {}, "initial": {}, "final": {}}, f)
        out_dir = os.path.join(d, "out")
        r = subprocess.run(
            [sys.executable, "-m", "lega.train", "--config", cfg_path,
             "--variant", "lega", "--data_dir", data_dir,
             "--out_dir", out_dir, "--device", "cpu", "--seed", "5",
             "--sanity_file", sanity_path],
            cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(os.path.join(out_dir, "run_manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["sanity"]["present"] is True
        assert manifest["sanity"]["passed"] is True
        assert manifest["sanity"]["variant"] == "lega"
        # 精度开关必须如实记录（顺序错误会记成训练前默认值）
        assert isinstance(manifest["runtime"]["tf32_matmul"], bool)
        assert isinstance(manifest["runtime"]["tf32_cudnn"], bool)
        with open(os.path.join(out_dir, "eval.json"), encoding="utf-8") as f:
            ev = json.load(f)
        assert ev["sanity_passed"] is True
    print("  [ok] train manifest：sanity 证据 + 精度开关已记录")
