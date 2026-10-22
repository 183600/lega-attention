"""可复现性与溯源测试：种子覆盖、数据批采样确定、文件/数据指纹稳定。"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

from lega.data import CACHE_SCHEMA, get_train_batch
from lega.repro import (build_run_manifest, data_fingerprint, git_info,
                        load_provenance, provenance_drift, runtime_info,
                        sanity_identity_check, sha256_file,
                        test_data_fingerprint as _test_data_fingerprint,
                        worktree_state, write_worktree_patch)
from lega.utils import (data_rng, data_rng_state_fingerprint, set_seed,
                        seed_report)


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
    # 数据顺序指纹：同一 seed 必须逐字节一致；不同 seed 必须不同。
    h1, h2, h3 = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for hasher, seed in ((h1, 7), (h2, 7), (h3, 8)):
        rng = np.random.default_rng(seed)
        for _ in range(3):
            get_train_batch(data, 16, 4, "cpu", rng, order_hasher=hasher)
    assert h1.hexdigest() == h2.hexdigest()
    assert h1.hexdigest() != h3.hexdigest()
    print("  [ok] set_seed 覆盖 random/np/torch，批采样顺序与顺序指纹由 seed 唯一确定")


def test_set_seed_overwrites_stale_child_hashseed():
    """子进程环境必须对齐本次 seed，而不是保留进程启动时的旧值。

    PYTHONHASHSEED 不能改变当前解释器，但设置 os.environ 后启动的子进程会
    继承它。若用 setdefault，环境里已有的 PYTHONHASHSEED=0 会让 set_seed(4321)
    的子进程仍以 0 启动，与 run_manifest 声称的 seed 不一致。
    """
    old = os.environ.get("PYTHONHASHSEED")
    try:
        os.environ["PYTHONHASHSEED"] = "0"
        set_seed(4321)
        assert os.environ["PYTHONHASHSEED"] == "4321"
        r = subprocess.run(
            [sys.executable, "-c",
             "import os; print(os.environ['PYTHONHASHSEED'])"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=os.environ.copy(), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "4321", r.stdout
    finally:
        if old is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old
    print("  [ok] set_seed 覆盖旧 PYTHONHASHSEED，子进程继承本次 seed")


def test_set_seed_and_seed_report_reject_fractional_and_bool_seeds():
    """种子身份不能被 int() 截断：True 当 1、1.5 当 1 都会让审计说谎。

    set_seed(True) 会静默播种到 1 且报告 "seed": 1；set_seed(1.5) 会在
    np.random.seed 处以 TypeError 半途失败（PYTHONHASHSEED 已被写成 "1"）。
    两者都必须在入口就显式拒绝。
    """
    for bad in (True, False, 1.5, -0.5, -1, 2 ** 32, float("nan"),
                float("inf"), "1.5", "abc", "", [1], {"seed": 1}, None):
        for fn in (set_seed, seed_report):
            try:
                fn(bad)
                raised = False
            except ValueError:
                raised = True
            assert raised, (fn.__name__, bad)
    from lega.utils import MAX_SEED, seed_int
    assert seed_int(0) == 0 and seed_int(MAX_SEED) == MAX_SEED
    for bad in (-1, MAX_SEED + 1):
        try:
            seed_int(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    # 整数值浮点/整数字符串语义明确，规范化成同一个 int（不再半途崩溃）
    set_seed(1337.0)
    rep = seed_report("1337")
    assert rep["seed"] == 1337 and rep["seed_verified"] is True
    assert seed_report(1337)["rng_fingerprints"] == rep["rng_fingerprints"]
    # 环境里的 PYTHONHASHSEED 不可解析为整数时如实标 None，不截断比较
    assert seed_report(1337)["pythonhashseed_at_start"] in (None, "1337")
    print("  [ok] set_seed/seed_report 拒绝 bool/小数/非整数字符串种子")


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


def test_data_fingerprint_detects_mtime_preserving_edit():
    """同尺寸替换内容并把 mtime 改回旧值，也必须触发重算。

    只看 size+mtime 的缓存会把这种替换误判为“数据未变”，让 run manifest
    用旧 SHA 冒充当前数据版本；ctime/inode 元数据变化是不可回填的证据。
    """
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "train.bin")
        with open(p, "wb") as f:
            f.write(b"aaaa")
        fp1 = data_fingerprint(d)
        st = os.stat(p)
        with open(p, "wb") as f:
            f.write(b"bbbb")                       # 同尺寸、不同内容
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        fp2 = data_fingerprint(d)
        assert fp2["digest"] != fp1["digest"]
        assert fp2["files"][0]["sha256"] != fp1["files"][0]["sha256"]
    print("  [ok] 数据指纹识别 mtime 回填的同尺寸内容替换")


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


def test_data_fingerprint_rejects_tampered_cache_metadata():
    """缓存文件 stat 未变但摘要/分词器字段被改坏时，不能直接信任缓存。

    缓存是派生数据；若只核对文件大小/mtime/ctime，损坏的 digest 或
    tokenizer 会被原样返回并写入 run_manifest，让溯源信息自相矛盾。
    """
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(bytes(32))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char"}, f)
        ref = data_fingerprint(d)
        cache = os.path.join(d, ".lega_fingerprint.json")
        with open(cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["digest"] = "0" * len(ref["digest"])
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        assert data_fingerprint(d)["digest"] == ref["digest"]
        with open(cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["tokenizer"] = "bogus"
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        assert data_fingerprint(d)["tokenizer"] == "char"
    print("  [ok] 指纹缓存摘要/分词器被篡改时重建")


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
    # 数据顺序 RNG 不仅是"声明用了 seed"：初始状态指纹可由 seed 独立复算，
    # 且与全局 NumPy RNG 相互独立（全局状态被模型初始化消耗也不影响）。
    assert rep["data_order_rng_fingerprint"] == \
        data_rng_state_fingerprint(2024)
    json.dumps(rep)
    print("  [ok] seed_report 覆盖 Python/NumPy/Torch(CPU/CUDA)、数据顺序 RNG "
          "与确定性开关")


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
    stale = seed_report(11)
    assert stale["rng_fingerprints"] is None   # 不冒充旧 seed
    # 逐随机源的 seed 也不能复述请求参数：当前 RNG 属于 seed 12，
    # 请求 seed 11 的报告必须把各源标为未验证/未知，而不是伪造 11。
    assert stale["seed_verified"] is False
    for key in ("python_random", "numpy_global", "torch_cpu",
                "data_order_seed", "data_order_rng_fingerprint"):
        assert stale[key] is None, (key, stale)
    json.dumps(fp_a)
    print("  [ok] seed_report 内嵌 RNG 状态指纹（同 seed 可复算/异 seed 不同/"
          "不冒充）")


def test_data_rng_is_seeded_and_reproducible():
    """数据顺序 RNG：同 seed 初始状态/采样逐位一致，异 seed 不同。

    这是“dataloader 种子已固定”的可执行证据：训练不再各自调用
    np.random.default_rng，而统一走 data_rng(seed)，其状态指纹可由第三方
    用同一 seed 复算。
    """
    data = np.arange(4096, dtype=np.uint16)
    for seed in (0, 7, 2 ** 32 - 1):
        r1, r2 = data_rng(seed), data_rng(seed)
        x1, y1 = get_train_batch(data, 16, 4, "cpu", r1)
        x2, y2 = get_train_batch(data, 16, 4, "cpu", r2)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert data_rng_state_fingerprint(seed) == \
            data_rng_state_fingerprint(seed)
    assert data_rng_state_fingerprint(7) != data_rng_state_fingerprint(8)
    # 非法种子不能被 np.random.default_rng 静默接受（bool/负数/小数）
    for bad in (True, -1, 1.5, 2 ** 32):
        try:
            data_rng(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] data_rng 同 seed 可复算、异 seed 不同且拒绝非法种子")


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


def test_set_seed_repairs_invalid_cublas_workspace_config():
    """继承的非法 CUBLAS 配置会让确定性 CUDA 算法启动失败，必须改写。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import os\n"
            "from lega.utils import set_seed\n"
            "set_seed(11)\n"
            "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])\n")
    for bad, want in ((":bad:", ":4096:8"), ("", ":4096:8"),
                      (":16:8", ":16:8")):
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = bad
        r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == want, (bad, r.stdout)
    print("  [ok] set_seed 修正非法 CUBLAS_WORKSPACE_CONFIG 且保留合法值")


def test_run_manifest_embeds_seed_report_and_runtime():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "train.bin"), "wb") as f:
            f.write(bytes(8))
        cfg = os.path.join(d, "cfg.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"seed": 5}, f)
        # manifest 只记录已实际固定到该 seed 的 RNG；调用方须先 set_seed
        set_seed(5)
        manifest = build_run_manifest(tag="t", variant="lega", seed=5,
                                      config_path=cfg, data_dir=d, device="cpu")
        assert manifest["seed_report"]["seed_verified"] is True
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
        assert base["seed_report"] is None
        assert base["training_precision"] is None
        # 语法合法但只是空壳/缺字段：缺失项补算，而不是返回一整排 None
        _manifest({"tag": "legacy"})
        p1 = load_provenance(d)
        assert p1["config_sha256"] == sha256_file(cfg)
        assert p1["code_commit"] == base["code_commit"]
        # 已有字段优先于当前环境，数据摘要缺失时保持 None 而不是乱填
        _manifest({"git": {"commit": "abc", "branch": "main"},
                   "config_sha256": "deadbeef",
                   "seed_report": {"seed": 5, "seed_verified": True},
                   "precision": {"amp_enabled": True,
                                 "amp_dtype": "bfloat16"}})
        p2 = load_provenance(d)
        assert p2["code_commit"] == "abc" and p2["code_branch"] == "main"
        assert p2["config_sha256"] == "deadbeef"
        assert p2["data_fingerprint"] is None
        assert p2["seed_report"]["seed_verified"] is True
        assert p2["training_precision"]["amp_dtype"] == "bfloat16"
        # git/data 不是对象（结构畸形）：不崩，回退到当前环境
        _manifest({"git": "oops", "data": [1, 2], "config_sha256": "x"})
        p3 = load_provenance(d)
        assert p3["config_sha256"] == "x"
        assert p3["code_commit"] == base["code_commit"]
    print("  [ok] load_provenance：缺字段/畸形清单逐字段回退补算")


def test_load_provenance_keeps_eval_and_training_data_fingerprints_apart():
    """用另一份数据评测 checkpoint 时，不能把清单里的训练指纹当评测指纹。"""
    with tempfile.TemporaryDirectory() as ckpt, \
            tempfile.TemporaryDirectory() as data:
        with open(os.path.join(data, "train.bin"), "wb") as f:
            f.write(bytes(32))
        eval_fp = data_fingerprint(data)["digest"]
        train_fp = "0" * 64
        mpath = os.path.join(ckpt, "run_manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump({"data": {"digest": train_fp}}, f)

        p = load_provenance(ckpt, data_dir=data)
        assert p["data_fingerprint"] == eval_fp
        assert p["eval_data_fingerprint"] == eval_fp
        assert p["train_data_fingerprint"] == train_fp
        assert p["data_fingerprint_matches_training"] is False

        # 同一份数据：显式确认一致。
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump({"data": {"digest": eval_fp}}, f)
        p = load_provenance(ckpt, data_dir=data)
        assert p["data_fingerprint"] == eval_fp
        assert p["data_fingerprint_matches_training"] is True

        # 未提供 data_dir 的只读审计仍回退到清单，但不声称已核对。
        p = load_provenance(ckpt)
        assert p["data_fingerprint"] == eval_fp
        assert p["eval_data_fingerprint"] is None
        assert p["data_fingerprint_matches_training"] is None
    print("  [ok] load_provenance 区分评测/训练数据指纹并标记是否一致")


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
        # 非布尔字符串/数字不能经 bool() 伪装成 PASS。
        fake = _write({"present": True, "passed": "false",
                       "identity": {"hard_mismatches": [],
                                    "verified": "false"}})
        assert fake["sanity_passed"] is None
        assert fake["sanity_verified"] is None
        fake2 = _write({"present": True, "passed": 1,
                        "identity": {"hard_mismatches": [], "verified": 1}})
        assert fake2["sanity_passed"] is None
        assert fake2["sanity_verified"] is None
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
            json.dump({"tokenizer": "char", "vocab_size": 64,
                       "cache_schema": CACHE_SCHEMA, "fit_on": "train"}, f)
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


def test_provenance_drift_flags_code_and_data_changes_without_false_positive():
    """结束快照必须发现 commit/工作区 diff/数据指纹变化，缺失时记 unknown。"""
    start = {
        "git": {"commit": "a" * 40},
        "worktree": {"available": True, "diff_sha256": "d1"},
        "data_fingerprint": "fp1",
    }
    same = {
        "git": {"commit": "a" * 40},
        "worktree": {"available": True, "diff_sha256": "d1"},
        "data_fingerprint": "fp1",
    }
    d0 = provenance_drift(start, same)
    assert d0["changed"] is False and d0["mismatches"] == []
    assert all(v is True for v in d0["checked"].values())
    moved = {
        "git": {"commit": "b" * 40},
        "worktree": {"available": True, "diff_sha256": "d2"},
        "data_fingerprint": "fp2",
    }
    d1 = provenance_drift(start, moved)
    assert d1["changed"] is True
    assert {m["field"] for m in d1["mismatches"]} == {
        "code_commit", "worktree_diff_sha256", "data_fingerprint"}
    # 非 git/工作区不可用/数据缺失：unknown 而不是“未变化”
    unknown = provenance_drift(
        {"git": {}, "worktree": {"available": False},
         "data_fingerprint": None},
        {"git": {}, "worktree": {"available": False},
         "data_fingerprint": None})
    assert unknown["changed"] is False
    assert set(unknown["unknown"]) == {
        "code_commit", "worktree_diff_sha256", "data_fingerprint"}
    # commit 不变但工作区改动同样必须被抓到
    dirty = dict(same, worktree={"available": True, "diff_sha256": "d9"})
    assert provenance_drift(start, dirty)["changed"] is True
    print("  [ok] 结束身份核对：commit/工作区/数据变化可发现，缺失记 unknown")


def test_provenance_drift_tolerates_malformed_snapshots():
    """畸形快照必须记为 unknown，而不是 AttributeError 中断收尾。"""
    res = provenance_drift([], "bad")
    assert res["changed"] is False
    assert set(res["unknown"]) == {
        "code_commit", "worktree_diff_sha256", "data_fingerprint"}
    assert all(v is None for v in res["checked"].values())
    print("  [ok] 畸形溯源快照按 unknown 降级")


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


def test_data_fingerprint_empty_cache_cannot_bypass_missing_files():
    """目录里只有 files=[] 的缓存时仍须报“无数据文件”，不能返回空指纹。"""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ".lega_fingerprint.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"algo": "sha256-16",
                       "digest": "e3b0c44298fc1c14",  # sha256("")[:16]
                       "tokenizer": None, "files": []}, f)
        try:
            data_fingerprint(d)
            raise AssertionError("空目录 + 空缓存必须报 FileNotFoundError")
        except FileNotFoundError:
            pass
    print("  [ok] 数据指纹：空缓存不能绕过无数据文件报错")


def test_test_data_fingerprint_returns_none_when_unreadable():
    """test.bin 存在但不可读（如目录占位）时返回 None，而不是崩溃。"""
    with tempfile.TemporaryDirectory() as d:
        assert _test_data_fingerprint(d) is None
        os.mkdir(os.path.join(d, "test.bin"))
        assert _test_data_fingerprint(d) is None
    print("  [ok] test_data_fingerprint：缺失/不可读均返回 None")


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
            json.dump({"tokenizer": "char", "vocab_size": 64,
                       "cache_schema": CACHE_SCHEMA, "fit_on": "train"}, f)
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
            json.dump({"tokenizer": "char", "vocab_size": 64,
                       "cache_schema": CACHE_SCHEMA, "fit_on": "train"}, f)
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


def test_provenance_drift_truncated_worktree_is_unknown_not_unchanged():
    """工作区补丁被截断时，相同哈希不能证明“未变化”，必须记 unknown。"""
    start = {
        "git": {"commit": "a" * 40},
        "worktree": {"available": True, "diff_sha256": "d1",
                     "diff_truncated": True},
        "data_fingerprint": "fp1",
    }
    same_truncated = {
        "git": {"commit": "a" * 40},
        "worktree": {"available": True, "diff_sha256": "d1",
                     "diff_truncated": True},
        "data_fingerprint": "fp1",
    }
    d = provenance_drift(start, same_truncated)
    assert d["checked"]["worktree_diff_sha256"] is None, d
    assert "worktree_diff_sha256" in d["unknown"], d
    assert d["changed"] is False
    # 哈希已不同时仍能确认发生变化，不因截断而漏报。
    moved = dict(same_truncated, worktree={
        "available": True, "diff_sha256": "d2", "diff_truncated": True})
    assert provenance_drift(start, moved)["changed"] is True
    print("  [ok] 截断的工作区补丁按 unknown 处理，哈希不同仍可确认变化")
