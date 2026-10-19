"""实验可追溯性（reproducibility & provenance）。

回答“这次结果对应哪份代码、哪份数据、什么环境”：
  * git_info        —— 代码版本（commit/branch/工作区是否 dirty）；
  * sha256_file     —— 任意产物文件的 SHA-256；
  * data_fingerprint—— 数据目录内容指纹（train/val/test bin + meta，含缓存）；
  * runtime_info    —— Python/PyTorch/NumPy/CUDA 版本与确定性开关；
  * load_provenance —— 下游评测脚本从 run_manifest.json 继承同一份溯源信息。

设计约定：所有字段必须是 JSON 原生类型（str/bool/int/float/None），
以便直接写入结果 JSON 或 torch checkpoint（weights_only=True 反序列化）。
"""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np
import torch

from .utils import deterministic_algorithms_warn_only

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINGERPRINT_ALGO = "sha256-16"
_CACHE_NAME = ".lega_fingerprint.json"


def _git(args, cwd):
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                           text=True, timeout=15)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info(root: str = None) -> dict:
    """代码版本；非 git 仓库或 git 不可用时字段为 None，绝不阻塞实验。"""
    root = root or ROOT
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return {"commit": None, "commit_short": None, "branch": None,
                "dirty": None}
    status = _git(["status", "--porcelain"], root)
    return {
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        # dirty=True 表示结果对应“未提交的工作区状态”，仅凭 commit 无法重建
        "dirty": bool(status) if status is not None else None,
    }


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _file_entry(path: str) -> dict:
    st = os.stat(path)
    return {"name": os.path.basename(path), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "sha256": sha256_file(path)}


def _digest_of(entries) -> str:
    canon = "|".join(f"{e['name']}:{e['size']}:{e['sha256']}"
                     for e in sorted(entries, key=lambda e: e["name"]))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def data_fingerprint(data_dir: str, use_cache: bool = True) -> dict:
    """数据目录指纹：对实际参与训练/评测的文件做 SHA-256 并给合并摘要。

    缓存写于 <data_dir>/.lega_fingerprint.json：逐文件记录 size/mtime_ns/sha256，
    任一文件大小或 mtime 变化即整体重算。缓存仅供重复调用复用（多种子/多脚本），
    不影响正确性——缓存命中要求文件集与 stat 完全一致。
    """
    names = [n for n in ("train.bin", "val.bin", "test.bin", "meta.json")
             if os.path.exists(os.path.join(data_dir, n))]
    cache_path = os.path.join(data_dir, _CACHE_NAME)
    cached = None
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, ValueError):
            cached = None
    # 结构校验：缓存是纯派生数据，语法合法但结构畸形（非对象、files 非列表、
    # 元素非对象）时必须视为未命中并重建，而不是抛 AttributeError/TypeError。
    if isinstance(cached, dict) and cached.get("algo") == FINGERPRINT_ALGO \
            and isinstance(cached.get("digest"), str) \
            and isinstance(cached.get("files"), list) \
            and all(isinstance(e, dict) for e in cached["files"]) \
            and [e.get("name") for e in cached["files"]] == names:
        fresh = True
        for e in cached["files"]:
            path = os.path.join(data_dir, e["name"])
            try:
                st = os.stat(path)
            except OSError:
                fresh = False
                break
            if int(st.st_size) != e.get("size") or int(st.st_mtime_ns) != e.get("mtime_ns"):
                fresh = False
                break
        if fresh:
            return cached
    if not names:
        raise FileNotFoundError(f"{data_dir} 下没有可指纹化的数据文件（train.bin 等）")
    entries = [_file_entry(os.path.join(data_dir, n)) for n in names]
    tokenizer = None
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # 语法合法但结构畸形（数组/标量）的 meta 视为无 tokenizer，
            # 与缓存结构校验同口径，不让指纹计算崩溃。
            tokenizer = (meta.get("tokenizer")
                         if isinstance(meta, dict) else None)
        except (OSError, ValueError):
            tokenizer = None
    fp = {"algo": FINGERPRINT_ALGO, "digest": _digest_of(entries),
          "tokenizer": tokenizer, "files": entries}
    if use_cache:
        try:
            with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(fp, f, ensure_ascii=False, indent=2)
            os.replace(cache_path + ".tmp", cache_path)
        except OSError:
            pass
    return fp


def _pkg_version(name: str):
    """已安装包版本；未安装（可选依赖）时返回 None。"""
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def runtime_info(device: str = None) -> dict:
    """运行环境版本与确定性/精度开关（跨机器不可复现时用于排查差异来源）。"""
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
                  else None),
        "device": device,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        # 影响产物的可选依赖版本（分词器变更会改变 token 流，绘图库影响图表），
        # 未安装时显式为 None，避免“环境已记录”的错觉
        "packages": {name: _pkg_version(name)
                     for name in ("tiktoken", "matplotlib")},
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        # True=仅告警而非严格模式；只记 deterministic_algorithms 会夸大数据一致性
        "deterministic_algorithms_warn_only":
            deterministic_algorithms_warn_only(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        # cudnn conv 默认 allow_tf32=True：不记录会把“训练实际用 TF32”漏掉
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        # 线程数影响归约顺序与浮点结果；CVD 决定实际可见的加速器
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # 机器级上下文：同机重跑是否稳定、跨机差异是否来自环境，都靠这些字段判断
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                        if os.environ.get(k) is not None},
    }
    return info


def sanity_summary(path: str) -> dict:
    """sanity 结果 JSON 的紧凑摘要，供训练 run 的 manifest 内嵌代码正确性闸门。

    每个 checkpoint 旁都应能回答“这个模型开训前是否通过了小批量过拟合检查、
    检查了哪些项、当时结果文件是什么”。文件缺失/损坏时不阻塞训练，但摘要
    显式 present=False（不假装通过）。
    """
    out = {"file": os.path.abspath(path) if path else None, "present": False,
           "sha256": None, "passed": None, "variant": None, "seed": None,
           "failed_checks": None, "initial_loss": None, "final_loss": None,
           "token_acc": None}
    if not path or not os.path.exists(path):
        return out
    out["present"] = True
    try:
        out["sha256"] = sha256_file(path)
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return out
    if not isinstance(rec, dict):
        return out
    checks = rec.get("checks")
    if isinstance(checks, dict):
        out["failed_checks"] = sorted(k for k, v in checks.items() if not v)
    for key in ("passed", "variant", "seed"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    initial = rec.get("initial")
    final = rec.get("final")
    if isinstance(initial, dict):
        out["initial_loss"] = initial.get("loss")
    if isinstance(final, dict):
        out["final_loss"] = final.get("loss")
        out["token_acc"] = final.get("token_acc")
    return out


def build_run_manifest(tag: str, variant: str, seed: int, config_path: str,
                       data_dir: str, device: str, argv=None,
                       sanity: dict = None) -> dict:
    """单个 (变体, 种子) run 的完整溯源清单。

    sanity（可选）为 lega.sanity 结果的紧凑摘要（见 sanity_summary）：把
    “开训前通过小批量过拟合闸门”的代码正确性证据绑定到本 run 的 checkpoint。
    """
    from .utils import now_str, seed_report
    config_path = os.path.abspath(config_path)
    data_dir = os.path.abspath(data_dir)
    return {
        "tag": tag,
        "variant": variant,
        "seed": int(seed),
        "seed_report": seed_report(seed),
        "time": now_str(),
        "config_file": config_path,
        "config_sha256": sha256_file(config_path),
        "git": git_info(),
        "data": data_fingerprint(data_dir),
        "runtime": runtime_info(device),
        "sanity": sanity,
        "argv": [str(a) for a in (argv if argv is not None else sys.argv)],
        "cwd": os.getcwd(),
    }


def load_provenance(ckpt_dir: str, data_dir: str = None) -> dict:
    """下游评测统一继承上游 run 的溯源字段；缺失时按当前环境/数据即时补算。

    返回平铺字段便于直接 update 进结果 JSON：
      code_commit / code_dirty / code_branch / data_fingerprint / config_sha256
    """
    manifest = None
    mpath = os.path.join(ckpt_dir, "run_manifest.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = None
    # 结构校验：语法合法但结构畸形（非对象，或 git/data 不是对象）的
    # run_manifest 必须视为“无清单”并按当前环境补算，而不是抛
    # AttributeError 阻塞下游评测（与 data_fingerprint 的缓存校验同口径）。
    # 先按当前环境/配置算出兜底值，再让清单逐字段覆盖：旧版清单可能缺少
    # config_sha256 等字段（或整个 git/data 不是对象），缺失字段必须补算
    # 而不是留 None 污染 eval.json 的溯源与受控变量审计。
    cfg = os.path.join(ckpt_dir, "config_used.json")
    git = git_info()
    out = {
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "code_branch": git.get("branch"),
        "data_fingerprint": None,
        "config_sha256": (sha256_file(cfg) if os.path.exists(cfg) else None),
        # 训练 run 是否通过了小批量过拟合 sanity 闸门（无清单/旧清单为 None）
        "sanity_passed": None,
        "sanity_variant": None,
        "sanity_file": None,
    }
    if data_dir:
        try:
            out["data_fingerprint"] = data_fingerprint(data_dir)["digest"]
        except FileNotFoundError:
            pass
    if isinstance(manifest, dict) and manifest:
        mgit = manifest.get("git")
        mdata = manifest.get("data")
        mgit = mgit if isinstance(mgit, dict) else {}
        mdata = mdata if isinstance(mdata, dict) else {}
        for key, src, field in (
                ("code_commit", mgit, "commit"),
                ("code_dirty", mgit, "dirty"),
                ("code_branch", mgit, "branch"),
                ("data_fingerprint", mdata, "digest"),
                ("config_sha256", manifest, "config_sha256")):
            val = src.get(field)
            if val is not None:
                out[key] = val
        msanity = manifest.get("sanity")
        if isinstance(msanity, dict):
            if msanity.get("passed") is not None:
                out["sanity_passed"] = bool(msanity["passed"])
            if msanity.get("variant") is not None:
                out["sanity_variant"] = msanity["variant"]
            if msanity.get("file") is not None:
                out["sanity_file"] = msanity["file"]
    return out
