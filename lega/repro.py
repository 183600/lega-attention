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
    if cached and cached.get("algo") == FINGERPRINT_ALGO \
            and [e.get("name") for e in cached.get("files", [])] == names:
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
                tokenizer = json.load(f).get("tokenizer")
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
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
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


def build_run_manifest(tag: str, variant: str, seed: int, config_path: str,
                       data_dir: str, device: str, argv=None) -> dict:
    """单个 (变体, 种子) run 的完整溯源清单。"""
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
    if manifest:
        git = manifest.get("git") or {}
        data = manifest.get("data") or {}
        return {
            "code_commit": git.get("commit"),
            "code_dirty": git.get("dirty"),
            "code_branch": git.get("branch"),
            "data_fingerprint": data.get("digest"),
            "config_sha256": manifest.get("config_sha256"),
        }
    cfg = os.path.join(ckpt_dir, "config_used.json")
    git = git_info()
    out = {
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "code_branch": git.get("branch"),
        "data_fingerprint": None,
        "config_sha256": (sha256_file(cfg) if os.path.exists(cfg) else None),
    }
    if data_dir:
        try:
            out["data_fingerprint"] = data_fingerprint(data_dir)["digest"]
        except FileNotFoundError:
            pass
    return out
