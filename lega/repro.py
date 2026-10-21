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


def _git_bytes(args, cwd):
    """git 的原始字节输出（diff --binary 必须按字节哈希/落盘，不能经过文本解码）。"""
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                           timeout=15)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


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


def _worktree_diff_bytes(root: str, max_untracked_bytes: int = 8 << 20):
    """工作区可重放补丁：已跟踪改动 + 未跟踪文件（受字节上限限制）。

    只取 `git diff HEAD` 会漏掉未跟踪的新代码文件；这里对
    `git ls-files --others --exclude-standard` 列出的每个文件生成
    `git diff --no-index --binary /dev/null <file>` 并拼接。上限用尽时在
    返回值中标记 truncated，调用方可据此知道补丁不完整（绝不静默冒充完整）。
    """
    tracked = _git_bytes(["diff", "HEAD", "--binary"], root) or b""
    raw = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"],
                     root) or b""
    files = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    chunks = [tracked] if tracked else []
    used = len(tracked)
    truncated = False
    for rel in files:
        if used >= max_untracked_bytes:
            truncated = True
            break
        try:
            r = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--",
                 os.devnull, rel],
                cwd=root, capture_output=True, timeout=15)
        except Exception:
            continue
        if r.returncode not in (0, 1):
            continue
        chunk = r.stdout
        if used + len(chunk) > max_untracked_bytes:
            truncated = True
            break
        chunks.append(chunk)
        used += len(chunk)
    return b"".join(chunks), truncated


def worktree_state(root: str = None, max_files: int = 500) -> dict:
    """未提交工作区的可重建证据（dirty 时仅凭 commit 无法还原代码）。

    记录：
      * changed_files / untracked_files：git status 的路径（截断到 max_files，
        并给出完整计数）；
      * diff_sha256 / diff_bytes：已跟踪改动 + 未跟踪文件补丁的 SHA-256 与
        字节数（原始字节；未跟踪文件受 8 MiB 上限限制，见 diff_truncated）；
      * head：快照对应的 commit。
    非 git 仓库/无提交/git 不可用时 available=False，字段为 None，绝不阻塞实验。
    """
    root = root or ROOT
    commit = _git(["rev-parse", "--verify", "HEAD"], root)
    if not commit:
        return {"available": False, "head": None, "dirty": None,
                "changed_files": None, "untracked_files": None,
                "n_changed": None, "n_untracked": None,
                "diff_sha256": None, "diff_bytes": None,
                "diff_truncated": False}
    # 必须保留原始行首空格：porcelain 的 " M path" 前两个字符是状态列，
    # 若先 strip 整段输出，第一行会变成 "M path" 而把路径截掉一个字符。
    raw_status = _git_bytes(["status", "--porcelain"], root)
    if raw_status is None:
        return {"available": False, "head": commit, "dirty": None,
                "changed_files": None, "untracked_files": None,
                "n_changed": None, "n_untracked": None,
                "diff_sha256": None, "diff_bytes": None,
                "diff_truncated": False}
    status = raw_status.decode("utf-8", "replace")
    changed, untracked = [], []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        # 重命名行格式为 "R  old -> new"：记录新路径
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        (untracked if code.strip() == "??" else changed).append(path)
    diff, truncated = _worktree_diff_bytes(root)
    return {
        "available": True,
        "head": commit,
        "dirty": bool(status),
        "changed_files": changed[:max_files],
        "untracked_files": untracked[:max_files],
        "n_changed": len(changed),
        "n_untracked": len(untracked),
        "diff_sha256": (hashlib.sha256(diff).hexdigest() if diff else None),
        "diff_bytes": len(diff),
        "diff_truncated": truncated,
    }


def provenance_drift(start: dict, end: dict) -> dict:
    """比较实验开始/结束时的代码与数据身份，识别长实验期间被改动。

    start/end 各含 ``git``（git_info 结果）、``worktree``（worktree_state
    结果）与 ``data_fingerprint``。只有双方都有可比较值且不同才判 changed；
    无法核对（非 git、数据缺失）记 unknown，绝不用“没检查”冒充“没变化”。
    worktree 用 diff_sha256 比较（含未跟踪文件补丁），因此 commit 不变但
    工作区代码被改动也能发现。
    """
    def _sub(d, key):
        v = (d or {}).get(key)
        return v if isinstance(v, dict) else {}

    s_git, e_git = _sub(start, "git"), _sub(end, "git")
    s_wt, e_wt = _sub(start, "worktree"), _sub(end, "worktree")
    checks, mismatches, unknown = {}, [], []

    s_commit, e_commit = s_git.get("commit"), e_git.get("commit")
    if s_commit is None or e_commit is None:
        checks["code_commit"] = None
        unknown.append("code_commit")
    else:
        checks["code_commit"] = s_commit == e_commit
        if s_commit != e_commit:
            mismatches.append({"field": "code_commit", "start": s_commit,
                               "end": e_commit})

    s_avail, e_avail = s_wt.get("available"), e_wt.get("available")
    if not s_avail or not e_avail:
        checks["worktree_diff_sha256"] = None
        unknown.append("worktree_diff_sha256")
    else:
        same = s_wt.get("diff_sha256") == e_wt.get("diff_sha256")
        if not same:
            checks["worktree_diff_sha256"] = False
            mismatches.append({
                "field": "worktree_diff_sha256",
                "start": s_wt.get("diff_sha256"),
                "end": e_wt.get("diff_sha256")})
        elif s_wt.get("diff_truncated") or e_wt.get("diff_truncated"):
            checks["worktree_diff_sha256"] = None
            unknown.append("worktree_diff_sha256")
        else:
            checks["worktree_diff_sha256"] = True

    s_data = (start or {}).get("data_fingerprint")
    e_data = (end or {}).get("data_fingerprint")
    if s_data is None or e_data is None:
        checks["data_fingerprint"] = None
        unknown.append("data_fingerprint")
    else:
        checks["data_fingerprint"] = s_data == e_data
        if s_data != e_data:
            mismatches.append({"field": "data_fingerprint", "start": s_data,
                               "end": e_data})

    return {
        "checked": checks,
        "unknown": unknown,
        "changed": bool(mismatches),
        "mismatches": mismatches,
        "note": ("changed=True 表示实验期间代码或数据身份发生变化；该轮结果"
                 "不能声称全程对应开始时记录的版本，需要按完整时长重跑或"
                 "在台账中显式标注。"),
    }


def write_worktree_patch(path: str, root: str = None) -> dict:
    """把当前工作区补丁（含未跟踪文件）原子写入 path，返回哈希/大小。

    与 worktree_state 的 diff_sha256 使用同一字节口径，因此可由哈希比对
    确认补丁与清单记录的是同一份工作区状态。无任何改动时不写文件，返回
    patch_file=None；未跟踪文件超过上限时 patch_truncated=True。
    """
    diff, truncated = _worktree_diff_bytes(root or ROOT)
    if not diff:
        return {"patch_file": None, "patch_sha256": None, "patch_bytes": 0,
                "patch_truncated": truncated}
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(diff)
    os.replace(tmp, path)
    return {"patch_file": os.path.abspath(path),
            "patch_sha256": hashlib.sha256(diff).hexdigest(),
            "patch_bytes": len(diff), "patch_truncated": truncated}


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
    # ctime 记录的是 inode 元数据变更时间，普通用户无法像 mtime 那样用
    # os.utime 回填。缓存校验同时看 mtime 与 ctime：同尺寸内容替换后再把
    # mtime 改回旧值，仍会因 ctime 变化而重算，避免拿旧 SHA 冒充当前数据。
    return {"name": os.path.basename(path), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "ctime_ns": int(st.st_ctime_ns), "sha256": sha256_file(path)}


def _digest_of(entries) -> str:
    canon = "|".join(f"{e['name']}:{e['size']}:{e['sha256']}"
                     for e in sorted(entries, key=lambda e: e["name"]))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def data_fingerprint(data_dir: str, use_cache: bool = True) -> dict:
    """数据目录指纹：对实际参与训练/评测的文件做 SHA-256 并给合并摘要。

    缓存写于 <data_dir>/.lega_fingerprint.json：逐文件记录
    size/mtime_ns/ctime_ns/sha256，任一变化即整体重算。缓存仅供重复调用复用
    （多种子/多脚本），不影响正确性——缓存命中要求文件集与 stat 完全一致。
    ctime 用于识别“同尺寸写入后把 mtime 改回旧值”的内容替换；旧缓存没有
    ctime 字段时视为未命中并重建，而不是沿用可能失真的 SHA。
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
            if (e.get("ctime_ns") is None
                    or int(st.st_size) != e.get("size")
                    or int(st.st_mtime_ns) != e.get("mtime_ns")
                    or int(st.st_ctime_ns) != e.get("ctime_ns")):
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
        # 线程数影响归约顺序与浮点结果；CVD 决定实际可见的设备
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
           "token_acc": None,
           # 证据身份：只有与当前 run 的 config/data/git 一致时，PASS 才能
           # 算作本 run 的代码正确性闸门（见 sanity_identity_check）
           "config_sha256": None, "data_fingerprint": None,
           "git_commit": None, "git_dirty": None,
           "worktree_diff_sha256": None, "worktree_recorded": False,
           "worktree_available": None}
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
    manifest = rec.get("manifest")
    if isinstance(manifest, dict):
        for key in ("config_sha256",):
            if manifest.get(key) is not None:
                out[key] = manifest[key]
        git = manifest.get("git")
        if isinstance(git, dict):
            out["git_commit"] = git.get("commit")
            out["git_dirty"] = git.get("dirty")
        data = manifest.get("data")
        if isinstance(data, dict):
            out["data_fingerprint"] = data.get("digest")
        worktree = manifest.get("worktree")
        if isinstance(worktree, dict):
            out["worktree_recorded"] = True
            out["worktree_available"] = worktree.get("available")
            out["worktree_diff_sha256"] = worktree.get("diff_sha256")
    return out


def sanity_identity_check(sanity: dict, run: dict) -> dict:
    """核对 sanity 证据与当前训练 run 是否针对同一代码/配置/数据。

    旧版 sanity 文件可能没有 manifest 身份字段：缺失记为 unknown（不冒充
    已验证，也不因“无法核对”直接推翻旧证据）。只要双方都有值且不同，就是
    hard mismatch——旧配置/旧数据/旧代码的 PASS 不得绑定到本 run。
    seed 只作展示：run_all 用首个种子跑 sanity，其余种子共享同一闸门是
    设计行为，不参与 hard mismatch。
    """
    sanity = sanity if isinstance(sanity, dict) else {}
    run = run if isinstance(run, dict) else {}
    git = run.get("git") if isinstance(run.get("git"), dict) else {}
    data = run.get("data") if isinstance(run.get("data"), dict) else {}
    worktree = (run.get("worktree")
                if isinstance(run.get("worktree"), dict) else {})
    current = {
        "variant": run.get("variant"),
        "config_sha256": run.get("config_sha256"),
        "data_fingerprint": data.get("digest"),
        "git_commit": git.get("commit"),
        "seed": run.get("seed"),
    }
    hard_fields = ("variant", "config_sha256", "data_fingerprint",
                   "git_commit")
    matches, mismatches, unknown = {}, [], []
    for key, want in current.items():
        got = sanity.get(key)
        if got is None or want is None:
            matches[key] = None
            if key in hard_fields:
                unknown.append(key)
        elif got == want:
            matches[key] = True
        else:
            matches[key] = False
            if key in hard_fields:
                mismatches.append(key)
    # 工作区补丁是 commit 之外的可重建证据：只有 sanity 清单确实记录了
    # worktree 才参与比较（旧文件缺失记 unknown）。clean 工作区两边都是
    # None，是合法相等而不是“缺字段”。
    recorded = bool(sanity.get("worktree_recorded"))
    # 旧记录没有 worktree_available 字段：按“已记录即可核对”处理（当时能从
    # manifest 取到 worktree 才会记录 diff），保持向后兼容。
    sanity_avail = sanity.get("worktree_available", recorded)
    if not recorded or sanity_avail is False or not worktree.get("available"):
        # 任一侧工作区不可用（如非 git 目录）时无法核对：记 unknown，
        # 不因“无法核对”直接推翻旧证据。
        matches["worktree_diff_sha256"] = None
        unknown.append("worktree_diff_sha256")
    elif worktree.get("diff_sha256") == sanity.get("worktree_diff_sha256"):
        matches["worktree_diff_sha256"] = True
    else:
        matches["worktree_diff_sha256"] = False
        mismatches.append("worktree_diff_sha256")
    return {
        "matches": matches,
        "hard_mismatches": mismatches,
        "unknown_fields": unknown,
        "verified": not mismatches and not unknown,
        "seed_mismatch": (matches.get("seed") is False),
    }


def build_run_manifest(tag: str, variant: str, seed: int, config_path: str,
                       data_dir: str, device: str, argv=None,
                       sanity: dict = None) -> dict:
    """单个 (变体, 种子) run 的完整溯源清单。

    sanity（可选）为 lega.sanity 结果的紧凑摘要（见 sanity_summary）：把
    “开训前通过小批量过拟合闸门”的代码正确性证据绑定到本 run 的 checkpoint。
    """
    from .utils import now_str, seed_report, strict_int
    config_path = os.path.abspath(config_path)
    data_dir = os.path.abspath(data_dir)
    # manifest 的种子字段必须与 seed_report 同口径：int(1.5)==1、int(True)==1
    # 会让 manifest 记下与实际播种不同的种子（seed_report 已严格拒绝）。
    seed = strict_int(seed, "seed")
    return {
        "tag": tag,
        "variant": variant,
        "seed": seed,
        "seed_report": seed_report(seed),
        "time": now_str(),
        "config_file": config_path,
        "config_sha256": sha256_file(config_path),
        "git": git_info(),
        # dirty 时 commit 无法重建代码：额外记录 diff 指纹/文件清单
        "worktree": worktree_state(),
        "data": data_fingerprint(data_dir),
        "runtime": runtime_info(device),
        "sanity": sanity,
        "argv": [str(a) for a in (argv if argv is not None else sys.argv)],
        "cwd": os.getcwd(),
    }


def test_data_fingerprint(data_dir: str):
    """test.bin 的内容指纹；不存在/不可读时返回 None。

    访问账本与评测溯源统一用这一个值作为“测试集版本”，避免 evaluate、
    n-gram 基线、ruler 语料读取各自哈希不同文件而无法互相对账。
    """
    path = os.path.join(data_dir, "test.bin")
    if not os.path.exists(path):
        return None
    return sha256_file(path)


def load_provenance(ckpt_dir: str, data_dir: str = None) -> dict:
    """下游评测统一继承上游 run 的溯源字段；缺失时按当前环境/数据即时补算。

    返回平铺字段便于直接 update 进结果 JSON：
      code_commit / code_dirty / code_branch / data_fingerprint / config_sha256

    数据身份必须区分两端：``train_data_fingerprint`` 来自 run_manifest（训练
    时数据），``eval_data_fingerprint``/``data_fingerprint`` 来自本次
    ``data_dir``（实际评测数据）。旧实现让清单里的训练指纹覆盖当前数据指纹，
    用另一份数据评测 checkpoint 时 eval.json 仍声称是训练数据，数据错配
    无法被发现。两者不一致时显式记录
    ``data_fingerprint_matches_training=False``；无法核对时为 None。
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
    current_data = None
    if data_dir:
        try:
            current_data = data_fingerprint(data_dir)["digest"]
        except FileNotFoundError:
            pass
    out = {
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "code_branch": git.get("branch"),
        # 兼容字段：优先表示本次实际评测数据；没有 data_dir 时才回退到清单。
        "data_fingerprint": current_data,
        "eval_data_fingerprint": current_data,
        "train_data_fingerprint": None,
        "data_fingerprint_matches_training": None,
        "config_sha256": (sha256_file(cfg) if os.path.exists(cfg) else None),
        # 训练 run 是否通过了小批量过拟合 sanity 闸门（无清单/旧清单为 None）。
        # sanity_verified 区分“PASS 已核对身份”与“旧证据/无法核对”：
        # 硬身份不一致时不得把旧 PASS 继承到下游结果。
        "sanity_passed": None,
        "sanity_verified": None,
        "sanity_variant": None,
        "sanity_file": None,
    }
    if isinstance(manifest, dict) and manifest:
        mgit = manifest.get("git")
        mdata = manifest.get("data")
        mgit = mgit if isinstance(mgit, dict) else {}
        mdata = mdata if isinstance(mdata, dict) else {}
        for key, src, field in (
                ("code_commit", mgit, "commit"),
                ("code_dirty", mgit, "dirty"),
                ("code_branch", mgit, "branch"),
                ("config_sha256", manifest, "config_sha256")):
            val = src.get(field)
            if val is not None:
                out[key] = val
        train_data = mdata.get("digest")
        if train_data is not None:
            out["train_data_fingerprint"] = train_data
            if current_data is None:
                # 没有当前数据目录可核对：保留旧行为（供只读审计使用），
                # 但不冒充“已核对一致”。
                out["data_fingerprint"] = train_data
            else:
                out["data_fingerprint_matches_training"] = (
                    current_data == train_data)
        msanity = manifest.get("sanity")
        if isinstance(msanity, dict):
            identity = msanity.get("identity")
            identity = identity if isinstance(identity, dict) else {}
            hard_mismatches = identity.get("hard_mismatches")
            # 与 train.py 的 sanity_passed 口径一致：只有硬身份不一致时
            # 才拒绝继承 PASS；旧清单缺 identity 时保留 PASS 但标记未验证。
            if (msanity.get("passed") is not None
                    and not hard_mismatches):
                out["sanity_passed"] = bool(msanity["passed"])
            if identity.get("verified") is not None:
                out["sanity_verified"] = bool(identity["verified"])
            elif hard_mismatches:
                # 身份已确认不一致时，即使旧清单没有 verified 字段也不能
                # 留 None 让下游误以为“尚未核对”。
                out["sanity_verified"] = False
            if msanity.get("variant") is not None:
                out["sanity_variant"] = msanity["variant"]
            if msanity.get("file") is not None:
                out["sanity_file"] = msanity["file"]
    return out
