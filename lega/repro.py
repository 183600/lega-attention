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

from .utils import deterministic_algorithms_warn_only, strict_int

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINGERPRINT_ALGO = "sha256-16"
_CACHE_NAME = ".lega_fingerprint.json"

# 流水线每次调用分配的唯一 run_id 通过环境变量传给所有子进程；结果文件
# 内嵌同一值后，汇总可以区分“本轮产物”与结果目录里上一轮/手工留下的
# 同名文件，而不是仅凭变体名与种子猜测来源。
PIPELINE_RUN_ID_ENV = "LEGA_PIPELINE_RUN_ID"

# 科学代码身份：git 不可用（源码包/快照目录）或工作区状态记录缺失时，
# 仍能对“产生结果的代码与配置”给出可复算的结构指纹。
SOURCE_TREE_DIRS = ("lega", "scripts", "configs", "preregistration")
SOURCE_TREE_FILES = ("run_all.py", "requirements.txt", "requirements-ci.txt")
_SOURCE_TREE_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
_SOURCE_TREE_SKIP_SUFFIXES = (".pyc", ".pyo", ".tmp")


def current_pipeline_run_id():
    """当前流水线 run_id（由 run_all 通过环境变量注入）；独立运行返 None。"""
    value = os.environ.get(PIPELINE_RUN_ID_ENV)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def source_tree_fingerprint(root: str = None) -> dict:
    """科学代码/配置目录树的稳定内容指纹（不依赖 git）。

    覆盖训练与评测实际执行的代码、实验配置、预注册表与依赖声明；显式排除
    ``__pycache__``/``.pyc`` 等派生文件以及结果目录。路径统一为 POSIX 风格、
    按字典序排列，摘要只由“相对路径 + 文件 SHA-256”决定，因此在源码压缩包、
    无 git 的容器或工作区状态记录缺失时仍可被第三方独立复算。

    返回结构包含逐文件条目，便于回答“到底哪份文件变了”；``digest`` 是整体
    身份。文件在遍历后消失/不可读时跳过，绝不把半份树冒充完整（调用方可
    通过 ``files`` 与实际树比对）。
    """
    root = os.path.abspath(root or ROOT)
    rels = []
    for top in SOURCE_TREE_DIRS:
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames if d not in _SOURCE_TREE_SKIP_DIRS)
            for name in sorted(filenames):
                if name.endswith(_SOURCE_TREE_SKIP_SUFFIXES):
                    continue
                rels.append(os.path.relpath(os.path.join(dirpath, name), root))
    for name in SOURCE_TREE_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            rels.append(name)
    entries = []
    seen = set()
    for rel in sorted(rels, key=lambda p: p.replace(os.sep, "/")):
        rel_norm = rel.replace(os.sep, "/")
        if rel_norm in seen:
            continue
        seen.add(rel_norm)
        path = os.path.join(root, rel)
        try:
            sha = sha256_file(path)
            size = int(os.path.getsize(path))
        except OSError:
            continue
        entries.append({"path": rel_norm, "size": size, "sha256": sha})
    h = hashlib.sha256()
    for entry in entries:
        h.update(entry["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(entry["sha256"].encode("ascii"))
        h.update(b"\0")
    return {"algo": "sha256-tree-v1", "digest": h.hexdigest()[:16],
            "files": len(entries), "entries": entries}


# 完整 sanity 结果必须覆盖的“代码正确性”检查项（与 lega.sanity 的输出同名）。
# 只在结果文件带 batch_sha256（可区分“最小手写/旧格式”与真实 CLI 产物）
# 时强制要求，避免把 checks 子集当成完整闸门；旧文件缺失时保持兼容，
# 但会记录 missing_checks=None 而不是伪造“完整”。
SANITY_REQUIRED_CHECKS = frozenset({
    "batch_contract_ok",
    "loss_below_threshold", "token_acc_above_threshold", "loss_decreased",
    "finite", "loss_requires_grad",
    "gradient_nonzero_first_step", "all_params_get_grads",
    "no_zero_grads_first_step", "grads_finite",
    "train_loss_matches_direct_logits", "next_token_shift_consistent",
    "initial_loss_near_log_vocab", "regularization_disabled",
})


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


def _display_path(path: str) -> str:
    """文件系统路径 → JSON/UTF-8 安全的展示字符串。

    POSIX 文件名允许任意字节；git status -z 返回原始字节，经 os.fsdecode 后
    可能含 surrogateescape 字符。直接把 surrogate 写进 UTF-8 JSON 会在
    ensure_ascii=False 时抛 UnicodeEncodeError。展示层用反斜杠转义保留原始
    字节信息，实际 git 调用仍使用 os.fsencode 可还原的路径。
    """
    return path.encode("utf-8", "backslashreplace").decode("utf-8")


def _worktree_diff_bytes(root: str, max_untracked_bytes: int = 8 << 20):
    """工作区可重放补丁：已跟踪改动 + 未跟踪文件（受字节上限限制）。

    只取 `git diff HEAD` 会漏掉未跟踪的新代码文件；这里对
    `git ls-files --others --exclude-standard` 列出的每个文件生成
    `git diff --no-index --binary /dev/null <file>` 并拼接。上限用尽时在
    返回值中标记 truncated，调用方可据此知道补丁不完整（绝不静默冒充完整）。

    路径必须保留原始字节：非 UTF-8 文件名经 os.fsdecode（surrogateescape）
    后再传给 subprocess 可逐字节还原；任何一次 git diff 失败也标记
    truncated，避免“dirty 但补丁为空”被误读为工作区未变化。
    """
    tracked_raw = _git_bytes(["diff", "HEAD", "--binary"], root)
    if tracked_raw is None:
        # git 本身失败（而非“无差异”返回空字节）：补丁不可信。
        tracked = b""
        truncated = True
    else:
        tracked = tracked_raw
        truncated = False
    raw = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"],
                     root)
    if raw is None:
        # 无法列出未跟踪文件时补丁不完整：不能把“空列表”当成“没有未跟踪
        # 文件”，否则 diff_sha256 会静默漏掉新代码且 truncated 仍为 False。
        truncated = True
        raw = b""
    files = [os.fsdecode(p) for p in raw.split(b"\0") if p]
    chunks = [tracked] if tracked else []
    used = len(tracked)
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
            truncated = True
            continue
        if r.returncode not in (0, 1):
            # 路径怪异/权限不足等导致 git 无法生成补丁：必须标记不完整，
            # 不能跳过该文件后仍声称 diff_sha256 覆盖整个工作区。
            truncated = True
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
    # 用 -z 取出 NUL 分隔的原始字节路径：普通 porcelain 会给含空格/非 ASCII
    # 的路径加引号并做 C 风格转义，line[3:] 记录下来的不是真实文件名。
    # -z 同时让重命名条目以 `XY new\0old\0` 输出，不会把 " -> " 混进路径。
    raw_status = _git_bytes(["status", "--porcelain", "-z"], root)
    if raw_status is None:
        return {"available": False, "head": commit, "dirty": None,
                "changed_files": None, "untracked_files": None,
                "n_changed": None, "n_untracked": None,
                "diff_sha256": None, "diff_bytes": None,
                "diff_truncated": False}
    changed, untracked = [], []
    fields = raw_status.split(b"\0")
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if len(field) < 4:
            continue
        code = field[:2].decode("utf-8", "replace")
        # 文件名保留原始字节（surrogateescape），供内部定位；对外字段改用
        # _display_path 的 ASCII 转义形式，避免 surrogate 进入 UTF-8 JSON。
        path = os.fsdecode(field[3:])
        # 重命名/复制条目：紧随其后的 NUL 字段是原路径（此处只记录新路径）。
        if code and code[0] in ("R", "C") and i < len(fields):
            i += 1
        (untracked if code.strip() == "??" else changed).append(path)
    diff, truncated = _worktree_diff_bytes(root)
    # dirty 但补丁为空：正常情况只有“未跟踪的空文件”等无内容差异，此时
    # diff_sha256 无法代表工作区状态；按不完整处理，绝不与 clean 的 None
    # 判为相同。git diff 失败也已在 _worktree_diff_bytes 中置 truncated。
    if raw_status and not diff:
        truncated = True
    return {
        "available": True,
        "head": commit,
        "dirty": bool(raw_status),
        "changed_files": [_display_path(p) for p in changed[:max_files]],
        "untracked_files": [_display_path(p) for p in untracked[:max_files]],
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
    def _get(d, key):
        return d.get(key) if isinstance(d, dict) else None

    def _sub(d, key):
        v = _get(d, key)
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

    s_data = _get(start, "data_fingerprint")
    e_data = _get(end, "data_fingerprint")
    if s_data is None or e_data is None:
        checks["data_fingerprint"] = None
        unknown.append("data_fingerprint")
    else:
        checks["data_fingerprint"] = s_data == e_data
        if s_data != e_data:
            mismatches.append({"field": "data_fingerprint", "start": s_data,
                               "end": e_data})

    # 源码树指纹独立于 git：旧快照没有该字段时保持兼容（不把“未记录”变成
    # 新的 unknown，否则历史台账会被误判为不可核对）；只有一端记录时才记
    # unknown，两端都有则内容哈希不同即代码/配置在实验期间被改动。
    s_src = _get(start, "source_tree")
    e_src = _get(end, "source_tree")
    if s_src is None and e_src is None:
        pass
    elif not isinstance(s_src, dict) or not isinstance(e_src, dict):
        checks["source_tree_digest"] = None
        unknown.append("source_tree_digest")
    else:
        s_digest, e_digest = s_src.get("digest"), e_src.get("digest")
        if not isinstance(s_digest, str) or not isinstance(e_digest, str):
            checks["source_tree_digest"] = None
            unknown.append("source_tree_digest")
        else:
            checks["source_tree_digest"] = s_digest == e_digest
            if s_digest != e_digest:
                mismatches.append({"field": "source_tree_digest",
                                   "start": s_digest, "end": e_digest})

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
    # size/mtime/ctime 只是缓存新鲜度的廉价提示，不能当作内容未变的证据：
    # overlayfs 等文件系统时间戳粒度粗，同尺寸内容替换后 ctime/mtime 可能
    # 完全不变（ctime 也可能被粗粒度时钟吞掉）。缓存命中时仍会重算 SHA-256，
    # 见 data_fingerprint。
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
    （多种子/多脚本），不影响正确性——缓存命中要求文件集与 stat 完全一致，
    并且会强制重算参与指纹的文件的 SHA-256。时间戳（含 ctime）在 overlayfs
    等粗粒度文件系统上可能在“同尺寸内容替换”后完全不变，因此 stat 三相等
    不能直接信任缓存里的旧 SHA；重算结果不一致时整体重建。旧缓存没有
    ctime 字段时同样视为未命中并重建。
    """
    names = [n for n in ("train.bin", "val.bin", "test.bin", "meta.json")
             if os.path.exists(os.path.join(data_dir, n))]
    # 先确认目录里确有可指纹化文件，再读取缓存：否则一个 files=[] 的空缓存
    # 能通过下面的“文件集完全一致”校验并直接返回，绕过“无数据文件”报错。
    if not names:
        raise FileNotFoundError(f"{data_dir} 下没有可指纹化的数据文件（train.bin 等）")
    cache_path = os.path.join(data_dir, _CACHE_NAME)
    meta_path = os.path.join(data_dir, "meta.json")
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
            and all(isinstance(e, dict)
                    and isinstance(e.get("sha256"), str)
                    and len(e["sha256"]) == 64
                    for e in cached["files"]) \
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
            # stat 三相等仍不足以证明内容未变：粗粒度时间戳文件系统上，
            # 同尺寸替换可能连 ctime 都不变。强制重算 SHA，避免缓存里的
            # 旧摘要冒充当前数据版本。
            if sha256_file(path) != e["sha256"]:
                fresh = False
                break
        if fresh:
            # 缓存是纯派生数据：文件 stat 与内容 SHA 都未变只说明 entries
            # 可能仍有效，不代表缓存里的合并摘要/分词器字段没有被损坏或
            # 篡改。返回前必须验证 digest 与 entries 自洽，且 tokenizer 与
            # 当前 meta.json 一致；否则重建，避免把错误指纹写进
            # run_manifest。
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                current_tokenizer = (meta.get("tokenizer")
                                     if isinstance(meta, dict) else None)
            except (OSError, ValueError):
                current_tokenizer = None
            if (cached.get("digest") == _digest_of(cached["files"])
                    and cached.get("tokenizer") == current_tokenizer):
                return cached
    entries = [_file_entry(os.path.join(data_dir, n)) for n in names]
    tokenizer = None
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


_INSTALLED_DIST_CACHE = None


def _installed_distributions() -> dict:
    """完整已安装 Python 包版本清单（只读缓存）。

    只记录 torch/numpy 等少数包无法回答“结果是否在另一套依赖栈上产生”：
    tokenizer、绘图库乃至间接依赖都可能改变数据/图表。名称按不区分大小写
    排序，值为发行版声明的版本；读取失败时返回空 dict 而不是阻塞实验。
    """
    global _INSTALLED_DIST_CACHE
    if _INSTALLED_DIST_CACHE is not None:
        return dict(_INSTALLED_DIST_CACHE)
    try:
        from importlib import metadata
        dists = {}
        for dist in metadata.distributions():
            try:
                name = dist.metadata.get("Name")
                version = dist.version
            except Exception:
                continue
            if isinstance(name, str) and name.strip() and version:
                dists.setdefault(name.strip(), str(version))
        _INSTALLED_DIST_CACHE = dict(
            sorted(dists.items(), key=lambda kv: kv[0].lower()))
    except Exception:
        _INSTALLED_DIST_CACHE = {}
    return dict(_INSTALLED_DIST_CACHE)


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
        # 完整依赖栈（含间接依赖）：requirements 只声明直接依赖，环境漂移
        # 往往发生在未固定的传递依赖上。
        "installed_distributions": _installed_distributions(),
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


# 环境身份只比较会影响数值的依赖/硬件字段。平台/主机名/线程数也会记录在
# 完整 runtime_info 中，但它们变化不必然意味着结果不可比；把它们混进
# “是否同一运行环境”的硬判定会产生大量假阳性。cuda/cudnn/gpu 为 None 时
# 表示该构建/主机没有对应组件（如 CPU-only），两端都为 None 视为一致。
RUNTIME_IDENTITY_FIELDS = ("python", "torch", "numpy", "cuda", "cudnn",
                           "gpu")


def runtime_drift(train_runtime, eval_runtime) -> dict:
    """比较训练清单与本次评测进程的运行环境身份。

    为什么需要单独比较：``run_manifest.json`` 记录了训练时的
    Python/PyTorch/NumPy/CUDA/cuDNN 版本，但评测可能在另一台机器、另一个
    虚拟环境或升级后的依赖栈上运行。此时代码 commit 与数据指纹都可以完全
    一致，浮点内核/算子实现却已改变；只把“训练环境”继承到 eval.json，会让
    读者误以为评测也在同一环境下完成。

    返回：
      * ``matches``：True=所有身份字段（Python/PyTorch/NumPy/CUDA/cuDNN/
        GPU 型号）都存在且一致；False=至少一个字段不同；None=清单缺少
        runtime/身份字段（旧产物），不能声称一致。
      * ``mismatches``：逐字段的 train/eval 值（仅不同项）。
      * ``unknown_fields``：无法核对的字段（任一端缺失），不当作一致。
    """
    if not isinstance(train_runtime, dict) or not isinstance(eval_runtime, dict):
        return {"matches": None, "mismatches": [],
                "unknown_fields": list(RUNTIME_IDENTITY_FIELDS),
                "note": "缺少可比较的训练/评测运行环境记录"}
    mismatches, unknown = [], []
    for field in RUNTIME_IDENTITY_FIELDS:
        # 字段存在但值为 None 是有效事实（如 CPU-only 构建没有 CUDA/cuDNN），
        # 两端都为 None 应判一致；只有字段整个缺失才是“无法核对”。
        if field not in train_runtime or field not in eval_runtime:
            unknown.append(field)
            continue
        train_value = train_runtime.get(field)
        eval_value = eval_runtime.get(field)
        if str(train_value) != str(eval_value):
            mismatches.append({"field": field, "train": train_value,
                               "eval": eval_value})
    if mismatches:
        matches = False
    elif unknown:
        matches = None
    else:
        matches = True
    return {"matches": matches, "mismatches": mismatches,
            "unknown_fields": unknown,
            "note": ("matches=False 表示评测使用的 Python/PyTorch/NumPy/"
                     "CUDA/cuDNN 或 GPU 型号与训练清单不同；同一 commit+数据"
                     "也可能因浮点内核差异得到不同数值，结果不得声称与训练"
                     "环境完全相同。")}


def sanity_summary(path: str) -> dict:
    """sanity 结果 JSON 的紧凑摘要，供训练 run 的 manifest 内嵌代码正确性闸门。

    每个 checkpoint 旁都应能回答“这个模型开训前是否通过了小批量过拟合检查、
    检查了哪些项、当时结果文件是什么”。文件缺失/损坏时不阻塞训练，但摘要
    显式 present=False（不假装通过）。
    """
    out = {"file": os.path.abspath(path) if path else None, "present": False,
           "sha256": None, "passed": None, "variant": None, "seed": None,
           "failed_checks": None,
           # passed 与 checks 是否自洽。损坏/手改的 sanity JSON 可能写
           # "passed": true 却保留失败检查项；只有两者同时通过才算证据。
           "checks_consistent": None,
           # 非 null 表示字段结构损坏：调用方必须按“未通过/不可验证”
           # 处理，不能把 malformed 当成 legacy 的“无法核对”而放行。
           "malformed": False,
           # 完整检查覆盖：缺项时即使其余项全 true 也不能算“完整闸门”。
           "missing_checks": None, "checks_complete": None,
           # 固定批内容指纹与实际执行的优化步数（计划 vs 实际）。
           "batch_sha256": None,
           "steps_planned": None, "steps_executed": None,
           "steps_complete": None,
           "initial_loss": None, "final_loss": None,
           "token_acc": None,
           # 证据身份：只有与当前 run 的 config/data/git 一致时，PASS 才能
           # 算作本 run 的代码正确性闸门（见 sanity_identity_check）
           "config_sha256": None, "data_fingerprint": None,
           "git_commit": None, "git_dirty": None,
           "worktree_diff_sha256": None, "worktree_recorded": False,
           "worktree_available": None,
           # 种子生效证据（sanity 自己也是 set_seed 后的产物）
           "seed_verified": None, "data_order_rng_fingerprint": None}
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
        if not checks:
            # 空对象虽然类型正确，但不构成任何检查证据：真实 sanity 结果
            # 永远至少包含过拟合/损失/梯度等检查项。把空 checks 当成
            # “全部通过”会让 {"passed": true, "checks": {}} 绕过训练闸门。
            out["failed_checks"] = []
            out["checks_consistent"] = False
            out["malformed"] = True
        else:
            # 只有真正的 JSON true 才算通过：0/1/"true"/null 都不能伪装成功。
            out["failed_checks"] = sorted(k for k, v in checks.items()
                                          if v is not True)
            out["checks_consistent"] = not out["failed_checks"]
    elif checks is not None:
        # checks 存在但不是对象（数组/字符串等）：结构已损坏，不能当作
        # “旧文件没有 checks”而放行。
        out["checks_consistent"] = False
        out["malformed"] = True
    batch = rec.get("batch_sha256")
    if batch is not None:
        if (not isinstance(batch, str) or len(batch) != 64
                or any(ch not in "0123456789abcdef" for ch in batch)):
            out["malformed"] = True
        else:
            out["batch_sha256"] = batch
    modern_marker = out["batch_sha256"] is not None
    if modern_marker:
        # batch_sha256 是真实 sanity CLI 的产物标记：此时 checks 必须是完整
        # 的现代检查集，不能被删掉整个 checks 字段或裁剪成“只剩 loss”仍冒充
        # 通过。不能只在 checks 非空时才校验——删除 checks 恰好绕过了校验。
        present_checks = set(checks) if isinstance(checks, dict) else set()
        missing = sorted(SANITY_REQUIRED_CHECKS - present_checks)
        out["missing_checks"] = missing
        out["checks_complete"] = not missing
        if missing:
            out["malformed"] = True
            out["failed_checks"] = out["failed_checks"] or []
            out["checks_consistent"] = False
    for key in ("steps_planned", "steps_executed"):
        value = rec.get(key)
        if value is None:
            continue
        try:
            value = strict_int(value, f"sanity.{key}")
        except (TypeError, ValueError, OverflowError):
            out["malformed"] = True
            continue
        if value < 0:
            out["malformed"] = True
            continue
        out[key] = value
    if modern_marker:
        # 现代 sanity CLI 一定同时写计划/执行步数：只保留其中一个字段即可
        # 绕过“计划步数未跑完”的一致性检查，必须按不完整证据处理。
        if (out["steps_planned"] is None
                or out["steps_executed"] is None):
            out["steps_complete"] = False
            out["malformed"] = True
        else:
            out["steps_complete"] = (out["steps_planned"] > 0
                                     and out["steps_executed"]
                                     == out["steps_planned"])
            if out["steps_complete"] is False:
                # 计划步数未跑完（NaN/断流提前 break）却声明 PASS：证据自相矛盾。
                out["malformed"] = True
    elif (out["steps_planned"] is not None
          and out["steps_executed"] is not None):
        out["steps_complete"] = (out["steps_planned"] > 0
                                 and out["steps_executed"]
                                 == out["steps_planned"])
        if out["steps_complete"] is False:
            out["malformed"] = True
    raw_passed = rec.get("passed")
    if isinstance(raw_passed, bool):
        out["passed"] = raw_passed
    elif raw_passed is not None:
        # 数字/字符串形式的 "passed" 是损坏记录，不得经 bool() 变成 PASS。
        out["malformed"] = True
    # 交叉校验：声明 passed=true 但存在失败检查项时，以检查结果为准并
    # 显式标记不一致。这样 train/汇总/下游继承都不会被伪造的 PASS 骗过。
    if out["passed"] is True and out["checks_consistent"] is False:
        out["passed"] = False
        out["malformed"] = True
    for key in ("variant", "seed"):
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
    if manifest is not None and not isinstance(manifest, dict):
        # 身份清单存在但不是对象：无法核对来源，属于损坏证据。
        out["malformed"] = True
        if out["passed"] is True:
            out["passed"] = False
    if isinstance(manifest, dict):
        for key in ("config_sha256",):
            if manifest.get(key) is not None:
                out[key] = manifest[key]
        seed_rep = manifest.get("seed_report")
        if isinstance(seed_rep, dict):
            if isinstance(seed_rep.get("seed_verified"), bool):
                out["seed_verified"] = seed_rep["seed_verified"]
            elif "seed_verified" in seed_rep:
                # 种子生效证据必须是真正的布尔值：字符串 "false"/0/1 若被
                # 静默忽略，会退化成“旧文件无法核对”，从而让一个明确自报
                # 未验证（或伪造）的清单继续以 PASS 身份进入训练闸门。
                out["malformed"] = True
            fp = seed_rep.get("data_order_rng_fingerprint")
            if isinstance(fp, str):
                # 指纹必须是本项目的 16 位小写十六进制短哈希；任意字符串
                # （空串/64 位 SHA/拼写错误）不能冒充可核对的随机源证据。
                if (len(fp) != 16
                        or any(ch not in "0123456789abcdef" for ch in fp)):
                    out["malformed"] = True
                else:
                    out["data_order_rng_fingerprint"] = fp
            elif "data_order_rng_fingerprint" in seed_rep:
                # 指纹存在但类型错误同样不能当成“没有该证据”。
                out["malformed"] = True
        elif seed_rep is not None:
            # seed_report 存在但不是对象：种子审计证据结构损坏。
            out["malformed"] = True
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
    # 结构损坏的证据不能同时声称 PASS：即使失败来自 seed_report 等不会被
    # checks/passed 交叉校验覆盖的字段，也必须按“不可验证”处理。
    if out["malformed"] and out["passed"] is True:
        out["passed"] = False
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
    # 必须是真正的 JSON true：字符串 "false"/1 若按 truthiness 变成“已记录”，
    # 会让损坏/手改的工作区证据绕过 diff 核对并冒充已验证。
    recorded = sanity.get("worktree_recorded") is True
    # 旧记录没有 worktree_available 字段：按“已记录即可核对”处理（当时能从
    # manifest 取到 worktree 才会记录 diff），保持向后兼容；但显式写了
    # 非布尔值说明记录已损坏，只能按 unknown 处理。
    sanity_avail = sanity.get("worktree_available", recorded)
    if (not recorded or sanity_avail is not True
            or worktree.get("available") is not True):
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
        "pipeline_run_id": current_pipeline_run_id(),
        "seed_report": seed_report(seed),
        "time": now_str(),
        "config_file": config_path,
        "config_sha256": sha256_file(config_path),
        "git": git_info(),
        # dirty 时 commit 无法重建代码：额外记录 diff 指纹/文件清单
        "worktree": worktree_state(),
        # 与 git 互补的代码/配置树指纹：源码包或无 git 环境也可复算。
        "source_tree": source_tree_fingerprint(),
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
    try:
        return sha256_file(path)
    except OSError:
        # 存在但不可读（权限/目录占位/文件系统错误）时按“无可用测试集指纹”
        # 处理，不能在下游评测/访问账本写入路径上崩溃。
        return None


def load_provenance(ckpt_dir: str, data_dir: str = None,
                    device: str = None) -> dict:
    """下游评测统一继承上游 run 的溯源字段；缺失时按当前环境/数据即时补算。

    返回平铺字段便于直接 update 进结果 JSON：
      code_commit / code_dirty / code_branch / data_fingerprint / config_sha256

    数据身份必须区分两端：``train_data_fingerprint`` 来自 run_manifest（训练
    时数据），``eval_data_fingerprint``/``data_fingerprint`` 来自本次
    ``data_dir``（实际评测数据）。旧实现让清单里的训练指纹覆盖当前数据指纹，
    用另一份数据评测 checkpoint 时 eval.json 仍声称是训练数据，数据错配
    无法被发现。两者不一致时显式记录
    ``data_fingerprint_matches_training=False``；无法核对时为 None。

    运行环境同样区分两端：``train_runtime`` 来自清单，``eval_runtime`` 是
    本次进程的实际环境；``runtime_matches_training`` 只在身份字段齐全且
    一致时为 True（见 runtime_drift）。这样 eval/ruler/analysis 的结果即使
    脱离 out/ 目录，也能回答“是否在训练同版本依赖下产生”。
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
    eval_runtime = runtime_info(device)
    try:
        current_source = source_tree_fingerprint()
    except Exception:
        current_source = None
    current_data = None
    if data_dir:
        try:
            current_data = data_fingerprint(data_dir)["digest"]
        except FileNotFoundError:
            pass
    # 只读一次 config_used：两次读之间文件若被替换，config_sha256 与
    # config_used_sha256 会指向不同内容，违背“同一份解析后配置”的语义。
    cfg_sha = sha256_file(cfg) if os.path.exists(cfg) else None
    out = {
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "code_branch": git.get("branch"),
        # 兼容字段：优先表示本次实际评测数据；没有 data_dir 时才回退到清单。
        "data_fingerprint": current_data,
        "eval_data_fingerprint": current_data,
        "train_data_fingerprint": None,
        "data_fingerprint_matches_training": None,
        # 训练清单里的环境 vs 本次评测进程的环境：结果文件必须同时带上
        # 两端，不能只继承训练时的版本而掩盖评测发生在另一依赖栈上。
        "train_runtime": None,
        "eval_runtime": eval_runtime,
        "runtime_matches_training": None,
        "runtime_mismatches": [],
        "runtime_unknown_fields": [],
        "config_sha256": cfg_sha,
        # 训练实际生效的 config_used.json 哈希（CLI/调优覆盖后与源配置不同）。
        # 旧清单没有该字段时回退到上面同一份 config_used 文件；有清单时优先
        # 采用训练时记录的值，避免 checkpoint 目录被覆盖后误用新配置。
        "config_used_sha256": cfg_sha,
        # 流水线 run_id 与两端源码树身份：run_id 用于汇总时排除同目录的
        # 上一轮遗留产物；source_tree 为无 git 场景提供可复算的代码身份。
        "pipeline_run_id": current_pipeline_run_id(),
        "train_source_tree_digest": None,
        "eval_source_tree_digest": (current_source or {}).get("digest"),
        "source_tree_matches_training": None,
        # 训练 run 是否通过了小批量过拟合 sanity 闸门（无清单/旧清单为 None）。
        # sanity_verified 区分“PASS 已核对身份”与“旧证据/无法核对”：
        # 硬身份不一致时不得把旧 PASS 继承到下游结果。
        "sanity_passed": None,
        "sanity_verified": None,
        "sanity_present": None,
        "sanity_variant": None,
        "sanity_file": None,
        # 闸门证据的自洽性：失败检查项/结构损坏不能继承成 PASS
        "sanity_failed_checks": None,
        "sanity_checks_consistent": None,
        "sanity_malformed": None,
        # 训练 run 的种子生效证据与实际精度：下游评测结果继承后，单看
        # eval.json 也能核对“哪个 seed/bf16 还是 fp32”而不是只信目录名。
        "seed_report": None,
        "training_precision": None,
        # 训练完成时写回 run_manifest 的 checkpoint 内容哈希。评测方可用
        # 它核对“当前被评测权重”是否就是训练清单记录的那一份。
        "manifest_checkpoint_hashes": None,
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
                ("config_sha256", manifest, "config_sha256"),
                ("config_used_sha256", manifest, "config_used_sha256")):
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
        mruntime = manifest.get("runtime")
        if isinstance(mruntime, dict) and mruntime:
            out["train_runtime"] = mruntime
            drift = runtime_drift(mruntime, eval_runtime)
            out["runtime_matches_training"] = drift["matches"]
            out["runtime_mismatches"] = drift["mismatches"]
            out["runtime_unknown_fields"] = drift["unknown_fields"]
        mrid = manifest.get("pipeline_run_id")
        if (out.get("pipeline_run_id") is None
                and isinstance(mrid, str) and mrid.strip()):
            # 单独/手工评测没有父流水线环境变量时，继承训练清单的 run_id，
            # 使结果仍能绑定到产出该 checkpoint 的那一轮实验；有当前环境
            # run_id 时保留当前值（表示本次评测属于新的流水线调用）。
            out["pipeline_run_id"] = mrid.strip()
        msrc = manifest.get("source_tree")
        if isinstance(msrc, dict):
            train_src = msrc.get("digest")
            if isinstance(train_src, str):
                out["train_source_tree_digest"] = train_src
                eval_src = out.get("eval_source_tree_digest")
                if isinstance(eval_src, str):
                    out["source_tree_matches_training"] = (
                        train_src == eval_src)
        mseed = manifest.get("seed_report")
        if isinstance(mseed, dict):
            out["seed_report"] = mseed
        mprec = manifest.get("precision")
        if isinstance(mprec, dict):
            out["training_precision"] = mprec
        arts = manifest.get("artifacts")
        if isinstance(arts, dict):
            ck = {name: entry.get("sha256")
                  for name, entry in arts.items()
                  if isinstance(name, str) and name.endswith(".pt")
                  and isinstance(entry, dict)
                  and isinstance(entry.get("sha256"), str)
                  and len(entry["sha256"]) == 64}
            out["manifest_checkpoint_hashes"] = ck or None
        msanity = manifest.get("sanity")
        if isinstance(msanity, dict):
            identity = msanity.get("identity")
            identity = identity if isinstance(identity, dict) else {}
            hard_mismatches = identity.get("hard_mismatches")
            # 与 train.py 的 sanity_passed 口径一致：只有硬身份不一致时
            # 才拒绝继承 PASS；旧清单缺 identity 时保留 PASS 但标记未验证。
            # 只继承真正的 JSON 布尔值："false"/"0"/1 经 bool() 会变成
            # True，把畸形清单伪装成已通过的 sanity 闸门。
            failed_checks = msanity.get("failed_checks")
            malformed = msanity.get("malformed")
            checks_consistent = msanity.get("checks_consistent")
            passed = msanity.get("passed")
            gate_broken = False
            # present=False 表示结果文件缺失/不可读，此时任何内嵌的
            # passed=true 都是自相矛盾（或伪造）的证据，绝不能继承为 PASS。
            # 旧清单没有 present 字段时保持兼容。
            present = msanity.get("present")
            out["sanity_present"] = present
            if present is False:
                gate_broken = True
                out["sanity_malformed"] = True
            elif present is not None and present is not True:
                gate_broken = True
                out["sanity_malformed"] = True
            if isinstance(failed_checks, list):
                out["sanity_failed_checks"] = failed_checks
                gate_broken = bool(failed_checks)
            elif failed_checks is not None:
                # 字段存在但不是列表：结构已损坏，不能静默当作“无失败项”。
                gate_broken = True
                out["sanity_malformed"] = True
            if isinstance(checks_consistent, bool):
                out["sanity_checks_consistent"] = checks_consistent
                gate_broken = gate_broken or checks_consistent is False
            elif checks_consistent is not None:
                gate_broken = True
                out["sanity_malformed"] = True
            if isinstance(malformed, bool):
                out["sanity_malformed"] = malformed
                gate_broken = gate_broken or malformed is True
            elif malformed is not None:
                gate_broken = True
                out["sanity_malformed"] = True
            # 失败检查项/损坏记录/自相矛盾的 checks 与 passed=true 矛盾时，
            # 以检查结果为准：旧 manifest 可能由旧版 train 写入，不能因字段
            # 存在就继承 PASS。
            if isinstance(passed, bool) and not hard_mismatches:
                out["sanity_passed"] = bool(passed) and not gate_broken
            elif gate_broken:
                out["sanity_passed"] = False
            verified = identity.get("verified")
            if isinstance(verified, bool):
                # 身份核对为真也不能把缺失/损坏/硬不匹配的证据标成已验证：
                # “已验证的 PASS”必须是 present 且闸门自身未被破坏。
                out["sanity_verified"] = bool(
                    verified and not gate_broken
                    and present is not False and not hard_mismatches)
            elif hard_mismatches:
                # 身份已确认不一致时，即使旧清单没有 verified 字段也不能
                # 留 None 让下游误以为“尚未核对”。
                out["sanity_verified"] = False
            if msanity.get("variant") is not None:
                out["sanity_variant"] = msanity["variant"]
            if msanity.get("file") is not None:
                out["sanity_file"] = msanity["file"]
    return out
