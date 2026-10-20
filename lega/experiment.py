"""追加式实验台账（append-only experiment ledger）：每轮实验一行 JSON。

解决“结果散落在各自目录、重跑即覆盖、无法回答历史上一共跑过哪些实验”：
  * 每轮流水线开始时追加 status="started" 记录，正常结束时追加同一 run_id
    的 status="completed" 记录（含最终指标与全部产物的 SHA-256）；
  * 只有 started 没有 completed 的 run_id 表示该轮被中止/崩溃 —— 失败实验
    也是历史的一部分，不是从记录里消失；
  * 记录代码版本（git commit/dirty）、超参数（config 路径+SHA-256）、随机
    种子、数据版本（数据指纹）、最终结果与训练曲线文件的哈希，可复现可追溯；
  * 追加写用文件锁保护，多个流水线并发也不会互相覆盖；
  * 台账路径固定为 results/experiments.jsonl（results/ 已在 .gitignore
    中忽略），不随 --results_dir 变化，所有实验汇聚到同一份历史。

读取：`read_records(path)` 逐行解析，畸形行跳过；消费方按 run_id 取最新的
status="completed" 记录作为该轮实验的最终档案。
"""
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .repro import ROOT, sha256_file

LEDGER_FILENAME = "experiments.jsonl"

try:  # POSIX（本项目 CI 与实验环境）用 flock；其它平台退化为无锁追加。
    import fcntl
except ImportError:  # pragma: no cover - 取决于平台
    fcntl = None


def ledger_path(results_root: str = None) -> str:
    """台账文件路径；默认 <repo>/results/experiments.jsonl。"""
    root = results_root or os.path.join(ROOT, "results")
    return os.path.join(root, LEDGER_FILENAME)


_RUN_SEQ = itertools.count()


def new_run_id(tag: str, config_sha256: str = None, ts: str = None) -> str:
    """生成可读且唯一的 run_id：时间-标签-配置哈希前缀-微秒·进程内序号。

    不占用任何科学随机源（用 time_ns 而非 random），避免台账生成影响实验
    本身的随机流。后缀 = 秒内微秒计数(6 位) + 进程内单调递增序号：同一时刻
    连续调用也绝不重复；跨进程则由微秒计数区分。
    """
    ts = ts or time.strftime("%Y%m%d-%H%M%S")
    sha = (config_sha256 or "noconfig")[:8]
    micros = (time.time_ns() // 1000) % 1_000_000
    return f"{ts}-{tag}-{sha}-{micros:06d}{next(_RUN_SEQ):03d}"


def append_record(path: str, record: dict) -> None:
    """把一条记录以一行 JSON 追加到台账；加锁 + fsync，保证并发安全与落盘。"""
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_records(path: str) -> list:
    """读取台账；文件不存在返回 []，畸形行（非 JSON/非对象）跳过不抛错。"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def collect_artifacts(paths, rel_to: str = None, known: dict = None) -> dict:
    """对产物文件计算 SHA-256 与大小；缺失文件跳过。

    rel_to 给定时用相对路径作为键（避免把机器绝对路径写进台账，且跨机器
    对照时键稳定）。键顺序与输入一致。known（path -> sha256）给出同一进程
    流水线内已计算过的哈希（如 eval.json 记录的 checkpoint 指纹）时直接
    使用；大小以当前文件为准。
    """
    entries = [(p, os.path.relpath(p, rel_to) if rel_to else p)
               for p in paths if os.path.isfile(p)]
    if not entries:
        return {}

    def _one(entry):
        p, key = entry
        h = (known or {}).get(p)
        return key, {"sha256": h if h else sha256_file(p),
                     "size": int(os.path.getsize(p))}

    if len(entries) == 1:
        return dict([_one(entries[0])])
    with ThreadPoolExecutor(max_workers=min(8, len(entries))) as ex:
        return dict(ex.map(_one, entries))


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def snapshot_train_run(variant: str, seed: int, out_dir: str,
                       known_ckpt_sha256=None, known_ckpt_path=None) -> dict:
    """把一个 (变体, 种子) run 压缩成台账条目。

    最终指标取自 eval.json；同时记录 config_used.json / train_log.jsonl /
    run_manifest.json / eval.json 的路径与哈希，以及被评测 checkpoint
    （model_best.pt / model.pt）的内容哈希 —— 训练曲线或结果文件后续被
    覆盖、手改、甚至换掉 checkpoint，都能通过哈希比对发现。

    evaluate 阶段已对同一 checkpoint 做过全量 SHA-256（写在 results_dir
    的 eval_<variant><suffix>.json 及 test 访问账本）：调用方通过
    known_ckpt_sha256/known_ckpt_path 传入。仅当路径确为本 run 的
    model_best.pt 时采信（防止复用旧评测文件导致错配），否则重算。
    out_dir/eval.json 自带的同名字段同样可用。
    """
    names = ("config_used.json", "train_log.jsonl", "run_manifest.json",
             "eval.json", "model_best.pt", "model.pt")
    ev = _load_json(os.path.join(out_dir, "eval.json"))
    cfg_used = _load_json(os.path.join(out_dir, "config_used.json"))
    known = {}
    best = os.path.join(out_dir, "model_best.pt")
    candidates = []
    if known_ckpt_sha256 and known_ckpt_path:
        candidates.append((known_ckpt_path, known_ckpt_sha256))
    if isinstance(ev, dict) and ev.get("ckpt_sha256") and ev.get("ckpt"):
        candidates.append((ev["ckpt"], ev["ckpt_sha256"]))
    for path, sha in candidates:
        try:
            if os.path.realpath(path) == os.path.realpath(best):
                known[best] = str(sha)
                break
        except (OSError, TypeError, ValueError):
            continue
    rec = {"variant": variant, "seed": int(seed), "out_dir": out_dir,
           "files": collect_artifacts([os.path.join(out_dir, n) for n in names],
                                      rel_to=out_dir, known=known)}
    if isinstance(cfg_used, dict):
        # 超参数内嵌进台账条目：即使 config_used.json 之后被覆盖/删除，
        # 该 run 实际生效的配置仍可从台账还原
        rec["config_used"] = cfg_used
    if isinstance(ev, dict) and ev:
        rec["result"] = {k: ev.get(k) for k in (
            "val_ppl", "val_loss", "gate_mean", "steps", "steps_planned",
            "early_stopped", "tokens_seen",
            "n_params", "code_commit", "code_dirty", "data_fingerprint",
            "config_sha256", "sanity_passed")}
    return rec


def build_experiment_record(*, run_id: str, status: str, tag: str,
                            config_path: str, config_sha256: str,
                            variants, seeds, smoke: bool, argv, git,
                            data_fingerprint, runtime, out_root: str,
                            results_dir: str, time_start: str,
                            time_end: str = None, sanity=None, runs=None,
                            artifacts=None, extra=None,
                            registration=None, config=None) -> dict:
    """组装标准台账记录；字段与 config_used.json/run_manifest.json 同口径。

    registration（可选）为预注册完整性快照：登记表指纹 + 锁链校验结论
    （见 lega/hypotheses.py）。写入台账后，“本轮实验开始时登记表是什么
    状态、是否通过防篡改校验”成为不可回填的历史记录。

    config（可选）为解析后的超参数字典（含 CLI/调优注入后的实际值）：
    台账因此自含超参数，config 文件之后被改写/删除也不影响历史记录的
    可复核性（config_sha256 只能证明“变了”，无法还原“原来是什么”）。
    """
    rec = {
        "run_id": run_id,
        "status": status,
        "time_start": time_start,
        "time_end": time_end,
        "tag": tag,
        "config_file": config_path,
        "config_sha256": config_sha256,
        "config": config,
        "variants": [str(v) for v in variants],
        "seeds": [int(s) for s in seeds],
        "smoke": bool(smoke),
        "argv": [str(a) for a in argv],
        "git": git,
        "data_fingerprint": data_fingerprint,
        "runtime": runtime,
        "out_root": out_root,
        "results_dir": results_dir,
        "sanity": sanity,
        "runs": runs,
        "artifacts": artifacts,
        "registration": registration,
    }
    if extra:
        rec["extra"] = extra
    return rec
