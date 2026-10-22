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
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .repro import ROOT, sha256_file
from .utils import data_rng_state_fingerprint, seed_int, strict_int

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
    try:
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
    except (OSError, UnicodeError):
        # 台账损坏（含非 UTF-8 字节）时保留已成功解析的记录，而不是让
        # verify_ledger/snapshot 因一个坏字节整体崩溃。
        pass
    return out


def _seed_evidence_issues(run, run_seed: int) -> list:
    """核验台账 run 内嵌的 seed_report 是否真的对应声明的 run_seed。

    旧台账可能完全没有 seed_report（保留兼容，不误报）；但只要该字段存在，
    seed 身份、是否实际播种、数据顺序 seed 与可复算指纹就不能与 run.seed
    冲突。否则手改/拼接一条 provenance 即可让“已固定种子”的证据说谎。
    """
    prov = run.get("provenance")
    sr = prov.get("seed_report") if isinstance(prov, dict) else None
    if not isinstance(sr, dict):
        return []
    issues = []

    def _bad_seed(value):
        try:
            return strict_int(value, "seed_report.seed") != int(run_seed)
        except (TypeError, ValueError, OverflowError):
            return True

    if "seed" in sr and _bad_seed(sr.get("seed")):
        issues.append("seed_report_seed_mismatch")
    if sr.get("seed_verified") is False:
        issues.append("seed_report_unverified")
    if "data_order_seed" in sr and _bad_seed(sr.get("data_order_seed")):
        issues.append("seed_report_data_order_seed_mismatch")
    fp = sr.get("data_order_rng_fingerprint")
    if isinstance(fp, str) and fp:
        if fp != data_rng_state_fingerprint(int(run_seed)):
            issues.append("seed_report_data_order_fingerprint_mismatch")
    return issues


def _ledger_seed(value):
    # 与 build_experiment_record 共用 seed_int：字符串 "1_0"/空串、bool、
    # 小数都不能在核验侧被 int()/float() 静默折叠成另一个合法种子；
    # 负数/超出 2^32-1 的种子在 set_seed 中也不可能被执行，同样按非法身份处理。
    return seed_int(value, "seed")


def verify_ledger(path: str = None, check_artifacts: bool = False,
                  root: str = None, max_reported: int = 20) -> dict:
    """核验台账完整性：生命周期链与（可选）产物哈希。

    每个 run_id 应有且仅有 1 条 started、1 条终态（completed/aborted），且
    终态之后不再出现该 run 的记录；只有 started 表示崩溃/中止未留痕，多条
    终态表示重复写入。同一 run_id 的 config_sha256 不得变化。completed
    记录的 runs 必须恰好覆盖 variants×seeds（缺失/多余/重复/畸形均可
    发现），避免“少跑种子却声称完整”或重复计权。check_artifacts=True 时
    抽样重算 completed 记录中
    artifacts/runs[].files 的 SHA-256（相对路径按 root 或 out_dir 解析），
    产物被覆盖/手改/删除即可发现。

    返回 JSON 原生结构，可直接落盘；issues 为问题列表（空表示通过），
    report_capped=True 表示问题数超过 max_reported 已截断。
    """
    path = path or ledger_path()
    root = root or ROOT
    recs = read_records(path)
    grouped = {}
    issues = []
    for i, rec in enumerate(recs):
        rid = rec.get("run_id")
        if not isinstance(rid, str) or not rid:
            issues.append({"index": i, "issue": "missing_run_id"})
            continue
        grouped.setdefault(rid, []).append((i, rec))
    run_details = []
    for rid, items in grouped.items():
        statuses = [r.get("status") for _, r in items]
        run_issues = []
        if statuses.count("started") == 0:
            run_issues.append("no_started_record")
        if statuses.count("started") > 1:
            run_issues.append("multiple_started_records")
        terminals = [s for s in statuses if s in ("completed", "aborted")]
        if not terminals:
            run_issues.append("dangling_started")
        if len(terminals) > 1:
            run_issues.append("multiple_terminal_records")
        seen_terminal = False
        for _, r in items:
            st = r.get("status")
            if st in ("completed", "aborted"):
                seen_terminal = True
            elif seen_terminal:
                run_issues.append("record_after_terminal")
        # 同一 run_id 的配置哈希必须始终一致：重建同名实验却改写记录（或
        # 误把不同配置的 started/completed 拼在一起）会让“run_id 即配置”
        # 的假设失效。非空字符串以外的值不能因为“不是字符串”就被静默忽略，
        # 否则把 completed 的哈希改成 123/数组即可绕过一致性核对。
        raw_shas = [r.get("config_sha256") for _, r in items]
        if any(v is not None
               and (not isinstance(v, str) or not v) for v in raw_shas):
            run_issues.append("config_sha256_malformed")
        shas = sorted({v for v in raw_shas
                       if isinstance(v, str) and v})
        if len(shas) > 1:
            run_issues.append("config_sha256_rewritten")
        # started/completed 声明的 variants×seeds 必须一致：否则可以用
        # “开始时跑 2 个种子、完成时只声明 1 个”的方式让覆盖核对自洽，
        # 把少跑的种子洗成合法实验。重复声明同样会重复计权。
        declared_scopes = []
        for _, r in items:
            dv, ds = r.get("variants"), r.get("seeds")
            if dv is None and ds is None:
                continue
            if not (isinstance(dv, list) and isinstance(ds, list)):
                run_issues.append("scope_malformed")
                continue
            try:
                v_norm = frozenset(str(x) for x in dv)
                s_norm = frozenset(_ledger_seed(x) for x in ds)
            except (TypeError, ValueError, OverflowError):
                run_issues.append("scope_unparseable")
                continue
            if len(v_norm) != len(dv):
                run_issues.append("declared_variants_duplicate")
            if len(s_norm) != len(ds):
                run_issues.append("declared_seeds_duplicate")
            declared_scopes.append((v_norm, s_norm))
        if len(set(declared_scopes)) > 1:
            run_issues.append("scope_rewritten")
        # completed 的 runs 必须完整覆盖 variants×seeds，且不得重复：
        # 少跑 seed 会让“N 个种子”的结论虚胖，重复条目会重复计权。
        for _, r in items:
            if r.get("status") != "completed":
                continue
            variants, seeds, runs = (r.get("variants"), r.get("seeds"),
                                     r.get("runs"))
            if not (isinstance(variants, list) and isinstance(seeds, list)
                    and isinstance(runs, list)):
                # 旧台账可能三者都没有（只核验生命周期）；但只要声明了其中
                # 任一字段，缺字段/类型错误就是不可核对的覆盖声明，不能静默
                # 跳过而让 completed 记录冒充完整。
                if any(v is not None for v in (variants, seeds, runs)):
                    run_issues.append("runs_malformed")
                continue
            try:
                expected = {(str(v), _ledger_seed(s))
                            for v in variants for s in seeds}
            except (TypeError, ValueError, OverflowError):
                run_issues.append("runs_expected_unparseable")
                continue
            actual_list, malformed = [], 0
            for run in runs:
                if not isinstance(run, dict):
                    malformed += 1
                    continue
                v, s = run.get("variant"), run.get("seed")
                if v is None or s is None:
                    malformed += 1
                    continue
                try:
                    s_int = _ledger_seed(s)
                    actual_list.append((str(v), s_int))
                    run_issues.extend(_seed_evidence_issues(run, s_int))
                except (TypeError, ValueError, OverflowError):
                    malformed += 1
            actual = set(actual_list)
            if expected - actual:
                run_issues.append("runs_coverage_missing")
            if actual - expected:
                run_issues.append("runs_coverage_unexpected")
            if len(actual_list) != len(actual):
                run_issues.append("runs_duplicate")
            if malformed:
                run_issues.append("runs_malformed")
        if run_issues:
            issues.append({"run_id": rid, "issues": sorted(set(run_issues)),
                           "statuses": statuses})
        run_details.append({"run_id": rid, "statuses": statuses,
                            "issues": run_issues})

    artifact_issues = []
    if check_artifacts:
        def _check(rel_key, entry, base):
            p = rel_key if os.path.isabs(rel_key) else os.path.join(base, rel_key)
            if not os.path.exists(p):
                return f"missing: {rel_key}"
            want = entry.get("sha256") if isinstance(entry, dict) else None
            if not want:
                return None
            if sha256_file(p) != want:
                return f"sha256_mismatch: {rel_key}"
            return None

        for rec in recs:
            if rec.get("status") != "completed":
                continue
            arts = rec.get("artifacts")
            if isinstance(arts, dict):
                for key, entry in arts.items():
                    problem = _check(key, entry, root)
                    if problem:
                        artifact_issues.append(problem)
            runs = rec.get("runs")
            for run in (runs if isinstance(runs, list) else []):
                if not isinstance(run, dict):
                    continue
                out_dir = run.get("out_dir") or ""
                base = out_dir if os.path.isabs(out_dir) \
                    else os.path.join(root, out_dir)
                files = run.get("files")
                if not isinstance(files, dict):
                    continue
                for key, entry in files.items():
                    problem = _check(key, entry, base)
                    if problem:
                        artifact_issues.append(
                            f"{run.get('variant')}/seed{run.get('seed')}: "
                            f"{problem}")
    capped = len(issues) + len(artifact_issues) > max_reported
    return {
        "path": os.path.abspath(path),
        "n_records": len(recs),
        "n_runs": len(grouped),
        "ok": not issues and not artifact_issues,
        "issues": (issues + [{"issue": a} for a in artifact_issues])[
            :max_reported],
        "runs": run_details,
        "artifacts_checked": bool(check_artifacts),
        "report_capped": capped,
    }


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


def _load_train_curve(out_dir: str, max_points: int = 1000):
    """把 train_log.jsonl 内嵌进台账快照（防止重跑覆盖后曲线历史丢失）。

    文件哈希只能证明“当时有过这份日志”，无法在文件被覆盖后还原曲线。
    这里保留逐条数值记录；超过 max_points 时按固定步长下采样并保留末点，
    报告 n_records/returned_records/downsampled，绝不把截断冒充完整历史。
    """
    path = os.path.join(out_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return None
    records = []
    try:
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
                    records.append(rec)
    except (OSError, UnicodeError):
        return None
    if not records:
        return None
    n = len(records)
    sampled = records
    cap = max(1, strict_int(max_points, "max_points"))
    if n > cap:
        # 均匀抽取并强制保留首尾（final/early_stop 标记通常在末尾）。
        # 用整除索引生成恰好 cap 条，避免“步长采样 + 追加末点”得到 cap+1 条、
        # 突破调用方声明的上限。
        if cap == 1:
            sampled = [records[-1]]
        else:
            idx = [(i * (n - 1)) // (cap - 1) for i in range(cap)]
            sampled = [records[i] for i in idx]
    return {
        "source": "train_log.jsonl",
        "n_records": n,
        "returned_records": len(sampled),
        "downsampled": len(sampled) != n,
        "records": sampled,
    }


def _eval_identity_mismatches(ev: dict, variant: str, seed) -> list:
    """返回评测结果与本 run 身份不符的字段描述（空表示一致）。"""
    mismatches = []
    ev_variant = ev.get("variant")
    if ev_variant is not None and ev_variant != variant:
        mismatches.append(f"variant={ev_variant}")
    ev_seed = ev.get("seed")
    if ev_seed is not None:
        try:
            if _ledger_seed(ev_seed) != int(seed):
                mismatches.append(f"seed={ev_seed}")
        except (TypeError, ValueError, OverflowError):
            mismatches.append(f"seed={ev_seed!r}")
    return mismatches


def _compact_eval(ev: dict) -> dict:
    """评测结果内嵌字段：保留可复核指标，剔除可能很大的逐样本/worst 列表。

    独立评测（held-out test）和训练内 val 评测共用同一口径，额外保留 split
    与 checkpoint 身份字段，避免只有文件哈希、文件被覆盖后最终结论无法还原。
    """
    return {k: ev.get(k) for k in (
        "split", "val_ppl", "val_loss", "ppl", "loss",
        "val_ppl_clamped", "eval_ctx", "eval_iters_requested", "eval_iters",
        "val_windows", "val_tokens_evaluated", "val_loss_se",
        "val_loss_ci95", "val_ppl_ci95", "val_loss_min", "val_loss_max",
        "gate_mean", "gate_std", "gate_frac_low", "gate_frac_high",
        "n_params", "n_gate_extra", "n_gate_per_head", "n_heads",
        "n_layers", "d_head", "steps", "steps_planned",
        "early_stopped", "tokens_seen", "ckpt", "ckpt_sha256",
        "selection_set_is_eval_set", "code_commit", "code_dirty",
        "data_fingerprint", "train_data_fingerprint",
        "eval_data_fingerprint", "data_fingerprint_matches_training",
        "config_sha256", "sanity_passed", "sanity_verified",
        "data_order_sha256", "train_step_time_ms", "train_tokens_per_sec",
        "train_peak_mem_mb", "epochs_seen", "training_precision")}


def snapshot_train_run(variant: str, seed: int, out_dir: str,
                       known_ckpt_sha256=None, known_ckpt_path=None,
                       heldout_path: str = None) -> dict:
    """把一个 (变体, 种子) run 压缩成台账条目。

    训练内指标取自 out_dir/eval.json；heldout_path 给定时，把独立评测
    （通常是 held-out test）的结果也内嵌为 final_eval。后者不是文件哈希的
    替代品：结果文件被覆盖后，最终指标仍可从台账还原。同时记录
    config_used.json / train_log.jsonl / run_manifest.json / eval.json 的
    路径与哈希，以及被评测 checkpoint（model_best.pt / model.pt）的内容
    哈希。训练曲线本身也内嵌进台账（train_curve，超长时显式下采样）：同
    tag 重跑覆盖文件后，历史曲线仍可从台账还原。

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
    manifest = _load_json(os.path.join(out_dir, "run_manifest.json"))
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
    # 台账条目自含的种子必须是精确整数：int(1.5)==1、int(True)==1 会让
    # 台账记录的身份与真正跑的 run 不一致（且台账是历史证据，写错无法回填）。
    seed = strict_int(seed, "seed")
    rec = {"variant": variant, "seed": seed, "out_dir": out_dir,
           "files": collect_artifacts([os.path.join(out_dir, n) for n in names],
                                      rel_to=out_dir, known=known)}
    curve = _load_train_curve(out_dir)
    if curve is not None:
        rec["train_curve"] = curve
    if isinstance(cfg_used, dict):
        # 超参数内嵌进台账条目：即使 config_used.json 之后被覆盖/删除，
        # 该 run 实际生效的配置仍可从台账还原
        rec["config_used"] = cfg_used
    if isinstance(manifest, dict):
        # 逐 run 溯源内嵌：manifest 被覆盖/删除后，台账仍能自证该 run 固定了
        # 哪些随机源、用了什么环境、代码/数据版本以及 sanity 闸门身份。
        mdata = manifest.get("data")
        mdata = mdata if isinstance(mdata, dict) else {}
        rec["provenance"] = {
            "git": manifest.get("git"),
            "worktree": manifest.get("worktree"),
            "worktree_patch": manifest.get("worktree_patch"),
            "config_sha256": manifest.get("config_sha256"),
            "data_fingerprint": mdata.get("digest"),
            "seed_report": manifest.get("seed_report"),
            "precision": manifest.get("precision"),
            "init_state_fingerprint": manifest.get("init_state_fingerprint"),
            "runtime": manifest.get("runtime"),
            "sanity": manifest.get("sanity"),
        }
    if isinstance(ev, dict) and ev:
        # eval 身份与本次快照不符（旧 eval.json 被复用）时显式标记：
        # 不能把别的 (variant, seed) 的结果当作本 run 的最终指标。
        mismatches = _eval_identity_mismatches(ev, variant, seed)
        if mismatches:
            rec["eval_identity_mismatches"] = mismatches
        rec["result"] = {k: ev.get(k) for k in (
            "val_ppl", "val_loss", "gate_mean", "steps", "steps_planned",
            "early_stopped", "tokens_seen",
            "n_params", "code_commit", "code_dirty", "data_fingerprint",
            "config_sha256", "sanity_passed", "sanity_verified",
            "data_order_sha256")}
        # 训练实际精度嵌在 provenance.training_precision（eval.json 顶层），
        # 台账 result 提升为标量，便于跨 run 直接核对 bf16/fp32。
        prec = ev.get("training_precision")
        if isinstance(prec, dict):
            rec["result"]["amp_enabled"] = prec.get("amp_enabled")
            rec["result"]["amp_dtype"] = prec.get("amp_dtype")
    # 独立评测结果（held-out test）：不只留文件哈希，指标本身也进台账。
    fe = _load_json(heldout_path) if heldout_path else None
    if isinstance(fe, dict) and fe:
        rec["final_eval"] = _compact_eval(fe)
        # 结果文件的路径不写绝对路径（跨机器不可移植），但哈希/大小写进
        # 台账：内嵌指标与当时文件内容可用哈希互相校验。
        rec["final_eval"]["source"] = "independent_eval"
        try:
            rec["final_eval_file"] = {
                "sha256": sha256_file(heldout_path),
                "size": int(os.path.getsize(heldout_path)),
            }
        except OSError:
            pass
        mismatches = _eval_identity_mismatches(fe, variant, seed)
        if mismatches:
            rec["final_eval_identity_mismatches"] = mismatches
    elif heldout_path:
        # 文件存在但无法解析/为空：显式记录，不能静默当作“未评测”。
        rec["final_eval"] = {"source": "independent_eval", "available": False,
                             "reason": ("missing" if not os.path.exists(heldout_path)
                                        else "unparseable_or_empty")}
    return rec


def build_experiment_record(*, run_id: str, status: str, tag: str,
                            config_path: str, config_sha256: str,
                            variants, seeds, smoke: bool, argv, git,
                            data_fingerprint, runtime, out_root: str,
                            results_dir: str, time_start: str,
                            time_end: str = None, sanity=None, runs=None,
                            artifacts=None, extra=None,
                            registration=None, config=None,
                            worktree=None, seed_audit=None) -> dict:
    """组装标准台账记录；字段与 config_used.json/run_manifest.json 同口径。

    registration（可选）为预注册完整性快照：登记表指纹 + 锁链校验结论
    （见 lega/hypotheses.py）。写入台账后，“本轮实验开始时登记表是什么
    状态、是否通过防篡改校验”成为不可回填的历史记录。

    config（可选）为解析后的超参数字典（含 CLI/调优注入后的实际值）：
    台账因此自含超参数，config 文件之后被改写/删除也不影响历史记录的
    可复核性（config_sha256 只能证明“变了”，无法还原“原来是什么”）。

    worktree（可选）为未提交工作区快照（见 repro.worktree_state）：dirty
    时仅凭 git commit 无法复现代码，diff 指纹/补丁路径必须随台账保存。
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
        "seeds": [seed_int(s, "seed") for s in seeds],
        # 声明式逐 seed 随机源计划（见 utils.seed_plan）：started 记录因此
        # 在训练尚未产出 run_manifest 时也带种子审计；实际 RNG 状态指纹
        # 仍在各 run 的 seed_report 中。
        "seed_audit": seed_audit,
        "smoke": bool(smoke),
        "argv": [str(a) for a in argv],
        "git": git,
        "worktree": worktree,
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


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="实验台账核验：started/completed/aborted 生命周期与产物哈希")
    p.add_argument("--path", default=None,
                   help="台账路径（默认 results/experiments.jsonl）")
    p.add_argument("--check_artifacts", action="store_true",
                   help="重算 completed 记录中全部产物的 SHA-256（较慢）")
    args = p.parse_args(argv)
    report = verify_ledger(args.path, check_artifacts=args.check_artifacts)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["ok"]:
        print(f"[ledger] 通过：{report['n_runs']} 个 run，"
              f"{report['n_records']} 条记录")
        return 0
    print(f"[ledger] 发现 {len(report['issues'])} 个问题"
          + ("（已截断）" if report["report_capped"] else ""))
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
