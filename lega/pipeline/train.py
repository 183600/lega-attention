"""一键流水线主流程编排（argparse、阶段调度与实验台账）。"""
from ._shared import _install as _shared_install

_shared_install(globals())

def main():
    p = argparse.ArgumentParser(description="一键实验流水线（多种子 + 受控对比 + 完整消融）")
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--data_dir", default="data/dummy")
    p.add_argument("--out_root", default="out")
    p.add_argument("--results_dir", default=None, help="默认 results/<tag>")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS),
                   help="逗号分隔的变体；默认主实验矩阵。可额外点名 "
                        "lega_placebo（参数匹配安慰剂），或直接用 "
                        "--param_control 追加")
    p.add_argument("--param_control", action="store_true",
                   help="追加训练期参数匹配安慰剂 lega_placebo：门控参数/"
                        "FLOPs 与 lega 相同，但 B/C 证据替换为确定性噪声；"
                        "汇总页给出 lega vs lega_placebo 的容量对照。"
                        "默认关闭（不影响主实验规模）")
    p.add_argument("--seeds", default=None,
                   help="逗号分隔的随机种子；默认取配置 seeds（冒烟模式仅取首个）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sanity", action="store_true",
                   help="跳过过拟合小批量 sanity check（默认每变体训练前必跑，"
                        "失败即中止全量训练）")
    p.add_argument("--sanity_steps", type=int, default=None,
                   help="sanity check 的优化步数（默认 300，冒烟 60）")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="验证损失连续 N 次评测无实质改善则早停（透传 lega.train；"
                        "默认读配置 early_stop_patience，缺省关闭）。启用后各 run "
                        "实际步数可不同，受控变量审计按计划步数核对并在汇总页"
                        "如实标记")
    p.add_argument("--worst_n", type=int, default=10,
                   help="每个 run 转储的最差语言建模窗口数（坏案例归因用；"
                        "默认 10，少于 3 条的模式分析不具统计意义）")
    p.add_argument("--worst_on_test", action="store_true",
                   help="允许在 test split 上转储最差窗口（默认关闭：test 只能"
                        "最终评估一次，把 test 错误清单交给人工迭代会让测试集"
                        "变成隐式训练集；开启后写访问账本并要求报告披露）")
    p.add_argument("--allow_repeat_test", action="store_true",
                   help="允许同一 checkpoint 在 test 上重复评测（默认拒绝，这是"
                        "防止“测试集变隐式训练集”的闸门；--skip_train 重跑会命中，"
                        "确需重测时显式放行，账本仍留痕）")
    p.add_argument("--skip_ruler", action="store_true")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--skip_ablation", action="store_true",
                   help="跳过推理期因果病灶消融（默认对 --ablate_variants "
                        "指定的门控变体，在同一 checkpoint/同一批 val 窗口上"
                        "做逐窗口配对干预；不访问 test）")
    p.add_argument("--ablate_variants", default="lega,qgate",
                   help="做推理期病灶消融的变体（逗号分隔；默认 lega,qgate，"
                        "让核心对比的已有门控基线也接受同一套病灶；"
                        "可再加 gate_only 作对照；standard/null 无输出门，"
                        "会被拒绝）")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--tune_baseline", action="store_true",
                   help="训练前对基线（standard）做 lr/权重衰减网格调优（仅验证集）；"
                        "选定组合注入配置并统一用于全部变体，证据写入 "
                        "results/<tag>/baseline_tuning.json")
    p.add_argument("--tune_steps", type=int, default=None,
                   help="调优 trial 的训练步数（默认与正式训练相同；显式缩小步数时"
                        "汇总页会如实记录，不能当作与正式同步数的调优）")
    p.add_argument("--tune_lr_mults", default="0.5,1,2",
                   help="调优网格的 lr 乘子（逗号分隔，默认 0.5,1,2）")
    p.add_argument("--tune_wd_list", default=None,
                   help="调优网格的 weight_decay 列表（逗号分隔；默认仅配置值）")
    p.add_argument("--tune_seeds", default=None,
                   help="调优选择用的种子（逗号分隔；默认只用首个正式种子）。"
                        "单种子选择可能被 val_loss 噪声左右（winner's curse），"
                        "正式实验建议给 ≥3 个（网格配置数 × 种子数 次训练）")
    p.add_argument("--skip_ngram", action="store_true",
                   help="跳过简单统计基线（一元/二元/插值二元平滑 n-gram PPL）")
    p.add_argument("--ngram_split", default="auto",
                   choices=["auto", "val", "test"],
                   help="n-gram 基线评测 split；auto（默认）与 lega.evaluate 的 "
                        "--split auto 同规则：有非空 test.bin 用 held-out test，"
                        "否则回退 val——保证与神经模型 PPL 同 split 可比；"
                        "test 会写访问账本")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 检测到打乱/字符切分/文档跨 split 也继续"
                        "训练（默认拒绝；仅用于复现历史实验，不推荐）")
    p.add_argument("--jobs", type=int, default=None,
                   help="sanity/训练/机制分析阶段的并发子进程数；默认 CPU 上 "
                        "min(4, 核数)（<=4 核时为核数+1），CUDA 或 --wandb 下为 1")
    args = p.parse_args()
    if args.allow_data_protocol_violations:
        # 训练/调优/sanity 子进程继承该环境变量；否则 run_all 放行了、
        # lega.train 仍会按默认策略拒绝同一份数据。
        os.environ[ALLOW_ENV] = "1"

    args.config = _repo_path(args.config)
    args.data_dir = _repo_path(args.data_dir)
    py = sys.executable
    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_root = os.path.join(_repo_path(args.out_root), tag)
    results_dir = _repo_path(args.results_dir) if args.results_dir else os.path.join(ROOT, "results", tag)
    os.makedirs(results_dir, exist_ok=True)
    variants = parse_variants(args.variants)
    if not variants:
        # 空 --variants 会让流水线“成功”跑完但没有任何 run，下游汇总为空；
        # 这是显式配置错误，不能当作合法实验。
        p.error("--variants 不能为空：至少指定一个变体（可选："
                + ", ".join(RUNNABLE_VARIANTS) + "）")
    if args.param_control and PARAM_CONTROL_VARIANT not in variants:
        variants.append(PARAM_CONTROL_VARIANT)
    invalid = sorted(set(variants) - set(RUNNABLE_VARIANTS))
    if invalid:
        p.error(f"未知变体: {', '.join(invalid)}；可选: "
                f"{', '.join(RUNNABLE_VARIANTS)}")
    ablate_variants = ([] if args.skip_ablation
                       else parse_variants(args.ablate_variants))
    invalid_ablate = sorted(set(ablate_variants) - set(GATE_FACTORS))
    if invalid_ablate:
        p.error("--ablate_variants 含无输出门/未知变体: "
                + ", ".join(invalid_ablate)
                + "；可选: " + ", ".join(sorted(GATE_FACTORS)))
    ablate_variants = [v for v in ablate_variants if v in variants]
    if not args.skip_ablation and not ablate_variants:
        print("[run_all] 警告：--ablate_variants 与本次 --variants 无交集，"
              "跳过推理期病灶消融")

    if not os.path.exists(args.config):
        p.error(f"配置文件不存在: {args.config}")
    with open(args.config, "r", encoding="utf-8") as cf:
        cfg = json.load(cf)
    if not isinstance(cfg, dict):
        p.error(f"配置文件顶层必须是 JSON 对象，收到 {type(cfg).__name__}: "
                f"{args.config}")

    # 多种子协议：显式 --seeds 优先；否则用配置 seeds；冒烟模式只跑首个种子。
    # 种子一律严格解析：int("1.5") 会直接抛错，而 int(1.5)/int(True) 会把
    # 1.5、true 静默截断成 1——结果目录与台账都会用错种子，且两个"不同"
    # 种子经 dict.fromkeys 去重后只剩一个，N 个种子的结论凭空变少。
    if args.seeds is not None:
        seeds = _parse_seed_list(args.seeds, p, "--seeds")
    else:
        # 只有“未提供 seeds”才回退到单个 seed；显式 seeds=[]（或 --seeds ""）
        # 是“没有种子”的配置错误，不能被 `or` 静默替换成默认种子。
        cfg_seeds = cfg.get("seeds")
        if cfg_seeds is None:
            cfg_seeds = [cfg.get("seed", 1337)]
        if not isinstance(cfg_seeds, (list, tuple)):
            p.error(f"配置 seeds 需要整数列表，收到 {cfg_seeds!r}")
        if args.smoke and cfg_seeds:
            seeds = [_strict_seed_int(cfg_seeds[0], p, "配置 seed")]
        else:
            seeds = [_strict_seed_int(s, p, "配置 seeds") for s in cfg_seeds]
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        p.error("种子列表为空")
    n_seeds = len(seeds)
    print(f"[run_all] 变体={variants}")
    print(f"[run_all] 种子={seeds}（{n_seeds} 个）tag={tag}")
    jobs = _resolve_jobs(args)
    if jobs > 1:
        print(f"[run_all] jobs={jobs}（sanity/训练/评测/机制分析并发；--jobs 可覆盖）")

    # 数据协议前置闸门：报告缺失只告警（兼容旧数据），但一旦报告声明的划分
    # 协议违反“时序/群组不跨侧”，默认直接中止——已知泄露的数据不应进入
    # 全量训练；报告与 meta 的哈希绑定也必须一致，避免陈旧/被改报告冒充
    # 当前自检结果。
    protocol_override = allow_protocol_violations(
        args.allow_data_protocol_violations)
    data_preflight = data_protocol_preflight(
        args.data_dir, allow_override=protocol_override)
    critical_data_alerts = data_preflight["critical_violations"]
    if critical_data_alerts:
        detail = "；".join(
            f"[{a.get('code')}] {a.get('message')}"
            for a in critical_data_alerts[:5])
        print(f"[run_all] ⚠ 数据协议自检发现 {len(critical_data_alerts)} 个"
              f"严重问题，但 --allow_data_protocol_violations 已放行：{detail}")
    elif not data_preflight["report_present"]:
        print("[run_all] 警告：数据目录缺少 data_report.json，"
              "无法做训练前防泄漏闸门（旧数据或未走统一入口）")

    # 子进程随机性/确定性环境：PYTHONHASHSEED 必须在解释器启动前设置；
    # 这里给出基准值（无 --seed 的命令用），带 --seed 的命令由 _cmd_env 逐条
    # 覆盖为自身种子。CUBLAS 工作区配置须在任何 CUDA 初始化前生效。
    os.environ["PYTHONHASHSEED"] = str(seeds[0])
    ensure_cublas_workspace_config()

    # 基线调优（可选）：先调 standard 的 lr/权重衰减（只用验证集，test 不加载），
    # 把最优组合注入配置并统一用于所有变体——保证对比是“调过的基线 vs 新方法”，
    # 而不是把超参数红利误归因给模块。证据落盘 results/<tag>/baseline_tuning.json。
    if args.tune_baseline and args.skip_train:
        print("[tuning] --skip_train：跳过基线调优（没有要训练的新 run）")
    elif args.tune_baseline:
        tune_root = os.path.join(out_root, "_baseline_tuning")
        tune_seeds = (_parse_seed_list(args.tune_seeds, p, "--tune_seeds")
                      if args.tune_seeds else None)
        tune_report = tune_baseline(
            args.config, args.data_dir, tune_root, variant="standard",
            seed=seeds[0], seeds=tune_seeds, steps=args.tune_steps,
            lr_mults=parse_floats(args.tune_lr_mults),
            wd_list=(parse_floats(args.tune_wd_list)
                     if args.tune_wd_list else None),
            device=args.device, tolerance=TUNE_TOLERANCE)
        save_json(tune_report,
                  os.path.join(results_dir, "baseline_tuning.json"))
        best = tune_report["selected"]
        cfg["lr"] = float(best["lr"])
        cfg["weight_decay"] = float(best["weight_decay"])
        tuned_path = os.path.join(results_dir, "config_tuned.json")
        save_json(cfg, tuned_path)
        args.config = tuned_path
        print(f"[tuning] 全部变体改用调优后配置: lr={cfg['lr']:.3g} "
              f"wd={cfg['weight_decay']:.3g} -> {tuned_path}")

    # 简单统计基线：一元/二元/插值二元平滑 n-gram 的 held-out PPL，给出
    # “数据绝对难度”参照（零训练；平滑系数/插值权重只在 val 上选）。split
    # 默认 auto：与神经
    # 模型的评测 split 完全一致（test.bin 存在且非空 → held-out test），
    # 否则汇总页的“神经 vs 统计基线”会变成 test-vs-val 的错位比较。
    if not args.skip_ngram:
        ngram_split = resolve_ngram_split(args.data_dir, args.ngram_split)
        # 窗口规格来自本次配置（含冒烟覆盖）：同 split 还不够，必须同 token——
        # n-gram 额外在神经评测窗口的目标 token 上打分，供汇总做逐位置对比。
        window_spec = _neural_eval_window_spec(cfg, args.smoke)
        try:
            ng = ngram_evaluate(args.data_dir, split=ngram_split,
                                allow_test=(ngram_split == "test"),
                                window_spec=window_spec)
            save_json(ng, os.path.join(results_dir, "ngram_baseline.json"))
            wm = ng.get("ppl_window_matched") or {}
            print(f"[ngram] 简单基线 {ngram_split} PPL: "
                  f"unigram={ng['ppl'].get('unigram')} "
                  f"bigram={ng['ppl'].get('bigram')} "
                  f"interp={ng['ppl'].get('interpolated_bigram')} "
                  f"(α={ng['selected_alpha']}, "
                  f"interp={ng.get('selected_interpolated')}, "
                  f"selected={ng.get('selected_model')})；"
                  f"窗口对齐 PPL: unigram={wm.get('unigram')} "
                  f"bigram={wm.get('bigram')} "
                  f"interp={wm.get('interpolated_bigram')}"
                  f"（spec={ng.get('window_spec')}）")
        except (OSError, ValueError) as e:
            print(f"[run_all] ⚠ n-gram 基线跳过：{e}")

    # 全局溯源清单：本次流水线的代码版本、数据指纹、运行环境与命令。
    git = git_info()
    config_sha256 = sha256_file(args.config)
    # 预注册完整性：把登记表指纹与锁链校验结论写进 repro 与实验台账，
    # 使“本轮实验对应哪份登记表、是否发生静默改写”可事后审计。
    reg_doc = load_registration()
    reg_integrity = verify_registration()
    reg_info = {
        "path": os.path.relpath(HYPOTHESES_PATH, ROOT),
        "present": reg_doc is not None,
        "fingerprint": (registration_fingerprint(reg_doc)
                        if reg_doc is not None else None),
        "integrity": reg_integrity,
    }
    if not reg_integrity.get("confirmatory_valid"):
        print(f"[repro] ⚠ 预注册完整性未通过（status="
              f"{reg_integrity.get('status')}）：本轮全部对比按探索性对待")
    else:
        print(f"[repro] 预注册完整性 {reg_integrity.get('status')}"
              f"（假设 {reg_integrity.get('n_hypotheses')} 条，"
              f"修订链 {reg_integrity.get('chain_length')} 环）")

    # 先验设计计划：在正式训练/评测之前落盘“计划内 (变体, 种子) 单元、
    # 种子数/RULER 每格样本量对 SESOI 的灵敏度、分析人群与排除规则”。
    # 它不读取任何结果或 test，故可作为“开跑前声明”的证据；汇总时据此
    # 核对实际分析人群，计划内单元缺失会使受影响确认性判定降级。
    try:
        design_plan = build_design_plan(
            cfg, seeds, variants, reg_doc,
            config_file=args.config, config_sha256=config_sha256,
            ruler_n_samples=(2 if args.smoke else None),
            smoke=bool(args.smoke), known_variants=set(RUNNABLE_VARIANTS),
            analysis_variants=[v for v in variants
                               if v in GATED_FOR_ANALYSIS])
    except ValueError as e:
        p.error(f"设计计划生成失败（检查配置的 design/ruler 字段）：{e}")
    design_plan_path = os.path.join(results_dir, "design_plan.json")
    save_design_plan(design_plan, design_plan_path)
    print(f"[design] 先验计划 status={design_plan.get('status')} "
          f"digest={str(design_plan.get('plan_digest'))[:12]} -> "
          f"{design_plan_path}")
    _design_hist = load_design_plan_history(results_dir)
    if len(_design_hist) > 1:
        print(f"[design] ⚠ 本结果目录已有 {len(_design_hist)} 份设计计划记录"
              "（重跑/改种子/改样本量）；历史见 design_plan_history.jsonl，"
              "当前计划以最新 digest 为准")
    for _c in design_plan.get("checks", []):
        if isinstance(_c, dict) and _c.get("status") in ("warn", "fail"):
            print(f"[design] {_c['status']}: {_c.get('title')} — "
                  f"{_c.get('detail')}")

    # 训练 run 的顺序由设计计划预先给出：种子为区组、区块内随机化，避免
    # “变体主序”把方法效应与时间/硬件漂移混在一起。计划缺失/损坏时不能
    # 静默退回旧顺序并伪装成随机化，必须在开跑前显式报错。
    run_order_plan = design_plan.get("run_order")
    if isinstance(run_order_plan, dict):
        order_problems = validate_run_order(run_order_plan, variants, seeds)
        if order_problems:
            p.error("设计计划的 run_order 非法："
                    + "；".join(order_problems))
        order_cells = [(c["variant"], c["seed"])
                       for c in run_order_plan.get("cells", [])]
    else:
        # 兼容极旧版本计划：只能退回旧的变体主序，并在启动日志中明确标注。
        print("[design] ⚠ 计划缺少 run_order：退回 variant_major 旧顺序，"
              "变体与时间漂移可能混淆")
        run_order_plan = build_run_order(
            variants, seeds, mode="variant_major")
        order_cells = [(c["variant"], c["seed"])
                       for c in run_order_plan.get("cells", [])]
    print(f"[design] run 顺序 mode={run_order_plan.get('mode')} "
          f"seed={run_order_plan.get('order_seed')} "
          f"（{run_order_plan.get('seed_source')}）："
          + " -> ".join(f"{v}@{s}" for v, s in order_cells[:6])
          + (" ..." if len(order_cells) > 6 else ""))
    if run_order_plan.get("mode") == "randomized_blocked" and jobs > 1:
        print("[design] ⚠ jobs>1：随机化的是提交顺序，run 仍会重叠执行；"
              "严格区组解释请用 --jobs 1")

    try:
        data_fp = data_fingerprint(args.data_dir)
        data_digest = data_fp["digest"]
    except FileNotFoundError as e:
        # 数据 bin 尚未生成（如 --skip_train 复用历史产物）时不阻塞流水线，
        # 但在计数指纹处显式标注缺失，避免汇总页假装数据版本已知。
        data_fp, data_digest = None, None
        print(f"[repro] 警告：无法计算数据指纹（{e}）")
    # dirty 工作区：commit 之外还需要 diff 指纹与可重放补丁，否则结果无法
    # 仅凭 commit 复原。补丁写入 results_dir 并纳入 completed 台账的产物哈希。
    worktree = worktree_state()
    if worktree.get("dirty") and worktree.get("diff_bytes"):
        patch = write_worktree_patch(
            os.path.join(results_dir, "worktree.patch"))
        if patch.get("patch_file"):
            patch["patch_file"] = os.path.relpath(patch["patch_file"], ROOT)
            if patch.get("patch_sha256") != worktree.get("diff_sha256"):
                patch["differs_from_manifest"] = True
            worktree["patch"] = patch
    repro = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "argv": sys.argv,
        "config_file": args.config,
        "config_sha256": config_sha256,
        "seeds": seeds,
        # 逐 seed 的声明式审计：各随机源将使用的 seed 与可独立复算的数据
        # 顺序 RNG 初始指纹。实际生效证据在各 run_manifest.seed_report。
        "seed_audit": {str(s): seed_plan(s) for s in seeds},
        "variants": variants,
        # 推理期病灶消融是否真的在本轮执行、覆盖哪些变体。汇总据此过滤
        # 旧 ablation_*.json：--skip_ablation 时不能把上一轮的病灶证据
        # 静默当成本轮结果（旧 repro 无此字段时保持向后兼容的全量发现）。
        "ablation": {
            "executed": bool(not args.skip_ablation and ablate_variants),
            "variants": list(ablate_variants),
            "split": "val",
        },
        "smoke": bool(args.smoke),
        "git": git,
        "worktree": worktree,
        "data": data_fp,
        "runtime": runtime_info(args.device),
        "registration": reg_info,
        # 先验设计计划摘要 + 各阶段是否实际执行：汇总据此区分“计划内但
        # 被 --skip_* 关闭”与“产物缺失”，不把跳过的阶段当成完整分析人群。
        "design": {
            "plan_path": os.path.relpath(design_plan_path, ROOT),
            "plan_digest": design_plan.get("plan_digest"),
            "status": design_plan.get("status"),
            "checks": [
                {"id": c.get("id"), "status": c.get("status")}
                for c in design_plan.get("checks", [])
                if isinstance(c, dict)],
            "stages": {
                "ruler": bool(not args.skip_ruler),
                "analysis": bool(not args.skip_analysis),
                "latency": bool(not args.skip_latency),
                "ablation": bool(not args.skip_ablation and ablate_variants),
            },
            "run_order": {
                "mode": run_order_plan.get("mode"),
                "order_seed": run_order_plan.get("order_seed"),
                "seed_source": run_order_plan.get("seed_source"),
                "n_cells": run_order_plan.get("n_cells"),
                "cells": [[v, s] for v, s in order_cells],
                "jobs": jobs,
                "order_is_dispatch_only": bool(jobs > 1),
            },
        },
        "data_protocol_preflight": data_preflight,
    }
    save_json(repro, os.path.join(results_dir, "repro.json"))
    print(f"[repro] code={git.get('commit_short')} dirty={git.get('dirty')} "
          f"data={data_digest} -> {os.path.join(results_dir, 'repro.json')}")

    # 实验台账：本轮 started 先落盘（崩溃轮次也留痕），结束时追加同一 run_id
    # 的 completed 记录（含最终指标与产物哈希），跨轮次可追溯，见 lega/experiment.py。
    ledger = ledger_path()
    run_id = new_run_id(tag, config_sha256)
    ledger_common = dict(
        run_id=run_id, tag=tag, config_path=args.config,
        config_sha256=config_sha256,
        # 快照而不是引用：started/completed 两条记录必须对应同一份配置
        config=json.loads(json.dumps(cfg)), variants=variants,
        seeds=seeds,
        # started 就落盘种子计划：即使进程随后崩溃、逐 run manifest 尚未
        # 生成，台账仍能回答“准备用哪些 seed、数据顺序 RNG 应是什么”。
        seed_audit=repro["seed_audit"],
        smoke=bool(args.smoke), argv=sys.argv, git=git,
        worktree=worktree,
        data_fingerprint=data_digest, runtime=repro["runtime"],
        out_root=out_root, results_dir=results_dir,
        time_start=repro["time"], registration=reg_info)
    append_record(ledger, build_experiment_record(status="started",
                                                  **ledger_common))
    print(f"[repro] 实验台账 run_id={run_id} -> {ledger}")

    # 冒烟：少量样例，ctx 取配置 block_size（离线字符分词下 needle+问题
    # 模板约 211 token，128 会让 RULER 全部任务与机制分析被跳过）
    smoke_ctx = (min(256, strict_int(cfg.get("block_size", 256),
                                    "block_size")) if args.smoke else 0)

    # 过拟合小批量 sanity check：全量训练前的代码正确性闸门（梯度/损失/标签）。
    # 每个变体跑一次（用首个种子），失败立即中止——连小批量都无法过拟合时不应
    # 继续全量训练。通过不代表数据增强/对齐/映射等问题不存在。
    sanity = {}
    if args.skip_sanity or args.skip_train:
        print("[run_all] sanity check 已跳过（--skip_sanity/--skip_train）")
    else:
        cmds = []
        for v in variants:
            out_json = os.path.join(results_dir, f"sanity_{v}.json")
            cmd = [py, "-m", "lega.sanity", "--config", args.config,
                   "--data_dir", args.data_dir, "--variant", v,
                   "--seed", str(seeds[0]), "--out", out_json,
                   "--device", args.device]
            if args.sanity_steps is not None:
                cmd += ["--steps", str(args.sanity_steps)]
            if args.smoke:
                cmd += ["--smoke"]
            print("+", " ".join(cmd), flush=True)
            cmds.append(cmd)
        sanity_started = time.time()
        try:
            run_many(cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            v = variants[cmds.index(e.cmd)]
            # 失败实验也要完整留痕：aborted 记录内嵌 sanity 判定与结果文件摘要
            record_aborted_sanity(ledger, ledger_common, v, e.returncode,
                                  results_dir)
            print(f"[repro] 实验台账已记录 aborted（run_id={run_id}，"
                  f"sanity 闸门失败）-> {ledger}")
            raise SystemExit(
                f"sanity check 失败（变体 {v}，退出码 {e.returncode}）：无法过拟合小批量，"
                "说明梯度反传/损失计算/标签对齐存在严重问题；请先排查再开全量"
                f"训练。结果见 {os.path.join(results_dir, f'sanity_{v}.json')}。")
        except BaseException as e:
            # 非“子进程非零退出”的异常（OSError/ValueError/KeyboardInterrupt
            # 等）同样会把已 started 的轮次留在台账里；补 aborted 后再抛。
            append_aborted_record(
                ledger, ledger_common, error="sanity_stage_failed",
                stage="sanity", returncode=getattr(e, "returncode", None))
            raise
        # 不能只信子进程退出码：逐个读回本次 sanity 文件，确认它真的
        # present=true 且 passed=true（旧/空/被手改的文件签不出 PASS），
        # 否则在全量训练前就中止，而不是等 train --sanity_file 再报错。
        sanity_verdict, sanity_failures = check_sanity_results(
            variants, results_dir, not_before=sanity_started,
            expected_seed=seeds[0])
        if sanity_failures:
            first = sanity_failures[0]
            v = first["variant"]
            record_aborted_sanity(ledger, ledger_common, v, 0, results_dir,
                                  reason=first["reason"])
            summary = first.get("summary") or {}
            raise SystemExit(
                f"sanity check 退出码为 0，但结果文件未通过复核："
                f"{first.get('path')}（reason={first['reason']}, "
                f"passed={summary.get('passed')!r}, "
                f"failed={summary.get('failed_checks')!r}）。"
                "不能把没有证据的退出码当作闸门通过。")
        sanity = sanity_verdict
        print(f"[run_all] sanity check 全部通过: {', '.join(sanity.keys())}")

    # 每个 (变体, 种子) 是一个独立 run：训练 → 独立评测 → RULER → 机制分析。
    # 四个阶段由 lega.postrun 在同一进程内顺序执行：单 run 的行为与逐条独立
    # 进程完全相同（各阶段 main() 自行 set_seed/加载模型/写自己的产物，评测/
    # RULER/分析只依赖本 run 的 checkpoint）。evaluate 追加 test 访问账本，
    # append_test_access 已用进程间文件锁保证不丢记录。
    chain = []
    chain_cmds = []
    for v, seed in order_cells:
        vdir = _variant_dir(out_root, v, seed, n_seeds)
        suffix = _seed_tag(seed, n_seeds)
        chain.append((v, seed, vdir, suffix))
        cmd = [py, "-m", "lega.postrun",
               "--config", args.config, "--variant", v, "--seed", str(seed),
               "--data_dir", args.data_dir, "--out_dir", vdir,
               "--results_dir", results_dir, "--suffix", suffix,
               "--device", args.device]
        if args.skip_train:
            cmd += ["--skip_train"]
        if args.wandb:
            cmd += ["--wandb"]
        if sanity:
            # 把开训前过拟合闸门的判定与哈希绑定进 run_manifest；
            # 跳过 sanity 时不透传，避免旧结果文件被当作本次证据。
            cmd += ["--sanity_file",
                    os.path.join(results_dir, f"sanity_{v}.json")]
        if args.early_stop_patience is not None:
            cmd += ["--early_stop_patience", str(args.early_stop_patience)]
        if args.worst_n is not None:
            cmd += ["--worst_n", str(args.worst_n)]
        if args.worst_on_test:
            cmd += ["--worst_on_test"]
        if args.allow_repeat_test:
            cmd += ["--allow_repeat_test"]
        if args.smoke:
            cmd += ["--smoke", "--ruler_ctx_lens", str(smoke_ctx),
                    "--ruler_n_samples", "2",
                    "--analysis_n_examples", "2",
                    "--analysis_ctx", str(smoke_ctx)]
        if args.skip_ruler:
            cmd += ["--skip_ruler"]
        if args.skip_analysis or v not in GATED_FOR_ANALYSIS:
            cmd += ["--skip_analysis"]
        print("+", " ".join(cmd), flush=True)
        chain_cmds.append(cmd)
    if chain_cmds:
        try:
            run_many(chain_cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            # 失败可能出现在链的任一阶段（postrun stderr 已标明阶段名与等价
            # 命令）；训练中止最常见的原因是 NaN 防护触发（train.py 会先落盘
            # nan_report.json 再退出），把排查入口直接指给用户
            v0 = chain[chain_cmds.index(e.cmd)][0]
            append_aborted_record(
                ledger, ledger_common, error="run_chain_failed",
                stage="train/eval/ruler/analysis", variant=v0,
                returncode=e.returncode,
                runs=_snapshot_existing_runs(ledger_common) or None)
            print(f"[repro] 实验台账已记录 aborted（run_id={run_id}，"
                  f"run 链失败）-> {ledger}")
            raise SystemExit(
                f"run 链失败（变体 {v0}，退出码 {e.returncode}）：训练/评测/"
                "机制分析任一阶段失败即中止；若为训练 NaN 中止，对应 run 目录"
                "下已生成 nan_report.json（数据→代码→混合精度→梯度/学习率的"
                "系统排查路径；报告含检测点前的整个 log 区间候选批次与区间内"
                "首个非有限损失步 first_bad_step，数据/激活排查针对失败现场而"
                f"非最后一步批次），排查后重跑；命令: {' '.join(e.cmd)}")
        except BaseException as e:
            # 非“子进程非零退出”的异常（OSError/ValueError/KeyboardInterrupt）
            # 不能让已 started 的轮次永久悬空：补 aborted 终态后再抛。
            append_aborted_record(
                ledger, ledger_common, error="run_chain_failed",
                stage="train/eval/ruler/analysis",
                returncode=getattr(e, "returncode", None),
                runs=_snapshot_existing_runs(ledger_common) or None)
            raise

    # 坏案例报告：把每个变体预测错误的样本逐一归因成 Markdown；
    # 错误样本是改进思路的最重要来源，不能只留聚合分数。
    # 解析结果进 preloaded 缓存：build_summary 汇总时使用同一份 payload。
    preloaded = {}
    if not args.skip_ruler:
        for fp in sorted(glob.glob(os.path.join(results_dir, "ruler_*.json"))):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                ana = process_ruler_json(fp, payload=payload)
            except (OSError, ValueError, KeyError) as e:
                print(f"[badcase] 跳过 {fp}: {e}")
                continue
            preloaded[fp] = payload
            if ana["n_badcases"]:
                top = ", ".join(f"{BADCASE_CN.get(k, k)}×{n}"
                                for k, n in sorted(ana["category_counts"].items(),
                                                   key=lambda kv: -kv[1])[:3])
                print(f"[badcase] {os.path.basename(fp)}: "
                      f"{ana['n_badcases']} 条错误样本 —— {top}"
                      f" -> badcases_*.md")

    # 语言建模坏案例：把 PPL 最差窗口（next-token 预测错误）逐一归因成
    # Markdown，与 RULER 坏案例同等对待——PPL 平均分看不出模型错在哪句话
    for fp in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ana = process_eval_json(fp, payload=payload)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
        preloaded[fp] = payload
        if not ana["n_windows"]:
            continue
        top = ", ".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{n}"
                        for k, n in sorted(ana["category_counts"].items(),
                                           key=lambda kv: -kv[1])[:3])
        print(f"[badcase] {os.path.basename(fp)}: 最差窗口 {ana['n_windows']} 个"
              f" —— {top} -> worstcases_*.md")

    with abort_ledger_on_failure(ledger, ledger_common, lambda: post_stage):
        # 推理期因果病灶消融：必须在汇总前完成，结果由 build_summary 读取。
        post_stage = "ablation"
        if not args.skip_ablation and ablate_variants:
            ablate_cmds = []
            for v, seed, vdir, suffix in chain:
                if v not in ablate_variants:
                    continue
                ckpt = os.path.join(vdir, "model_best.pt")
                if not os.path.exists(ckpt):
                    ckpt = os.path.join(vdir, "model.pt")
                if not os.path.exists(ckpt):
                    print(f"[ablate] 跳过 {v}/seed{seed}：找不到 checkpoint")
                    continue
                out = os.path.join(results_dir, f"ablation_{v}{suffix}.json")
                cmd = [py, "-m", "lega.ablate", "--ckpt", ckpt,
                       "--data_dir", args.data_dir, "--out", out,
                       "--seed", str(seed), "--device", args.device]
                if args.smoke:
                    # 冒烟时训练配置已被覆盖为小 ctx/少窗口；显式对齐，
                    # 避免病灶分析退回默认 50 窗口拖慢冒烟。
                    cmd += ["--ctx", str(smoke_ctx), "--eval_iters", "2"]
                print("+", " ".join(cmd), flush=True)
                ablate_cmds.append(cmd)
            if ablate_cmds:
                run_many(ablate_cmds, jobs, _jobs_env(jobs))

        post_stage = "latency"
        if not args.skip_latency:
            # 显式给出基准的变体清单：lega.latency 的默认值不含消融变体，
            # 不传会让汇总表的消融变体对应列恒为 "-"。多种子时按各变体首个
            # 种子的 checkpoint 测量（该指标由架构决定，与训练种子无关）。
            lat_variants = ",".join(dict.fromkeys(["standard", "standard_sdpa"] + variants))
            run([py, "-m", "lega.latency", "--config", args.config,
                 "--data_dir", args.data_dir, "--ckpt_dir", out_root,
                 "--out", os.path.join(results_dir, "latency.json"),
                 "--variants", lat_variants, "--seed", str(seeds[0]),
                 "--device", args.device] + (["--smoke"] if args.smoke else []))

        post_stage = "plots"
        if not args.skip_plots:
            run([py, "-m", "lega.plots", "--results_dir", results_dir,
                 "--train_root", out_root])

        # 训练曲线诊断：曲线体检（过拟合/欠拟合/发散/NaN）在汇总前完成，
        # 结果落盘 curve_diagnosis.json 并进入 summary.md
        post_stage = "curve_diagnosis"
        diagnose_training_curves(chain, args.data_dir, results_dir)

        # 数据协议防泄漏报告（prepare_bins 生成）：在所有评测结束后重新读取，连同
        # 实时测试集访问账本复制进 results，供汇总与审计（测试账本已在本轮更新）。
        post_stage = "data_report"
        data_report = load_data_report(args.data_dir)
        if data_report is None:
            print("[repro] 警告：数据目录缺少可解析的 data_report.json"
                  "（旧数据或未走统一入口），汇总将标注防泄漏自检缺失")
        if data_report is not None:
            data_report["test_access"] = summarize_test_access(
                read_test_access_log(args.data_dir))
            save_json(data_report, os.path.join(results_dir, "data_report.json"))
            print(f"[repro] 防泄漏报告 -> {os.path.join(results_dir, 'data_report.json')}")
        else:
            print("[repro] 警告：数据目录缺少 data_report.json（旧数据或未走统一入口），"
                  "汇总将标注防泄漏自检缺失")

        post_stage = "audit"
        audit = verify_controlled_setup(out_root, variants, seeds, n_seeds,
                                        expected_data_digest=data_digest)
        post_stage = "summary"
        build_summary(results_dir, audit=audit, data_report=data_report,
                      preloaded=preloaded)

        # 收尾台账：最终指标（逐 run 的 eval.json）+ 全部结果文件的 SHA-256。
        # 曲线/结果文件后续被覆盖或手改，都能通过哈希与当时记录比对发现。
        # evaluate 阶段已对本 run 的 model_best.pt 做过全量哈希（写在
        # results_dir/eval_<variant><suffix>.json 的 ckpt/ckpt_sha256）；
        # 路径匹配时台账使用该值，否则（未评测/旧评测文件）重算。
        post_stage = "ledger_finalize"
        known_ckpt = {}
        for v, seed, _vdir, suffix in chain:
            try:
                with open(os.path.join(results_dir, f"eval_{v}{suffix}.json"),
                          "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError):
                rec = None
            if isinstance(rec, dict) and rec.get("ckpt_sha256") and rec.get("ckpt"):
                known_ckpt[(v, seed)] = (str(rec["ckpt_sha256"]), str(rec["ckpt"]))
        run_snapshots = []
        for v, seed, vdir, suffix in chain:
            sha, ckpt = known_ckpt.get((v, seed), (None, None))
            run_snapshots.append(snapshot_train_run(
                v, seed, vdir, known_ckpt_sha256=sha, known_ckpt_path=ckpt,
                heldout_path=os.path.join(results_dir, f"eval_{v}{suffix}.json")))
        # 收尾身份核对：长实验期间代码/数据若被改动，开始时记录的 commit/指纹
        # 已不足以代表整轮实验。把结束快照与漂移结论一并写入 completed 记录，
        # 让“全程同一版本”成为可核对的事实而不是假设。
        end_git = git_info()
        end_worktree = worktree_state()
        try:
            end_data_digest = data_fingerprint(args.data_dir)["digest"]
        except FileNotFoundError:
            end_data_digest = None
        drift = provenance_drift(
            {"git": git, "worktree": worktree,
             "data_fingerprint": data_digest},
            {"git": end_git, "worktree": end_worktree,
             "data_fingerprint": end_data_digest})
        tracking_end = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": end_git,
            "worktree": end_worktree,
            "data_fingerprint": end_data_digest,
            "drift": drift,
        }
        # 把逐 run 的**实际**种子审计回填到 repro.json：启动时只有 seed_plan
        # 声明，这里补上每个 (变体, 种子) run_manifest.seed_report 的真实 RNG
        # 状态指纹。repro.json 因此即使脱离 out/ 目录也能自证种子覆盖；失败
        # 轮次不会走到这里，仍保留启动期的计划与台账 aborted 记录。
        repro["seed_reports"] = {
            f"{v}/seed{seed}": ((snap.get("provenance") or {}).get("seed_report"))
            for (v, seed, _vdir, _suffix), snap
            in zip(chain, run_snapshots)}
        repro["tracking_end"] = tracking_end
        repro["status"] = ("completed_with_drift" if drift["changed"]
                           else "completed")
        save_json(repro, os.path.join(results_dir, "repro.json"))
        print(f"[repro] 逐 run 种子审计已回填 -> "
              f"{os.path.join(results_dir, 'repro.json')}")
        # repro.json 必须在最终哈希前写入：completed 台账的产物哈希要覆盖
        # 回填后的版本，否则重算会显示“repro 被改写”。
        result_files = [p for p in glob.glob(os.path.join(results_dir, "*"))
                        if os.path.isfile(p)]
        if drift["changed"]:
            print("[repro] ⚠ 实验期间代码或数据身份发生变化："
                  + "; ".join(f"{m['field']} {m['start']} -> {m['end']}"
                              for m in drift["mismatches"])
                  + "；该轮 completed 记录已显式标注，结果不得声称对应单一版本")
        append_record(ledger, build_experiment_record(
            status="completed", time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
            sanity={"executed": bool(sanity), "verdict": sanity},
            runs=run_snapshots,
            artifacts=collect_artifacts(result_files, rel_to=ROOT),
            extra={"tracking_end": tracking_end},
            **ledger_common))
        print(f"[repro] 实验台账已记录 completed（run_id={run_id}）-> {ledger}")
    print(f"\n[run_all] 全部完成；结果目录: {results_dir}")
