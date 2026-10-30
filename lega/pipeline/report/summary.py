"""汇总统计与 Markdown/JSON 报告生成（第一部分：数据聚合与主表）。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def _summary_part_a(results_dir: str, audit: dict = None,
                  data_report: dict = None,
                  preloaded: dict = None) -> dict:
    """汇总 PPL / RULER / 参数量等指标为 Markdown + CSV + JSON。

    多种子口径：每个变体的每个指标先按种子聚合为 Mean ± Std；变体间比较使用
    同一批种子上的配对 t 检验（同种子共享初始化与数据顺序，配对消除公共方差）。
    消融矩阵显式列出全因子 8 个组合的覆盖情况，缺失组合会被标注。
    data_report（可选）来自数据目录的防泄漏自检报告，汇总为独立小节。
    preloaded（可选）为 {results 下结果文件路径: 已解析 JSON} 缓存：坏案例
    阶段已解析过的 eval/ruler 文件在此共用同一份结果。
    """
    audit = audit if isinstance(audit, dict) else None
    data_report = data_report if isinstance(data_report, dict) else None

    def load(p):
        if preloaded and p in preloaded:
            return preloaded[p]
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            # 坏/半写入的结果文件按“未运行”处理，不能毁掉整份汇总。
            return None

    # --- “本次 run”范围限定：results/<tag> 会被同一 tag 的多次运行复用
    # （改种子/变体集、--skip_* 等），聚合按 repro.json 记录的本次
    # variants/seeds 过滤，不同实验的观测不混入同一均值/配对检验；
    # 没有 repro.json（直接调用/手工构建目录）时保持全量发现。
    repro_doc = load(os.path.join(results_dir, "repro.json"))
    # 合法 JSON 但非对象（数组/标量）的 repro.json 按“无 repro”处理，
    # 不能让汇总在 .get 上崩溃。
    repro = repro_doc if isinstance(repro_doc, dict) else {}
    scope_variants = repro.get("variants") if isinstance(
        repro.get("variants"), list) else None
    scope_seeds = repro.get("seeds") if isinstance(
        repro.get("seeds"), list) else None
    # 同一 results/<tag> 会被多次运行复用。仅按 (variant, seed) 过滤时，
    # --skip_eval/--skip_ruler 等会让上一轮的同名产物被当成本轮观测；新产物
    # 内嵌 pipeline run_id 后，本轮有 run_id 就要求产物同源（旧产物缺该字段
    # 时明确视为“无法证明属于本轮”而排除，避免默认信任陈旧文件）。
    scope_run_id = repro.get("run_id")
    scope_run_id = scope_run_id if isinstance(scope_run_id, str) and scope_run_id else None
    scope_data_digest = (repro.get("data").get("digest")
                         if isinstance(repro.get("data"), dict) else None)
    scope_data_digest = (scope_data_digest
                         if isinstance(scope_data_digest, str)
                         and scope_data_digest else None)

    def _artifact_run_id(d):
        rid = d.get("pipeline_run_id")
        if isinstance(rid, str) and rid:
            return rid
        prov = d.get("provenance")
        if isinstance(prov, dict):
            rid = prov.get("pipeline_run_id")
            if isinstance(rid, str) and rid:
                return rid
        return None

    def _artifact_data_digest(d):
        # eval/ruler/analysis 用平铺字段；ablation 把 load_provenance 放在
        # provenance 下，两种结构都要覆盖。
        for key in ("data_fingerprint", "eval_data_fingerprint"):
            val = d.get(key)
            if isinstance(val, str) and val:
                return val
        prov = d.get("provenance")
        if isinstance(prov, dict):
            for key in ("data_fingerprint", "eval_data_fingerprint"):
                val = prov.get(key)
                if isinstance(val, str) and val:
                    return val
        return None

    def in_run_scope(fp, d):
        # 变体名必须是可哈希字符串：外部 JSON 的数组/对象既不能作为 runs
        # 的键，也不是合法的变体标识，按损坏产物跳过。
        if not isinstance(d, dict) or not isinstance(d.get("variant"), str):
            return False
        if scope_variants is not None and d["variant"] not in scope_variants:
            return False
        if scope_seeds is not None and _seed_key(fp, d) not in scope_seeds:
            return False
        if scope_run_id is not None:
            if _artifact_run_id(d) != scope_run_id:
                return False
        if scope_data_digest is not None:
            actual = _artifact_data_digest(d)
            if actual is not None and actual != scope_data_digest:
                return False
        return True

    # --- 发现所有 run：（变体, 种子） -> {eval, ruler, analysis}
    runs = {}

    def slot(v, s):
        return runs.setdefault(v, {}).setdefault(
            s, {"eval": None, "ruler": None, "analysis": None})

    for fp in _result_files(results_dir, "eval", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["eval"] = d
    for fp in _result_files(results_dir, "ruler", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["ruler"] = d
    for fp in _result_files(results_dir, "analysis", scope_seeds):
        if "records" in os.path.basename(fp):
            continue
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["analysis"] = d

    # 推理期因果病灶消融（每 (变体, 种子) 一个 ablation_*.json）；单独存放，
    # 不混入 runs 槽位——它不是训练变体，也不参与受控变量审计。
    ablations = {}
    ablation_scope = repro.get("ablation")
    if isinstance(ablation_scope, dict):
        # 本轮显式关闭消融时，绝不复用旧 ablation 文件冒充本轮证据。
        # executed 必须是真正的 bool true：字符串 "false"/1 经 truthiness
        # 会打开复用，把上一轮的病灶结果算进本轮结论。
        allow_ablation = ablation_scope.get("executed") is True
        ablation_variants = (ablation_scope.get("variants")
                             if isinstance(ablation_scope.get("variants"), list)
                             else [])
    else:
        # 旧 repro 没有该字段：保持向后兼容的全量发现。
        allow_ablation = True
        ablation_variants = None
    for fp in _result_files(results_dir, "ablation", scope_seeds):
        d = load(fp)
        if not in_run_scope(fp, d):
            continue
        if not allow_ablation:
            continue
        if (ablation_variants is not None
                and d["variant"] not in ablation_variants):
            continue
        ablations.setdefault(d["variant"], {})[_seed_key(fp, d)] = d

    variants = sorted(v for v in runs if v in RUNNABLE_VARIANTS)
    eval_splits = set()
    invalid_eval_split = False
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            if s.get("eval"):
                try:
                    eval_splits.add(s["eval"].get("split", "val"))
                except TypeError:
                    invalid_eval_split = True

    # 汇总运行时指标时必须使用所有变体共有的同一上下文档，不允许各取自己的最大 ctx。
    lat = load(os.path.join(results_dir, "latency.json"))
    if not isinstance(lat, dict):
        lat = {}
    if scope_run_id is not None and lat:
        lat_meta = lat.get("meta")
        lat_meta = lat_meta if isinstance(lat_meta, dict) else {}
        if lat_meta.get("pipeline_run_id") != scope_run_id:
            lat = {}
    lat_variants = lat.get("variants")
    lat_variants = lat_variants if isinstance(lat_variants, dict) else {}
    lat_entries = {v: (lat_variants.get(v)
                       if isinstance(lat_variants.get(v), dict) else None)
                   for v in variants}

    def _lat_metric_at(e, key, ctx):
        if not isinstance(e, dict) or ctx is None:
            return None
        by_ctx = e.get(key)
        if not isinstance(by_ctx, dict):
            return None
        return _num_or_none(by_ctx.get(str(ctx)))

    def _prefill_at(e, ctx):
        if not isinstance(e, dict) or ctx is None:
            return None
        by_ctx = e.get("prefill_ms")
        if not isinstance(by_ctx, dict):
            return None
        cell = by_ctx.get(str(ctx))
        return _num_or_none(cell.get("1")) if isinstance(cell, dict) else None

    common_ctxs = None
    for v in variants:
        e = lat_entries.get(v)
        if not isinstance(e, dict):
            continue
        prefill = e.get("prefill_ms")
        decode = e.get("decode_ms_per_tok")
        if not isinstance(prefill, dict) or not isinstance(decode, dict):
            continue
        usable = set()
        for c, by_bs in prefill.items():
            if (not isinstance(c, str)
                    or not isinstance(by_bs, dict)
                    or _num_or_none(by_bs.get("1")) is None
                    or _num_or_none(decode.get(c)) is None):
                continue
            # int("1_0")==10、int("040")==40，但这些非规范 key 在
            # _lat_metric_at 里按 str(ctx) 查找时并不存在；若拿来求公共
            # ctx，会选中一个没有真实量测的“最大上下文”。
            ctx = _integral_value(c)
            if ctx is None or c != str(ctx):
                continue
            usable.add(ctx)
        common_ctxs = usable if common_ctxs is None else (common_ctxs & usable)
    common_ctx = max(common_ctxs) if common_ctxs else None

    def _clock_extractor(key):
        return lambda d: d.get(key) if isinstance(d, dict) else None

    def _eval_window_fraction(d):
        if not isinstance(d, dict):
            return None
        frac = _num_or_none(d.get("eval_window_fraction"))
        if frac is not None:
            return frac
        used = _num_or_none(d.get("eval_iters"))
        total = _num_or_none(d.get("val_windows"))
        if used is None or not total:
            return None
        return used / total

    metric_specs = [
        ("val_ppl", lambda m: _stat_series(m, "eval", _ppl_of)),
        ("gate_mean", lambda m: _stat_series(m, "eval", _clock_extractor("gate_mean"))),
        ("val_loss", lambda m: _stat_series(m, "eval", _clock_extractor("val_loss"))),
        # 评估不确定性（单 run 内窗口抽样）与训练数据覆盖率：这两个量决定
        # “数值差异是否值得解释”，必须与点估计一起进入 summary.json/csv。
        ("ppl_ci_rel_pct", lambda m: _stat_series(m, "eval", _ppl_ci_rel_pct)),
        ("epochs_seen", lambda m: _stat_series(m, "eval", _eval_scalar("epochs_seen"))),
        # model_best.pt 的选型步：同一最大预算下，各变体可能在差异很大的步
        # 被 val 选中。记录后才能审计“同预算”之外的选型差异。
        ("ckpt_step", lambda m: _stat_series(m, "eval", _eval_scalar("ckpt_step"))),
        ("selection_step_fraction", lambda m: _stat_series(
            m, "eval", _eval_scalar("selection_step_fraction"))),
        ("eval_tokens", lambda m: _stat_series(
            m, "eval", _eval_scalar("val_tokens_evaluated"))),
        ("eval_windows", lambda m: _stat_series(
            m, "eval", _eval_scalar("val_windows"))),
        ("eval_window_fraction", lambda m: _stat_series(
            m, "eval", _eval_window_fraction)),
        ("ruler_avg", lambda m: _stat_series(m, "ruler", _ruler_avg_of)),
        ("ruler_avg_micro", lambda m: _stat_series(
            m, "ruler", _ruler_avg_micro)),
        ("ruler_min_n", lambda m: _stat_series(m, "ruler", _ruler_min_cell_n)),
        # NIAH 位置分层：聚合 EM 会把早期高分与中部塌陷互相抵消；分层
        # 指标进入 summary.json/csv，Markdown 汇总另设小节（旧产物缺失）。
        ("ruler_depth_early", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("early"))),
        ("ruler_depth_middle", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("middle"))),
        ("ruler_depth_late", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("late"))),
        ("ruler_depth_early_minus_late", lambda m: _stat_series(
            m, "ruler", _ruler_depth_gap)),
    ]
    ruler_tasks = RULER_TASKS
    for task in ruler_tasks:
        metric_specs.append(
            (f"ruler_{task}",
             (lambda t: lambda m: _stat_series(
                 m, "ruler", lambda d: _ruler_task_means(d).get(t)))(task)))
    # fwe 召回率仅作诊断单列：recall 不惩罚“把上下文词全部列出”的灌水
    # 策略，不能进入预注册的 RULER 平均（主口径为严格集合 EM）。
    metric_specs.append(
        ("ruler_fwe_recall",
         lambda m: _stat_series(m, "ruler", _ruler_fwe_recall)))
    # fwe precision/F1 同为诊断：precision 才惩罚“多答”，与 recall 并排
    # 才能区分“没找到高频词”与“找对但灌水/漏答”；均不进主平均。
    metric_specs.append(
        ("ruler_fwe_precision",
         lambda m: _stat_series(
             m, "ruler", lambda d: _ruler_task_means(d, "precision").get("fwe"))))
    metric_specs.append(
        ("ruler_fwe_f1",
         lambda m: _stat_series(
             m, "ruler", lambda d: _ruler_task_means(d, "f1").get("fwe"))))
    for key in ("auroc_logit", "auroc_qterm", "misleading_rate",
                "misleading_given_conc", "p_hit_given_conc",
                "auprc_logit", "auprc_qterm", "hit_base_rate",
                "brier_g", "ece_g", "ece_g_adaptive",
                "brier_g_constant", "brier_g_skill",
                "brier_g_reliability", "brier_g_resolution",
                "brier_g_uncertainty", "brier_g_within_score",
                "brier_g_within_cov", "brier_g_binning"):
        metric_specs.append(
            (key, (lambda k: lambda m: _stat_series(
                m, "analysis", _clock_extractor(k)))(key)))
    # 门控决策点 F1（g>0.5）：显式只取 macro 进汇总表（不平衡下对稀有类
    # 最诚实的口径）；micro/weighted 与逐类明细在 analysis_*.json 中
    # 的 gate_f1 字段并列完整保留，避免“只写 F1 不注明平均口径”。
    metric_specs.append(
        ("gate_f1_macro", lambda m: _stat_series(
            m, "analysis",
            lambda d: _as_dict(d.get("gate_f1")).get("macro"))))
    # 恒写（全判 hit）规则的 macro F1：门控决策的“稻草人”对照。
    # 不平衡下门控 F1 若不明显高于该固定规则，就没有决策价值。
    metric_specs.append(
        ("gate_f1_macro_always_write", lambda m: _stat_series(
            m, "analysis",
            lambda d: _as_dict(
                _as_dict(d.get("gate_f1_trivial")).get("always_write")
            ).get("macro"))))
    # 线性探针（有标签监督的简单基线）OOF 指标：进入 summary.json/csv，
    # 供“门控 vs 最简线性模型”的补充性对照审计（渲染见基线合理性小节）。
    metric_specs.append(
        ("probe_auprc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auprc"))))
    metric_specs.append(
        ("probe_auroc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auroc"))))
    metric_specs.append(
        ("probe_f1_fixed", lambda m: _stat_series(
            m, "analysis",
            lambda d: _probe_oof(d, "f1", "macro"))))
    metric_specs.append(
        ("probe_f1_tuned", lambda m: _stat_series(
            m, "analysis",
            lambda d: _probe_oof(d, "f1_tuned", "macro"))))
    metric_specs.append(
        ("probe_brier", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "brier"))))
    metric_specs.append(
        ("probe_ece", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "ece"))))

    rows = []
    for v in variants:
        m = _variant_seed_map(runs, v)
        row = {"variant": v, "variant_cn": VARIANT_CN.get(v, v),
               "n_seeds": len(m)}
        for key, fn in metric_specs:
            st = fn(m)
            if st:
                row[key] = st["mean"]
                row[f"{key}_std"] = st["std"]
                row[f"{key}_n"] = st["n"]
            else:
                row[key] = row[f"{key}_std"] = row[f"{key}_n"] = None
        ev_example = next((s["eval"] for s in m.values() if s.get("eval")), None)
        row["params"] = (ev_example or {}).get("n_params")
        row["gate_extra"] = (ev_example or {}).get("n_gate_extra", 0)
        row["n_gate_per_head"] = (ev_example or {}).get("n_gate_per_head")
        row["n_heads"] = (ev_example or {}).get("n_heads")
        row["n_layers"] = (ev_example or {}).get("n_layers")
        row["d_head"] = (ev_example or {}).get("d_head")
        row["val_ppl_clamped"] = any(
            (s.get("eval") or {}).get("val_ppl_clamped") for s in m.values())
        # 训练/评估覆盖的“最坏种子”兜底值：均值可能掩盖某个种子只跑了
        # 极少数据或只评测了极少窗口的情况，汇总警告以最小值触发。
        evals_v = [s["eval"] for s in m.values() if s.get("eval")]
        eps = [x for x in (_num_or_none(e.get("epochs_seen"))
                           for e in evals_v) if x is not None]
        row["epochs_seen_min"] = min(eps) if eps else None
        toks = [x for x in (_num_or_none(e.get("val_tokens_evaluated"))
                            for e in evals_v) if x is not None]
        row["eval_tokens_min"] = min(toks) if toks else None
        row["val_loss_ci95"] = (ev_example or {}).get("val_loss_ci95")
        row["val_ppl_ci95"] = (ev_example or {}).get("val_ppl_ci95")
        rns = [_ruler_min_cell_n(s.get("ruler")) for s in m.values()
               if s.get("ruler") is not None]
        rns = [n for n in rns if n is not None]
        row["ruler_min_n_min"] = min(rns) if rns else None
        ruler_payloads = [s.get("ruler") for s in m.values()
                          if isinstance(s.get("ruler"), dict)]
        raci = _ruler_macro_absolute_ci(ruler_payloads)
        row["ruler_avg_ci95"] = raci["ci95"] if raci else None
        row["ruler_avg_ci_n_seeds"] = raci["n_seeds"] if raci else None
        # 运行时指标（取全部变体的最大公共 ctx、batch=1）；不受种子影响，测一次
        e = lat_entries.get(v)
        if e and common_ctx is not None:
            row["prefill_ms"] = _prefill_at(e, common_ctx)
            row["decode_ms_per_tok"] = _lat_metric_at(
                e, "decode_ms_per_tok", common_ctx)
            row["decode_prefill_ctx"] = _lat_metric_at(
                e, "decode_prefill_ctx", common_ctx)
            row["latency_ctx"] = common_ctx
        rows.append(row)

    # 运行时基线：优先取 standard_sdpa，缺失时退回显式 standard。
    lat_all = lat_variants
    base_ms, base_name = None, None
    for cand in ("standard_sdpa", "standard"):
        if _prefill_at(lat_all.get(cand), common_ctx) is not None:
            base_ms, base_name = _prefill_at(lat_all.get(cand), common_ctx), cand
            break
    for r in rows:
        if (base_ms and base_ms > 0 and r.get("prefill_ms") is not None
                and r["prefill_ms"] > 0
                and r["variant"] not in ("standard", "standard_sdpa")):
            # 两端有限但比值不可表示时按缺失处理，不能把 inf 写进汇总行。
            rel = _rel_pct(base_ms, r["prefill_ms"])
            if rel is not None:
                r["prefill_rel_pct"] = round(rel, 2)

    # --- 配对显著性检验（同种子）：PPL 与 RULER 平均
    # 单因素消融（去一法 + 全因子增量）与整体对比分开构造；B-only/C-only
    # 不再与 Full 配对（那会同时改变 A+C / A+B，不能归因给单个组件）。
    registered = _as_dict(load_registered_hypotheses())
    sf_audit = single_factor_audit()
    if not sf_audit["ok"]:
        print("[run_all] ⚠ 单因素消融审计未通过："
              + "；".join(sf_audit["problems"]))
    pairs = build_comparison_pairs(runs, registered)
    tests = []
    tests += _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    # PPL 是比值尺度：算术均值差会被高 PPL 种子主导。log-PPL 配对给出
    # 几何均值比（exp(Δ)），作为独立敏感性家族；预注册主判定仍是原始 PPL。
    log_tests = _paired_comparisons(
        runs, pairs, "PPL(log)", lambda slot: _log_ppl_of(slot.get("eval")),
        lower_better=True)
    for t in log_tests:
        md = t.get("mean_diff")
        ci = t.get("mean_diff_ci95")
        # log-PPL 差是两个有限 log 值之差，但 exp(Δ) 仍可能超出 float 上限
        # （例如候选 PPL=1e300、参照 PPL=1e-300）。比值不可表示时记为缺失，
        # 不能抛 OverflowError 让整份汇总失败；原始 log 差与 p 值仍保留。
        ratio_overflow = False
        t["ppl_ratio"] = None
        if md is not None and math.isfinite(float(md)):
            try:
                t["ppl_ratio"] = math.exp(float(md))
            except OverflowError:
                ratio_overflow = True
        t["ppl_ratio_ci95"] = None
        if (isinstance(ci, (list, tuple)) and len(ci) == 2
                and all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in ci)):
            try:
                t["ppl_ratio_ci95"] = [math.exp(float(ci[0])),
                                       math.exp(float(ci[1]))]
            except OverflowError:
                ratio_overflow = True
        t["ppl_ratio_overflow"] = bool(ratio_overflow)
        t["ppl_delta_pct"] = None
        if t["ppl_ratio"] is not None:
            delta_pct = (t["ppl_ratio"] - 1.0) * 100.0
            # exp(Δ) 本身可表示时，再乘 100 仍可能溢出；派生百分比按缺失
            # 处理，不能把 inf 写进 significance.json。
            if math.isfinite(delta_pct):
                t["ppl_delta_pct"] = delta_pct
        t["exploratory"] = True
    tests += log_tests
    tests += _paired_comparisons(
        runs, pairs, "RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
        lower_better=False)
    # RULER 逐任务分解（探索性）：综合平均可能让方向相反的任务互相抵消。
    # 每个任务单独作为指标家族做同种子配对 + Holm；综合平均仍是预注册
    # 主判定，逐任务结果只用于定位“效果来自哪个任务/是否被平均掩盖”。
    for task in RULER_TASKS:
        task_tests = _paired_comparisons(
            runs, pairs, f"RULER:{task}",
            (lambda t: lambda slot: _ruler_task_means(
                slot.get("ruler")).get(t))(task),
            lower_better=False)
        for t in task_tests:
            t["ruler_task"] = task
            t["exploratory"] = True
        tests += task_tests
    # 机制主张的正式检验族：AUROC(logit) − AUROC(qterm) 同种子配对差，
    # 以及不平衡口径下的 PR-AUC 配对差（两族分别 Holm 校正）
    tests += _auroc_diff_tests(runs)
    tests += _ap_diff_tests(runs)
    tests += _factorial_interaction_tests(runs)
    tests += _factorial_shapley_tests(runs)
    tests = _apply_holm(tests)
    # RULER 上下文长度剂量-反应（探索性）：只报跨长度平均会把“短涨长跌”
    # 与“全长度小涨”混成同一个数。独立 Holm 家族，不进入主检验表，也不
    # 改变预注册假设的确认性判定。
    length_analysis = _ruler_length_analysis(runs, pairs)
    # 把 vs standard 的长度剂量-反应摘要并入 summary.json/csv 的变体行，
    # 使“跨长度平均是否掩盖长上下文效应”可被程序消费（完整逐 ctx/逐种子
    # 明细仍在 significance.json）。没有配对（standard 本身/未跑 RULER）时
    # 保持 None，不伪造趋势。
    length_by_pair = {(p.get("candidate"), p.get("reference")): p
                      for p in length_analysis.get("pairs", [])
                      if isinstance(p, dict)}

    def _length_pp(v):
        f = _finite_number(v)
        if f is None:
            return None
        scaled = f * 100.0
        # 外部 significance.json 的有限值乘 100 仍可能溢出；None 比 JSON
        # Infinity（严格 JSON 无法解析）安全。
        return round(scaled, 4) if math.isfinite(scaled) else None

    for row in rows:
        pair = length_by_pair.get((row.get("variant"), "standard"))
        if not isinstance(pair, dict):
            continue
        per_ctx = pair.get("per_context") or []
        slope = pair.get("slope_per_doubling") or {}
        span = pair.get("long_minus_short") or {}
        row["ruler_length_contexts"] = pair.get("contexts")
        row["ruler_length_overall_delta_pp"] = _length_pp(
            pair.get("overall_mean_diff"))
        row["ruler_length_longest_delta_pp"] = _length_pp(
            per_ctx[-1].get("mean_diff") if per_ctx else None)
        row["ruler_length_long_minus_short_pp"] = _length_pp(
            span.get("mean_diff"))
        row["ruler_length_slope_pp_per_doubling"] = _length_pp(
            slope.get("mean_diff"))
        row["ruler_length_slope_p_adj"] = slope.get("p_adj")
        row["ruler_length_span_p_adj"] = span.get("p_adj")
        row["ruler_length_sign_flip"] = bool(
            pair.get("overall_vs_longest_sign_flip")
            or pair.get("overall_vs_slope_sign_flip")
            or pair.get("overall_vs_span_sign_flip"))
        row["ruler_length_masking"] = bool(pair.get("long_context_masking"))
        row["ruler_length_loo_sign_stable"] = pair.get(
            "slope_loo_sign_stable")
        row["ruler_length_balanced"] = pair.get("balanced_contexts")
    shapley_tests = [t for t in tests if t.get("shapley_factor")]
    # 训练预算敏感性（探索性）：只比较已有 train_log.jsonl 中的 val_loss
    # 轨迹，检查终点结论是否在训练预算之间发生方向翻转。它不重新选
    # checkpoint，也不改变任何终点/预注册判定；缺日志时显式不可用。
    budget_curves = curves_from_runs(runs)
    budget_sensitivity = analyze_budget_sensitivity(
        budget_curves, pairs, alpha=ALPHA, min_seeds=MIN_SEEDS)
    # 参数化稳健性（探索性）：比较归一化 c 与原始 log max_p 两套完整
    # 2^3 网格上的单因子边与 Shapley 归因。只有显式运行原始对照时才有
    # 内容；缺格/协议不一致会返回可机读原因，不伪造“无差异”。
    reparameterization = _reparameterization_analysis(runs)
    # 推理期因果病灶消融：与训练期消融回答不同的因果问题（见 lega.ablate
    # 模块文档）。独立指标家族、独立 Holm，不混入上面的变体间配对表。
    causal_tests = _ablation_lesion_records(ablations)
    # 活动匹配对照：原始病灶 − 匹配病灶。单独 Holm 家族，回答“原始 ΔNLL
    # 有多少可由平均门控开度变化解释”，不混入原始病灶因果检验。
    matched_ablation_contrasts = _matched_ablation_contrasts(ablations)
    # 同一 checkpoint 内的 2^3 干预 → 推理期 Shapley；与训练期 Shapley 并列，
    # 用于检验“重新训练后的组件结构”是否也存在于最终前向。探索性证据。
    inference_shapley = _inference_shapley_tests(ablations)
    train_phi = {}
    for t in shapley_tests:
        key = _test_label_key(t.get("shapley_factor"))
        if key is not None and t.get("shapley_metric") == "PPL":
            train_phi[key] = _finite_number(t.get("mean_diff"))
    for t in inference_shapley:
        tr = train_phi.get(_test_label_key(t.get("shapley_factor")))
        ph = _finite_number(t.get("mean_diff"))
        t["training_phi_mean"] = tr
        t["training_inference_sign_agree"] = (
            None if tr is None or ph is None else bool(
                (tr > 0) == (ph > 0) or (tr == 0 and ph == 0)))
    # 推理期交互：与训练期 PPL交互 同尺度并列，检查“训练期交互结构是否在
    # 最终前向复现”。只作探索性证据，不进入确认性判定。
    inference_interactions = _inference_interaction_tests(ablations)
    train_inter = {}
    for t in tests:
        key = _test_label_key(t.get("interaction"))
        if key is not None and t.get("metric") == "PPL交互":
            train_inter[key] = _finite_number(t.get("mean_diff"))
    for t in inference_interactions:
        tr = train_inter.get(_test_label_key(t.get("interaction")))
        ph = _finite_number(t.get("mean_diff"))
        t["training_interaction_mean"] = tr
        t["training_inference_sign_agree"] = (
            None if tr is None or ph is None else bool(
                (tr > 0) == (ph > 0) or (tr == 0 and ph == 0)))
    # 参数匹配安慰剂（lega vs 同参数/同 FLOPs、但证据为噪声的 lega_placebo）。
    # 只有显式运行时才有这条对比；未运行时汇总页显式提示，不默认“容量已排除”。
    param_control_test = next(
        (t for t in tests
         if t.get("metric") == "PPL"
         and t.get("candidate") == PARAM_CONTROL_PAIR[0]
         and t.get("reference") == PARAM_CONTROL_PAIR[1]), None)
    if param_control_test is not None:
        param_control_test["capacity_control"] = True
        param_control_test["exploratory"] = True
    capacity_only_test = next(
        (t for t in tests
         if t.get("metric") == "PPL"
         and t.get("candidate") == PARAM_CONTROL_VARIANT
         and t.get("reference") == "qgate"), None)
    # 样例级配对（McNemar）：固定题面上的评测抽样不确定性，与上面的
    # 跨种子聚合 EM 检验互补；独立 Holm 家族、探索性证据。
    ruler_sample_tests = _ruler_sample_paired_tests(runs, pairs)
    # PPL 的窗口级配对：评估抽样不确定性，与种子级 PPL 检验互补。
    ppl_window_tests = _ppl_window_paired_tests(runs, pairs)
    comp = component_evidence(tests, audit=audit)
    # 单因素审计失败时，任何“组件有用”的统计证据都不能升级为确认性结论：
    # 对比本身不是单因素，归因对象不成立。证据数值保留供排查。
    for c in comp:
        c["single_factor_audit_ok"] = bool(sf_audit["ok"])
        if not sf_audit["ok"]:
            c["confirmatory_supported"] = False
            c["evidence_status"] = "exploratory"
    # 预注册假设判定：confirmatory（登记主张）与 exploratory（其余对比）。
    # 完整性锁链把“事后静默改写登记表”变成可检测事件：校验未通过时
    # 全部确认性判定被显式降级（apply_integrity_gate），不会假装边界仍成立。
    integrity = verify_registration()
    # 范围门：登记表完整还不够——本次 run 必须真的落在登记的实验范围内
    # （config 规模/数据/种子计划），否则同样不能给 confirmatory 判定。
    scope_check = registration_scope_check(registered, repro)
    # 设计符合性门：先验计划里的 (变体, 种子) 单元必须齐全；缺失只影响
    # 涉及该变体/指标端点的假设，不搞“一刀切”降级，也不能把缺失当无效应。
    design_plan = load_design_plan(results_dir)
    design_stages = _as_dict(
        _as_dict(repro.get("design")).get("stages"))
    design_conf = design_conformance(
        design_plan, repro, runs, audit=audit, stages=design_stages)
    design_hist = load_design_plan_history(results_dir)
    if len(design_hist) > 1:
        design_conf["plan_history_count"] = len(design_hist)
        design_conf.setdefault("warnings", []).append(
            f"本结果目录已有 {len(design_hist)} 份设计计划记录"
            "（重跑/改种子/改样本量）：历史见 design_plan_history.jsonl；"
            "当前判定只对应最新计划，不得把旧计划下的产物混入")
    hyp_eval = apply_design_gate(
        apply_budget_gate(
            apply_scope_gate(
                apply_integrity_gate(
                    evaluate_registered_hypotheses(registered, tests),
                    integrity),
                scope_check),
            audit),
        design_conf, design_plan)

    # --- Markdown
    cols = [("variant_cn", "变体"), ("n_seeds", "种子数"), ("params", "参数量"),
            ("gate_extra", "门控额外参数"), ("n_gate_per_head", "门控参数/头"),
            ("val_ppl", "PPL↓"),
            ("ppl_ci_rel_pct", "PPL评估CI±%"),
            ("epochs_seen", "训练覆盖(epoch)"),
            ("ckpt_step", "选型步"),
            ("selection_step_fraction", "选型步/计划"),
            ("ruler_avg", "RULER平均↑"),
            ("ruler_avg_ci95", "RULER评估CI"),
            ("ruler_niah_synth", "niah_synth↑"),
            ("ruler_niah_corpus", "niah_corpus↑"),
            ("ruler_niah_multikey", "multikey↑"),
            ("ruler_fwe", "fwe EM↑"),
            ("ruler_fwe_recall", "fwe recall↑"),
            ("ruler_fwe_precision", "fwe prec↑"),
            ("ruler_fwe_f1", "fwe F1↑"),
            ("ruler_vt", "vt↑"), ("prefill_ms", "Prefill ms↓"),
            ("prefill_rel_pct", "相对基线%"),
            ("decode_ms_per_tok", "Decode ms/tok↓"),
            ("gate_mean", "门控均值"), ("misleading_rate", "集中但看错率"),
            ("misleading_given_conc", "集中后看错率"),
            ("hit_base_rate", "命中基线率"),
            ("auroc_logit", "AUROC(logit)"), ("auroc_qterm", "AUROC(仅查询)"),
            ("auprc_logit", "PR-AUC(logit)↑"), ("auprc_qterm", "PR-AUC(仅查询)↑"),
            ("gate_f1_macro", "门控F1(macro)↑"),
            ("gate_f1_macro_always_write", "恒写基线F1(macro)")]

    def fmt_stat_cell(r, key):
        st = {"mean": r.get(key), "std": r.get(f"{key}_std"),
              "n": r.get(f"{key}_n")}
        # 评估 CI 半宽与 epoch 覆盖用 2 位小数：0.03 epoch 的差异不应被
        # 四舍五入成 0.0 而看起来像“没有训练”；CI ±12.5% 也需要小数位。
        cell = _fmt_stat(st, nd=2 if key in ("ppl_ci_rel_pct", "epochs_seen")
                         else 3)
        # PPL 被钳制在 exp(20) 的 run（发散）必须显式标注：不能与正常 PPL
        # 在同一列表头中静默比较（行级 val_ppl_clamped 由评测写入）。
        if key == "val_ppl" and r.get("val_ppl_clamped"):
            cell = f"{cell} ⚠含截断值"
        return cell

    def _fmt_scalar(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    md = ["# LEGA 实验汇总", "",
          f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；"
          f"PPL/RULER 为跨随机种子的 Mean ± Std（变体间同种子配对，"
          f"显著性经 Holm–Bonferroni 校正），运行时指标在相同配置下测得。", ""]
    if rows:
        stat_keys = {"val_ppl", "ruler_avg", "ruler_avg_micro",
                     "gate_mean", "misleading_rate",
                     "misleading_given_conc", "auroc_logit", "auroc_qterm",
                     "auprc_logit", "auprc_qterm", "hit_base_rate",
                     "gate_f1_macro", "gate_f1_macro_always_write",
                     "ppl_ci_rel_pct", "epochs_seen", "ckpt_step",
                     "selection_step_fraction", "ruler_fwe_recall",
                     "ruler_fwe_precision", "ruler_fwe_f1"} \
            | {f"ruler_{t}" for t in ruler_tasks}
        header = "| " + " | ".join(c for _, c in cols) + " |"
        sep = "|" + "---|" * len(cols)
        md += [header, sep]
        for r in rows:
            cells = []
            for k, _ in cols:
                if k == "ruler_avg_ci95":
                    ci = r.get(k)
                    cells.append(
                        f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                        if isinstance(ci, (list, tuple)) and len(ci) == 2
                        else "-")
                elif k in stat_keys:
                    cells.append(fmt_stat_cell(r, k))
                else:
                    cells.append(_fmt_scalar(r.get(k)))
            md.append("| " + " | ".join(cells) + " |")
        if invalid_eval_split or len(eval_splits) != 1:
            shown_splits = sorted((str(x) for x in eval_splits))
            split_note = (
                "⚠ PPL 评测 split 缺失、非法或不一致"
                + (f"（{', '.join(shown_splits)}）" if shown_splits else "")
                + "：跨变体数值可能来自不同切片，不能直接比较；请统一用 "
                "held-out test（或在同一 val 上）重跑评测")
        elif "test" in eval_splits:
            split_note = ("PPL 在 held-out test 集上评测（checkpoint 选择在 "
                          "val 上完成）")
        elif "val" in eval_splits:
            split_note = ("⚠ PPL 在 val 集上评测，该集同时用于 best-checkpoint "
                          "选择，数值偏乐观；请用 prepare_data.py 生成 "
                          "held-out test 后重跑评测")
        else:
            only_split = next(iter(eval_splits))
            split_note = (f"⚠ PPL 在 {only_split} 集上评测（非 val/test 口径）："
                          "该切片的绝对 PPL 不能与 val/test 数值直接比较，"
                          "也不能用于 held-out 泛化结论")
        min_n = min((r.get("n_seeds") or 1) for r in rows)
        seed_note = (f"关键指标至少 {min_n} 个随机种子；变体间比较为同种子配对。"
                     if min_n >= 3 else
                     f"⚠ 当前最少只有 {min_n} 个随机种子（<3），"
                     "无法可靠评估方差，结论只能是探索性的；"
                     "正式实验请用 ≥3 个种子（configs 默认已达要求）。")
        cols_notes = [
            "注: 统计单元格为 Mean ± Std (n=种子数)；n=1 时 Std 不可估计。",
            "注: `PPL评估CI` 为单次 run 内验证窗口抽样的 95% 相对半宽"
            "（窗口作为抽样单元，未计窗口间空间相关）；它与跨种子 Mean ± Std "
            "回答不同问题，比较变体时两者都要看；逐 run 区间见 eval.json 的 "
            "`val_ppl_ci95`。",
            "注: `训练覆盖` 为 tokens_seen / train_tokens 的等价 epoch 数；"
            "低于 1 时曲线下降主要反映“还没学完”，不能当作收敛。",
            "注: `选型步`/`选型步/计划` 是 `model_best.pt` 在 val 上被选中时"
            "的训练步数与占计划预算比例（跨种子均值）。所有变体共用同一最大"
            "预算与同一选型规则；选型步差异属 val 波动而非条件失控，但某变体"
            "长期在极早步被选中时，它的“同预算”比较实质是“更少训练”，需结合"
            "训练覆盖与曲线谨慎解读。",
            "注: niah/fwe/vt 为各上下文档位平均后的跨种子均值，"
            "每格样本数见 ruler_*.json（各变体同 seed 使用同一批样本）；",
            "注: `RULER评估CI` 为绝对 macro EM 的分层 bootstrap 95% 区间："
            "固定已观测种子，在每个 (任务, 上下文) 单元格内对样例有放回"
            "重采样，任务等权平均后跨种子平均；它只覆盖评测样例抽样，"
            "不覆盖训练种子随机性（后者看 Mean ± Std 与配对检验）。"
            "旧产物缺逐样例记录时显示 '-'，不伪造区间。",
            "“集中但看错率”= P(max_p>τ 且注意力落点不在答案数值跨度 value_span 内)，"
            "见机制分析。",
            "注: PPL 单元格标注 ⚠含截断值时，该变体至少有 1 个种子验证损失 >20，"
            "PPL 已按 exp(20) 截断；发散的 run 不能与正常 PPL 直接比较。",
            "注（类别不平衡与指标口径）：“命中基线率”= 机制分析中 hit 正类率，"
            "即 PR-AUC 的随机猜基线；正类率 <5% 时 AUROC 会因负样本基数巨大、"
            "FPR 增长缓慢而过于乐观，结论应以 PR-AUC 为主（极度不平衡 <0.1% "
            "时更是如此），两者并报供对照。F1 一律为 **macro**（各类等权，对"
            "稀有类最诚实）；micro/weighted 与逐类明细见 analysis_*.json 的 "
            "`gate_f1`（门控 g>0.5 决策点）。门控 F1 必须对照"
            "“恒写基线F1(macro)”（恒判写入的固定规则）：不高于这个无信息"
            "基线时，门控的决策价值不成立（`gate_f1_trivial` 另含恒不写"
            "规则）。RULER 的 EM 是生成式准确率、无决策阈值，不受此口径"
            "影响；fwe 另报召回率。",
            seed_note, split_note, ""]
        if common_ctx is not None:
            cols_notes.insert(1, f"“相对基线%”以 {base_name or 'standard'} 的 prefill 为基线"
                                 f"（公共 ctx={common_ctx}、batch=1，见 latency.json）；")
        md += [""] + cols_notes
        # 训练覆盖门：不足 1 epoch 的 PPL 下降只是“还没学完”，与“方法有效”
        # 是两回事；在表格外单独给醒目警告，避免读者把可跑通当结论。
        insufficient = [r for r in rows
                        if r.get("epochs_seen_min") is not None
                        and r["epochs_seen_min"] < 1.0]
        missing_cov = [r for r in rows if r.get("epochs_seen_min") is None]
        if insufficient:
            worst_cov = min(r["epochs_seen_min"] for r in insufficient)
            md += ["", f"⚠ 训练覆盖不足：{len(insufficient)}/{len(rows)} 个变体"
                      f"的最少训练覆盖仅约 {worst_cov:.2f} 个 epoch（<1）——"
                      "训练/验证曲线同时下降只说明“还在学”，不能作为收敛证据；"
                      "下表的 PPL/RULER 差异先按“没学完”处理，延长训练后重评。"]
        if missing_cov:
            md += ["", f"注：{len(missing_cov)}/{len(rows)} 个变体的 eval.json "
                      "没有训练覆盖字段（旧结果或手工评测），无法核对训练覆盖。"]
        sampled = [r for r in rows
                   if r.get("eval_window_fraction") is not None
                   and r["eval_window_fraction"] < 1.0]
        if sampled:
            worst = min(r["eval_window_fraction"] for r in sampled)
            md += ["", f"⚠ PPL 为窗口抽样：{len(sampled)}/{len(rows)} 个变体"
                      f"未评测全部不重叠窗口（最低覆盖 {worst:.1%}）。小于"
                      "`PPL评估CI` 或种子间差异的 PPL 差值先按评估噪声处理；"
                      "正式 PPL 结论请用 `run_all.py --full_eval` 重跑。"]
        # 评估噪声门：单 run 的 PPL 窗口抽样 CI 相对半宽若与跨种子差异同量级，
        # 变体间的小 PPL 差异不足以支撑方法结论。
        ci_vals = [r["ppl_ci_rel_pct"] for r in rows
                   if r.get("ppl_ci_rel_pct") is not None]
        if ci_vals and max(ci_vals) > 5.0:
            md += ["", f"注：单 run 内 PPL 评估 95% CI 相对半宽最大 "
                      f"±{max(ci_vals):.1f}%（窗口抽样）；小于该量级的 PPL "
                      "差异不能脱离评估噪声解读。"]
        # RULER 每格样本数门：n=2 与 n=30 的相同 EM 不是同样可信的证据；
        # 汇总层显式提示最小 n，读者需对照 ruler_*.json 的 Wilson CI。
        ruler_ns = [r["ruler_min_n_min"] for r in rows
                    if r.get("ruler_min_n_min") is not None]
        if ruler_ns and min(ruler_ns) < 10:
            md += ["", f"⚠ RULER 每格样本数不足：最小 n={min(ruler_ns)}（<10）——"
                      "EM 的 Wilson 95% CI 在该样本量下很宽（n=2 时接近 [0,0.84]），"
                      "RULER 的变体差异只能作方向性线索；逐格区间见 "
                      "ruler_*.json 的 `em_ci95_wilson`，正式结论需增大 "
                      "`ruler.n_samples` 后重跑。"]
        # RULER 平均口径：macro 与 micro 可能因任务/格大小不均而不同。
        # “平均”不注明定义就等于给读者留了事后择优的空间，这里显式固定
        # 预注册口径（macro）并给出样本加权口径供对照。
        rul_rows = [r for r in rows if r.get("ruler_avg") is not None]
        if rul_rows:
            md += ["", "### RULER 平均口径（macro / micro）", "",
                   "macro = 各任务内先对上下文档位等权平均，再对任务等权平均"
                   "（预注册“RULER平均”固定指此口径）；micro = 按每格样本数 n "
                   "加权（所有评测样例等权）。两者统一用各任务 EM；fwe 用"
                   "严格集合 EM（与预注册 H4 的“RULER 平均 EM”一致），其"
                   "recall 仅作诊断单列、不进入平均——recall 不惩罚“把上下文"
                   "词全部列出”的灌水策略。两者差异大说明任务难度/每格样本数"
                   "不均衡会改变“平均分”的读法，结论应同时对照逐任务列与"
                   "Wilson CI。", "",
                   "| 变体 | RULER平均 macro↑ | RULER平均 micro↑ | "
                   "macro−micro |", "|---|---|---|---|"]
            for r in rul_rows:
                macro, micro = r.get("ruler_avg"), r.get("ruler_avg_micro")
                gap = (macro - micro
                       if macro is not None and micro is not None else None)
                md.append(f"| {r.get('variant_cn') or r.get('variant')} | "
                          f"{_fmt4(macro)} | {_fmt4(micro)} | {_fmt4(gap)} |")
        # NIAH 位置分层：只报总 EM 会让“early/late 高、middle 塌陷”平均成
        # 看似正常的中等分；分层单独成表，并明确样本不可跨箱配对。
        depth_rows = [r for r in rows
                      if any(r.get(k) is not None for k in (
                          "ruler_depth_early", "ruler_depth_middle",
                          "ruler_depth_late"))]
        if depth_rows:
            md += ["", "### RULER NIAH 位置分层（early / middle / late）", "",
                   "按 needle 在上下文中的相对位置三分位分箱（[0,1/3)、"
                   "[1/3,2/3)、[2/3,1]），对 niah_* 任务的 (任务, 上下文) "
                   "单元格等权平均 EM。总 EM 掩盖位置差异：早期与末期的高分"
                   "会抵消中部塌陷（lost-in-the-middle），因此位置分层是"
                   "机制诊断的必报项。三个箱使用**不同评测样例**，不能直接"
                   "当作同批配对样本做显著性检验；每箱样本数与 Wilson 95% "
                   "CI 见 ruler_*.json 的 `em_by_depth`。", "",
                   "| 变体 | early↑ | middle↑ | late↑ | early−late |",
                   "|---|---|---|---|---|"]
            for r in depth_rows:
                e, m, l = (r.get("ruler_depth_early"),
                           r.get("ruler_depth_middle"),
                           r.get("ruler_depth_late"))
                gap = r.get("ruler_depth_early_minus_late")
                md.append(f"| {r.get('variant_cn') or r.get('variant')} | "
                          f"{_fmt4(e)} | {_fmt4(m)} | {_fmt4(l)} | "
                          f"{_fmt4(gap)} |")
            flags = []
            for r in depth_rows:
                e, m, l = (r.get("ruler_depth_early"),
                           r.get("ruler_depth_middle"),
                           r.get("ruler_depth_late"))
                name = r.get("variant_cn") or r.get("variant")
                if None not in (e, m, l):
                    mid_drop = min(e, l) - m
                    if mid_drop >= 0.10:
                        flags.append(
                            f"{name} 的 middle 比两端较低者低 "
                            f"{mid_drop * 100:.1f} pp")
                gap = r.get("ruler_depth_early_minus_late")
                if gap is not None and gap >= 0.10:
                    flags.append(f"{name} 的 early−late 落差 "
                                 f"{gap * 100:.1f} pp")
            if flags:
                md += ["", "⚠ 位置不均衡（描述性提示，非显著性判定）："
                           + "；".join(flags)
                           + "。这会把总 EM 抬到看似可接受的水平；需结合逐箱 "
                           "Wilson CI 与样本文本复核，不能只用 RULER平均 下结论。"]
        else:
            niah_rows = [r for r in rows
                         if any(r.get(f"ruler_{t}") is not None for t in (
                             "niah_synth", "niah_corpus", "niah_multikey"))]
            if niah_rows:
                md += ["", "⚠ 本次 RULER 产物不含 needle 位置分层"
                           "（`em_by_depth`，旧格式）：总 EM 可能掩盖 "
                           "lost-in-the-middle。旧结果不能补算，需用当前版本"
                           "重跑 RULER 后才能给出位置分层结论。"]
        # 上下文长度剂量-反应：位置分层回答“上下文内部的 needle 在哪”，
        # 长度分层回答“上下文本身变长时方法是否更有效”——两者是不同轴，
        # 不能互相替代；方法名为 length-calibrated，这一轴是核心主张。
        md += _render_ruler_length(length_analysis)
    else:
        md.append("（暂无结果，先运行训练评测）")

    # --- 门控概率校准：排序好 ≠ 概率数值可信。g 被当作“该不该写入”的
    #     软权重/概率使用，Brier/ECE 必须与 AUROC/PR-AUC 分开报告。
    cal_rows = [r for r in rows
                if r.get("brier_g") is not None
                or r.get("probe_brier") is not None]
    if cal_rows:
        md += ["", "### 门控概率校准（Brier / ECE）", "",
               "排序指标回答“能不能区分 hit/¬hit”，校准回答“g 的数值是否"
               "可信”。Brier 越小越好；常数基率预测（永远输出 hit_base_rate）"
               "的 Brier 是无信息基线，`Brier skill=1−Brier/Brier_常数` >0 "
               "才说明概率预测优于常数。ECE 为 10 个等宽分箱的加权平均"
               "|置信−准确率|；等频 ECE 按分数排序均分样本，避免分数聚集/"
               "小样本下大量空箱或每个样本独占一箱把 ECE 机械压低。两者"
               "都只作描述性证据；线性探针为有标签监督的 OOF 概率，门控 g "
               "为语言建模副产品，两者校准口径不同，不能只凭 Brier 判定"
               "方法优劣。", "",
               "| 变体 | 命中基线率 | 门控Brier↓ | 常数基率Brier | "
               "Brier skill↑ | 门控ECE↓ | 门控ECE(等频)↓ | "
               "探针Brier↓ | 探针ECE↓ |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in cal_rows:
            md.append(
                f"| {r.get('variant_cn') or r.get('variant')} | "
                f"{_fmt4(r.get('hit_base_rate'))} | "
                f"{_fmt4(r.get('brier_g'))} | "
                f"{_fmt4(r.get('brier_g_constant'))} | "
                f"{_fmt4(r.get('brier_g_skill'))} | "
                f"{_fmt4(r.get('ece_g'))} | "
                f"{_fmt4(r.get('ece_g_adaptive'))} | "
                f"{_fmt4(r.get('probe_brier'))} | "
                f"{_fmt4(r.get('probe_ece'))} |")
        md += ["",
               "注：门控 Brier/ECE 与固定 0.5 阈值无关；它们评估的是 g 的"
               "概率数值，而不是某个决策点的 F1。逐分箱的置信/准确率、"
               "n 与 gap 见 `analysis_*.json` 的 `calibration_g.bins`；"
               "Brier 的 Murphy 分解（reliability/resolution/uncertainty，"
               "原始分数非箱内常数时另报 brier_binning 箱内项）"
               "与等频分箱明细见 `calibration_g.brier_*` 与 `ece_adaptive`。"]

    # --- 基线合理性（简单统计基线 + 基线调优证据 + 逐变体 vs 统计基线
    #     + 机制判别的线性探针简单监督基线）
    ngram_cmp = ngram_baseline_comparison(runs, results_dir)
    probe_cmp = linear_probe_baseline_comparison(runs)
    md += _render_baseline_sections(results_dir,
                                    smoke=((repro or {}).get("smoke") is True),
                                    ngram_cmp=ngram_cmp,
                                    probe_cmp=probe_cmp,
                                    variants=sorted(runs))

    # --- c~log2(n) 长度零假设诊断（README 要求的长度主张必要证据）
    length_null = load(os.path.join(results_dir, "length_null.json"))
    length_null = length_null if isinstance(length_null, dict) else None
    length_null_stale = bool(
        length_null and scope_run_id
        and length_null.get("pipeline_run_id") != scope_run_id)
    md += _render_length_null(length_null, stale=length_null_stale)

    # --- 先验设计计划与分析人群（训练前声明，汇总时核对）
    md += render_design_section(
        design_plan, design_conf,
        observed=observed_design_sensitivity(tests))

    # --- 受控变量自检
    md += ["", "## 对比实验受控变量自检", ""]
    if audit:
        exempt_keys = _as_text_list(audit.get("exempt_keys"))
        stopped = _as_text_list(audit.get("early_stop_runs"))
        if audit.get("ok"):
            if audit.get("checked"):
                md.append(f"- ✅ 通过：{audit['checked']} 个 run 的优化器、学习率、"
                          f"训练步数、数据顺序等受控设置逐项一致；逐 run 仅 "
                          f"`{'`, `'.join(exempt_keys)}` 不同。")
            else:
                md.append("- ⚠ 未检查受控变量：没有 run 可用于核对"
                          "（全部缺少 config_used.json），不能据此声称一致性。")
            if audit.get("meta_checked"):
                digest = audit.get("expected_data_digest")
                md.append(f"- ✅ 训练溯源一致：{audit['meta_checked']} 个 run 的"
                          f"数据指纹、代码提交、配置 SHA-256、计划训练步数"
                          f"逐项一致"
                          + ("；实际步数差异为早停所致（见下条）"
                             if stopped else "")
                          + (f"（数据指纹 `{digest}`，并与本次流水线数据比对）"
                             if digest else "。"))
            if stopped:
                md.append(
                    f"- ⏹ 早停启用：{len(stopped)} 个 run 未跑满"
                    f"计划步数（{', '.join(stopped[:8])}"
                    + (" …" if len(stopped) > 8 else "")
                    + "）：实际步数/tokens_seen 因验证曲线不同而不同，"
                    "有效训练量不等；受控变量按计划步数核对，但预注册"
                    "确认性判定已自动降级为探索性（见 `significance.json` "
                    "的 registered_hypotheses）。逐 run 实际步数与 tokens_seen "
                    "见各 eval.json。")
        else:
            md.append(f"- ❌ 未通过：{audit.get('mismatches') or audit}")
        missing_configs = _as_text_list(audit.get("missing_configs"))
        if missing_configs:
            md.append(f"- ⚠ 缺少配置文件的 run：{', '.join(missing_configs[:8])}"
                      + (" …" if len(missing_configs) > 8 else ""))
        missing_evals = _as_text_list(audit.get("missing_evals"))
        if missing_evals:
            md.append(f"- ⚠ 缺少 eval.json、无法核对数据/代码/训练步数的 run："
                      f"{', '.join(missing_evals[:8])}"
                      + (" …" if len(missing_evals) > 8 else ""))
    else:
        md.append("- ⚠ 未执行（直接调用 build_summary 或缺少 out 目录）。")

    # --- 数据协议与防泄漏自检（由 lega.data.prepare_bins 生成 data_report.json）
    md += ["", "## 数据协议与防泄漏自检", ""]
    if data_report:
        proto = _as_dict(data_report.get("protocol"))
        md.append(
            f"- 协议：先划分后预处理={proto.get('split_first_then_preprocess')}、"
            f"拟合统计只来自 {proto.get('fit_on')}、"
            f"划分时打乱={proto.get('shuffle_at_split_time')}；"
            "训练阶段 batch 级随机采样（shuffle）不构成泄露。")
        sp = _as_dict(data_report.get("split_protocol"))
        if sp:
            md.append(f"- 划分方式：{sp.get('mode', '-')}；"
                      f"顺序 {sp.get('order', '-')}；"
                      f"文档边界回退={sp.get('fallback_char_cut', '-')}"
                      + ("（⚠ 字符切分可能让同一文档跨 split）"
                         if sp.get("fallback_char_cut") else ""))
            # 声明与文本核对结果：document_disjoint 是上游声明，verified_disjoint
            # 是对实际写出的三份文本做跨 split 长行核对后的结论。
            if "verified_disjoint" in sp:
                vd = sp.get("verified_disjoint")
                md.append(f"- 划分声明核对：无共享长行={vd}"
                          + ("" if vd else
                             "（⚠ 实际文本存在跨 split 共享长行，声明被证伪，"
                             "见 overlap_long_lines）"))
            group = sp.get("group_unit")
            cross = sp.get("documents_cross_cut")
            pcross = sp.get("paragraphs_cross_cut")
            if group or cross is not None or pcross is not None:
                warn = ""
                if cross:
                    warn += "（⚠ 文档群组跨 split，见告警 split_documents_cross_cut）"
                if pcross:
                    warn += "（⚠ 段落群组跨 split，见告警 split_paragraphs_cross_cut）"
                md.append(f"- 群组完整性：群组单位={group or '-'}；"
                          f"标题数={sp.get('n_headings', '-')}；"
                          f"段落边界数={sp.get('n_paragraph_boundaries', '-')}；"
                          f"文档跨切点={cross if cross is not None else '-'}；"
                          f"段落跨切点={pcross if pcross is not None else '-'}"
                          + warn)
            actual = _as_dict(sp.get("actual_fractions"))
            target = _as_dict(sp.get("target_fractions"))
            dev = _finite_number(sp.get("max_abs_fraction_deviation"))
            if actual:
                def _fr(d, k):
                    f = _finite_number(d.get(k))
                    if f is None:
                        return "-"
                    # 有限比例乘 100 仍可能溢出（如 1e307），不能写出 inf%。
                    return "溢出" if not math.isfinite(f * 100.0) else f"{f:.1%}"
                dev_s = ("溢出" if not math.isfinite(dev * 100.0)
                         else f"{dev:.1%}") if dev is not None else None
                md.append(
                    "- 划分比例：实际 "
                    + " / ".join(f"{k}={_fr(actual, k)}"
                                 for k in ("train", "val", "test"))
                    + ("；目标 " + " / ".join(
                        f"{k}={_fr(target, k)}"
                        for k in ("train", "val", "test")) if target else "")
                    + (f"；最大绝对偏差 {dev_s}" if dev_s is not None else "")
                    + (f"；切点相对目标偏移 {sp.get('cut_deviation_chars')} 字符"
                       if sp.get("cut_deviation_chars") is not None else "")
                    + "。切点吸附到文档/段落边界会改变实际训练/评测规模，"
                      "解读固定比例与检验功效时须按实际值")
        tok = _as_dict(data_report.get("tokenizer"))
        unk = _as_dict(tok.get("unk_rate"))
        unk_s = "、".join(f"{k}={_fmt_rate(v)}" for k, v in unk.items())
        md.append(f"- 分词器：{tok.get('kind')}（vocab={tok.get('vocab_size')}，"
                  f"拟合集={tok.get('fit_on')}）；OOV 字符率：{unk_s or 'N/A'}"
                  "（未知字符映射 <unk>，val/test 不参与词表拟合）")
        splits = _as_dict(data_report.get("splits"))
        md.append("- token 流硬校验：" + "；".join(
            f"{k} {_int_or_zero(_as_dict(v).get('tokens')):,} tokens"
            f"（id∈[{_as_dict(v).get('min_id')},{_as_dict(v).get('max_id')}]，"
            f"NaN={_as_dict(v).get('nan')}，Inf={_as_dict(v).get('inf')}）"
            for k, v in splits.items()))
        dup = _as_dict(data_report.get("duplicates"))
        cont = _as_dict(data_report.get("contamination"))
        doc = _as_dict(dup.get("document_level"))
        if doc:
            md.append(
                f"- 块级去污染：与 val/test 重复的多行块移除 "
                f"{doc.get('train_blocks_dropped_as_duplicates_of_eval', '-')} 个；"
                f"train 内部重复块移除 "
                f"{doc.get('train_intra_split_duplicate_blocks_removed', '-')} 个"
                f"（共 {doc.get('train_lines_dropped', '-')} 行）；"
                f"val↔test 多行块重叠 {doc.get('val_test_overlap_blocks', '-')} 个"
                "（只改 train，val/test 保持尘封）")
        near = _as_dict(dup.get("near_duplicate"))
        if near:
            md.append(
                f"- 近似重复去污染（MinHash/LSH，字符 "
                f"{near.get('shingle_size', '-')}-gram，协议阈值 "
                f"{near.get('threshold', '-')}）：与 val/test 近似块移除 "
                f"{_int_or_zero(near.get('train_blocks_dropped_as_near_duplicates_of_eval')):,} 个、"
                f"train 内部近似重复块移除 "
                f"{_int_or_zero(near.get('train_intra_split_near_duplicates_removed')):,} 个"
                f"（共 {_int_or_zero(near.get('train_lines_dropped')):,} 行；"
                "只改 train，val/test 保持尘封）；"
                f"val↔test 近似对 "
                f"{_int_or_zero(near.get('val_test_near_duplicate_pairs')):,}；"
                f"LSH 理论召回率 {_fmt_rate(near.get('lsh_recall_at_threshold'))}；"
                f"候选对 {_int_or_zero(near.get('candidate_pairs')):,}"
                "（阈值敏感性见 data_report.json）")
            cov = _finite_number(near.get("train_chars_coverage_fraction"))
            if cov is not None:
                cov_s = ("溢出" if not math.isfinite(cov * 100.0)
                         else f"{cov:.1%}")
                md.append(
                    f"- 近似重复扫描覆盖：{cov_s} 的 train 字符达到 "
                    f"min_chars={near.get('min_chars', '-')} 且 shingle 数达标；"
                    f"未扫描字符：短块 "
                    f"{_int_or_zero(near.get('train_chars_skipped_short')):,}、"
                    f"少 shingle "
                    f"{_int_or_zero(near.get('train_chars_skipped_few_shingles')):,}、"
                    f"超大块 "
                    f"{_int_or_zero(near.get('train_chars_skipped_too_large')):,}"
                    "（近重复结论只对已覆盖部分成立）")
            removal = _as_dict(near.get("threshold_sensitivity_removal"))
            if removal:
                def _tau_key(x):
                    try:
                        return (0, float(x))
                    except (TypeError, ValueError):
                        return (1, str(x))
                parts = []
                for key in sorted(removal, key=_tau_key):
                    r = _as_dict(removal.get(key))
                    parts.append(
                        f"τ={key}: 删 "
                        f"{_int_or_zero(r.get('train_blocks_removed_total'))} 块/"
                        f"{_int_or_zero(r.get('train_chars_removed')):,} 字符/"
                        f"{_fmt_rate(r.get('removal_fraction_chars'))}")
                md.append("- 近似重复阈值敏感性（实际删除量，不只是判定对数）："
                          + "；".join(parts)
                          + "。阈值是分析前声明的协议参数，不能按下游结果回头挑选")
            if (near.get("candidate_pairs_truncated")
                    or _int_or_zero(near.get("oversized_groups_truncated"))):
                md.append("- ⚠ 近似重复扫描触发候选簇/候选总数上限：删除量是下界，"
                          "须降低语料规模或提高上限后重跑")
        post = _as_dict(dup.get("post_cleanup"))
        if post.get("checked_pairs"):
            clean = post.get("train_eval_clean")
            if clean is True:
                md.append("- 清洗后复核：train 与 val/test 已无共享精确/归一化"
                          "长行或多行块（短行与近似重复另见对应条目）")
            elif clean is False:
                md.append("- ⚠ 清洗后复核：train 仍与 val/test 共享精确/归一化"
                          "长行或多行块，去污染未闭合，须重建数据")
        acct = _as_dict(data_report.get("accounting"))
        if acct:
            md.append(
                "- 语料核算（与分词器无关的分母）：" + "；".join(
                    f"{k} {_int_or_zero(_as_dict(v).get('tokens')):,} tokens / "
                    f"{_int_or_zero(_as_dict(v).get('chars')):,} 字符 / "
                    f"{_int_or_zero(_as_dict(v).get('utf8_bytes')):,} UTF-8 字节"
                    f"（{_fmt4(_as_dict(v).get('bytes_per_token'))} bytes/token）"
                    for k, v in acct.items())
                + "。字符级兜底与 BPE 的 token PPL 不可直接比较，"
                  "规模与分母以字节/字符核算为准")
        md.append(
            f"- 去污染：与 val/test 重复的训练长行移除 "
            f"{dup.get('train_lines_dropped_as_duplicates_of_eval', '-')} 行；"
            f"train 内部重复长行移除 "
            f"{dup.get('train_intra_split_duplicate_lines_removed', '-')} 行"
            f"（保留首次出现）；val↔test 长行重叠 "
            f"{dup.get('val_test_overlap_long_lines', '-')} 条；"
            f"采样 {_as_dict(cont.get('train_val')).get('n', '-')}-gram 重叠率 "
            f"val={_fmt_rate(_as_dict(cont.get('train_val')).get('overlap_rate'))}、"
            f"test={_fmt_rate(_as_dict(cont.get('train_test')).get('overlap_rate'))}、"
            f"val↔test={_fmt_rate(_as_dict(cont.get('val_test')).get('overlap_rate'))}"
            "（采样筛查下界，>0 需人工核查）")
        n_short_tv = _int_or_zero(dup.get("train_val_overlap_short_lines"))
        n_short_tt = _int_or_zero(dup.get("train_test_overlap_short_lines"))
        n_short_vt = _int_or_zero(dup.get("val_test_overlap_short_lines"))
        if n_short_tv or n_short_tt or n_short_vt:
            md.append(
                f"- 短行跨 split 重复（<{dup.get('min_chars', '-')} 字符，"
                "不自动删除以免误删标题/固定用语）："
                f"train↔val {n_short_tv} 条、train↔test {n_short_tt} 条、"
                f"val↔test {n_short_vt} 条"
                f"（空白归一化后额外 "
                f"{_int_or_zero(dup.get('train_val_overlap_normalized_short_lines'))}/"
                f"{_int_or_zero(dup.get('train_test_overlap_normalized_short_lines'))}/"
                f"{_int_or_zero(dup.get('val_test_overlap_normalized_short_lines'))}）；"
                "重复样本会削弱 held-out 独立性，须人工核查")
        cshort = _as_dict(data_report.get("contamination_short_ngram"))
        if cshort:
            md.append(
                f"- 短 n-gram 重叠（{_as_dict(cshort.get('train_val')).get('n', '-')}-gram，"
                "抓 <64 token 整句/模板复用，仅告警）："
                f"val={_fmt_rate(_as_dict(cshort.get('train_val')).get('overlap_rate'))}、"
                f"test={_fmt_rate(_as_dict(cshort.get('train_test')).get('overlap_rate'))}、"
                f"val↔test={_fmt_rate(_as_dict(cshort.get('val_test')).get('overlap_rate'))}"
                "（短 n 易有自然搭配假阳性，不进硬闸门）")
        dist = _as_dict(data_report.get("distribution_shift"))
        dv = _as_dict(dist.get("val_vs_train"))
        dt = _as_dict(dist.get("test_vs_train"))
        dvt = _as_dict(dist.get("test_vs_val"))
        md.append(
            f"- 分布漂移（一元 token，默认精确计数）：val−train TV={_fmt4(dv.get('tv'))}、"
            f"JS={_fmt4(dv.get('js_bits'))} bit；test−train TV={_fmt4(dt.get('tv'))}、"
            f"JS={_fmt4(dt.get('js_bits'))} bit；test−val TV={_fmt4(dvt.get('tv'))}、"
            f"JS={_fmt4(dvt.get('js_bits'))} bit；test OOV token 率="
            f"{_fmt_rate(dt.get('eval_oov_token_rate'))}")
        san = _as_dict(data_report.get("sanitization"))
        noise = sum(_int_or_zero(_as_dict(san.get(s)).get(
            "replacement_char_replaced_with_space")) for s in san)
        ctrl = sum(_int_or_zero(_as_dict(san.get(s)).get(
            "control_chars_deleted")) for s in san)
        md.append(f"- 文本清洗（标签噪声）：U+FFFD 替换 {noise:,} 处、"
                  f"非法控制符删除 {ctrl:,} 处；清洗为逐字符确定性规则，"
                  "不依赖跨样本统计量。")
        ts = _as_dict(data_report.get("text_stats"))
        if ts:
            md.append("- 行长度异常检查：" + "；".join(
                f"{k} 行数={_int_or_zero(_as_dict(v).get('lines')):,}、"
                f"最长行={_int_or_zero(_as_dict(v).get('line_chars_max')):,} 字符、"
                f"超长行(>10万字符)={_int_or_zero(_as_dict(v).get('lines_gt_huge_threshold'))}"
                for k, v in ts.items()))
        alerts = _as_dict_list(data_report.get("alerts"))
        if alerts:
            n_warn = sum(1 for a in alerts if a.get("level") == "warning")
            shown = "；".join(f"[{a.get('code')}] {a.get('message')}"
                              for a in alerts[:5])
            md.append(f"- ⚠ 自检告警（{n_warn} warning / 共 {len(alerts)} 条）："
                      + shown + (" …" if len(alerts) > 5 else "")
                      + "（阈值见 data_report.json 的 alert_thresholds）")
        else:
            md.append("- 自检告警：无（全部指标在 alert_thresholds 阈值内；"
                      "阈值是复核触发线而非显著性检验）")
        ta = _as_dict(data_report.get("test_access"))
        if ta:
            repeated = _as_text_list(
                ta.get("checkpoints_evaluated_multiple_times"))
            warn = ("" if not repeated
                    else "；⚠ 存在被重复评测的 checkpoint："
                         + "、".join(repeated[:4]))
            md.append(f"- 测试集尘封账本：累计 test 评测 {ta.get('evaluations', 0)} 次、"
                      f"不同 checkpoint {ta.get('distinct_checkpoints', 0)} 个"
                      f"{warn}。测试集只用于最终评估；据此调参即视为泄露。")
        else:
            md.append("- 测试集尘封账本：暂无 test 评测记录。")
    else:
        md.append("- ⚠ 未找到 `data_report.json`（旧数据或未走统一数据入口）；"
                  "请用 `scripts/prepare_data.py` / `scripts/make_dummy_data.py` "
                  "重建数据以生成防泄漏自检。")

    _assigned = {
        "_",
        "_clock_extractor",
        "_fmt_scalar",
        "_fr",
        "_lat_metric_at",
        "_length_pp",
        "_prefill_at",
        "_tau_key",
        "ablation_scope",
        "ablation_variants",
        "ablations",
        "acct",
        "actual",
        "alerts",
        "allow_ablation",
        "audit",
        "base_ms",
        "base_name",
        "budget_curves",
        "budget_sensitivity",
        "by_bs",
        "c",
        "cal_rows",
        "cand",
        "capacity_only_test",
        "causal_tests",
        "matched_ablation_contrasts",
        "cells",
        "ci",
        "ci_vals",
        "clean",
        "cols",
        "cols_notes",
        "common_ctx",
        "common_ctxs",
        "comp",
        "cont",
        "cov",
        "cross",
        "cshort",
        "ctrl",
        "ctx",
        "d",
        "data_report",
        "decode",
        "delta_pct",
        "depth_rows",
        "design_conf",
        "design_hist",
        "design_plan",
        "design_stages",
        "dev",
        "digest",
        "dist",
        "doc",
        "dt",
        "dup",
        "dv",
        "dvt",
        "e",
        "eps",
        "ev_example",
        "eval_splits",
        "evals_v",
        "exempt_keys",
        "flags",
        "fmt_stat_cell",
        "fn",
        "fp",
        "gap",
        "group",
        "header",
        "hyp_eval",
        "in_run_scope",
        "inference_interactions",
        "inference_shapley",
        "insufficient",
        "integrity",
        "invalid_eval_split",
        "k",
        "key",
        "l",
        "lat",
        "lat_all",
        "lat_entries",
        "lat_variants",
        "length_analysis",
        "length_by_pair",
        "length_null",
        "load",
        "log_tests",
        "m",
        "macro",
        "md",
        "metric_specs",
        "micro",
        "mid_drop",
        "min_n",
        "missing_configs",
        "missing_cov",
        "missing_evals",
        "n_short_tt",
        "n_short_tv",
        "n_short_vt",
        "n_warn",
        "name",
        "near",
        "ngram_cmp",
        "niah_rows",
        "noise",
        "only_split",
        "pair",
        "pairs",
        "param_control_test",
        "parts",
        "pcross",
        "per_ctx",
        "ph",
        "post",
        "ppl_window_tests",
        "prefill",
        "probe_cmp",
        "proto",
        "r",
        "raci",
        "ratio_overflow",
        "registered",
        "rel",
        "reparameterization",
        "removal",
        "repeated",
        "repro",
        "repro_doc",
        "rns",
        "row",
        "rows",
        "rul_rows",
        "ruler_ns",
        "ruler_payloads",
        "ruler_sample_tests",
        "ruler_tasks",
        "runs",
        "s",
        "san",
        "scope_check",
        "scope_seeds",
        "scope_variants",
        "seed_note",
        "sep",
        "sf_audit",
        "shapley_tests",
        "shown",
        "shown_splits",
        "slope",
        "slot",
        "sp",
        "span",
        "split_note",
        "splits",
        "st",
        "stat_keys",
        "stopped",
        "t",
        "ta",
        "target",
        "task",
        "task_tests",
        "tests",
        "tok",
        "toks",
        "tr",
        "train_inter",
        "train_phi",
        "ts",
        "unk",
        "unk_s",
        "usable",
        "v",
        "variants",
        "vd",
        "warn",
        "worst_cov",
    }
    return {k: v for k, v in locals().items() if k in _assigned}

def build_summary(results_dir: str, audit: dict = None,
                  data_report: dict = None, preloaded: dict = None) -> dict:
    """汇总 PPL / RULER / 参数量等指标为 Markdown + CSV + JSON。

    拆分后由多个阶段函数顺序生成；对外接口与返回结构保持不变。
    """
    from .sections import _summary_part_b
    from .tables import _summary_part_c

    ctx = {
        "results_dir": results_dir,
        "audit": audit,
        "data_report": data_report,
        "preloaded": preloaded,
    }
    ctx.update(_summary_part_a(results_dir, audit, data_report, preloaded))
    ctx.update(_summary_part_b(**ctx))
    ctx.update(_summary_part_c(**ctx))
    return {
        'rows': ctx["rows"],
        'markdown': ctx["md_text"],
        'tests': ctx["tests"],
        'ruler_sample_tests': ctx["ruler_sample_tests"],
        'ruler_length': ctx["length_analysis"],
        'length_null': ctx.get("length_null"),
        'ppl_window_tests': ctx["ppl_window_tests"],
        'factorial_shapley': ctx["shapley_tests"],
        'reparameterization': ctx["reparameterization"],
        'inference_factorial_shapley': ctx["inference_shapley"],
        'inference_interactions': ctx["inference_interactions"],
        'parameter_matched_control': ctx["param_control_test"],
        'causal_ablation': ctx["causal_tests"],
        'matched_ablation_contrasts': ctx["matched_ablation_contrasts"],
        'budget_sensitivity': ctx["budget_sensitivity"],
    }

