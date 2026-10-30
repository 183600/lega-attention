"""设计计划与多重比较规划。"""
from __future__ import annotations
from ._shared import _install as _shared_install

_shared_install(globals())

from .scheduling import (DEFAULT_RUN_ORDER, RUN_ORDER_MODES,  # noqa: E402
                         build_run_order, validate_run_order)
from lega.testreuse import DEFAULT_TEST_LOOK_BUDGET  # noqa: E402


def _mechanism_endpoint_sensitivity(design_cfg: dict, metric_name: str,
                                    grid_key: str, declared_key: str,
                                    n_seeds: int, alpha: float,
                                    power: float) -> dict:
    """机制端点（AUROC/AP 的 logit−仅查询配对差）的先验灵敏度。

    这里的 SD 是**跨种子配对差**的标准差，与 PPL 端点同一推断单位；单 run
    内由有限样例造成的测量误差不在这个公式里，必须另看机制分析的样例聚类
    bootstrap。把两者混成一个“功效”会高估确认性证据。
    """
    margin = design_cfg.get("mechanism_sesoi")
    if margin is not None:
        margin = _finite_float(margin, "design.mechanism_sesoi", positive=True)
    else:
        margin = 0.01
        try:
            from lega.equivalence import metric_margin
            spec = metric_margin(metric_name)
            if isinstance(spec, dict) and spec.get("margin") is not None:
                margin = _finite_float(spec.get("margin"),
                                       f"{metric_name} SESOI", positive=True)
        except Exception:  # pragma: no cover - 保持模块可独立使用
            pass
    raw_grid = design_cfg.get(grid_key, list(MECHANISM_SD_GRID))
    grid = sorted({_finite_float(v, grid_key, positive=True)
                   for v in _finite_sequence(raw_grid, grid_key)})
    if not grid:
        raise ValueError(f"设计网格不能为空：{grid_key}")
    can_power = n_seeds >= 2
    rows = []
    for sd in grid:
        if can_power:
            mde = min_detectable_effect(sd, n_seeds, alpha, power)
            pw = paired_t_power(margin, sd, n_seeds, alpha)
        else:
            mde = pw = None
        rows.append({
            "paired_sd_absolute": sd,
            "mde_absolute": mde,
            "power_for_sesoi": pw,
            "required_seeds_for_sesoi": required_pairs(
                margin, sd, alpha, power),
            "mde_absolute_confirmatory": None,
            "power_for_sesoi_confirmatory": None,
            "required_seeds_for_sesoi_confirmatory": None,
        })
    declared = None
    if declared_key in design_cfg:
        declared = _finite_float(design_cfg.get(declared_key), declared_key,
                                 positive=True)
    return {
        "metric": metric_name,
        "sesoi_absolute": margin,
        "sesoi_source": "predeclared_protocol",
        "planned_seeds": n_seeds,
        "declared_paired_sd": declared,
        "sensitivity": rows,
        "measurement_note": (
            "SD 为跨种子配对差的标准差；单 run 内有限样例的测量误差由"
            "机制分析的样例聚类 bootstrap 单独报告，不能与本表互相替代"),
    }


def _fill_confirmatory_rows(rows, alpha_effective: float, n_seeds: int,
                            margin: float, power: float) -> None:
    """把机制端点表补成确认性 α_eff 口径（原地写入并行字段）。"""
    for row in rows:
        sd = row.get("paired_sd_absolute")
        if n_seeds >= 2 and isinstance(sd, (int, float)):
            row["mde_absolute_confirmatory"] = min_detectable_effect(
                sd, n_seeds, alpha_effective, power)
            row["power_for_sesoi_confirmatory"] = paired_t_power(
                margin, sd, n_seeds, alpha_effective)
            row["required_seeds_for_sesoi_confirmatory"] = required_pairs(
                margin, sd, alpha_effective, power)


_DESIGN_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

_LENGTH_NULL_MODES = ("random", "repeated", "uniform")


def _optional_positive_int(value, name: str):
    """返回 (值, 错误文本)；缺省字段不报错，非法值留给计划检查显形。"""
    if value is None:
        return None, None
    try:
        return _strict_int(value, name, minimum=1), None
    except ValueError as e:
        return None, str(e)


def _accounting_tokens(data_report, split: str):
    """从 data_report.accounting 读取某 split 的 token 数（损坏则 None）。"""
    report = _as_dict(data_report)
    accounting = _as_dict(report.get("accounting"))
    entry = _as_dict(accounting.get(split))
    value = entry.get("tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _design_data_budget(config: dict, data_report=None) -> tuple:
    """开跑前的数据/评测预算摘要与检查。

    设计计划此前只看“要跑多少 seed”，不看“这些 seed 会消费多少 token、
    val/test 是否够组成计划中的窗口”。这里把 data_report 的语料规模核算
    （prepare_data 已落盘）接到计划里：报告 token 触碰次数（steps × batch ×
    grad_accum × train_ctx 相对 train token 数）、预期唯一 token 覆盖比例，
    以及 val/test 可组成的窗口数与计划 eval_iters 的对比。它不读取原始文本，
    不访问 test，也不改变任何功效公式。
    """
    block_size, bs_err = _optional_positive_int(
        config.get("block_size"), "block_size")
    train_ctx, tc_err = _optional_positive_int(
        config.get("train_ctx"), "train_ctx")
    steps, steps_err = _optional_positive_int(config.get("steps"), "steps")
    batch_size, batch_err = _optional_positive_int(
        config.get("batch_size"), "batch_size")
    grad_accum, ga_err = _optional_positive_int(
        config.get("grad_accum", 1), "grad_accum")
    eval_iters, ei_err = _optional_positive_int(
        config.get("eval_iters"), "eval_iters")
    eval_ctx = (min(block_size, train_ctx)
                if block_size is not None and train_ctx is not None
                else (block_size if block_size is not None else train_ctx))

    train_tokens = _accounting_tokens(data_report, "train")
    val_tokens = _accounting_tokens(data_report, "val")
    test_tokens = _accounting_tokens(data_report, "test")
    tokens_seen = None
    if None not in (steps, batch_size, grad_accum, train_ctx):
        tokens_seen = steps * batch_size * grad_accum * train_ctx
    epochs = (tokens_seen / train_tokens
              if tokens_seen is not None and train_tokens else None)
    unique_fraction = (1.0 - math.exp(-epochs)
                       if epochs is not None and math.isfinite(epochs)
                       else None)
    val_windows = ((val_tokens - 1) // eval_ctx
                   if val_tokens is not None and eval_ctx else None)
    test_windows = ((test_tokens - 1) // eval_ctx
                    if test_tokens is not None and eval_ctx else None)

    summary = {
        "source": ("data_report.accounting" if data_report is not None
                    else None),
        "report_present": data_report is not None,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens,
        "block_size": block_size,
        "train_ctx": train_ctx,
        "eval_ctx": eval_ctx,
        "steps": steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "eval_iters_planned": eval_iters,
        "planned_tokens_seen_per_seed": tokens_seen,
        "token_touches_per_train_token": epochs,
        "expected_unique_token_fraction": unique_fraction,
        "val_windows_available": val_windows,
        "test_windows_available": test_windows,
    }
    checks = []
    if data_report is None:
        checks.append(_check(
            "data_budget", "训练 token 预算", "info",
            "未提供 data_report.json：无法核对计划 token 触碰次数与 train "
            "语料规模；正式实验应传入 prepare_data 生成的报告"))
        checks.append(_check(
            "eval_budget", "评测窗口预算", "info",
            "未提供 data_report.json：无法核对 val/test 是否足以组成计划中的"
            "评测窗口"))
        return summary, checks

    config_errors = [e for e in (bs_err, tc_err, steps_err, batch_err,
                                 ga_err, ei_err) if e]
    if config_errors:
        checks.append(_check(
            "data_budget", "训练 token 预算", "warn",
            "配置字段非法，无法计算 token 触碰次数：" + "；".join(config_errors)))
    elif train_tokens is None:
        checks.append(_check(
            "data_budget", "训练 token 预算", "warn",
            "data_report 缺少可读的 accounting.train.tokens：无法核对计划"
            "训练量"))
    elif train_tokens <= 0:
        checks.append(_check(
            "data_budget", "训练 token 预算", "fail",
            "data_report 的 accounting.train.tokens=0：训练集为空，"
            "训练/评测无法按计划执行"))
    elif tokens_seen is None:
        checks.append(_check(
            "data_budget", "训练 token 预算", "warn",
            "配置缺少 steps/batch_size/train_ctx，无法计算计划 token 触碰次数"))
    else:
        touches = tokens_seen / train_tokens if train_tokens else None
        status = ("warn" if touches is not None and touches > 4.0
                  else "info" if touches is not None and touches > 1.0
                  else "pass")
        detail = (
            f"train tokens={train_tokens:,}；计划每种子 token 触碰="
            f"{tokens_seen:,}"
            + (f"（≈{touches:.2f}×train，预期覆盖约 "
               f"{unique_fraction:.1%} 的唯一 token）" if touches is not None
               else "")
            + "。>1× 表示训练会重复使用语料；这不自动推翻 held-out 对比，"
            "但必须在报告中披露，且不能与“更多独立数据”混为一谈")
        checks.append(_check("data_budget", "训练 token 预算", status, detail))

    budget_errors = [e for e in (bs_err, tc_err, ei_err) if e]
    if budget_errors:
        checks.append(_check(
            "eval_budget", "评测窗口预算", "warn",
            "配置字段非法，无法核对评测窗口：" + "；".join(budget_errors)))
    elif val_tokens is None or eval_ctx is None:
        checks.append(_check(
            "eval_budget", "评测窗口预算", "warn",
            "data_report 缺少 accounting.val.tokens 或配置缺少 "
            "block_size/train_ctx：无法核对 val 评测窗口"))
    elif val_tokens <= eval_ctx:
        checks.append(_check(
            "eval_budget", "评测窗口预算", "fail",
            f"val tokens={val_tokens}，不足一个 eval_ctx={eval_ctx} 的窗口："
            "训练内验证/checkpoint 选择无法执行，PPL 端点不可检验"))
    else:
        detail = (f"val tokens={val_tokens:,}，eval_ctx={eval_ctx}，可用窗口="
                  f"{val_windows}")
        if eval_iters is not None and val_windows < eval_iters:
            checks.append(_check(
                "eval_budget", "评测窗口预算", "warn",
                detail + f"，少于计划 eval_iters={eval_iters}：实际会评估全部"
                "可用窗口，PPL 的评测抽样误差大于计划口径"))
        else:
            checks.append(_check(
                "eval_budget", "评测窗口预算", "pass",
                detail + (f"，覆盖计划 eval_iters={eval_iters}"
                          if eval_iters is not None else "")))
    if test_tokens is not None and eval_ctx is not None:
        if test_tokens <= eval_ctx:
            checks.append(_check(
                "test_eval_budget", "held-out test 预算", "warn",
                f"test tokens={test_tokens}，不足一个 eval_ctx={eval_ctx} 的"
                "窗口：最终 test PPL 无法计算，会回退 val（此时 test 不再是"
                "独立确认集）"))
        else:
            checks.append(_check(
                "test_eval_budget", "held-out test 预算", "info",
                f"test tokens={test_tokens:,}，可用窗口={test_windows}；"
                "test 只在最终评估访问"))
    return summary, checks


def _scope_readiness(registered: dict, config_file, data_report=None) -> tuple:
    """开跑前核对预注册 scope 的配置规模与数据分词器。

    汇总阶段已有 registration_scope_check，但那时算力已经花掉。设计层在
    训练前把同一组“结论可迁移性”前提显形：配置规模字段（d_model/n_layers/
    steps 等）和 scope.data 声明的分词器。lr/weight_decay 不比较，因为基线
    调优按协议在 val 上选择后统一注入。
    """
    registered = _as_dict(registered)
    scope = _as_dict(registered.get("scope"))
    summary = {"declared_config": None, "config_diffs": [],
               "declared_data": scope.get("data"),
               "actual_tokenizer": None}
    checks = []
    want_rel = scope.get("config")
    if not isinstance(want_rel, str) or not want_rel:
        checks.append(_check(
            "scope_config", "预注册配置范围", "info",
            "登记表未声明 scope.config：无法在开跑前核对模型/训练规模"))
    elif config_file is None:
        checks.append(_check(
            "scope_config", "预注册配置范围", "info",
            "未提供实际配置路径：无法核对是否落在预注册配置规模内"))
    else:
        want_path = (want_rel if os.path.isabs(want_rel)
                     else os.path.join(_DESIGN_ROOT, want_rel))
        got_path = (config_file if os.path.isabs(str(config_file))
                    else os.path.abspath(str(config_file)))

        def _load(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                return None
            return doc if isinstance(doc, dict) else None

        want_cfg = _load(want_path)
        got_cfg = _load(got_path)
        summary["declared_config"] = want_rel
        if want_cfg is None:
            checks.append(_check(
                "scope_config", "预注册配置范围", "warn",
                f"登记 scope.config 不可读：{want_rel}；无法核对配置规模"))
        elif got_cfg is None:
            checks.append(_check(
                "scope_config", "预注册配置范围", "warn",
                f"实际配置不可读：{got_path}；无法核对配置规模"))
        else:
            def _effective(doc, key):
                if key == "train_ctx" and key not in doc:
                    return doc.get("block_size")
                return doc.get(key)

            diffs = [
                f"{k}: 登记={_effective(want_cfg, k)} vs "
                f"实际={_effective(got_cfg, k)}"
                for k in SCOPE_CONFIG_KEYS
                if k in want_cfg
                and _effective(want_cfg, k) != _effective(got_cfg, k)]
            summary["config_diffs"] = diffs
            if diffs:
                checks.append(_check(
                    "scope_config", "预注册配置范围", "fail",
                    "模型/训练规模与预注册配置不一致（" + "；".join(diffs)
                    + "）：本轮不在登记范围内，确认性判定将降级"))
            else:
                checks.append(_check(
                    "scope_config", "预注册配置范围", "pass",
                    f"配置规模与预注册 {want_rel} 一致（"
                    + ", ".join(SCOPE_CONFIG_KEYS) + "）"))

    scope_data = str(scope.get("data") or "")
    tokenizer = _as_dict(_as_dict(data_report).get("tokenizer"))
    actual_tok = tokenizer.get("kind") or tokenizer.get("name")
    summary["actual_tokenizer"] = actual_tok
    if not scope_data:
        checks.append(_check(
            "scope_data", "预注册数据范围", "info",
            "登记表未声明 scope.data：无法核对数据/分词器范围"))
    elif data_report is None or not actual_tok:
        checks.append(_check(
            "scope_data", "预注册数据范围", "info",
            "缺少可读的 data_report/tokenizer：无法在开跑前核对 scope.data"))
    elif ("gpt2" in scope_data.lower().replace("-", "")
          and "gpt2" not in str(actual_tok).lower().replace("-", "")):
        checks.append(_check(
            "scope_data", "预注册数据范围", "fail",
            f"登记 scope.data 声明 GPT-2 BPE，实际分词器={actual_tok}："
            "语言建模任务已改变，不在预注册范围内"))
    else:
        checks.append(_check(
            "scope_data", "预注册数据范围", "pass",
            f"数据分词器={actual_tok} 与 scope.data 声明一致"))
    return summary, checks


def build_design_plan(config: dict, seeds, variants, registered: dict = None,
                      *, config_file: str = None, config_sha256: str = None,
                      alpha: float = None, power: float = None,
                      ruler_n_samples: int = None, smoke: bool = False,
                      known_variants=None, analysis_variants=None,
                      multiplicity: str = None, data_report=None) -> dict:
    """根据配置/种子计划/预注册表生成训练前的设计计划。

    不训练、不访问 test；只做输入解析、解析式功效计算，以及可选的
    ``data_report`` 语料规模核算（token 预算/val 窗口），因此可以在正式
    训练之前（基线调优等准备步骤之后）生成并落盘。``registered`` 为登记表 dict（可空）；
    ``known_variants`` 用于把假设里的候选/参照映射到变体，缺省时惰性导入。

    返回 dict；``status`` 为 ok/warning/fail。``fail`` 不在此处中止实验，
    由调用方决定是否继续，但会进入汇总页与确认性降级证据。
    """
    config = _as_dict(config)
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds 必须是非空列表")
    seeds = [_seed_int(s, "seed") for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds 不能重复")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("variants 必须是非空列表")
    variants = [str(v) for v in variants]
    if len(set(variants)) != len(variants):
        raise ValueError("variants 不能重复")

    design_cfg = _as_dict(config.get("design"))
    alpha = (_open_unit(alpha, "alpha") if alpha is not None
             else _open_unit(design_cfg.get("alpha", DEFAULT_ALPHA), "alpha"))
    power = (_open_unit(power, "power") if power is not None
             else _open_unit(design_cfg.get("power", DEFAULT_POWER), "power"))
    order_mode = design_cfg.get("run_order", DEFAULT_RUN_ORDER)
    if not isinstance(order_mode, str) or order_mode not in RUN_ORDER_MODES:
        raise ValueError(
            f"design.run_order 必须是 {RUN_ORDER_MODES} 之一，"
            f"收到 {order_mode!r}")
    declared_order_seed = design_cfg.get("run_order_seed")
    if declared_order_seed is not None:
        declared_order_seed = _seed_int(declared_order_seed,
                                        "design.run_order_seed")
    run_order = build_run_order(variants, seeds, mode=order_mode,
                                order_seed=declared_order_seed)
    ppl_grid = design_cfg.get("ppl_paired_sd_rel_grid", list(PPL_SD_GRID))
    ruler_grid = design_cfg.get("ruler_discordance_grid",
                                list(RULER_DISCORDANCE_GRID))
    ruler_sd_grid = design_cfg.get("ruler_paired_sd_grid",
                                   list(RULER_SD_GRID))
    try:
        ppl_grid = sorted({_finite_float(x, "ppl_paired_sd_rel_grid",
                                         positive=True)
                           for x in _finite_sequence(ppl_grid,
                                                     "ppl_paired_sd_rel_grid")})
        ruler_grid = sorted({
            _finite_float(x, "ruler_discordance_grid", positive=True,
                          upper=1.0)
            for x in _finite_sequence(ruler_grid,
                                      "ruler_discordance_grid")})
        ruler_sd_grid = sorted({
            _finite_float(x, "ruler_paired_sd_grid", positive=True,
                          upper=1.0)
            for x in _finite_sequence(ruler_sd_grid,
                                      "ruler_paired_sd_grid")})
    except ValueError as e:
        raise ValueError(f"设计网格非法：{e}") from None
    if not ppl_grid or not ruler_grid or not ruler_sd_grid:
        raise ValueError("设计网格不能为空")

    ruler_cfg = _as_dict(config.get("ruler"))
    if ruler_n_samples is None:
        ruler_n_samples = ruler_cfg.get("n_samples", 30)
    ruler_n_samples = _strict_int(ruler_n_samples, "ruler n_samples",
                                 minimum=1)
    ctx_lens = ruler_cfg.get("ctx_lens")
    ctx_values = []
    ctx_errors = []
    if ctx_lens is not None:
        if not isinstance(ctx_lens, (list, tuple)):
            ctx_errors.append("ruler.ctx_lens 必须是整数列表")
        elif not ctx_lens:
            ctx_errors.append("ruler.ctx_lens 不能为空")
        else:
            for c in ctx_lens:
                try:
                    ctx_values.append(_strict_int(
                        c, "ruler.ctx_lens", minimum=1))
                except ValueError as e:
                    ctx_errors.append(str(e))
    n_contexts = len(set(ctx_values)) if ctx_values else 1

    registered = _as_dict(registered)
    hyps = [h for h in _as_list(registered.get("hypotheses"))
            if isinstance(h, dict)]
    scope = _as_dict(registered.get("scope"))
    scope_seeds = scope.get("seeds")
    scope_seeds = (scope_seeds if isinstance(scope_seeds, list) else None)
    scope_seeds_int = None
    if scope_seeds is not None:
        try:
            scope_seeds_int = [_seed_int(s, "预注册 scope seed")
                               for s in scope_seeds]
        except ValueError:
            scope_seeds_int = None

    known = (set(known_variants) if known_variants is not None
             else _known_variant_names())
    planned = set(variants)
    analysis_explicit = analysis_variants is not None
    if analysis_variants is None:
        # 默认只把真正会产出机制分析的门控变体列入分析人群；standard
        # 无输出门、不写 analysis_*.json，不能因此被判成“分析缺失”。
        analysis_variants = [v for v in variants
                             if v in _DEFAULT_GATED_FOR_ANALYSIS]
    if not isinstance(analysis_variants, (list, tuple)):
        raise ValueError("analysis_variants 必须是列表")
    analysis_variants = [str(v) for v in analysis_variants]
    unknown_analysis = sorted(set(analysis_variants) - planned)
    if unknown_analysis:
        raise ValueError(
            f"analysis_variants 含计划外变体：{unknown_analysis}")
    n_seeds = len(seeds)

    mode = (multiplicity if multiplicity is not None
            else design_cfg.get("multiplicity", DEFAULT_MULTIPLICITY))
    if not isinstance(mode, str) or mode not in MULTIPLICITY_MODES:
        raise ValueError(
            f"design.multiplicity 必须是 {MULTIPLICITY_MODES} 之一，"
            f"收到 {mode!r}")
    family_sizes = planned_family_sizes(
        variants, registered,
        analysis_variants if analysis_explicit else None)
    ppl_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ppl_family_size"],
        family_sizes["global_family_size"], mode)
    ruler_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ruler_family_size"],
        family_sizes["global_family_size"], mode)
    auroc_multiplicity = multiplicity_alpha(
        alpha, family_sizes["auroc_family_size"],
        family_sizes["global_family_size"], mode)
    ap_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ap_family_size"],
        family_sizes["global_family_size"], mode)
    ppl_alpha = ppl_multiplicity["alpha_effective"]
    ruler_alpha = ruler_multiplicity["alpha_effective"]
    auroc_alpha = auroc_multiplicity["alpha_effective"]
    ap_alpha = ap_multiplicity["alpha_effective"]

    # ---- 检查 1：最小种子数（低于 3 不判定显著性，属协议内但只能探索）
    if n_seeds >= MIN_SEEDS:
        checks = [_check("seed_minimum", "种子数下限", "pass",
                         f"计划 {n_seeds} 个种子（>= {MIN_SEEDS}）")]
    else:
        checks = [_check(
            "seed_minimum", "种子数下限", "warn",
            f"计划只有 {n_seeds} 个种子（< {MIN_SEEDS}）：只能描述，"
            "不能做显著性/组件有效性判定")]

    look_budget_raw = design_cfg.get("test_look_budget",
                                     DEFAULT_TEST_LOOK_BUDGET)
    try:
        test_look_budget = _strict_int(
            look_budget_raw, "design.test_look_budget", minimum=1)
    except ValueError as e:
        test_look_budget = DEFAULT_TEST_LOOK_BUDGET
        checks.append(_check(
            "test_look_budget", "测试集 look 预算", "fail", str(e)))
    else:
        explicit_look_budget = "test_look_budget" in design_cfg
        checks.append(_check(
            "test_look_budget", "测试集 look 预算",
            "pass" if explicit_look_budget else "info",
            f"允许 {test_look_budget} 个独立流水线运行访问同一测试版本；"
            "超出后依赖测试集的确认性判定降级为探索性"
            + ("" if explicit_look_budget else "（未显式声明，采用默认 1 次）")))

    # ---- 检查 2：与预注册种子计划一致（少跑不等于“范围外”，但确认性
    # 证据不再覆盖登记的全部种子；必须显式降级/告警）
    if scope_seeds_int is None:
        checks.append(_check(
            "seed_scope", "预注册种子计划", "warn",
            "登记表未提供可读的 scope.seeds，无法核对计划种子是否完整；"
            "确认性结论仍受范围门约束"))
    else:
        missing_scope = [s for s in scope_seeds_int if s not in seeds]
        extra_scope = [s for s in seeds if s not in scope_seeds_int]
        if missing_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "fail",
                f"计划缺少预注册种子 {missing_scope}：即使跑完，"
                "登记范围内仍有未观测种子；确认性判定应降级为探索性"))
        elif extra_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "warn",
                f"计划包含未登记种子 {extra_scope}：结论含登记计划外的抽样，"
                "证据强度打折"))
        else:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "pass",
                f"计划种子与登记表一致（{seeds}）"))

    # ---- 检查 2b：开跑前核对预注册 scope 的模型规模与数据分词器
    scope_readiness, scope_checks = _scope_readiness(
        registered, config_file, data_report)
    checks.extend(scope_checks)

    # ---- 检查 3：运行顺序必须预先随机化，避免变体与时间/硬件漂移混淆
    order_problems = validate_run_order(run_order, variants, seeds)
    if order_problems:
        checks.append(_check(
            "run_order", "run 顺序分配", "fail",
            "；".join(order_problems)))
    elif order_mode == "variant_major":
        checks.append(_check(
            "run_order", "run 顺序分配", "warn",
            "顺序为 variant_major（每个变体的全部种子连续跑）：变体效应与"
            "时间/硬件漂移完全混淆，正式实验应改用 randomized_blocked；"
            "本计划已记录顺序，便于审计但不能消除该混淆"))
    else:
        extra = (f"；order_seed={run_order.get('order_seed')}"
                 f"（{run_order.get('seed_source')}）"
                 if order_mode == "randomized_blocked" else "；未随机化")
        checks.append(_check(
            "run_order", "run 顺序分配", "pass",
            f"模式={order_mode}，种子为区组，共 {run_order.get('n_blocks')} "
            f"个区块 × {len(variants)} 个变体{extra}；顺序不读取任何结果"))

    # ---- 检查 4：登记假设的候选/参照变体都必须在计划矩阵内
    missing_hyps = []
    unmapped = []
    for h in hyps:
        hid = h.get("id", "?")
        cand = h.get("candidate")
        ref = h.get("reference")
        for role, name in (("候选", cand), ("参照", ref)):
            if not isinstance(name, str) or not name:
                missing_hyps.append(f"{hid} 的{role}未命名")
                continue
            if name in known and name not in planned:
                missing_hyps.append(f"{hid} 的{role} {name} 不在计划变体中")
            elif name not in known and name not in planned:
                # “仅查询项基线”等分析内对照不是训练变体：只要求候选可映射。
                if role == "参照":
                    unmapped.append(f"{hid} 的参照 {name} 为分析内对照（无对应变体）")
                else:
                    missing_hyps.append(
                        f"{hid} 的候选 {name} 既不是已知变体也不在计划矩阵中")
    if not hyps:
        checks.append(_check("hypothesis_variants", "预注册假设覆盖", "warn",
                             "登记表无假设可核对"))
    elif missing_hyps:
        checks.append(_check(
            "hypothesis_variants", "预注册假设覆盖", "fail",
            "计划矩阵无法检验全部登记假设：" + "；".join(missing_hyps)))
    else:
        detail = f"{len(hyps)} 条登记假设的候选/训练变体均在计划矩阵中"
        if unmapped:
            detail += "；" + "；".join(sorted(set(unmapped))[:3])
        checks.append(_check("hypothesis_variants", "预注册假设覆盖",
                             "pass", detail))

    # ---- 检查 5：先验功效必须按最终确认阈值规划，而非名义 α
    mult_detail = (
        f"模式={mode}；PPL/RULER 计划家族 m="
        f"{family_sizes['ppl_family_size']}，全局确认族 k="
        f"{family_sizes['global_family_size']}；"
        f"PPL α_eff={ppl_alpha:.6g}、RULER α_eff={ruler_alpha:.6g}、"
        f"AUROC α_eff={auroc_alpha:.6g}、AP α_eff={ap_alpha:.6g}"
        f"（名义 α={alpha:.6g}）。按 Holm 最坏情形 α/max(m,k) 规划，"
        "避免用未校正阈值高估确认性功效")
    if not (ppl_multiplicity["representable"]
            and ruler_multiplicity["representable"]):
        mult_detail += "；警告：家族负担过大导致 α/m 下溢，校正阈值不可表示"
        mult_status = "warn"
    else:
        mult_status = "info"
    checks.append(_check("multiplicity_plan", "确认性多重比较规划",
                         mult_status, mult_detail))

    # ---- PPL 先验灵敏度：对合理 SD 网格给 MDE/功效/所需种子数
    ppl_margin = 0.01  # 与 lega.equivalence.metric_margin("PPL") 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("PPL")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ppl_margin = _finite_float(spec.get("margin"), "PPL SESOI",
                                       positive=True)
    except Exception:  # pragma: no cover - 保持模块可独立使用
        pass
    ppl_rows = []
    can_power = n_seeds >= 2
    for sd in ppl_grid:
        if can_power:
            mde = min_detectable_effect(sd, n_seeds, alpha, power)
            pw = paired_t_power(ppl_margin, sd, n_seeds, alpha)
            mde_confirm = min_detectable_effect(
                sd, n_seeds, ppl_alpha, power)
            pw_confirm = paired_t_power(ppl_margin, sd, n_seeds, ppl_alpha)
            pw_nct = (paired_t_power_nct(
                ppl_margin, sd, n_seeds, alpha, n_sim=20000, seed=0)["power"]
                if 2 <= n_seeds <= 10 else None)
        else:
            mde = pw = mde_confirm = pw_confirm = pw_nct = None
        ppl_rows.append({
            "paired_sd_relative": sd,
            "mde_relative": mde,
            "power_for_sesoi": pw,
            "power_for_sesoi_nct": pw_nct,
            "required_seeds_for_sesoi": required_pairs(
                ppl_margin, sd, alpha, power),
            "mde_relative_confirmatory": mde_confirm,
            "power_for_sesoi_confirmatory": pw_confirm,
            "required_seeds_for_sesoi_confirmatory": required_pairs(
                ppl_margin, sd, ppl_alpha, power),
        })
    ppl_break_even = (max_sd_for_effect(ppl_margin, n_seeds, alpha, power)
                      if can_power else None)
    ppl_break_even_confirm = (
        max_sd_for_effect(ppl_margin, n_seeds, ppl_alpha, power)
        if can_power else None)
    declared_ppl_sd = None
    declared_disc = None
    if "ppl_paired_sd_rel" in design_cfg:
        declared_ppl_sd = design_cfg.get("ppl_paired_sd_rel")
        declared_ppl_sd = _finite_float(declared_ppl_sd,
                                        "design.ppl_paired_sd_rel",
                                        positive=True)
        need = required_pairs(ppl_margin, declared_ppl_sd, alpha, power)
        pw = (paired_t_power(ppl_margin, declared_ppl_sd, n_seeds, alpha)
              if can_power else None)
        ok = (can_power and need is not None and need <= n_seeds)
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）",
            "pass" if ok else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw:.3f}" if pw is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need} 个种子（当前 {n_seeds}）" if need is not None
               else "所需种子数超出可计算范围")
            + ("" if ok else "：设计对 SESOI 功效不足")))
    else:
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）", "info",
            "配置未声明 design.ppl_paired_sd_rel：不把任一 SD 假设当作已知，"
            "仅报告灵敏度曲线（见 ppl.sensitivity）"))

    if declared_ppl_sd is not None:
        need_c = required_pairs(ppl_margin, declared_ppl_sd,
                                ppl_alpha, power)
        pw_c = (paired_t_power(ppl_margin, declared_ppl_sd,
                              n_seeds, ppl_alpha)
                if can_power else None)
        ok_c = (can_power and need_c is not None and need_c <= n_seeds)
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "pass" if ok_c else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；按 {mode} 的 "
            f"α_eff={ppl_alpha:.6g}（家族 m="
            f"{family_sizes['ppl_family_size']}，全局 k="
            f"{family_sizes['global_family_size']}），{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw_c:.3f}" if pw_c is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need_c} 个种子（当前 {n_seeds}）"
               if need_c is not None else "所需种子数超出可计算范围")
            + ("" if ok_c else "：通过多重比较后对 SESOI 功效不足，"
               "当前设计不能确认该效应")))
    else:
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "info",
            "未声明 design.ppl_paired_sd_rel；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    # 参考 SD 不是数据估计，只用于展示“常见噪声水平下”的设计灵敏度。
    ref_sd = DEFAULT_PPL_SD_REF
    ref_pw = (paired_t_power(ppl_margin, ref_sd, n_seeds, alpha)
              if can_power else None)
    ref_pw_confirm = (paired_t_power(ppl_margin, ref_sd, n_seeds, ppl_alpha)
                      if can_power else None)
    ref_need = required_pairs(ppl_margin, ref_sd, alpha, power)
    ref_need_confirm = required_pairs(ppl_margin, ref_sd, ppl_alpha, power)
    ref_detail = (
        f"若配对差 SD={ref_sd:.1%}（参考值），{n_seeds} 种子对 "
        f"{ppl_margin:.1%} SESOI 的未校正功效≈"
        + (f"{ref_pw:.3f}" if ref_pw is not None else "无法计算（n<2）")
        + (f"；按确认性 α_eff={ppl_alpha:.6g} 仅≈{ref_pw_confirm:.3f}"
           if ref_pw_confirm is not None else "")
        + (f"，需 {ref_need} 个种子（校正后 {ref_need_confirm}）"
           if ref_need is not None and ref_need_confirm is not None
           else ""))
    if ppl_break_even is not None:
        ref_detail += f"；未校正 80% 功效下最大 SD≈{ppl_break_even:.3%}"
    if ppl_break_even_confirm is not None:
        ref_detail += f"（确认性 {ppl_break_even_confirm:.3%}）"
    checks.append(_check("ppl_reference_sensitivity",
                         "PPL 参考灵敏度（非数据估计）", "info",
                         ref_detail))
    if (ref_pw_confirm is not None and ref_pw_confirm < power):
        checks.append(_check(
            "ppl_confirmatory_reference", "PPL 确认性参考功效", "warn",
            f"参考 SD={ref_sd:.1%} 时，{n_seeds} 种子对 "
            f"{ppl_margin:.1%} SESOI 的确认性功效仅≈{ref_pw_confirm:.3f}"
            f"（目标 {power:.0%}；α_eff={ppl_alpha:.6g}，"
            f"需约 {ref_need_confirm} 个种子）。{n_seeds} 个种子只能"
            "可靠检出远大于 1% 的 PPL 差异；1% SESOI 结论应标为"
            "描述性/探索性"))

    # ---- 机制端点（H6/H7）的设计灵敏度：跨种子配对差的绝对尺度。
    # 只规划 PPL/RULER 会漏掉“机制主张是否值得跑”：AUROC/AP 的种子级
    # 配对差同样需要先验 MDE，否则 5 个种子只能在事后发现功效不足。
    mechanism_hyps = [
        h for h in hyps
        if isinstance(h.get("metric"), str)
        and (h["metric"].startswith("AUROC") or h["metric"].startswith("AP差"))]
    missing_mechanism = []
    for h in mechanism_hyps:
        cand = h.get("candidate")
        if isinstance(cand, str) and cand in planned and cand not in set(
                analysis_variants):
            missing_mechanism.append(f"{h.get('id', '?')} 的候选 {cand}")
    if mechanism_hyps and (not analysis_variants or missing_mechanism):
        checks.append(_check(
            "mechanism_endpoint_variants", "机制端点分析人群", "fail",
            "登记了机制假设，但计划内没有可产出机制分析的变体："
            + ("计划 analysis_variants 为空；" if not analysis_variants else "")
            + ("；".join(missing_mechanism) if missing_mechanism else "")
            + "。H6/H7 将无法检验，不能等跑完再把缺失解释成无效应"))
    elif mechanism_hyps:
        checks.append(_check(
            "mechanism_endpoint_variants", "机制端点分析人群", "pass",
            f"{len(mechanism_hyps)} 条机制假设的候选均在分析变体子集中"
            f"（{len(analysis_variants)} 个）"))
    else:
        checks.append(_check(
            "mechanism_endpoint_variants", "机制端点分析人群", "info",
            "登记表没有 AUROC/AP 机制假设；仍保留机制端点灵敏度表作参考"))

    analysis_cfg = _as_dict(config.get("analysis"))
    n_analysis_examples = analysis_cfg.get("n_examples")
    if n_analysis_examples is not None:
        n_analysis_examples = _strict_int(n_analysis_examples,
                                          "analysis.n_examples", minimum=1)
        if n_analysis_examples < 10:
            checks.append(_check(
                "mechanism_analysis_n", "机制分析样例数", "warn",
                f"analysis.n_examples={n_analysis_examples} 偏少：单 run 的"
                "AUROC/AP 及其差值会很不稳定；这是评测抽样误差，"
                "不会进入种子级 sd 假设表，须看样例聚类 bootstrap"))
        else:
            checks.append(_check(
                "mechanism_analysis_n", "机制分析样例数", "info",
                f"analysis.n_examples={n_analysis_examples}（单 run 评测"
                "抽样误差另见聚类 bootstrap，不并入种子级功效）"))
    else:
        checks.append(_check(
            "mechanism_analysis_n", "机制分析样例数", "info",
            "配置未声明 analysis.n_examples：无法核对机制端点的单 run "
            "测量精度；默认值由机制分析代码决定"))

    mechanism_endpoints = {}
    for key, metric_name, grid_key, declared_key, endpoint_alpha in (
            ("auroc", "AUROC差(logit−仅查询)", "auroc_paired_sd_grid",
             "auroc_paired_sd", auroc_alpha),
            ("ap", "AP差(logit−仅查询)", "ap_paired_sd_grid",
             "ap_paired_sd", ap_alpha)):
        ep = _mechanism_endpoint_sensitivity(
            design_cfg, metric_name, grid_key, declared_key,
            n_seeds, alpha, power)
        _fill_confirmatory_rows(ep["sensitivity"], endpoint_alpha, n_seeds,
                                ep["sesoi_absolute"], power)
        ep["multiplicity"] = dict(
            auroc_multiplicity if key == "auroc" else ap_multiplicity)
        mechanism_endpoints[key] = ep
        declared = ep.get("declared_paired_sd")
        title = "AUROC" if key == "auroc" else "AP"
        if declared is not None:
            need = required_pairs(ep["sesoi_absolute"], declared,
                                  endpoint_alpha, power)
            pw = (paired_t_power(ep["sesoi_absolute"], declared, n_seeds,
                                 endpoint_alpha) if can_power else None)
            ok = (can_power and need is not None and need <= n_seeds)
            checks.append(_check(
                f"{key}_declared_power", f"{title} 确认性计划功效",
                "pass" if ok else "fail",
                f"声明跨种子配对差 SD={declared:.4g}；{n_seeds} 种子对 "
                f"{ep['sesoi_absolute']:.3f} SESOI 的确认性功效≈"
                + (f"{pw:.3f}" if pw is not None else "无法计算（n<2）")
                + (f"，需 {need} 个种子" if need is not None else "")
                + ("" if ok else "：机制端点在最终多重比较阈值下功效不足，"
                   "H6/H7 的支持/拒绝只能作探索性线索")))
        else:
            checks.append(_check(
                f"{key}_declared_power", f"{title} 确认性计划功效", "info",
                f"配置未声明 design.{declared_key}：只报告灵敏度曲线，"
                "不把任一 SD 假设当作已知"))

    ref_sd_mech = DEFAULT_MECHANISM_SD_REF
    ref_details = []
    weak_mech = []
    for key, title in (("auroc", "AUROC"), ("ap", "AP")):
        ep = mechanism_endpoints[key]
        endpoint_alpha = (auroc_alpha if key == "auroc" else ap_alpha)
        try:
            pw = (paired_t_power(ep["sesoi_absolute"], ref_sd_mech, n_seeds,
                                 endpoint_alpha) if can_power else None)
            need = required_pairs(ep["sesoi_absolute"], ref_sd_mech,
                                  endpoint_alpha, power)
        except ValueError:
            pw, need = None, None
        ref_details.append(
            f"{title} 参考 SD={ref_sd_mech:.3g} 时确认性功效≈"
            + (f"{pw:.3f}" if pw is not None else "无法计算")
            + (f"、需 {need} 个种子" if need is not None else ""))
        if isinstance(pw, float) and pw < power:
            weak_mech.append(title)
    checks.append(_check(
        "mechanism_reference_sensitivity", "机制端点参考灵敏度（非数据估计）",
        "info", "；".join(ref_details)
        + "。参考值是展示用横轴，不是从当前数据估计的效应量"))
    if weak_mech and mechanism_hyps:
        checks.append(_check(
            "mechanism_confirmatory_reference", "机制端点确认性参考功效",
            "warn",
            "、".join(weak_mech) + f" 在参考 SD={ref_sd_mech:.3g} 下、按"
            "最终 Holm 阈值规划时低于目标功效：即使种子级检验显著，"
            "也不能把 H6/H7 当作稳健机制结论；应增加种子或降低端点方差"))

    # ---- 上下文档位设计：至少两档且跨度足够，才谈得上“长度校准”
    if ctx_errors:
        checks.append(_check(
            "ruler_context_values", "RULER 上下文取值", "fail",
            "；".join(ctx_errors) + "：RULER 实际入口要求每个档位为正整数"))
    elif ctx_lens is None:
        checks.append(_check(
            "ruler_context_values", "RULER 上下文取值", "info",
            "配置未声明 ruler.ctx_lens：RULER 端点可能不会运行"))
    else:
        checks.append(_check(
            "ruler_context_values", "RULER 上下文取值", "pass",
            f"上下文档位均为正整数：{sorted(set(ctx_values))}"))

    if not ctx_errors and ctx_values:
        unique_ctx = sorted(set(ctx_values))
        if len(unique_ctx) < 2:
            checks.append(_check(
                "ruler_context_span", "RULER 上下文档位", "warn",
                f"计划只有 {len(unique_ctx)} 个有效上下文档位：无法检验"
                "长度剂量-反应/长度校准主张，跨长度平均也不能替代分档结果"))
        else:
            ratio = max(unique_ctx) / min(unique_ctx)
            if ratio < 2.0:
                checks.append(_check(
                    "ruler_context_span", "RULER 上下文档位", "warn",
                    f"上下文档位跨度仅 {min(unique_ctx):g}–"
                    f"{max(unique_ctx):g}（×{ratio:.2f}）：长度效应可能被"
                    "限制在过窄范围，只能作该范围内的结论"))
            else:
                checks.append(_check(
                    "ruler_context_span", "RULER 上下文档位", "pass",
                    f"上下文档位 {min(unique_ctx):g}–{max(unique_ctx):g}"
                    f"（×{ratio:.2f}，{len(unique_ctx)} 档）：可检验长度趋势"))

    block_size, bs_err = _optional_positive_int(
        config.get("block_size"), "block_size")
    train_ctx, tc_err = _optional_positive_int(
        config.get("train_ctx"), "train_ctx")
    model_errors = [e for e in (bs_err, tc_err) if e]
    if model_errors:
        checks.append(_check(
            "model_context", "模型上下文能力", "fail",
            "；".join(model_errors) + "：无法核对训练/评测上下文长度"))
    elif block_size is None or train_ctx is None:
        checks.append(_check(
            "model_context", "模型上下文能力", "info",
            "配置未完整声明 block_size/train_ctx：跳过模型上下文能力核对"))
    elif train_ctx > block_size:
        checks.append(_check(
            "model_context", "模型上下文能力", "fail",
            f"train_ctx={train_ctx} > block_size={block_size}：训练窗口"
            "超过模型位置编码容量，训练无法按计划执行"))
    else:
        checks.append(_check(
            "model_context", "模型上下文能力", "pass",
            f"train_ctx={train_ctx} <= block_size={block_size}；"
            f"评测上下文按 min(block_size, train_ctx)={min(block_size, train_ctx)}"))

    if ctx_errors:
        checks.append(_check(
            "ruler_context_model", "RULER 上下文与模型容量", "info",
            "上下文取值非法，无法核对模型容量"))
    elif not ctx_values:
        checks.append(_check(
            "ruler_context_model", "RULER 上下文与模型容量", "info",
            "未声明有效 RULER 上下文档位"))
    elif block_size is None:
        checks.append(_check(
            "ruler_context_model", "RULER 上下文与模型容量", "info",
            "未声明 block_size：无法核对 RULER 最长上下文是否超出模型容量"))
    else:
        max_ctx = max(ctx_values)
        if max_ctx > block_size:
            checks.append(_check(
                "ruler_context_model", "RULER 上下文与模型容量", "fail",
                f"RULER 最大上下文={max_ctx} > block_size={block_size}："
                "该档位无法按计划评测；RULER 结论只能限定在可运行档位，"
                "确认性长度/长上下文主张应降级"))
        elif train_ctx is not None and max_ctx > train_ctx:
            checks.append(_check(
                "ruler_context_model", "RULER 上下文与模型容量", "warn",
                f"RULER 最大上下文={max_ctx} > train_ctx={train_ctx}："
                "属于长度外推测试，不能把结果当作训练分布内的长度效应；"
                "必须在报告中明确区分"))
        else:
            checks.append(_check(
                "ruler_context_model", "RULER 上下文与模型容量", "pass",
                f"RULER 最大上下文={max_ctx} <= block_size={block_size}"
                + (f" 且 <= train_ctx={train_ctx}" if train_ctx is not None
                   else "")))

    # ---- 长度零假设诊断：覆盖最长档位，且长度/模式在运行前合法
    ln_cfg = _as_dict(config.get("length_null"))
    if ln_cfg.get("enabled") is not True:
        checks.append(_check(
            "length_null_coverage", "长度零假设诊断", "info",
            "未启用 length_null：长度校准主张只能作为探索性结果，"
            "不能进入确认性判定"))
    else:
        ln_errors = []
        ln_lengths = []
        raw_lengths = ln_cfg.get("lengths")
        explicit_lengths = raw_lengths is not None
        if explicit_lengths:
            if isinstance(raw_lengths, str):
                raw_lengths = [part.strip() for part in raw_lengths.split(",")
                               if part.strip()]
            if not isinstance(raw_lengths, (list, tuple)) or not raw_lengths:
                ln_errors.append(
                    "length_null.lengths 必须是非空整数列表或逗号分隔字符串")
            else:
                for value in raw_lengths:
                    if isinstance(value, str):
                        text = value.strip()
                        if text and text.lstrip("+-").isdigit():
                            value = int(text)
                    try:
                        ln_lengths.append(_strict_int(
                            value, "length_null.lengths", minimum=2))
                    except ValueError as e:
                        ln_errors.append(str(e))
        raw_modes = ln_cfg.get("modes", "random,repeated,uniform")
        if isinstance(raw_modes, str):
            modes = [m.strip() for m in raw_modes.split(",") if m.strip()]
        elif isinstance(raw_modes, (list, tuple)):
            modes = [str(m).strip() for m in raw_modes if str(m).strip()]
        else:
            modes = []
            ln_errors.append(
                "length_null.modes 需要字符串或字符串列表")
        if not modes:
            ln_errors.append("length_null.modes 必须是非空列表")
        invalid_modes = sorted(set(modes) - set(_LENGTH_NULL_MODES))
        if invalid_modes:
            ln_errors.append(
                f"length_null.modes 含未知模式 {invalid_modes}，"
                f"允许 {list(_LENGTH_NULL_MODES)}")
        if ln_errors:
            checks.append(_check(
                "length_null_coverage", "长度零假设诊断", "fail",
                "；".join(ln_errors)))
        elif not explicit_lengths:
            checks.append(_check(
                "length_null_coverage", "长度零假设诊断", "info",
                f"未显式声明 length_null.lengths：运行时按 block_size/"
                f"RULER 档位生成默认长度；模式={modes} 合法。计划不复制"
                "默认长度生成规则，只核对显式配置"))
        else:
            max_ln = max(ln_lengths)
            if block_size is not None and max_ln > block_size:
                checks.append(_check(
                    "length_null_coverage", "长度零假设诊断", "fail",
                    f"length_null 最长长度={max_ln} > block_size="
                    f"{block_size}：该诊断无法运行，长度校准证据不完整"))
            elif ctx_values and max_ln < max(ctx_values):
                checks.append(_check(
                    "length_null_coverage", "长度零假设诊断", "warn",
                    f"length_null 最长长度={max_ln} < RULER 最大上下文="
                    f"{max(ctx_values)}：最长档位缺少同长度零假设，"
                    "该档位的长度校准主张证据不足"))
            elif len(set(ln_lengths)) < 2:
                checks.append(_check(
                    "length_null_coverage", "长度零假设诊断", "warn",
                    f"length_null 只有长度 {set(ln_lengths)}：无法区分"
                    "固定长度噪声与长度趋势"))
            else:
                checks.append(_check(
                    "length_null_coverage", "长度零假设诊断", "pass",
                    f"长度={sorted(set(ln_lengths))}，模式={modes}；"
                    "覆盖最长 RULER 档位"))

    # ---- 数据/评测预算：开跑前用 data_report 核对 token 与窗口
    data_budget, budget_checks = _design_data_budget(config, data_report)
    checks.extend(budget_checks)

    # ---- RULER 先验灵敏度：每格样本数决定 McNemar 能检出多大的边际差
    ruler_margin = 0.01  # 与 RULER平均 SESOI 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("RULER平均")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ruler_margin = _finite_float(spec.get("margin"), "RULER SESOI",
                                         positive=True)
    except Exception:  # pragma: no cover
        pass
    ruler_rows = []
    for disc in ruler_grid:
        try:
            cell = mcnemar_power(ruler_n_samples, disc, ruler_margin, alpha)
            mde = mcnemar_mde(ruler_n_samples, disc, alpha, power)
            need = mcnemar_required_samples(ruler_margin, disc, alpha, power)
            cell_confirm = mcnemar_power(
                ruler_n_samples, disc, ruler_margin, ruler_alpha)
            mde_confirm = mcnemar_mde(
                ruler_n_samples, disc, ruler_alpha, power)
            need_confirm = mcnemar_required_samples(
                ruler_margin, disc, ruler_alpha, power)
        except ValueError:
            continue
        ruler_rows.append({
            "discordance": disc,
            "power_for_sesoi": cell["power"],
            "mde_pp": mde,
            "required_samples_per_cell_for_sesoi": need,
            "power_for_sesoi_confirmatory": cell_confirm["power"],
            "mde_pp_confirmatory": mde_confirm,
            "required_samples_per_cell_for_sesoi_confirmatory": need_confirm,
        })
    ruler_break_even = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, alpha, power)
    ruler_break_even_confirm = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, ruler_alpha, power)
    declared_disc = design_cfg.get("ruler_discordance")
    if declared_disc is not None:
        declared_disc = _finite_float(declared_disc,
                                      "design.ruler_discordance",
                                      positive=True, upper=1.0)
        try:
            need = mcnemar_required_samples(ruler_margin, declared_disc,
                                            alpha, power)
            pw = mcnemar_power(ruler_n_samples, declared_disc,
                               ruler_margin, alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "fail", f"声明的不一致率无法计算：{e}"))
        else:
            ok = need is not None and need <= ruler_n_samples
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "pass" if ok else "warn",
                f"声明不一致率={declared_disc:.3g}；每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw:.3f}"
                + (f"，需每格 {need} 个样例" if need is not None else "")
                + ("" if ok else "：当前 RULER 样本量不足以确认该 SESOI，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_declared_power", "RULER 计划功效（声明假设）", "info",
            "配置未声明 design.ruler_discordance：仅报告灵敏度曲线"))

    if declared_disc is not None:
        try:
            need_c = mcnemar_required_samples(
                ruler_margin, declared_disc, ruler_alpha, power)
            pw_c = mcnemar_power(ruler_n_samples, declared_disc,
                                 ruler_margin, ruler_alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）", "fail",
                f"声明的 RULER 不一致率在确认性阈值下无法计算：{e}"))
        else:
            ok_c = need_c is not None and need_c <= ruler_n_samples
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）",
                "pass" if ok_c else "warn",
                f"声明不一致率={declared_disc:.3g}；按 {mode} 的 "
                f"α_eff={ruler_alpha:.6g}（家族 m="
                f"{family_sizes['ruler_family_size']}，全局 k="
                f"{family_sizes['global_family_size']}），每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw_c:.3f}"
                + (f"，需每格 {need_c} 个样例" if need_c is not None else "")
                + ("" if ok_c else "：通过多重比较后样本量不足，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_confirmatory_power",
            "RULER 确认性计划功效（多重比较校正）", "info",
            "未声明 design.ruler_discordance；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    ref_disc = DEFAULT_RULER_DISCORDANCE_REF
    try:
        ref_cell = mcnemar_power(ruler_n_samples, ref_disc,
                                 ruler_margin, alpha)
        ref_need = mcnemar_required_samples(ruler_margin, ref_disc,
                                            alpha, power)
        ref_mde = mcnemar_mde(ruler_n_samples, ref_disc, alpha, power)
        ref_cell_confirm = mcnemar_power(
            ruler_n_samples, ref_disc, ruler_margin, ruler_alpha)
        ref_need_confirm = mcnemar_required_samples(
            ruler_margin, ref_disc, ruler_alpha, power)
        ref_mde_confirm = mcnemar_mde(
            ruler_n_samples, ref_disc, ruler_alpha, power)
    except ValueError:
        ref_cell, ref_need, ref_mde = {"power": None}, None, None
        ref_cell_confirm, ref_need_confirm, ref_mde_confirm = (
            {"power": None}, None, None)
    cells = n_contexts * _TASKS_FOR_RULER
    pooled_n = ruler_n_samples * cells
    try:
        pooled_mde = mcnemar_mde(pooled_n, ref_disc, alpha, power)
    except ValueError:
        pooled_mde = None
    pooled_grid = []
    for disc in ruler_grid:
        try:
            pooled_grid.append(mcnemar_mde(pooled_n, disc, alpha, power))
        except ValueError:
            continue
    pooled_min = min((v for v in pooled_grid if v is not None),
                     default=None)
    checks.append(_check(
        "ruler_reference_sensitivity", "RULER 参考灵敏度（非数据估计）",
        "info",
        f"若不一致率={ref_disc:.0%}（参考值），每格 n={ruler_n_samples} "
        f"对 {ruler_margin:.1%} SESOI 的未校正功效≈"
        + ((f"{ref_cell.get('power'):.3f}")
           if isinstance(ref_cell.get("power"), float) else "-")
        + (f"；按确认性 α_eff={ruler_alpha:.6g} 仅≈"
           f"{ref_cell_confirm.get('power'):.3f}"
           if isinstance(ref_cell_confirm.get("power"), float) else "")
        + (f"，MDE≈{ref_mde:.2%}"
           + (f"（确认性 {ref_mde_confirm:.2%}）"
              if ref_mde_confirm is not None else "")
           if ref_mde is not None else "")
        + (f"，需每格 {ref_need} 个样例"
           + (f"（确认性 {ref_need_confirm}）"
              if ref_need_confirm is not None else "")
           if ref_need is not None else "")
        + (f"；在 80% 功效下能容忍的最大不一致率≈{ruler_break_even:.4%}"
           + (f"（确认性 {ruler_break_even_confirm:.4%}）"
              if ruler_break_even_confirm is not None else "")
           if ruler_break_even is not None else "")))
    # 参考不一致率下若连 50% 功效都不到，明确告警：这不是“没有效应”，
    # 而是当前每格样本量根本不足以支撑 1pp 级别的实际意义判定。
    ref_power = ref_cell.get("power")
    ref_power_confirm = ref_cell_confirm.get("power")
    if (isinstance(ref_power, float) and ref_power < 0.5
            or isinstance(ref_power_confirm, float)
            and ref_power_confirm < 0.5):
        best_case = (
            f"即使把 {cells} 个格子当作完全独立的 {pooled_n} 个样例"
            f"（对宏观平均最乐观），MDE≈{pooled_mde:.2%}"
            + (f"、网格内最乐观不一致率下仍≈{pooled_min:.2%}"
               if pooled_min is not None else "")
            + "，"
            if pooled_mde is not None else "")
        checks.append(_check(
            "ruler_sample_size", "RULER 每格样本量", "warn",
            f"每格 n={ruler_n_samples} 在参考不一致率={ref_disc:.0%} 下"
            + (f"对 {ruler_margin:.1%} SESOI 的未校正功效仅≈"
               f"{ref_power:.3f}" if isinstance(ref_power, float) else "")
            + (f"；通过多重比较后仅≈{ref_power_confirm:.3f}"
               if isinstance(ref_power_confirm, float) else "")
            + "；"
            + best_case
            + "仍远大于 1pp：RULER 的 SESOI/等价判定只能作描述性证据；"
            "如需确认 1pp 级差异，应按上表增加每格样本数并重新登记"))

    # ---- RULER 种子级先验灵敏度：确认性 H4 走跨种子配对 t（macro EM），
    # 不是逐格 McNemar。只规划每格样本数会让“确认性检验单位”没有先验
    # 功效证据：500 个样例也不能替代 5 个种子对种子方差的计划。
    declared_ruler_sd = None
    if "ruler_paired_sd" in design_cfg:
        declared_ruler_sd = _finite_float(
            design_cfg.get("ruler_paired_sd"), "design.ruler_paired_sd",
            positive=True, upper=1.0)
    ruler_seed_rows = []
    for sd in ruler_sd_grid:
        if can_power:
            mde = min_detectable_effect(sd, n_seeds, alpha, power)
            pw = paired_t_power(ruler_margin, sd, n_seeds, alpha)
            mde_confirm = min_detectable_effect(
                sd, n_seeds, ruler_alpha, power)
            pw_confirm = paired_t_power(
                ruler_margin, sd, n_seeds, ruler_alpha)
        else:
            mde = pw = mde_confirm = pw_confirm = None
        pw_nct = (paired_t_power_nct(
            ruler_margin, sd, n_seeds, alpha, n_sim=20000, seed=0)["power"]
            if 2 <= n_seeds <= 10 else None)
        ruler_seed_rows.append({
            "paired_sd_pp": sd,
            "mde_pp": mde,
            "power_for_sesoi": pw,
            "power_for_sesoi_nct": pw_nct,
            "required_seeds_for_sesoi": required_pairs(
                ruler_margin, sd, alpha, power),
            "mde_pp_confirmatory": mde_confirm,
            "power_for_sesoi_confirmatory": pw_confirm,
            "required_seeds_for_sesoi_confirmatory": required_pairs(
                ruler_margin, sd, ruler_alpha, power),
        })
    ruler_seed_break_even = (
        max_sd_for_effect(ruler_margin, n_seeds, alpha, power)
        if can_power else None)
    ruler_seed_break_even_confirm = (
        max_sd_for_effect(ruler_margin, n_seeds, ruler_alpha, power)
        if can_power else None)

    if declared_ruler_sd is not None:
        need = required_pairs(ruler_margin, declared_ruler_sd,
                              alpha, power)
        pw = (paired_t_power(ruler_margin, declared_ruler_sd,
                             n_seeds, alpha) if can_power else None)
        ok = can_power and need is not None and need <= n_seeds
        checks.append(_check(
            "ruler_seed_declared_power", "RULER 种子级计划功效（声明假设）",
            "pass" if ok else "fail",
            f"声明跨种子配对差 SD={declared_ruler_sd:.4g}（EM 绝对尺度）；"
            f"{n_seeds} 个种子对 {ruler_margin:.2%} SESOI 的功效≈"
            + (f"{pw:.3f}" if pw is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need} 个种子（当前 {n_seeds}）"
               if need is not None else "所需种子数超出可计算范围")
            + ("" if ok else "：种子级设计对 SESOI 功效不足；"
               "逐格 McNemar 的样本量不能补上这个缺口")))
        need_c = required_pairs(ruler_margin, declared_ruler_sd,
                                ruler_alpha, power)
        pw_c = (paired_t_power(ruler_margin, declared_ruler_sd,
                               n_seeds, ruler_alpha) if can_power else None)
        ok_c = can_power and need_c is not None and need_c <= n_seeds
        checks.append(_check(
            "ruler_seed_confirmatory_power",
            "RULER 种子级确认性计划功效（多重比较校正）",
            "pass" if ok_c else "fail",
            f"声明跨种子配对差 SD={declared_ruler_sd:.4g}；按 {mode} 的 "
            f"α_eff={ruler_alpha:.6g}（家族 m="
            f"{family_sizes['ruler_family_size']}，全局 k="
            f"{family_sizes['global_family_size']}），{n_seeds} 个种子对 "
            f"{ruler_margin:.2%} SESOI 的功效≈"
            + (f"{pw_c:.3f}" if pw_c is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need_c} 个种子（当前 {n_seeds}）"
               if need_c is not None else "所需种子数超出可计算范围")
            + ("" if ok_c else "：通过多重比较后种子级功效不足，"
               "H4 的支持/拒绝只能作探索性线索")))
    else:
        checks.append(_check(
            "ruler_seed_declared_power",
            "RULER 种子级计划功效（声明假设）", "info",
            "配置未声明 design.ruler_paired_sd：不把任一跨种子 SD 假设"
            "当作已知，仅报告种子级灵敏度曲线"))
        checks.append(_check(
            "ruler_seed_confirmatory_power",
            "RULER 种子级确认性计划功效（多重比较校正）", "info",
            "未声明 design.ruler_paired_sd；见 seed_level 中的 "
            "power_for_sesoi_confirmatory"))

    ref_ruler_sd = DEFAULT_RULER_SD_REF
    ref_ruler_pw = (paired_t_power(ruler_margin, ref_ruler_sd,
                                   n_seeds, alpha) if can_power else None)
    ref_ruler_pw_confirm = (
        paired_t_power(ruler_margin, ref_ruler_sd, n_seeds, ruler_alpha)
        if can_power else None)
    ref_ruler_need = required_pairs(
        ruler_margin, ref_ruler_sd, alpha, power)
    ref_ruler_need_confirm = required_pairs(
        ruler_margin, ref_ruler_sd, ruler_alpha, power)
    ref_ruler_detail = (
        f"若跨种子配对差 SD={ref_ruler_sd:.1%}（参考值），{n_seeds} 个种子对 "
        f"{ruler_margin:.1%} SESOI 的未校正功效≈"
        + (f"{ref_ruler_pw:.3f}" if ref_ruler_pw is not None
           else "无法计算（n<2）")
        + (f"；按确认性 α_eff={ruler_alpha:.6g} 仅≈"
           f"{ref_ruler_pw_confirm:.3f}"
           if ref_ruler_pw_confirm is not None else "")
        + (f"，需 {ref_ruler_need} 个种子（校正后 "
           f"{ref_ruler_need_confirm}）"
           if ref_ruler_need is not None
           and ref_ruler_need_confirm is not None else ""))
    if ruler_seed_break_even is not None:
        ref_ruler_detail += (
            f"；未校正 80% 功效下最大 SD≈{ruler_seed_break_even:.3%}")
    if ruler_seed_break_even_confirm is not None:
        ref_ruler_detail += (
            f"（确认性 {ruler_seed_break_even_confirm:.3%}）")
    checks.append(_check(
        "ruler_seed_reference_sensitivity",
        "RULER 种子级参考灵敏度（非数据估计）", "info",
        ref_ruler_detail
        + "。这是确认性 H4 的实际推断单位；逐格 McNemar 表只描述"
        "评测抽样精度，不能替代"))
    if (ref_ruler_pw_confirm is not None
            and ref_ruler_pw_confirm < power):
        checks.append(_check(
            "ruler_seed_confirmatory_reference",
            "RULER 种子级确认性参考功效", "warn",
            f"参考 SD={ref_ruler_sd:.1%} 时，{n_seeds} 个种子对 "
            f"{ruler_margin:.1%} SESOI 的确认性功效仅≈"
            f"{ref_ruler_pw_confirm:.3f}（目标 {power:.0%}；"
            f"α_eff={ruler_alpha:.6g}，需约 {ref_ruler_need_confirm} "
            "个种子）。当前设计对 RULER 的 1pp 种子级效应只能作"
            "描述性/探索性证据"))

    # ---- 分析人群/排除规则必须在开跑前声明（不是事后补的叙述）
    checks.append(_check(
        "analysis_protocol", "分析人群与排除规则", "pass",
        "计划内 (变体, 种子) 全集；只允许协议违规/产物损坏排除，"
        "不得因结果不利或离群排除"))

    if smoke:
        # 冒烟/伪数据本就不在预注册范围；把 fail 降为 warn，避免把
        # “流程连通性测试”误报成正式设计失败，但仍在计划里留痕。
        for c in checks:
            if c["status"] == "fail":
                c["status"] = "warn"
                c["detail"] += "（冒烟运行：仅检查流程，不作确认性设计）"

    endpoint_ppl = {
        "metric": "PPL", "sesoi_relative": ppl_margin,
        "sesoi_source": "predeclared_protocol",
        "planned_seeds": n_seeds,
        "max_paired_sd_for_sesoi": ppl_break_even,
        "max_paired_sd_for_sesoi_confirmatory": ppl_break_even_confirm,
        "multiplicity": dict(ppl_multiplicity),
        "sensitivity": ppl_rows,
    }
    endpoint_ruler = {
        "metric": "RULER平均", "sesoi_pp": ruler_margin,
        "sesoi_source": "predeclared_protocol",
        "samples_per_cell": ruler_n_samples,
        "n_contexts": n_contexts,
        "n_tasks_assumed": _TASKS_FOR_RULER,
        "cells": cells,
        "pooled_samples_best_case": pooled_n,
        "mde_pp_pooled_best_case_at_reference": pooled_mde,
        "mde_pp_pooled_best_case_min_over_grid": pooled_min,
        "max_discordance_for_sesoi": ruler_break_even,
        "max_discordance_for_sesoi_confirmatory":
            ruler_break_even_confirm,
        "multiplicity": dict(ruler_multiplicity),
        "sensitivity": ruler_rows,
        "inference_units_note": (
            "sensitivity 是评测样本级配对 McNemar（每格 n 个样例），"
            "只描述评测抽样精度；seed_level.sensitivity 是确认性 H4 "
            "实际使用的种子级配对 t（跨种子 macro EM 配对差），两者"
            "推断单位不同，不能互相替代"),
        "seed_level": {
            "inference_unit": "seed",
            "metric": "RULER平均",
            "sesoi_pp": ruler_margin,
            "sesoi_source": "predeclared_protocol",
            "planned_seeds": n_seeds,
            "paired_sd_grid_pp": list(ruler_sd_grid),
            "declared_paired_sd_pp": declared_ruler_sd,
            "max_paired_sd_for_sesoi_pp": ruler_seed_break_even,
            "max_paired_sd_for_sesoi_pp_confirmatory":
                ruler_seed_break_even_confirm,
            "multiplicity": dict(ruler_multiplicity),
            "sensitivity": ruler_seed_rows,
            "measurement_note": (
                "SD 为换训练种子时 macro RULER EM 配对差的标准差（绝对 "
                "EM 尺度）；单 run 内有限样例的评测误差由逐格 McNemar/"
                "样例聚类 bootstrap 回答，不并入本表"),
        },
    }

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "lega.design",
        "test_look_budget": test_look_budget,
        "inputs": {
            "config_file": config_file,
            "config_sha256": config_sha256,
            "analysis_code": _analysis_code_fingerprint(),
            "variants": list(variants),
            "seeds": list(seeds),
            "n_seeds": n_seeds,
            "alpha": alpha,
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "multiplicity": mode,
            "power": power,
            "smoke": bool(smoke),
            "registered_hypotheses": [h.get("id") for h in hyps],
            "registration_scope_seeds": scope_seeds_int,
            "analysis_variants": list(analysis_variants),
            "known_variants": sorted(known) if known else None,
            "run_order_mode": order_mode,
            "run_order_seed": run_order.get("order_seed"),
            "run_order_seed_source": run_order.get("seed_source"),
            "analysis_n_examples": n_analysis_examples,
            "data_report_present": data_report is not None,
        },
        "run_order": run_order,
        "scope_readiness": scope_readiness,
        "data_budget": data_budget,
        "analysis_protocol": dict(_ANALYSIS_PROTOCOL),
        # 确认性分析计划的只读快照：计划摘要因此同时绑定“登记了哪些
        # 假设、用什么指标/候选/参照”，不只是一个种子/样本量清单。
        "registered_plan": [
            {k: h.get(k) for k in ("id", "metric", "candidate", "reference",
                                   "direction")}
            for h in hyps],
        "assumptions": {
            "ppl_paired_sd_rel": declared_ppl_sd,
            "ruler_paired_sd_pp": declared_ruler_sd,
            "ruler_discordance": declared_disc,
            "reference_ppl_sd_rel": ref_sd,
            "reference_ruler_sd_pp": ref_ruler_sd,
            "reference_ruler_discordance": ref_disc,
            "reference_mechanism_sd_abs": ref_sd_mech,
            "auroc_paired_sd_abs": mechanism_endpoints["auroc"].get(
                "declared_paired_sd"),
            "ap_paired_sd_abs": mechanism_endpoints["ap"].get(
                "declared_paired_sd"),
            "multiplicity": mode,
            "run_order": order_mode,
            "test_look_budget": test_look_budget,
            "note": (
                "声明假设来自配置 design 块（分析前写死）；参考值是展示用"
                "横轴，不是从当前数据估计的效应量，也不参与任何推断。"
                "确认性功效按最终 Holm 判定阈值规划，不能与未校正功效混用"),
        },
        "multiplicity_plan": {
            "mode": mode,
            "family_sizes": {
                "PPL": family_sizes["ppl_family_size"],
                "RULER": family_sizes["ruler_family_size"],
                "AUROC": family_sizes["auroc_family_size"],
                "AP": family_sizes["ap_family_size"],
                "global_confirmatory": family_sizes["global_family_size"],
            },
            "registered_tests": family_sizes["registered_tests"],
            "planned_comparison_pairs": [list(p) for p in family_sizes["pairs"]],
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "note": (
                "家族大小按计划内全部同指标配对计算（含探索性对比），"
                "全局族按计划内可检验的预注册假设计算；缺失产物只会让"
                "实际家族变小，故这是设计上界/保守口径"),
        },
        "endpoints": {
            "ppl": endpoint_ppl,
            "ruler": endpoint_ruler,
            "auroc": mechanism_endpoints["auroc"],
            "ap": mechanism_endpoints["ap"],
        },
        "checks": checks,
        "status": _checklist_status(checks),
        "warnings": [c["detail"] for c in checks
                     if c["status"] in ("warn", "fail")],
    }
    plan["plan_digest"] = plan_digest(plan)
    return plan


