"""受控变量与数据/评测协议核对。"""
from ._shared import _install as _shared_install

_shared_install(globals())

def verify_controlled_setup(out_root: str, variants, seeds, n_seeds: int,
                            expected_data_digest: str = None) -> dict:
    """核对所有变体 × 种子确实除待验证模块外使用完全相同的训练条件。

    对比实验的公平性前提：优化器/LR 策略/步数/数据顺序等受控变量必须一致，
    且各 run 必须消费同一份数据、同一代码版本、同样的训练步数。该检查把
    “公平对比”从口头约定变成可执行断言，分两层：
     1. config_used.json：除 variant/seed/device/tag 外的字段逐项一致；
     2. eval.json 溯源：数据指纹、代码提交、配置 SHA-256、计划训练步数
        （steps_planned，旧格式回退 steps）跨 run 一致，并与本次流水线的
        数据指纹比对，避免"同一 config 跑了不同数据/代码版本"被静默当成可比。
        实际步数/tokens_seen 默认也必须逐项一致；启用早停（early_stopped）
        时各 run 可以停在不同的步数——这是早停的定义而非条件失控，此时按
        计划步数核对，实际步数逐 run 如实报告、不再要求相同。
     任一层不一致都会报错并列出差异。
    """
    ref = None
    ref_name = None
    mismatches = []
    missing = []
    checked = 0
    meta_keys = ("data_fingerprint", "code_commit", "config_sha256",
                 "steps_planned")
    meta_ref = None
    meta_ref_name = None
    meta_mismatches = []
    meta_missing = []
    meta_checked = 0
    early_stop_runs = []
    actual_meta = []
    eval_protocols = []
    selection_points = []
    eval_protocol_missing = []
    data_order_runs = []
    data_order_missing = []
    attention_backends = {}
    attention_key_seen = False
    attention_missing = []

    def _attention_backend(variant: str, sdpa_on: bool) -> str:
        # 含集中度项的门控必须读取 softmax 概率，无法走 SDPA；其它变体在
        # sdpa_on 时走 SDPA。若配置开启 SDPA，两类变体的注意力实现就不同，
        # PPL 差异不能全部归因于门控。集中度变体集合直接从 attention 的
        # _USE_CONC 派生，避免新增变体（如 lega_placebo）漏进审计时被误判
        # 成 SDPA 路径：lega_placebo 同样 use_conc=True，实际走显式路径。
        needs_probs = variant in _USE_CONC
        return "explicit" if needs_probs or not sdpa_on else "sdpa"
    devices = {}
    device_missing = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            cfg_path = os.path.join(vdir, "config_used.json")
            if not os.path.exists(cfg_path):
                missing.append(f"{v}/seed{seed}")
            else:
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except (OSError, ValueError):
                    cfg = None
                if not isinstance(cfg, dict):
                    missing.append(f"{v}/seed{seed}(config_used.json 不可解析)")
                else:
                    # 键缺失时沿用默认 seed；显式为 null/非数值时无法证明用了
                    # 预期种子，审计应如实报“不一致”。
                    cfg_seed = cfg.get("seed", seed)
                    seed_mismatch = (cfg_seed is None
                                     or not _same_seed(cfg_seed, seed))
                    if seed_mismatch:
                        mismatches.append(f"{v}/seed{seed}: config seed={cfg.get('seed')}")
                    if "device" in cfg:
                        raw_device = cfg.get("device")
                        if isinstance(raw_device, str) and raw_device:
                            devices[f"{v}/seed{seed}"] = raw_device
                        else:
                            device_missing.append(f"{v}/seed{seed}")
                    else:
                        device_missing.append(f"{v}/seed{seed}")
                    if "use_sdpa_standard" in cfg:
                        attention_key_seen = True
                        try:
                            sdpa_on = config_bool(
                                cfg.get("use_sdpa_standard"),
                                "use_sdpa_standard")
                        except (TypeError, ValueError) as e:
                            # 外部 config_used.json 的坏值不能让审计函数自己
                            # 崩溃（它的契约是报告问题并 SystemExit），也不能
                            # 静默按默认 True 推断。
                            mismatches.append(
                                f"{v}/seed{seed}: use_sdpa_standard 非法"
                                f"（{e}）；无法确认注意力实现路径")
                            attention_backends[v] = "invalid"
                        else:
                            attention_backends[v] = _attention_backend(
                                v, sdpa_on)
                    else:
                        # train.py 的默认值是 True；config_used 未写该键时
                        # 实际仍走 SDPA。先记录缺失，稍后在有任一 run 显式
                        # 记录该键时按默认值补全，否则“部分 run 缺键”会让
                        # 混合后端漏检。
                        attention_missing.append(f"{v}/seed{seed}")
                    ctrl = {k: val for k, val in cfg.items()
                            if k not in _AUDIT_EXEMPT_KEYS}
                    if ref is None:
                        ref, ref_name = ctrl, f"{v}/seed{seed}"
                    elif ctrl != ref:
                        keys = sorted(set(ctrl) | set(ref))
                        diffs = [f"{k}: {ref.get(k)!r} vs {ctrl.get(k)!r}"
                                 for k in keys if ref.get(k) != ctrl.get(k)]
                        mismatches.append(f"{v}/seed{seed} 相对 {ref_name} 的差异: "
                                          + "; ".join(diffs[:8]))
                    checked += 1
            ev_path = os.path.join(vdir, "eval.json")
            if not os.path.exists(ev_path):
                meta_missing.append(f"{v}/seed{seed}")
                continue
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
            except (OSError, ValueError):
                ev = None
            if not isinstance(ev, dict):
                meta_missing.append(f"{v}/seed{seed}(eval.json 不可解析)")
                continue
            meta = {k: ev.get(k) for k in meta_keys}
            if meta["steps_planned"] is None:
                meta["steps_planned"] = ev.get("steps")
            proto = _ppl_protocol(ev)
            if proto:
                eval_protocols.append((f"{v}/seed{seed}", proto))
            else:
                eval_protocol_missing.append(f"{v}/seed{seed}")
            # 早停标志必须是真正的 bool：字符串 "false"/"0" 若按 truthiness
            # 变成 True，会让“实际步数与计划步数不一致”的 run 逃过受控预算
            # 检查。非法值按未早停处理并显式记为元数据不一致。
            early_raw = ev.get("early_stopped")
            if isinstance(early_raw, bool):
                early_stopped = early_raw
            else:
                early_stopped = False
                if early_raw is not None:
                    meta_mismatches.append(
                        f"{v}/seed{seed}: early_stopped={early_raw!r} "
                        "不是布尔值；按未早停核对实际步数")
            actual_meta.append((f"{v}/seed{seed}", ev.get("steps"),
                                ev.get("tokens_seen"), early_stopped,
                                meta["steps_planned"]))
            order_sha = ev.get("data_order_sha256")
            if isinstance(order_sha, str) and order_sha:
                data_order_runs.append((seed, f"{v}/seed{seed}", order_sha,
                                        early_stopped))
            else:
                data_order_missing.append(f"{v}/seed{seed}")
            steps_planned_here = meta["steps_planned"]
            sel_step = _integral_value(ev.get("ckpt_step"))
            if sel_step is not None:
                # 计划步数必须是"可解释为整数"的量：字符串 "10"/10.0 可接受，
                # 但 bool（True 会被 / 当成 1）与小数（10.5）都不能作为分母，
                # 否则选型比例会用错误的预算做除法，且报告里分母与它处不一致。
                planned_int = _integral_value(steps_planned_here)
                sel_frac = None
                if planned_int is not None and planned_int > 0:
                    try:
                        sel_frac = sel_step / planned_int
                    except (OverflowError, ZeroDivisionError):
                        # 外部 JSON 的合法大整数（如 10**400）可精确解析为
                        # 整数，但转 float 做比例会溢出；比例按缺失处理，
                        # 不能让审计自己崩溃。
                        sel_frac = None
                selection_points.append((f"{v}/seed{seed}", sel_step,
                                         steps_planned_here, sel_frac))
            if early_stopped:
                early_stop_runs.append(f"{v}/seed{seed}(step {ev.get('steps')}"
                                       f"/{meta['steps_planned']})")
            here = []
            ev_seed = ev.get("seed", seed)
            ev_seed_bad = (ev_seed is None
                           or not _same_seed(ev_seed, seed))
            if ev_seed_bad:
                here.append(f"eval seed={ev.get('seed')}")
            if (expected_data_digest is not None
                    and meta["data_fingerprint"] != expected_data_digest):
                here.append(f"data_fingerprint={meta['data_fingerprint']!r} "
                            f"!= 本次流水线数据 {expected_data_digest!r}")
            if meta_ref is None:
                meta_ref, meta_ref_name = meta, f"{v}/seed{seed}"
            else:
                diffs = [f"{k}: {meta_ref.get(k)!r} vs {meta.get(k)!r}"
                         for k in meta_keys if meta_ref.get(k) != meta.get(k)]
                if diffs:
                    here.append(f"相对 {meta_ref_name} 的差异: " + "; ".join(diffs))
            if here:
                meta_mismatches.append(f"{v}/seed{seed}: " + "; ".join(here))
            meta_checked += 1
    # 实际步数/tokens_seen：只豁免真正早停的 run。其它 run 的实际步数必须
    # 等于计划步数，且彼此的实际步数/tokens_seen 必须一致；不能因为同批里
    # 有另一个 run 早停，就跳过所有 run 的实际训练量检查。
    def _same_quantity(a, b):
        # 旧产物可能把步数写成 "10"/10.0；语义相同即算一致。但 bool 是 int
        # 的子类（True 会被 float() 当成 1），小数则不能作为步数/tokens_seen：
        # 静默接受会让被污染的外部元数据冒充同一训练预算。
        if a is None and b is None:
            return True
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return False
        # float("1_0")==10.0：下划线写法不是合法训练量，不能与 10 相等。
        if ((isinstance(a, str) and "_" in a)
                or (isinstance(b, str) and "_" in b)):
            return False
        # 先按精确整数身份比较：>2**53 的合法 JSON 大整数转 float 会丢精度，
        # 10**400 更会直接 OverflowError；两者若表示同一个整数仍应判一致。
        ia, ib = _integral_value(a), _integral_value(b)
        if ia is not None and ib is not None:
            return ia == ib
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError, OverflowError):
            return a == b
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        if not (fa.is_integer() and fb.is_integer()):
            return False
        return fa == fb

    stable_actual = [(name, steps_run, tokens_run, planned)
                     for name, steps_run, tokens_run, early, planned in actual_meta
                     if not early]
    for name, steps_run, _tokens_run, planned in stable_actual:
        if not _same_quantity(steps_run, planned):
            meta_mismatches.append(
                f"{name}: 实际 steps={steps_run!r} != 计划 steps={planned!r}")
    if stable_actual:
        a_ref_name, a_ref_steps, a_ref_tokens = stable_actual[0][:3]
        for name, steps_run, tokens_run, _planned in stable_actual[1:]:
            diffs = []
            if not _same_quantity(a_ref_steps, steps_run):
                diffs.append(f"steps: {a_ref_steps!r} vs {steps_run!r}")
            if not _same_quantity(a_ref_tokens, tokens_run):
                diffs.append(f"tokens_seen: {a_ref_tokens!r} vs {tokens_run!r}")
            if diffs:
                meta_mismatches.append(f"{name} 相对 {a_ref_name} 的差异: "
                                       + "; ".join(diffs))
    # 实际训练数据顺序：同一 seed 下所有变体必须消费完全相同的窗口序列。
    # config/seed 相同只说明“应当”同序；若采样代码路径被改或 RNG 被提前
    # 消耗，实际 batch 序列可能不同，PPL 差异就混入了数据混杂。只比较非早停
    # run（早停会少消费批次，本来就不在同一训练量上）。
    by_seed_order = {}
    for seed_v, name, digest_v, early_v in data_order_runs:
        if not early_v:
            by_seed_order.setdefault(seed_v, []).append((name, digest_v))
    data_order_checked = 0
    for seed_v, items in sorted(by_seed_order.items()):
        if len(items) < 2:
            continue
        ref_name, ref_digest = items[0]
        for name, digest_v in items[1:]:
            data_order_checked += 1
            if digest_v != ref_digest:
                meta_mismatches.append(
                    f"{name}: 训练数据顺序指纹 {digest_v!r} 与 {ref_name} 的 "
                    f"{ref_digest!r} 不同（同 seed 下实际 batch 序列不一致，"
                    "PPL 差异混入数据混杂）")
    # PPL 评测协议：训练配置相同并不保证独立评测用了同一批 token。
    # 只比较双方都记录的字段（旧产物可能缺字段），缺失单独告警；只要共同
    # 字段有差异就直接失败，避免把不同窗口/上下文的 PPL 当成同条件配对。
    protocol_ref = None
    protocol_ref_name = None
    for name, proto in eval_protocols:
        if protocol_ref is None:
            protocol_ref, protocol_ref_name = proto, name
            continue
        common = sorted(set(proto) & set(protocol_ref))
        diffs = [f"{k}: {protocol_ref[k]!r} vs {proto[k]!r}"
                 for k in common
                 if not _same_protocol_value(protocol_ref[k], proto[k])]
        if diffs:
            meta_mismatches.append(
                f"{name} 相对 {protocol_ref_name} 的 PPL 评测协议差异: "
                + "; ".join(diffs)
                + "（不同评测 token 子集/上下文不可直接配对）")
    # 设备类型必须一致：CPU 与 CUDA（或不同精度后端）的数值/内核不同，
    # 若不同变体跑在不同设备上，PPL 差异不能归因于方法。只比较设备类型
    # （cuda:0/cuda:1 视为同一类型），不强制同一张卡。
    device_types = {}
    for name, raw_device in devices.items():
        device_types[name] = str(raw_device).split(":", 1)[0].strip().lower()
    mixed_device = []
    if len(set(device_types.values())) > 1:
        by_type = {}
        for name, dtype in device_types.items():
            by_type.setdefault(dtype, []).append(name)
        mixed_device = [f"{dtype}: {', '.join(sorted(names))}"
                        for dtype, names in sorted(by_type.items())]
        mismatches.append(
            "训练设备类型不一致（" + "；".join(mixed_device)
            + "）：不同设备/精度后端的数值行为不同，正式对比必须同设备类型")
    mixed_attention = []
    attention_inferred = []
    if attention_missing:
        # train.py 对缺键使用默认 True；因此无论是否有 run 显式记录该开关，
        # 缺键 run 都必须按默认值补全后再做混合后端判定。否则一份完全没有
        # 写 use_sdpa_standard 的配置会让 standard/qgate 实际走 SDPA、lega
        # 走显式 softmax，却因“没有可比较的键”被审计静默放行。
        for name in attention_missing:
            v = name.split("/seed", 1)[0]
            if v not in attention_backends:
                attention_backends[v] = _attention_backend(v, True)
                attention_inferred.append(name)
    if len(set(attention_backends.values())) > 1:
        by_backend = {}
        for v, b in attention_backends.items():
            by_backend.setdefault(b, []).append(v)
        mixed_attention = [f"{b}: {', '.join(sorted(vs))}"
                           for b, vs in sorted(by_backend.items())]
    if missing:
        print(f"[audit] 警告：缺少 {len(missing)} 个 run 的 config_used.json，"
              f"无法核对受控变量: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if meta_missing:
        print(f"[audit] 警告：缺少 {len(meta_missing)} 个 run 的 eval.json，"
              f"无法核对数据指纹/代码版本/训练步数: {', '.join(meta_missing[:6])}"
              + (" ..." if len(meta_missing) > 6 else ""))
    if eval_protocol_missing:
        print(f"[audit] 警告：{len(eval_protocol_missing)} 个 run 的 eval.json "
              "缺少评测协议字段（eval_ctx/eval_iters/val_windows），"
              "无法核对是否在同一批 token 上评测: "
              + ", ".join(eval_protocol_missing[:6])
              + (" ..." if len(eval_protocol_missing) > 6 else ""))
    if attention_missing:
        if attention_key_seen:
            print(f"[audit] 警告：{len(attention_missing)} 个 run 的 "
                  "config_used.json 缺少 use_sdpa_standard，已按 train.py 的"
                  "默认值 True 推断实际注意力后端: "
                  + ", ".join(attention_missing[:6])
                  + (" ..." if len(attention_missing) > 6 else ""))
        else:
            print(f"[audit] 警告：全部 config_used.json 都缺少 "
                  f"use_sdpa_standard（{len(attention_missing)} 个 run），"
                  "已按 train.py 的默认值 True 推断实际注意力后端；"
                  "正式实验建议在配置中显式写入该开关")
    if data_order_missing:
        print(f"[audit] 警告：{len(data_order_missing)} 个 run 的 eval.json "
              "缺少 data_order_sha256，无法证明实际训练批次序列一致: "
              + ", ".join(data_order_missing[:6])
              + (" ..." if len(data_order_missing) > 6 else ""))
    if device_missing:
        print(f"[audit] 警告：{len(device_missing)} 个 run 的 config_used.json "
              "缺少 device，无法核对训练设备类型一致: "
              + ", ".join(device_missing[:6])
              + (" ..." if len(device_missing) > 6 else ""))
    if mixed_attention:
        mismatches.append(
            "注意力实现路径不一致（来自 use_sdpa_standard=true）："
            + "；".join(mixed_attention)
            + "。含集中度项的变体必须走显式 softmax，而其它变体走 SDPA，"
              "两者数值/内核不同；正式对比请设 use_sdpa_standard=false，"
              "让全部变体共用显式路径（延迟评测仍可单独测 standard_sdpa）")
    if mismatches or meta_mismatches:
        msg = ("受控变量自检失败：对比实验除待验证模块外必须完全一致，"
               "否则观测差异不能归因于方法本身。\n  - "
               + "\n  - ".join(mismatches + meta_mismatches))
        print(f"[audit] {msg}")
        raise SystemExit(2)
    if checked:
        print(f"[audit] 受控变量自检通过：{checked} 个 run 的优化器/LR/步数等配置一致")
    if meta_checked:
        print(f"[audit] 训练溯源自检通过：{meta_checked} 个 run 的数据指纹/"
              f"代码版本/计划训练步数一致"
              + (f"（数据指纹 {expected_data_digest}）"
                 if expected_data_digest else ""))
    if data_order_checked:
        print(f"[audit] 训练数据顺序自检通过：同 seed 下 {data_order_checked} "
              "组变体对比的实际 batch 起点序列指纹一致")
    if early_stop_runs:
        print(f"[audit] 早停启用：{len(early_stop_runs)} 个 run 未跑满计划步数"
              f"（{', '.join(early_stop_runs[:6])}"
              + (" ..." if len(early_stop_runs) > 6 else "")
              + "）；实际训练步数不等，预注册判定将按计划步数核对并降级为探索性")
    # checkpoint 选型步分布：val-loss 最优点的训练步是否在变体间差异过大
    # （同一预算下“过早选中 vs 接近末步选中”可能不是公平对照）。只报告，
    # 不作为强制失败，以免污染既有实验流水线；但在汇总中已经有展示列。
    sel_spread = None
    if selection_points:
        fracs = [p[3] for p in selection_points if p[3] is not None]
        if len(fracs) >= 2:
            sel_spread = {
                "n": len(fracs),
                "min": min(fracs), "max": max(fracs),
                "span": max(fracs) - min(fracs),
            }
    return {"ok": not (mismatches or meta_mismatches), "checked": checked,
            "eval_protocol_checked": len(eval_protocols),
            "eval_protocol_missing": eval_protocol_missing,
            "missing_configs": missing, "mismatches": mismatches,
            "attention_backends": attention_backends,
            "mixed_attention": mixed_attention,
            "attention_key_seen": bool(attention_key_seen),
            "attention_missing": attention_missing,
            "attention_inferred": attention_inferred,
            "training_budget_comparable": not early_stop_runs,
            "meta_checked": meta_checked, "missing_evals": meta_missing,
            "meta_mismatches": meta_mismatches,
            "data_order_checked": data_order_checked,
            "data_order_missing": data_order_missing,
            "device_types": device_types,
            "device_missing": device_missing,
            "mixed_device": mixed_device,
            "early_stop_runs": early_stop_runs,
            "selection_step_points": selection_points,
            "selection_step_spread": sel_spread,
            "expected_data_digest": expected_data_digest,
            "exempt_keys": sorted(_AUDIT_EXEMPT_KEYS)}
