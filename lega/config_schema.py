"""实验配置的显式契约：已知键、类型/范围校验与语义指纹。

配置 SHA-256 只能证明“这份字节被用了”，不能发现两类常见问题：

* 键名拼写错误（``seed_list``/``length_calib``）被 ``dict.get`` 静默忽略，
  实验按代码默认值运行，而作者以为声明了另一套条件；
* ``true`` 被当成 ``1``、``1.5`` 被截断成 ``1``、``NaN`` 进入数值字段，
  运行值与落盘值/人工阅读值不一致。

本模块在解析后、开跑前对配置做一次只读审计，并把**规范化 JSON**（键排序、
无无关空白）的 SHA-256 作为语义指纹。它与原始文件 SHA-256 互补：后者绑定
字节，前者回答“这两份不同格式的配置是否声明了同一实验”。缺失键仍允许，
因为代码有显式默认值；但出现未知键或确定非法的值会在正式入口 fail closed。
"""

from __future__ import annotations

import hashlib
import json
import math
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

SCHEMA_VERSION = 1

COMMENT_PREFIX = "_"

# 顶层：每个键必须能被某个入口实际读取。未知键通常是拼写错误或从旧版本
# 复制来的失效旋钮，静默忽略会让“声明的实验”与“实际实验”分离。
TOP_LEVEL = {
    "d_model": "positive_int",
    "n_layers": "positive_int",
    "n_heads": "positive_int",
    "ffn_mult": "positive_number",
    "block_size": "positive_int",
    "dropout": "probability",
    "attn_dropout": "probability",
    "gate_bias_init": "finite_number",
    "use_sdpa_standard": "bool",
    "compile": "bool",
    "batch_size": "positive_int",
    "grad_accum": "positive_int",
    "steps": "positive_int",
    "lr": "nonnegative_number",
    "warmup": "nonnegative_int",
    "weight_decay": "nonnegative_number",
    "grad_clip": "nonnegative_number",
    "train_ctx": "positive_int",
    "amp": "bool",
    "seed": "int",
    "seeds": "int_list",
    "log_every": "positive_int",
    "eval_every": "positive_int",
    "eval_iters": "positive_int",
    "early_stop_patience": "nonnegative_int",
    "early_stop_min_delta": "nonnegative_number",
    "length_null": "object",
    "length_calibration": "object",
    "length_ppl": "object",
    "design": "object",
    "ruler": "object",
    "analysis": "object",
    "latency": "object",
    "sanity": "object",
}

NESTED = {
    "length_null": {
        "enabled": "bool",
        "lengths": "lengths",
        "modes": "str_or_str_list",
        "score_sd": "nonnegative_number",
        "n_trials": "positive_int",
        "seed": "int",
    },
    "length_calibration": {
        "enabled": "bool",
        "lengths": "lengths",
        "n_windows": "positive_int",
        "query_positions": "positive_int",
        "growth_modes": "str_or_str_list",
        "bootstrap": "nonnegative_int",
        "tolerance": "nonnegative_number",
        "seed": "int",
    },
    "length_ppl": {
        "enabled": "bool",
        "lengths": "lengths",
        "n_windows": "positive_int",
        "query_positions": "positive_int",
        "growth_modes": "str_or_str_list",
        "bootstrap": "nonnegative_int",
        "tolerance": "nonnegative_number",
        "seed": "int",
    },
    "design": {
        "alpha": "open_probability",
        "power": "open_probability",
        "multiplicity": "str",
        "run_order": "str",
        "run_order_seed": "int",
        "test_look_budget": "positive_int",
        "length_stratification": {
            "enabled": "bool",
            "lengths": "lengths",
            "n_windows": "positive_int",
            "query_positions": "positive_int",
            "growth_modes": "str_or_str_list",
            "bootstrap": "nonnegative_int",
            "tolerance": "nonnegative_number",
        },
        "ppl_paired_sd_rel": "positive_number",
        "ppl_paired_sd_rel_grid": "number_list",
        "ruler_paired_sd": "positive_number",
        "ruler_paired_sd_grid": "number_list",
        "ruler_discordance": "open_probability",
        "ruler_discordance_grid": "number_list",
        "mechanism_sesoi": "positive_number",
        "auroc_paired_sd": "positive_number",
        "auroc_paired_sd_grid": "number_list",
        "ap_paired_sd": "positive_number",
        "ap_paired_sd_grid": "number_list",
        "development_seeds": "int_list",
        "pilot_seeds": "int_list",
        "seed_overlap_policy": "str",
        "pilot_variance_file": "str",
        "compute_budget_gpu_hours": "positive_number",
        "compute_budget_policy": "str",
        "expected_step_time_ms": "positive_number",
        "calibration": {
            "enabled": "bool",
            "n_sim": "positive_int",
            "seed": "int",
        },
    },
    "ruler": {
        "ctx_lens": "int_list",
        "n_samples": "positive_int",
        "gen_batch": "positive_int",
        "seed": "int",
        "max_new": "positive_int",
        "fwe_max_new": "positive_int",
        "vt_chains": "positive_int",
        "vt_depth": "positive_int",
        "depth_stratified": "bool",
    },
    "analysis": {
        "n_examples": "positive_int",
        "tau": "probability",
        "tau_grid": "number_list",
        "probe_l2_grid": "number_list",
    },
    "latency": {
        "ctx_lens": "int_list",
        "batch_sizes": "int_list",
        "decode_tokens": "positive_int",
        "iters": "positive_int",
        "warmup": "nonnegative_int",
    },
    "sanity": {
        "steps": "positive_int",
        "batch_size": "positive_int",
        "ctx": "positive_int",
        "lr": "nonnegative_number",
        "loss_threshold": "finite_number",
        "acc_threshold": "probability",
        "initial_loss_tol": "finite_number",
        "seed": "int",
    },
}

# 仅作展示/注释的键可以任意出现，但必须以 "_" 开头（如 configs 里的 "_说明"）。
_NUMERIC_TYPES = {
    "int": (int,),
    "positive_int": (int,),
    "nonnegative_int": (int,),
    "finite_number": (int, float),
    "positive_number": (int, float),
    "nonnegative_number": (int, float),
    "probability": (int, float),
    "open_probability": (int, float),
}


def _is_bool(value) -> bool:
    return isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _check_scalar(kind: str, value):
    """返回 ``(ok, reason)``；不通过时 reason 说明期望与实际类型。"""
    if kind == "bool":
        return (_is_bool(value), None if _is_bool(value) else "需要 bool")
    if kind == "str":
        return (
            isinstance(value, str),
            None if isinstance(value, str) else "需要字符串",
        )
    if kind == "object":
        return (
            isinstance(value, dict),
            None if isinstance(value, dict) else "需要 JSON 对象",
        )
    if kind in _NUMERIC_TYPES:
        if not _is_number(value):
            return False, "需要有限数值（bool/NaN/Inf 不接受）"
        if kind in ("int", "positive_int", "nonnegative_int") and not _is_int(value):
            return False, "需要整数（bool/小数不接受）"
        if kind in ("positive_int", "positive_number") and float(value) <= 0:
            return False, "需要 > 0"
        if kind in ("nonnegative_int", "nonnegative_number") and float(value) < 0:
            return False, "需要 >= 0"
        if kind in ("probability", "open_probability") and not (
            0.0 <= float(value) <= 1.0
        ):
            return False, "需要位于 [0, 1]"
        if kind == "open_probability" and not (0.0 < float(value) < 1.0):
            return False, "需要位于 (0, 1)"
        return True, None
    if kind == "int_list":
        if not isinstance(value, (list, tuple)):
            return False, "需要整数列表"
        if not value:
            return False, "列表不能为空"
        if any(not _is_int(v) for v in value):
            return False, "列表元素必须是整数（bool/小数不接受）"
        return True, None
    if kind == "number_list":
        if not isinstance(value, (list, tuple)):
            return False, "需要数值列表"
        if not value:
            return False, "列表不能为空"
        if any(not _is_number(v) for v in value):
            return False, "列表元素必须是有限数值（bool/NaN/Inf 不接受）"
        return True, None
    if kind == "lengths":
        if isinstance(value, str):
            return True, None
        if isinstance(value, (list, tuple)):
            if not value:
                return False, "长度列表不能为空"
            if any(not _is_int(v) or v < 2 for v in value):
                return False, "每个长度必须是 >= 2 的整数"
            return True, None
        return False, "需要整数列表或逗号分隔字符串"
    if kind == "str_or_str_list":
        if isinstance(value, str):
            return True, None
        if isinstance(value, (list, tuple)):
            if not value or any(not isinstance(v, str) for v in value):
                return False, "需要非空字符串列表"
            return True, None
        return False, "需要字符串或字符串列表"
    return False, f"未知类型规格 {kind!r}"


def _check_object_subkeys(spec: dict, value: dict, path: str) -> tuple:
    """递归检查对象的键是否都在规格中且类型正确。

    返回 (ok, reason) 元组。
    """
    unknown_keys = []
    for k, v in value.items():
        if k.startswith(COMMENT_PREFIX):
            continue
        if k not in spec:
            unknown_keys.append(k)
            continue
        sub_kind = spec[k]
        if isinstance(sub_kind, dict):
            ok, reason = _check_object_subkeys(sub_kind, v, f"{path}.{k}")
            if not ok:
                return False, reason
        else:
            sub_ok, sub_reason = _check_scalar(sub_kind, v)
            if not sub_ok:
                return False, sub_reason
    if unknown_keys:
        return False, f"未知键 {', '.join(unknown_keys)}"
    return True, None


def _issue(path: str, problem: str, value) -> dict:
    return {
        "path": path,
        "problem": problem,
        "value_type": _type_name(value),
        "value": value
        if isinstance(value, (str, int, float, bool, type(None)))
        else None,
    }


def audit_config(config: dict, *, allow_unknown: bool = False) -> dict:
    """审计配置的键空间、类型与跨字段约束（只读，不修改 config）。

    ``allow_unknown=True`` 只把未知键从错误降为警告，用于诊断/迁移旧配置；
    类型、范围和跨字段错误仍然 fail，因为它们会直接改变数值语义。
    """
    config = config if isinstance(config, dict) else {}
    unknown, type_errors, range_errors, cross_field, warnings = [], [], [], [], []
    checked = 0

    for key, value in config.items():
        if key.startswith(COMMENT_PREFIX):
            continue
        path = key
        kind = TOP_LEVEL.get(key)
        if kind is None:
            unknown.append(_issue(path, "未知配置键（会被代码静默忽略）", value))
            continue
        checked += 1
        ok, reason = _check_scalar(kind, value)
        if not ok:
            item = _issue(path, reason or "类型/范围非法", value)
            (range_errors if _is_number(value) else type_errors).append(item)
        if kind == "object" and isinstance(value, dict):
            spec = NESTED.get(key) or {}
            for sub_key, sub_value in value.items():
                if sub_key.startswith(COMMENT_PREFIX):
                    continue
                sub_path = f"{key}.{sub_key}"
                sub_kind = spec.get(sub_key)
                if sub_kind is None:
                    unknown.append(
                        _issue(sub_path, "未知配置键（会被代码静默忽略）", sub_value)
                    )
                    continue
                if isinstance(sub_kind, dict):
                    checked += 1
                    sub_ok, sub_reason = _check_object_subkeys(sub_kind, sub_value, sub_path)
                    if not sub_ok:
                        item = _issue(sub_path, sub_reason or "类型/范围非法", sub_value)
                        (
                            range_errors
                            if isinstance(sub_value, (int, float))
                            and not isinstance(sub_value, bool)
                            and _is_number(sub_value)
                            else type_errors
                        ).append(item)
                else:
                    checked += 1
                    sub_ok, sub_reason = _check_scalar(sub_kind, sub_value)
                    if not sub_ok:
                        item = _issue(sub_path, sub_reason or "类型/范围非法", sub_value)
                        (
                            range_errors
                            if isinstance(sub_value, (int, float))
                            and not isinstance(sub_value, bool)
                            and _is_number(sub_value)
                            else type_errors
                        ).append(item)

    def _int(key):
        value = config.get(key)
        return value if _is_int(value) else None

    d_model = _int("d_model")
    n_heads = _int("n_heads")
    if d_model is not None and n_heads is not None:
        if d_model % n_heads:
            cross_field.append(
                {
                    "path": "d_model,n_heads",
                    "problem": f"d_model={d_model} 必须能被 n_heads={n_heads} 整除",
                }
            )
        elif (d_model // n_heads) % 2:  # d_model % n_heads == 0 validated above
            cross_field.append(
                {
                    "path": "d_model,n_heads",
                    "problem": (
                        f"d_model / max(n_heads, 1)={d_model // n_heads} 必须是偶数"
                        "（RoPE 实现要求）"
                    ),
                }
            )
    block_size = _int("block_size")
    train_ctx = _int("train_ctx")
    if block_size is not None and train_ctx is not None and train_ctx > block_size:
        cross_field.append(
            {
                "path": "train_ctx,block_size",
                "problem": (
                    f"train_ctx={train_ctx} > block_size={block_size}："
                    "训练窗口超过位置编码容量"
                ),
            }
        )
    steps = _int("steps")
    warmup = _int("warmup")
    if steps is not None and warmup is not None and warmup > steps:
        warnings.append(
            {
                "path": "warmup,steps",
                "problem": (
                    f"warmup={warmup} > steps={steps}：学习率尚未完成"
                    "预热训练就结束了"
                ),
            }
        )
    seeds = config.get("seeds")
    if isinstance(seeds, (list, tuple)) and not seeds:
        cross_field.append({"path": "seeds", "problem": "显式 seeds 为空"})
    elif isinstance(seeds, (list, tuple)) and len(set(seeds)) != len(seeds):
        warnings.append({"path": "seeds", "problem": "seeds 含重复项，会被去重"})
    ruler = config.get("ruler")
    if (
        isinstance(ruler, dict)
        and block_size is not None
        and isinstance(ruler.get("ctx_lens"), (list, tuple))
    ):
        too_long = [c for c in ruler["ctx_lens"] if _is_int(c) and c > block_size]
        if too_long:
            cross_field.append(
                {
                    "path": "ruler.ctx_lens,block_size",
                    "problem": (
                        f"RULER 上下文 {too_long} 超过 block_size="
                        f"{block_size}：评测会越过模型容量"
                    ),
                }
            )

    if allow_unknown:
        warnings.extend(unknown)
        unknown = []
    problems = unknown + type_errors + range_errors + cross_field
    status = "fail" if problems else ("warn" if warnings else "pass")
    return {
        "schema": SCHEMA_VERSION,
        "status": status,
        "checked_keys": checked,
        "unknown": unknown,
        "type_errors": type_errors,
        "range_errors": range_errors,
        "cross_field_errors": cross_field,
        "warnings": warnings,
        "allow_unknown": bool(allow_unknown),
        "summary": _summary(
            status, unknown, type_errors, range_errors, cross_field, warnings
        ),
    }


def _summary(status, unknown, type_errors, range_errors, cross_field, warnings) -> str:
    if status == "pass":
        return "配置键与类型全部通过审计"
    parts = []
    if unknown:
        paths = ", ".join(str(i.get("path")) for i in unknown[:5])
        parts.append(f"未知键 {len(unknown)} 个（{paths}）")
    if type_errors:
        paths = ", ".join(str(i.get("path")) for i in type_errors[:5])
        parts.append(f"类型错误 {len(type_errors)} 个（{paths}）")
    if range_errors:
        paths = ", ".join(str(i.get("path")) for i in range_errors[:5])
        parts.append(f"范围错误 {len(range_errors)} 个（{paths}）")
    if cross_field:
        paths = ", ".join(str(i.get("path")) for i in cross_field[:5])
        parts.append(f"跨字段冲突 {len(cross_field)} 个（{paths}）")
    if warnings:
        paths = ", ".join(str(i.get("path")) for i in warnings[:5])
        parts.append(f"警告 {len(warnings)} 个（{paths}）")
    return "；".join(parts) or "配置审计未通过"


def assert_config_valid(
    config: dict, *, source: str = None, allow_unknown: bool = False
) -> dict:
    """审计并在 ``status=fail`` 时抛 ``ValueError``；返回同一份报告。"""
    report = audit_config(config, allow_unknown=allow_unknown)
    if report["status"] == "fail":
        where = f"配置文件 {source}" if source else "配置"
        raise ValueError(f"{where}未通过语义审计：{report['summary']}")
    return report


def _semantic_view(value):
    """递归去掉下划线注释键，只保留会被实验读取的声明。"""
    if isinstance(value, dict):
        return {
            k: _semantic_view(v)
            for k, v in value.items()
            if not str(k).startswith(COMMENT_PREFIX)
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_view(v) for v in value]
    return value


def canonical_config_json(config: dict) -> str:
    """稳定序列化：去除注释键、键排序、紧凑分隔符、保留非 ASCII。"""
    return json.dumps(
        _semantic_view(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_canonical_sha256(config: dict) -> str:
    """配置的语义指纹（与键顺序/缩进/行尾/注释无关，与数值内容绑定）。

    它不是文件 SHA-256 的替代：字节哈希回答“是否同一份文件”，语义哈希
    回答“两份格式不同/被重排的配置是否声明了同一实验”。
    """
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(description="实验配置语义审计（未知键/类型/范围）")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--allow_unknown",
        action="store_true",
        help="把未知键降为警告（仅用于迁移/诊断旧配置）",
    )
    args = p.parse_args(argv)
    try:
        from .utils import load_config_json

        cfg = load_config_json(args.config, validate=False)
    except (OSError, ValueError) as e:
        print(f"[config] 读取配置失败：{e}", file=sys.stderr)
        return 2
    report = audit_config(cfg, allow_unknown=args.allow_unknown)
    report["config"] = os.path.abspath(args.config)
    report["config_canonical_sha256"] = config_canonical_sha256(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    from lega.bootstrap import ensure_hash_seed

    ensure_hash_seed()
    raise SystemExit(main())
