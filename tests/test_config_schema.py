"""配置契约：未知键、类型/范围、跨字段冲突与语义指纹。

回归动机：``dict.get`` 会静默忽略拼错的键，``int()`` 会把 ``true``/``1.5``
截断成另一个训练条件。配置审计必须在开跑前把这些“声明的实验”与“实际
实验”不一致的情形显形，并给出与文件格式无关的语义指纹。
"""
import json
import os
import tempfile

from lega.config_schema import (audit_config, assert_config_valid,
                                canonical_config_json,
                                config_canonical_sha256)
from lega.design import build_design_plan
from lega.utils import load_config_json


def _issue_paths(report):
    return {item["path"] for item in report["unknown"]}


def test_repository_configs_pass_audit():
    """仓库内所有正式/冒烟配置必须通过契约，避免审计误伤真实入口。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("tiny.json", "small.json"):
        path = os.path.join(root, "configs", name)
        cfg = load_config_json(path, validate=True)
        report = audit_config(cfg)
        assert report["status"] == "pass", (name, report)
        assert report["checked_keys"] > 0
    print("  [ok] configs/tiny.json 与 configs/small.json 通过配置审计")


def test_unknown_top_level_and_nested_keys_fail():
    cfg = {"steps": 10, "seed_list": [1, 2],
           "ruler": {"ctx_lens": [128], "sample_seed": 7}}
    report = audit_config(cfg)
    assert report["status"] == "fail"
    assert _issue_paths(report) == {"seed_list", "ruler.sample_seed"}, report
    try:
        assert_config_valid(cfg, source="probe.json")
    except ValueError as e:
        text = str(e)
        assert "probe.json" in text and "seed_list" in text, text
    else:
        raise AssertionError("未知键应使配置审计失败")
    print("  [ok] 未知顶层/嵌套键被拒绝并列出路径")


def test_comment_keys_are_allowed_everywhere():
    cfg = {"_说明": "这是注释", "steps": 2,
           "ruler": {"_note": "不会进入代码", "ctx_lens": [8]}}
    report = audit_config(cfg)
    assert report["status"] == "pass", report
    assert report["unknown"] == []
    print("  [ok] 下划线注释键不触发未知键错误")


def test_type_and_range_errors_are_explicit():
    cfg = {"steps": True, "lr": float("nan"), "batch_size": 0,
           "seeds": [1, 2.5], "design": {"alpha": 1.0, "power": -0.1}}
    report = audit_config(cfg)
    assert report["status"] == "fail"
    paths = {item["path"] for item in report["type_errors"]}
    paths |= {item["path"] for item in report["range_errors"]}
    assert {"steps", "lr", "batch_size", "seeds", "design.alpha",
            "design.power"} <= paths, report
    # bool 不能被当作整数 1；NaN 也不能通过“>= 0”比较而漏网
    assert any(i["path"] == "steps" and i["value_type"] == "bool"
               for i in report["type_errors"])
    print("  [ok] bool/NaN/小数/越界值被显式拒绝，不静默截断")


def test_cross_field_conflicts_fail_closed():
    cfg = {"d_model": 30, "n_heads": 4, "block_size": 128,
           "train_ctx": 256, "ruler": {"ctx_lens": [64, 512]}}
    report = audit_config(cfg)
    paths = {item["path"] for item in report["cross_field_errors"]}
    assert report["status"] == "fail"
    assert {"d_model,n_heads", "train_ctx,block_size",
            "ruler.ctx_lens,block_size"} <= paths, report
    print("  [ok] 模型维度/上下文/RULER 跨字段冲突被拒绝")


def test_allow_unknown_downgrades_only_unknown_keys():
    cfg = {"steps": 2, "typo_key": 1}
    strict = audit_config(cfg)
    relaxed = audit_config(cfg, allow_unknown=True)
    assert strict["status"] == "fail"
    assert relaxed["status"] == "warn"
    assert relaxed["unknown"] == []
    assert relaxed["warnings"]
    # 类型错误在 allow_unknown 下仍然 fail，不能被“兼容旧配置”洗掉
    bad = audit_config({"steps": True, "typo_key": 1}, allow_unknown=True)
    assert bad["status"] == "fail", bad
    print("  [ok] allow_unknown 只降级未知键，不放过类型错误")


def test_canonical_digest_ignores_format_but_tracks_content():
    a = {"steps": 10, "lr": 0.1, "ruler": {"ctx_lens": [8, 16]}}
    b = {"ruler": {"ctx_lens": [8, 16]}, "lr": 0.1, "steps": 10}
    assert canonical_config_json(a) == canonical_config_json(b)
    assert config_canonical_sha256(a) == config_canonical_sha256(b)
    c = dict(a, steps=11)
    assert config_canonical_sha256(c) != config_canonical_sha256(a)
    assert len(config_canonical_sha256(a)) == 64
    commented = dict(a, _说明="只改注释", ruler={
        "ctx_lens": [8, 16], "_note": "不影响语义"})
    assert config_canonical_sha256(commented) == config_canonical_sha256(a)
    print("  [ok] 语义指纹与键顺序/注释无关，但随数值内容变化")


def test_load_config_json_strict_and_escape_hatches():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"steps": 2, "steps_typo": 3}, f)
        raw = load_config_json(path)
        assert raw["steps_typo"] == 3
        try:
            load_config_json(path, validate=True)
        except ValueError as e:
            assert "steps_typo" in str(e), e
        else:
            raise AssertionError("显式 validate=True 应拒绝未知键")
        relaxed = load_config_json(path, validate=True, allow_unknown=True)
        assert relaxed["steps_typo"] == 3
    print("  [ok] load_config_json 默认兼容读取，validate=True 开启严格审计")


def test_design_plan_records_config_audit_and_canonical_digest():
    cfg = {"steps": 10, "block_size": 64, "train_ctx": 64,
           "seed": 1, "seeds": [1, 2, 3],
           "ruler": {"ctx_lens": [32, 64], "n_samples": 4},
           "design": {"alpha": 0.05, "power": 0.8}}
    plan = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"],
                             registered=None,
                             known_variants={"lega", "standard"})
    assert plan["inputs"]["config_canonical_sha256"] == \
        config_canonical_sha256(cfg)
    audit = plan["inputs"]["config_audit"]
    assert audit["status"] == "pass", audit
    check = next(c for c in plan["checks"] if c["id"] == "config_keys")
    assert check["status"] == "pass", check

    bad = dict(cfg, typo_knob=1)
    plan2 = build_design_plan(bad, [1, 2, 3], ["lega", "standard"],
                              registered=None,
                              known_variants={"lega", "standard"})
    check2 = next(c for c in plan2["checks"] if c["id"] == "config_keys")
    assert check2["status"] == "fail", check2
    assert plan2["status"] == "fail"
    print("  [ok] 设计计划绑定配置语义指纹并把未知键传成 fail 检查")
