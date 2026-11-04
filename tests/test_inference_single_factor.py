"""推理期单因素纪律：病灶的实际影响通道按变体裁剪，重复干预不算两次检验。

训练期的单因素机制已有审计覆盖；这里盯住**推理期**的病灶臂：病灶声明的
通道集合（``LESION_EXPECTED_CHANNELS``）是变体无关的上界，而变体实际注册
的门控项不同（``qgate`` 只有 A，``gate_only`` 一条都没有）。不按变体裁剪
就会出现两类实质错误：

  * 把只动了 A 的干预写成“去全部证据 A+B+C”，读者把单通道差异误读成
    整体证据；
  * 同一变体上两个签名相同的病灶（``qgate`` 的 ``no_evidence`` 实际只清零
    ``gate_w``，与 ``no_query`` 是同一个干预）被当成两次独立检验放进同一个
    Holm 家族，“两个病灶各自显著”其实是同一个干预的同一个数。
"""

import pytest

from lega.ablate import (
    LESION_CONTRAST_GATE,
    LESION_CONTRAST_MULTI,
    LESION_CONTRAST_SINGLE,
    LESION_EXPECTED_CHANNELS,
    LESION_ORDER,
    channels_from_capabilities,
    lesion_affected_channels,
    lesion_contrast_type,
    lesion_intervention_signature,
    probe_lesion_param_effects,
)
from lega.attention import variant_gate_capabilities
from lega.pipeline.common import single_factor_audit

Q_CAPS = {"gated": True, "query": True, "conc": False, "len": False}
GATE_ONLY_CAPS = {"gated": True, "query": False, "conc": False, "len": False}
LEGA_CAPS = {"gated": True, "query": True, "conc": True, "len": True}


def test_channels_follow_variant_registration():
    """变体实际注册的门控项决定它拥有哪些证据通道。"""
    assert channels_from_capabilities(Q_CAPS) == ("A",)
    assert channels_from_capabilities(GATE_ONLY_CAPS) == ()
    assert channels_from_capabilities(LEGA_CAPS) == ("A", "B", "C")
    assert channels_from_capabilities(variant_gate_capabilities("qgate")) == ("A",)
    assert channels_from_capabilities(variant_gate_capabilities("gate_only")) == ()


def test_affected_channels_are_clipped_to_the_variant():
    """声明的 A+B/C 只是上界：变体缺通道时实际影响集合必须被裁剪。"""
    assert lesion_affected_channels("no_evidence", LEGA_CAPS) == ("A", "B", "C")
    assert lesion_affected_channels("no_evidence", Q_CAPS) == ("A",)
    assert lesion_affected_channels("no_query", Q_CAPS) == ("A",)
    # gate_only 没有证据通道：门级干预不覆盖任何证据通道。
    assert lesion_affected_channels("gate_open", GATE_ONLY_CAPS) == ()
    assert lesion_affected_channels("gate_open", Q_CAPS) == ("A",)
    # 不适用的病灶（缺 requires 声明的能力）根本没被施加。
    assert lesion_affected_channels("no_conc", Q_CAPS) == ()
    assert lesion_affected_channels("query_only", Q_CAPS) == ()


def test_contrast_type_follows_actual_channel_count():
    """对比类型按**实际**影响通道数判定：1→单因素，≥2→多因素，0→门级。"""
    assert lesion_contrast_type(("A",)) == LESION_CONTRAST_SINGLE
    assert lesion_contrast_type(("A", "B")) == LESION_CONTRAST_MULTI
    assert lesion_contrast_type(()) == LESION_CONTRAST_GATE


def test_duplicate_interventions_share_one_signature():
    """签名相同 = 施加的是同一个干预，ΔNLL 恒等，不能算两次独立检验。"""
    q_evidence = lesion_intervention_signature("no_evidence", Q_CAPS)
    q_query = lesion_intervention_signature("no_query", Q_CAPS)
    assert q_evidence == q_query, (q_evidence, q_query)
    # 完整变体上 no_evidence 与 no_query 才是两个不同干预。
    assert lesion_intervention_signature("no_evidence", LEGA_CAPS) != (
        lesion_intervention_signature("no_query", LEGA_CAPS)
    )


def test_param_probe_matches_variant_clipped_expectation():
    """参数级实测（任意带门变体）必须与按变体裁剪的元数据口径一致。"""
    probe = probe_lesion_param_effects("no_evidence", variant="qgate")
    assert probe.get("error") is None, probe
    assert probe["applied"] is True, probe
    assert probe["affected"] == ["A"], probe
    assert probe["expected"] == ["A"], probe
    assert probe["declared"] == list(LESION_EXPECTED_CHANNELS["no_evidence"]), probe

    probe = probe_lesion_param_effects("no_evidence", variant="lega")
    assert probe.get("error") is None, probe
    assert probe["affected"] == ["A", "B", "C"], probe

    probe = probe_lesion_param_effects("no_query", variant="qgate")
    assert probe["affected"] == ["A"], probe

    # gate_only 没有任何证据通道：门级病灶实测 0 条证据通道。
    probe = probe_lesion_param_effects("gate_open", variant="gate_only")
    assert probe.get("error") is None, probe
    assert probe["applied"] is True, probe
    assert probe["affected"] == [], probe


def test_audit_covers_every_variant_and_flags_duplicates():
    """审计必须逐变体实测，并把重复干预显式列出来。"""
    audit = single_factor_audit()
    assert audit["ok"], audit["problems"]
    rows = audit["inference_lesion_variant_effects"]
    variants = {r["variant"] for r in rows}
    assert {"lega", "qgate", "gate_only"} <= variants, variants
    assert len(rows) == len(variants) * len(LESION_ORDER), len(rows)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]

    duplicates = audit["inference_lesion_duplicate_interventions"]
    pairs = {(r["variant"], r["lesion"], r["duplicate_of"]) for r in duplicates}
    assert ("qgate", "no_evidence", "no_query") in pairs, pairs
    assert ("lega_conc_only", "no_evidence", "no_conc") in pairs, pairs
    # 未施加的病灶不构成“重复干预”。
    assert not any(
        r["lesion"] == "no_conc" and r["variant"] == "qgate" for r in duplicates
    )


def test_audit_detects_variant_level_channel_drift(monkeypatch):
    """实现层多动一条通道时，逐变体审计必须抓出来。"""
    import lega.ablate as ablate_mod

    real = ablate_mod.probe_lesion_param_effects

    def drifted(lesion_id, **kwargs):
        probe = real(lesion_id, **kwargs)
        if lesion_id == "no_query" and kwargs.get("variant") == "qgate":
            probe["affected"] = ["A", "B"]
        return probe

    monkeypatch.setattr(ablate_mod, "probe_lesion_param_effects", drifted)
    audit = single_factor_audit()
    assert not audit["ok"], audit
    assert any("no_query" in p and "qgate" in p for p in audit["problems"]), audit


def _payload(variant, seed, lesion_deltas):
    return {
        "variant": variant,
        "seed": seed,
        "lesions": [
            {
                "lesion": lid,
                "applicable": True,
                "applied": True,
                "no_op": False,
                "delta_loss_mean": delta,
                "n_windows": 10,
            }
            for lid, delta in lesion_deltas.items()
        ],
    }


def test_causal_records_tag_actual_channels_and_drop_duplicates():
    """结果层：实际通道要写进记录，重复干预要排除出 Holm 家族。"""
    from run_all import _ablation_lesion_records

    ablations = {
        "qgate": {
            s: _payload("qgate", s, {"no_query": 0.10, "no_evidence": 0.10})
            for s in (1, 2, 3, 4, 5)
        }
    }
    tests = _ablation_lesion_records(ablations)
    by_lesion = {t["lesion"]: t for t in tests}
    assert set(by_lesion) == {"no_query", "no_evidence"}, by_lesion

    query = by_lesion["no_query"]
    assert query["changed_factors"] == ["A"]
    assert query["contrast_type"] == LESION_CONTRAST_SINGLE
    assert query["declared_channels"] == ["A"]
    assert query["channels_degraded"] is False
    assert query.get("excluded_from_family") is not True
    assert query["p_adj"] is not None

    evidence = by_lesion["no_evidence"]
    # 声明 A+B+C，实际只动了 A → 单因素，且显式标注降级。
    assert evidence["changed_factors"] == ["A"]
    assert evidence["contrast_type"] == LESION_CONTRAST_SINGLE
    assert evidence["declared_channels"] == ["A", "B", "C"]
    assert evidence["channels_degraded"] is True
    # 同一干预 → 排除出 Holm 家族，不计为第二次独立检验。
    assert evidence["duplicate_of"] == "no_query"
    assert evidence["excluded_from_family"] is True
    assert evidence["p_adj"] is None
    assert "重复干预" in str(evidence.get("verdict") or "")
    assert abs(evidence["mean_diff"] - query["mean_diff"]) < 1e-12


def test_full_variant_multi_factor_stays_multi_factor():
    """完整变体上 no_evidence 仍是多因素对照，不得被降级成单因素。"""
    from run_all import _ablation_lesion_records

    ablations = {
        "lega": {
            s: _payload("lega", s, {"no_query": 0.10, "no_evidence": 0.30})
            for s in (1, 2, 3, 4, 5)
        }
    }
    tests = _ablation_lesion_records(ablations)
    by_lesion = {t["lesion"]: t for t in tests}
    assert by_lesion["no_evidence"]["changed_factors"] == ["A", "B", "C"]
    assert by_lesion["no_evidence"]["contrast_type"] == LESION_CONTRAST_MULTI
    assert by_lesion["no_evidence"]["channels_degraded"] is False
    assert by_lesion["no_evidence"].get("excluded_from_family") is not True
    assert by_lesion["no_evidence"]["p_adj"] is not None


def test_report_table_shows_actual_channels_and_duplicate_flag():
    """汇总页必须写出“类型（实际影响通道）”，并标出重复干预。"""
    from run_all import _ablation_lesion_records, _render_causal_ablation_table

    ablations = {
        "qgate": {
            s: _payload("qgate", s, {"no_query": 0.10, "no_evidence": 0.10})
            for s in (1, 2, 3, 4, 5)
        }
    }
    tests = _ablation_lesion_records(ablations)
    md = "\n".join(_render_causal_ablation_table(tests))
    assert "类型（实际影响通道）" in md
    assert "单因素 A" in md
    assert "声明 A/B/C" in md
    assert "重复干预" in md
    assert "no_evidence≡no_query" in md


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
