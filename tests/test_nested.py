"""嵌套上下文配对 NIAH 的构造、分析与比较测试。"""
from types import SimpleNamespace

from lega.data import CharTokenizer
from lega.nested import (analyze_nested, build_nested_niah,
                         build_nested_sample, compare_nested,
                         run_nested_niah)
from lega.ruler import ContextTooSmall

_TEXT = ("The grass is green. The sky is blue. 0123456789 "
         "One of the special magic numbers for lyre is: 7021853. "
         "What is the special magic number for lyre mentioned in the provided text? "
         "Answer: The special magic number for lyre mentioned in the provided text is")


def _tok():
    return CharTokenizer(sorted(set(_TEXT)))


def _payload(growth_values, positions, distances):
    results = {}
    for growth, per_ctx in growth_values.items():
        results[growth] = {}
        for ctx, outcomes in per_ctx.items():
            recs = []
            for pid, em in outcomes.items():
                recs.append({
                    "pair_id": pid,
                    "em": float(em),
                    "actual_tokens": ctx,
                    "needle_pos": positions[growth][ctx],
                    "evidence_distance_tokens": distances[growth][ctx],
                    "core_context_sha256": "same-core",
                    "key": "lyre",
                    "value": "7021853",
                })
            results[growth][str(ctx)] = {
                "n": len(recs),
                "em": sum(int(r["em"]) for r in recs) / len(recs),
                "actual_tokens": ctx,
                "filler_stream_sha256": "same-filler",
                "needle_pos": positions[growth][ctx],
                "evidence_distance_tokens": distances[growth][ctx],
                "outcomes": recs,
            }
    return {"design": {"base_ctx": 256, "mode": "synth"},
            "results": results}


def test_build_nested_design_invariants():
    tok = _tok()
    ctxs = [400, 500, 600]
    for growth in ("suffix", "prefix", "split"):
        exs = build_nested_niah(
            tok, ctxs, growth, key="lyre", value="7021853", filler_seed=7)
        assert [len(e["ids"]) for e in exs] == ctxs
        assert len({e["pair_id"] for e in exs}) == 1
        assert len({e["meta"]["core_context_sha256"] for e in exs}) == 1
        assert len({e["meta"]["filler_stream_sha256"] for e in exs}) == 1
        for ex in exs:
            assert ex["meta"]["key"] == "lyre"
            assert ex["meta"]["value"] == "7021853"
            s, e = ex["value_span"]
            assert tok.decode(ex["ids"][s:e]).strip() == "7021853"
            qs, qe = ex["query_span"]
            assert 0 <= qs < qe == len(ex["ids"])
            assert ex["meta"]["actual_tokens"] == len(ex["ids"])
        pos = [e["meta"]["needle_pos"] for e in exs]
        dist = [e["meta"]["evidence_distance_tokens"] for e in exs]
        if growth == "suffix":
            assert len(set(pos)) == 1 and dist[0] < dist[1] < dist[2]
        elif growth == "prefix":
            assert len(set(dist)) == 1 and pos[0] < pos[1] < pos[2]
        else:
            assert pos[0] < pos[1] < pos[2]
            assert dist[0] < dist[1] < dist[2]
    print("  [ok] 嵌套构造：精确长度、局部上下文/填充流固定、位置与距离可控")


def test_build_nested_sample_shares_core_across_growth():
    tok = _tok()
    sample = build_nested_sample(
        tok, [400, 500], seed=11, sample_index=2,
        growth_modes=("suffix", "prefix", "split"))
    assert sample["ok"]
    key = sample["key"]
    value = sample["value"]
    cores = set()
    fillers = set()
    for growth, exs in sample["growth"].items():
        assert [len(e["ids"]) for e in exs] == [400, 500]
        for ex in exs:
            assert ex["meta"]["key"] == key
            assert ex["meta"]["value"] == value
            cores.add(ex["meta"]["core_context_sha256"])
            fillers.add(ex["meta"]["filler_stream_sha256"])
    assert len(cores) == 1, "三种 growth 必须共享同一局部上下文"
    assert len(fillers) == 1, "三种 growth 必须共享同一填充流"
    print("  [ok] 同一 pair 跨 growth 共享 key/value、填充流与局部上下文")


def test_build_nested_rejects_bad_inputs():
    tok = _tok()
    bad_calls = [
        lambda: build_nested_niah(tok, [50, 60], "suffix",
                                  key="lyre", value="7021853"),
        lambda: build_nested_niah(tok, [400, 400], "suffix",
                                  key="lyre", value="7021853"),
        lambda: build_nested_niah(tok, [400], "suffix",
                                  key="lyre", value="7021853"),
        lambda: build_nested_niah(tok, [400, 500], "sideways",
                                  key="lyre", value="7021853"),
        lambda: build_nested_niah(tok, [400, 500], "suffix",
                                  key="lyre", value="abc"),
        lambda: build_nested_niah(tok, [400, 500], "suffix",
                                  key="lyre", value="7021853",
                                  base_ctx=500),
        lambda: build_nested_niah(tok, [400, 500], "corpus",
                                  key="lyre", value="7021853"),
        lambda: build_nested_niah(tok, [400, 500], "suffix",
                                  key="lyre", value="7021853",
                                  filler_ids=[1, 2, 3]),
    ]
    for i, fn in enumerate(bad_calls):
        try:
            fn()
        except (ValueError, ContextTooSmall):
            pass
        else:
            raise AssertionError(f"非法嵌套构造未被拒绝: case {i}")
    print("  [ok] 嵌套构造拒绝过短/重复/单档/非法 growth/非数值/短填充流")


def test_analyze_nested_paired_transitions_and_checks():
    pairs = [f"p{i}" for i in range(5)]
    payload = _payload(
        {
            "suffix": {
                256: {p: 1 for p in pairs},
                512: {p: 0 for p in pairs},
            },
            "prefix": {
                256: {p: 0 for p in pairs},
                512: {p: 1 for p in pairs},
            },
        },
        positions={"suffix": {256: 10, 512: 10},
                   "prefix": {256: 10, 512: 20}},
        distances={"suffix": {256: 100, 512: 200},
                   "prefix": {256: 100, 512: 100}},
    )
    ana = analyze_nested(payload)
    checks = ana["checks"]
    assert checks["lengths_exact"] is True
    assert checks["core_context_consistent"] is True
    assert checks["pairing_complete"] is True
    assert checks["suffix_position_fixed"] is True
    assert checks["suffix_distance_varies"] is True
    assert checks["prefix_distance_fixed"] is True
    assert checks["prefix_position_varies"] is True
    suffix = ana["growth"]["suffix"]
    assert suffix["cells"]["256"]["em"] == 1.0
    assert suffix["cells"]["512"]["em"] == 0.0
    assert suffix["em_slope_per_log2_ctx"] == -1.0
    tr = suffix["paired_transitions"][0]
    assert tr["base_only"] == 5 and tr["ctx_only"] == 0
    assert abs(tr["mcnemar_p"] - 0.0625) < 1e-12
    assert abs(tr["p_holm"] - 0.125) < 1e-12
    prefix = ana["growth"]["prefix"]
    assert prefix["em_slope_per_log2_ctx"] == 1.0
    assert prefix["paired_transitions"][0]["ctx_only"] == 5

    bad = _payload(
        {"suffix": {256: {p: 1 for p in pairs},
                    512: {p: 0 for p in pairs}}},
        positions={"suffix": {256: 10, 512: 10}},
        distances={"suffix": {256: 100, 512: 200}},
    )
    bad["results"]["suffix"]["256"]["outcomes"].append(
        dict(bad["results"]["suffix"]["256"]["outcomes"][0]))
    try:
        analyze_nested(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("重复 pair_id 必须被拒绝")
    print("  [ok] 分析：配对转移、McNemar、斜率与设计自检")


def test_compare_nested_matches_pairs_and_holm():
    pairs = [f"p{i}" for i in range(5)]
    cand = _payload(
        {"suffix": {256: {p: 1 for p in pairs},
                    512: {p: 1 for p in pairs}},
         "prefix": {256: {p: 1 for p in pairs},
                    512: {p: 1 for p in pairs}}},
        positions={"suffix": {256: 10, 512: 10},
                   "prefix": {256: 10, 512: 20}},
        distances={"suffix": {256: 100, 512: 200},
                   "prefix": {256: 100, 512: 100}},
    )
    ref = _payload(
        {"suffix": {256: {p: 0 for p in pairs},
                    512: {p: 0 for p in pairs}},
         "prefix": {256: {p: 0 for p in pairs},
                    512: {p: 0 for p in pairs}}},
        positions={"suffix": {256: 10, 512: 10},
                   "prefix": {256: 10, 512: 20}},
        distances={"suffix": {256: 100, 512: 200},
                   "prefix": {256: 100, 512: 100}},
    )
    comp = compare_nested(cand, ref)
    assert comp["n_tests"] == 4
    for rec in comp["tests"]:
        assert rec["candidate_only"] == 5 and rec["reference_only"] == 0
        assert abs(rec["mcnemar_p"] - 0.0625) < 1e-12
        assert abs(rec["p_holm"] - 0.25) < 1e-12
    assert all(abs(v["slope_diff"]) < 1e-12 for v in comp["slopes"].values())

    cand["results"]["suffix"]["256"]["outcomes"][0][
        "core_context_sha256"] = "different"
    try:
        compare_nested(cand, ref)
    except ValueError:
        pass
    else:
        raise AssertionError("局部上下文不一致的 pair 必须拒绝比较")
    print("  [ok] 比较：pair 对齐、Holm 校正与题面一致性检查")


def test_run_nested_niah_tiny_model():
    import torch

    from lega.model import GPT

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(vocab_size=len(tok.stoi), block_size=400, d_model=16,
                n_layers=1, n_heads=2, ffn_mult=4, variant="standard")
    model.eval()
    res = run_nested_niah(
        model, tok, "cpu", [300, 360], n_samples=1, seed=5, max_new=2,
        growth_modes=("suffix", "prefix"), gen_batch=1, verbose=False)
    for growth in ("suffix", "prefix"):
        assert res["results"][growth]["300"]["n"] == 1
        assert res["results"][growth]["360"]["n"] == 1
        pid300 = res["results"][growth]["300"]["outcomes"][0]["pair_id"]
        pid360 = res["results"][growth]["360"]["outcomes"][0]["pair_id"]
        assert pid300 == pid360
    ana = analyze_nested(res)
    assert ana["checks"]["core_context_consistent"] is True
    assert ana["checks"]["pairing_complete"] is True
    assert ana["checks"]["lengths_exact"] is True
    assert ana["checks"]["suffix_position_fixed"] is True
    assert ana["checks"]["prefix_distance_fixed"] is True
    print("  [ok] tiny GPT 端到端：逐档 EM、pair_id 对齐与设计自检")


def test_run_nested_rejects_invalid_protocol():
    tok = _tok()
    model = SimpleNamespace(block_size=320)
    calls = [
        dict(ctx_lens=[300, 360], max_new=4),
        dict(ctx_lens=[300], max_new=2),
        dict(ctx_lens="300,360", max_new=2),
        dict(ctx_lens=[300, 360], n_samples=0, max_new=2),
        dict(ctx_lens=[300, 360], max_new=2, growth_modes=("sideways",)),
    ]
    for kwargs in calls:
        try:
            run_nested_niah(model, tok, "cpu", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法评测协议未被拒绝: {kwargs}")
    print("  [ok] 评测入口拒绝超 block_size/单档/字符串/零样本/非法 growth")

def test_nested_cli_writes_payload_and_design_checks(monkeypatch):
    import json
    import os
    import shutil
    import sys
    import tempfile

    import numpy as np
    import torch

    from lega.data import CACHE_SCHEMA
    from lega.model import GPT, save_checkpoint
    from lega.nested import main as nested_main

    d = tempfile.mkdtemp()
    try:
        data = os.path.join(d, "data")
        os.makedirs(data)
        text = _TEXT * 3
        itos = sorted(set(text))
        with open(os.path.join(data, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "itos": itos, "char_unk": False,
                       "vocab_size": len(itos),
                       "cache_schema": CACHE_SCHEMA, "fit_on": "train"}, f)
        arr = np.array([itos.index(c) for c in text], dtype=np.uint16)
        arr.tofile(os.path.join(data, "train.bin"))
        arr.tofile(os.path.join(data, "val.bin"))
        ckpt = os.path.join(d, "model.pt")
        torch.manual_seed(0)
        model = GPT(vocab_size=len(itos), block_size=400, d_model=16,
                    n_layers=1, n_heads=2, ffn_mult=4, variant="standard",
                    dropout=0.0, attn_dropout=0.0)
        save_checkpoint(ckpt, model, extra={"variant": "standard", "seed": 5})
        out = os.path.join(d, "nested.json")
        monkeypatch.setattr(sys, "argv", [
            "nested", "--ckpt", ckpt, "--data_dir", data, "--out", out,
            "--ctx_lens", "300,360", "--n_samples", "1", "--max_new", "2",
            "--device", "cpu"])
        nested_main()
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["task"] == "niah_nested"
        assert payload["design"]["ctx_lens"] == [300, 360]
        assert payload["analysis"]["checks"]["lengths_exact"] is True
        assert payload["analysis"]["checks"]["pairing_complete"] is True
        assert payload["analysis"]["checks"][
            "core_context_consistent"] is True
        assert payload["results"]["suffix"]["300"]["n"] == 1
        assert payload["ckpt"] == os.path.abspath(ckpt)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] nested CLI 落盘 payload、设计自检与 checkpoint 身份")
