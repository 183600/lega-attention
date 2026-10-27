"""注意力核心数学性质的单元测试。

重点验证：
  1. 均匀注意力 => max_j p_j = 1/n => c = log(n * 1/n) = 0（LEGA 的核心设计性质）
  2. 完全集中于单键 => c ≈ log n
  3. 门控公式逐项可复算
  4. RoPE 平移等变性：score(m, n) 只依赖 m - n
  5. 全部变体可前向/反向，且额外参数量符合理论值
"""
import math
from unittest import mock

import torch

import lega.attention as attention_mod
from lega.attention import (CONTROL_VARIANTS, CausalSelfAttention,
                            GATE_FACTORS, RUNNABLE_VARIANTS, VARIANTS,
                            _placebo_evidence)
from lega.model import (GPT, RoPE, expected_gate_extras,
                        gate_extra_params, gate_extra_params_per_head)


def _mk_attn(d_model=16, n_heads=1, variant="lega_no_query", **kw):
    torch.manual_seed(0)
    return CausalSelfAttention(d_model, n_heads, variant=variant, **kw)


def test_uniform_attention_gives_c_zero():
    """均匀注意力时 c = 0：与上下文长度无关（设计性质）。"""
    m = _mk_attn(variant="lega_no_query")
    torch.nn.init.zeros_(m.wq.weight)
    torch.nn.init.zeros_(m.wk.weight)   # q·k = 0 → softmax 均匀
    with torch.no_grad():
        m.gate_a.fill_(1.0)
        m.gate_b.fill_(0.0)
        m.gate_d.fill_(0.0)
    x = torch.randn(1, 8, 16)
    out, cache, stats = m(x, rope=None, need_stats=True)
    for t in range(8):
        n = t + 1
        assert abs(stats["max_p"][0, 0, t].item() - 1.0 / n) < 1e-5
        # logit = a*c + b*log n + d = 0（均匀时 c=0 且 b=0, d=0）
        assert abs(stats["logit"][0, 0, t].item()) < 1e-5
        assert abs(stats["g"][0, 0, t].item() - 0.5) < 1e-5
    assert out.shape == (1, 8, 16)
    assert torch.isfinite(out).all()


def test_peaked_attention_gives_c_logn():
    """完全集中于单键时 c ≈ log n。"""
    m = _mk_attn(variant="lega_no_query")
    torch.nn.init.zeros_(m.wq.weight)
    torch.nn.init.eye_(m.wk.weight)     # k_j = x_j
    with torch.no_grad():
        m.wq.weight.copy_(torch.eye(16) * 1.0)  # q_t = x_t
        m.gate_a.fill_(1.0)
        m.gate_b.fill_(0.0)
        m.gate_d.fill_(0.0)
    x = torch.randn(1, 10, 16)
    x[0, :, 0] = 1.0        # 保证 token3 与所有查询正相关
    x[0, 3, 0] = 50.0       # token3 的键远大于其它键 → 注意力集中于 token3
    out, cache, stats = m(x, rope=None, need_stats=True)
    for t in range(4, 10):
        n = t + 1
        assert stats["max_p"][0, 0, t].item() > 0.99, f"位置 {t} 未集中: {stats['max_p'][0,0,t]}"
        c = stats["logit"][0, 0, t].item()   # a=1 → logit = c = log n + log max_p
        assert abs(c - math.log(n)) < 0.05, f"c={c} vs log n={math.log(n)}"


def test_gate_formula_components():
    """门控 logit 各分量可复算（w·q + a*c + b*log n + d）。"""
    m = _mk_attn(d_model=16, n_heads=2, variant="lega")
    with torch.no_grad():
        m.gate_w.fill_(0.05)
        m.gate_a.fill_(0.3)
        m.gate_b.fill_(-0.1)
        m.gate_d.fill_(2.0)
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        out, cache, stats = m(x, rope=None, need_stats=True)
    import numpy as np
    max_p = stats["max_p"].float().numpy()
    log_n = stats["log_n"].numpy()
    qterm = stats["qterm"].numpy()
    g = stats["g"].numpy()
    c = log_n + np.log(np.clip(max_p, 1e-30, None))
    expected_logit = 2.0 + 0.3 * c - 0.1 * log_n
    # qterm 已含 w·q；下面完整复算 qterm 与 g
    q = m.wq(x).view(2, 6, 2, 8).transpose(1, 2).detach().float().numpy()
    qterm_manual = (q * 0.05).sum(-1)
    assert np.allclose(qterm, qterm_manual, atol=1e-5)
    full = expected_logit + qterm_manual
    assert np.allclose(g, 1.0 / (1.0 + np.exp(-full)), atol=1e-5)


def test_concentration_ablations_keep_length_normalization():
    """消融长度项 b*log n 时，集中度项仍是 c=log(n*max_p)，不能连带删掉
    a*log n；否则 Full−C 不再是单组件消融，B 格也不再是完整集中度项。"""
    for variant in ("lega_no_len", "lega_conc_only"):
        m = _mk_attn(variant=variant)
        with torch.no_grad():
            m.gate_a.fill_(0.5)
            m.gate_d.zero_()
            if m.use_query:
                m.gate_w.zero_()
        max_p = torch.full((1, 1, 1), 0.25)
        log_n = torch.full((1, 1, 1), math.log(8.0))
        qterm = torch.zeros(1, 1, 1) if m.use_query else None
        _, logit, _ = m._gate_logit(qterm, max_p, log_n, (1, 1, 1),
                                    keep_terms=True)
        expected = 0.5 * (math.log(0.25) + math.log(8.0))
        assert abs(logit.item() - expected) < 1e-5, (variant, logit.item())
    print("  [ok] 集中度消融保留 c=log(n*max_p) 的长度归一化")


def test_gate_factor_metadata_matches_variant_formulas():
    """单因素实验的因子元数据必须与变体实际启用的门控项一致。"""
    expected = {
        "gate_only": set(),
        "qgate": {"A"},
        "lega_conc_only": {"B"},
        "lega_len_only": {"C"},
        "lega_no_len": {"A", "B"},
        "lega_no_conc": {"A", "C"},
        "lega_no_query": {"B", "C"},
        "lega": {"A", "B", "C"},
    }
    assert set(GATE_FACTORS) == set(expected)
    # standard/null 不属于门控内 2^3，不能出现在因子表里
    assert "standard" not in GATE_FACTORS and "null" not in GATE_FACTORS
    for variant, factors in expected.items():
        assert GATE_FACTORS[variant] == frozenset(factors), variant
        m = _mk_attn(variant=variant)
        actual = {f for f, on in (("A", m.use_query), ("B", m.use_conc),
                                  ("C", m.use_len)) if on}
        assert actual == factors, (variant, actual, factors)
    print("  [ok] 门控因子元数据与各变体 A/B/C 实际开关一致")


def test_gate_query_term_uses_pre_rope_query():
    """门控查询项 w·q 不应混入 RoPE 绝对位置（注意力打分仍使用 RoPE）。"""
    torch.manual_seed(0)
    m = _mk_attn(d_model=16, n_heads=2, variant="lega")
    with torch.no_grad():
        m.gate_w.normal_(0, 0.5)
    rope = RoPE(head_dim=8, max_len=32)
    suffix = torch.randn(1, 5, 16)
    padded = torch.cat([torch.randn(1, 7, 16), suffix], dim=1)
    m.eval()
    with torch.no_grad():
        _, _, s1 = m(suffix, rope=rope, need_stats=True)
        _, _, s2 = m(padded, rope=rope, need_stats=True)
    assert torch.allclose(s1["qterm"], s2["qterm"][:, :, 7:], atol=1e-6), \
        "qterm 使用了 RoPE 后的 q，导致门控混入绝对位置"


def test_rope_translation_equivalence():
    """RoPE: score(q@pos m, k@pos n) 只依赖 m-n。"""
    rope = RoPE(head_dim=16, max_len=64)
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(mpos, npos):
        qr = rope.rotate(q, past_len=mpos)
        kr = rope.rotate(k, past_len=npos)
        return (qr @ kr.transpose(-1, -2)).item()

    assert abs(score(3, 7) - score(10, 14)) < 1e-4
    assert abs(score(0, 5) - score(20, 25)) < 1e-4
    assert abs(score(5, 0) - score(9, 4)) < 1e-4


def test_all_variants_forward_backward_and_params():
    """全部变体：前向有限、反向可传、额外参数量符合理论。"""
    for v in VARIANTS:
        torch.manual_seed(0)
        model = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2,
                    n_heads=2, ffn_mult=4, variant=v)
        assert gate_extra_params(model) == expected_gate_extras(v, 16, 2, 2), v
        per_head = gate_extra_params_per_head(model)
        assert per_head == expected_gate_extras(v, 16, 2, 2) / (2 * 2), v
        idx = torch.randint(0, 64, (2, 12))
        out = model(idx, targets=idx)
        assert torch.isfinite(out["loss"]).item(), v
        out["loss"].backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{v}:{n} 梯度非有限"
        print(f"  [ok] {v}: params={sum(p.numel() for p in model.parameters()):,} "
              f"extras={gate_extra_params(model)}")


def test_gate_stats_batch_shape_all_variants():
    """门控统计必须与输入批次对齐 (B,H,T)。

    lega_len_only 只含 b·log n 一项，若 (1,H,1) 参数与 (T,) 的 log_n 广播会
    把批次维压成 1，机制分析按批次行取统计时将越界；该测试同时覆盖
    显式完整路径（stats_argmax=True）与 SDPA 轻量路径（stats_argmax=False）。
    """
    torch.manual_seed(5)
    x = torch.randn(2, 12, 32)
    for variant in VARIANTS:
        for stats_argmax in (True, False):
            m = _mk_attn(d_model=32, n_heads=2, variant=variant)
            m.eval()
            with torch.no_grad():
                _, _, stats = m(x, rope=None, need_stats=True,
                                stats_argmax=stats_argmax)
            if not stats:
                continue
            for key in ("g", "logit", "qterm", "max_p", "argmax", "p_null"):
                v = stats.get(key)
                if v is None:
                    continue
                assert v.shape == (2, 2, 12), (variant, stats_argmax, key, v.shape)


def test_standard_sdpa_matches_explicit():
    """standard 的 SDPA 快速路径与显式路径数值一致。"""
    torch.manual_seed(0)
    m_fast = CausalSelfAttention(32, 2, variant="standard", use_sdpa_standard=True)
    m_slow = CausalSelfAttention(32, 2, variant="standard", use_sdpa_standard=False)
    m_slow.load_state_dict(m_fast.state_dict())
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        o1, _, _ = m_fast(x)
        o2, _, _ = m_slow(x)
    assert torch.allclose(o1, o2, atol=1e-5), (o1 - o2).abs().max()


def test_light_stats_parity_sdpa_vs_explicit():
    """轻量诊断（need_stats=True, stats_argmax=False）在 SDPA 与显式路径上
    必须返回相同口径的 stats：log_n 等键的存在性与数值都一致。"""
    torch.manual_seed(1)
    x = torch.randn(2, 20, 32)
    for variant in ("standard", "qgate", "lega_no_conc"):
        m_fast = CausalSelfAttention(32, 2, variant=variant, use_sdpa_standard=True)
        m_slow = CausalSelfAttention(32, 2, variant=variant, use_sdpa_standard=False)
        m_slow.load_state_dict(m_fast.state_dict())
        m_fast.eval()
        m_slow.eval()
        with torch.no_grad():
            o1, _, s1 = m_fast(x, need_stats=True, stats_argmax=False)
            o2, _, s2 = m_slow(x, need_stats=True, stats_argmax=False)
        assert torch.allclose(o1, o2, atol=1e-5), variant
        for key in ("max_p", "log_n", "argmax", "qterm", "g", "logit"):
            a, b = s1.get(key), s2.get(key)
            assert (a is None) == (b is None), (variant, key, a is not None, b is not None)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (variant, key)


def test_standard_sdpa_cache_forwards_dropout_in_training():
    """训练态 + KV 缓存的 SDPA 分支也必须启用 attention dropout。"""
    torch.manual_seed(0)
    m = CausalSelfAttention(32, 2, variant="standard", attn_dropout=0.3,
                            use_sdpa_standard=True)
    x = torch.randn(1, 6, 32)
    m.eval()
    with torch.no_grad():
        _, past, _ = m(x)
    m.train()
    with mock.patch("torch.nn.functional.scaled_dot_product_attention",
                    wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
        m(x[:, -1:], past_kv=past)
    kwargs = sdpa.call_args.kwargs
    assert kwargs["dropout_p"] == 0.3
    # 单查询步所有键必然可见，无需掩码（None 或全 True 均可）
    if kwargs["attn_mask"] is not None:
        assert bool(kwargs["attn_mask"].all())


def test_query_chunked_attention_matches_unchunked():
    """查询维分块（逐块 checkpoint / 直接重算两种模式）与单块显式注意力逐位一致（前向/反向）。"""
    torch.manual_seed(3)
    ref = CausalSelfAttention(32, 2, variant="lega")
    with torch.no_grad():
        ref.gate_w.normal_(0, 0.2)
        ref.gate_a.normal_(0, 0.2)
        ref.gate_b.normal_(0, 0.2)
    x = torch.randn(2, 96, 32)

    old = attention_mod._CHUNK_TARGET_ELEMS
    old_ckpt = attention_mod._CKPT_MIN_BYTES
    try:
        attention_mod._CHUNK_TARGET_ELEMS = 10**12
        ref.train()
        o_ref, _, _ = ref(x)
        (o_ref * o_ref).mean().backward()
        # 阈值 0 强制逐块 checkpoint；阈值极大走不重算的分块路径。
        for ckpt_bytes in (0, 10**18):
            attention_mod._CKPT_MIN_BYTES = ckpt_bytes
            chk = CausalSelfAttention(32, 2, variant="lega")
            chk.load_state_dict(ref.state_dict())
            attention_mod._CHUNK_TARGET_ELEMS = 1024
            chk.train()
            o_chk, _, _ = chk(x)
            (o_chk * o_chk).mean().backward()

            assert torch.equal(o_ref, o_chk)
            for (n0, p0), (n1, p1) in zip(ref.named_parameters(),
                                          chk.named_parameters()):
                assert n0 == n1
                if p0.grad is not None:
                    assert torch.allclose(p0.grad, p1.grad,
                                          atol=1e-5, rtol=1e-5), n0
    finally:
        attention_mod._CHUNK_TARGET_ELEMS = old
        attention_mod._CKPT_MIN_BYTES = old_ckpt


def test_chunked_full_stats_matches_unchunked():
    """完整诊断统计也可分块产生，且与单块结果一致。"""
    old = attention_mod._CHUNK_TARGET_ELEMS
    for variant in ("standard", "null", "lega"):
        torch.manual_seed(4)
        ref = CausalSelfAttention(32, 2, variant=variant)
        chk = CausalSelfAttention(32, 2, variant=variant)
        chk.load_state_dict(ref.state_dict())
        x = torch.randn(1, 96, 32)
        try:
            attention_mod._CHUNK_TARGET_ELEMS = 10**12
            with torch.no_grad():
                o_ref, _, s_ref = ref(x, need_stats=True)
            attention_mod._CHUNK_TARGET_ELEMS = 1024
            with torch.no_grad():
                o_chk, _, s_chk = chk(x, need_stats=True)
        finally:
            attention_mod._CHUNK_TARGET_ELEMS = old

        assert torch.allclose(o_ref, o_chk, atol=1e-5, rtol=1e-5), variant
        for key in ("max_p", "log_n", "qterm", "g", "logit", "p_null"):
            a, b = s_ref.get(key), s_chk.get(key)
            if a is None:
                assert b is None, (variant, key)
            else:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (variant, key)
        assert torch.equal(s_ref["argmax"], s_chk["argmax"]), variant


def test_null_variant_max_p_includes_null_slot():
    """null 主导时 max_p 必须是含 null 槽的联合概率最大值（与 argmax=-1 一致）。"""
    old = attention_mod._CHUNK_TARGET_ELEMS
    torch.manual_seed(0)
    ref = CausalSelfAttention(16, 1, variant="null")
    chk = CausalSelfAttention(16, 1, variant="null")
    with torch.no_grad():
        ref.wq.weight.copy_(torch.eye(16))
        ref.wk.weight.copy_(torch.eye(16))
        # k_null 与查询方向高度对齐 → null 槽以接近 1 的概率主导。
        ref.k_null.copy_(torch.eye(16)[0] * 1000.0)
        ref.v_null.zero_()
    chk.load_state_dict(ref.state_dict())
    x = torch.zeros(1, 40, 16)
    x[..., 0] = 10.0
    try:
        attention_mod._CHUNK_TARGET_ELEMS = 10**12
        with torch.no_grad():
            _, _, s_ref = ref(x, need_stats=True)
        attention_mod._CHUNK_TARGET_ELEMS = 1024
        with torch.no_grad():
            _, _, s_chk = chk(x, need_stats=True)
    finally:
        attention_mod._CHUNK_TARGET_ELEMS = old
    for s in (s_ref, s_chk):
        for t in range(x.shape[1]):
            assert s["argmax"][0, 0, t].item() == -1
            assert abs(s["max_p"][0, 0, t].item()
                       - s["p_null"][0, 0, t].item()) < 1e-5
    assert torch.allclose(s_ref["max_p"], s_chk["max_p"],
                          atol=1e-5, rtol=1e-5)


def test_null_zero_sdpa_keeps_null_slot():
    """null_zero 的 SDPA 快路也必须并入空槽，不能只对真实键做 softmax。

    null 的 v_null 初始为 0，与 null_zero 的固定零 value 等价；两者在同一
    权重下应给出完全相同的输出（prefill 与增量解码）。
    """
    torch.manual_seed(0)
    fast = CausalSelfAttention(32, 2, variant="null_zero",
                               use_sdpa_standard=True)
    slow = CausalSelfAttention(32, 2, variant="null_zero",
                               use_sdpa_standard=False)
    ref = CausalSelfAttention(32, 2, variant="null",
                              use_sdpa_standard=True)
    with torch.no_grad():
        fast.k_null.normal_(0, 0.5)
        slow.load_state_dict(fast.state_dict())
        ref.load_state_dict(fast.state_dict())
        # 给 null 一个非零 value 并同步到 null_zero 无法表达（buffer 固定为零），
        # 因此这里只验证零 value 下的路径等价性。
        ref.v_null.zero_()
    fast.eval()
    slow.eval()
    ref.eval()
    x = torch.randn(1, 12, 32)
    with torch.no_grad():
        o_fast, cache_fast, _ = fast(x)
        o_slow, cache_slow, _ = slow(x)
        o_ref, cache_ref, _ = ref(x)
    assert torch.allclose(o_fast, o_slow, atol=1e-5, rtol=1e-5)
    assert torch.allclose(o_fast, o_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cache_fast[0], cache_slow[0])
    assert torch.allclose(cache_fast[0], cache_ref[0])
    # 增量解码：单查询步走显式路径，也必须包含 null 槽。
    nxt = torch.randn(1, 1, 32)
    with torch.no_grad():
        o1, _, _ = fast(nxt, past_kv=cache_fast)
        o2, _, _ = slow(nxt, past_kv=cache_slow)
    assert torch.allclose(o1, o2, atol=1e-5, rtol=1e-5)


def test_mask_cache_byte_limit_uses_lru():
    """全局掩码缓存按字节上限驱逐最旧条目。"""
    old_cap = attention_mod._CACHE_CAP
    old_limit = attention_mod._MASK_CACHE_MAX_BYTES
    old_tri_limit = attention_mod._TRI_CACHE_MAX_BYTES
    mask0 = None
    try:
        attention_mod._MASK_CACHE.clear()
        attention_mod._MASK_CACHE_BYTES = 0
        attention_mod._TRI_CACHE.clear()
        attention_mod._TRI_CACHE_BYTES = 0
        attention_mod._CACHE_CAP = 8
        attention_mod._MASK_CACHE_MAX_BYTES = 20
        attention_mod._TRI_CACHE_MAX_BYTES = 6
        for i in range(2):
            entry = [torch.ones(1, 1, 4, 4, dtype=torch.bool),
                     torch.ones(1, 1, 4, 1, dtype=torch.bool), None, None]
            if i == 0:
                mask0 = entry
            attention_mod._mask_cache_store(("cpu", i), entry)
        assert len(attention_mod._MASK_CACHE) == 1
        assert attention_mod._MASK_CACHE_BYTES == 20
        assert not any(entry is mask0 for entry in attention_mod._MASK_CACHE.values())
        first_tri = attention_mod._tri_band(2, 2, torch.device("cpu"))
        attention_mod._tri_band(2, 3, torch.device("cpu"))
        assert len(attention_mod._TRI_CACHE) == 1
        assert not any(tri is first_tri for tri in attention_mod._TRI_CACHE.values())
    finally:
        attention_mod._MASK_CACHE.clear()
        attention_mod._MASK_CACHE_BYTES = 0
        attention_mod._TRI_CACHE.clear()
        attention_mod._TRI_CACHE_BYTES = 0
        attention_mod._CACHE_CAP = old_cap
        attention_mod._MASK_CACHE_MAX_BYTES = old_limit
        attention_mod._TRI_CACHE_MAX_BYTES = old_tri_limit


def test_all_variants_support_zero_length_input():
    """空查询序列对所有变体都必须返回空输出/缓存，而不是只在部分变体崩溃。"""
    C = 16
    x = torch.empty(2, 0, C)
    for v in VARIANTS:
        attn = CausalSelfAttention(C, 2, variant=v)
        out, cache, stats = attn(x)
        assert out.shape == (2, 0, C), (v, out.shape)
        assert cache[0].shape == (2, 2, 0, C // 2), (v, cache[0].shape)
        assert stats is None
        out2, cache2, stats2 = attn(x, need_stats=True)
        assert out2.shape == (2, 0, C)
        assert isinstance(stats2, dict) and stats2["g"] is None


def test_placebo_control_matches_capacity_but_not_evidence():
    """参数匹配安慰剂：同门控参数/FLOPs，但 B/C 证据换成无信息噪声。

    该对照回答“LEGA 的收益是否只是多几个门控参数的优化红利”。测试必须
    同时验证 (1) 参数预算与 LEGA 完全一致；(2) 它确实不是 LEGA 的等价
    前向（否则对照没有区分力）；(3) 安慰剂特征确定、有界、不消耗 RNG。
    """
    assert "lega_placebo" in RUNNABLE_VARIANTS
    assert "lega_placebo" in CONTROL_VARIANTS
    assert "lega_placebo" not in VARIANTS
    assert "lega_placebo" not in GATE_FACTORS
    torch.manual_seed(0)
    real = CausalSelfAttention(32, 2, variant="lega", use_sdpa_standard=False)
    torch.manual_seed(0)
    plac = CausalSelfAttention(32, 2, variant="lega_placebo",
                               use_sdpa_standard=False)
    assert expected_gate_extras("lega_placebo", 16, 2, 2) == \
        expected_gate_extras("lega", 16, 2, 2)
    for name in ("gate_w", "gate_a", "gate_b", "gate_d"):
        assert getattr(real, name).shape == getattr(plac, name).shape, name
    assert sum(p.numel() for p in real.parameters()) == \
        sum(p.numel() for p in plac.parameters())

    with torch.no_grad():
        for m in (real, plac):
            m.gate_w.normal_(0.0, 0.02)
            m.gate_a.fill_(0.5)
            m.gate_b.fill_(-0.3)
            m.gate_d.fill_(1.0)
    x = torch.randn(2, 12, 32)
    real.eval(); plac.eval()
    with torch.no_grad():
        o1, _, _ = real(x)
        o2, _, _ = plac(x)
    assert o1.shape == o2.shape
    assert not torch.allclose(o1, o2, atol=1e-6), \
        "安慰剂与 LEGA 前向完全相同，说明证据没有被置换"

    # 特征确定、有界、同形状；同一输入两次调用逐位相同（不消耗全局 RNG）。
    z = torch.zeros(2, 3, 7)
    f1 = _placebo_evidence(z, 3, 1.0)
    f2 = _placebo_evidence(z, 3, 1.0)
    assert torch.equal(f1, f2)
    assert f1.shape == z.shape and float(f1.abs().max()) <= 1.0
    ln = torch.zeros(9)
    assert _placebo_evidence(ln, 4, 2.0).shape == ln.shape

    # 梯度仍能经 a/b 回传（不是被 detach 的死支路）。
    plac.train()
    out, _, _ = plac(x)
    out.sum().backward()
    for name in ("gate_a", "gate_b"):
        g = getattr(plac, name).grad
        assert g is not None and torch.isfinite(g).all(), name
    print("  [ok] lega_placebo：参数匹配、证据被置换、梯度可达")
