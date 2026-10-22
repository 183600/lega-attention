"""模型级测试：KV 缓存一致性、贪心生成确定性、检查点保存与恢复。"""
import os
import tempfile

import numpy as np
import torch

from lega.attention import CausalSelfAttention, VARIANTS
from lega.model import GPT, load_model, save_checkpoint
from lega.train import _val_eval_indices, evaluate_val


def _tiny(variant):
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2, n_heads=2,
               ffn_mult=4, variant=variant)


def test_model_constructor_rejects_invalid_regularization():
    """模型/注意力构造不能把 bool/NaN/Inf 静默转成 dropout 或门控偏置。

    train 已在配置入口校验，但直接构造 GPT/CausalSelfAttention（sanity、
    latency、外部脚本）会绕过它：``bool`` 被 float 成 0/1、NaN 写入
    gate_d，都会让后续训练/评测结果失去意义。
    """
    bad_probs = (True, False, np.bool_(True), -0.1, 1.1, float("nan"),
                 float("inf"), None, "nan")
    for kw in ("dropout", "attn_dropout"):
        for bad in bad_probs:
            try:
                GPT(vocab_size=32, block_size=16, d_model=16, n_layers=1,
                    n_heads=2, variant="lega", **{kw: bad})
                raise AssertionError(f"GPT({kw}={bad!r}) 应被拒绝")
            except ValueError as exc:
                assert kw in str(exc)
    for bad in (True, np.bool_(False), float("nan"), float("inf"), None, "nan"):
        try:
            GPT(vocab_size=32, block_size=16, d_model=16, n_layers=1,
                n_heads=2, variant="lega", gate_bias_init=bad)
            raise AssertionError(f"GPT(gate_bias_init={bad!r}) 应被拒绝")
        except ValueError as exc:
            assert "gate_bias_init" in str(exc)
    for bad in bad_probs:
        try:
            CausalSelfAttention(16, 2, attn_dropout=bad)
            raise AssertionError(f"CausalSelfAttention(attn_dropout={bad!r}) 应被拒绝")
        except ValueError as exc:
            assert "attn_dropout" in str(exc)
    for bad in (True, float("nan"), float("inf"), None, "nan"):
        try:
            CausalSelfAttention(16, 2, gate_bias_init=bad)
            raise AssertionError(f"CausalSelfAttention(gate_bias_init={bad!r}) 应被拒绝")
        except ValueError as exc:
            assert "gate_bias_init" in str(exc)

    model = GPT(vocab_size=32, block_size=16, d_model=16, n_layers=1,
                n_heads=2, variant="lega", dropout="0.25",
                attn_dropout="0.5", gate_bias_init="1.5")
    assert model.dropout_p == 0.25
    assert model.blocks[0].attn.attn_dropout == 0.5
    assert model.gate_bias_init == 1.5
    assert torch.equal(model.blocks[0].attn.gate_d, torch.full((2,), 1.5))
    print("  [ok] 模型构造拒绝 bool/NaN/Inf dropout 与 gate 偏置")


def test_kv_cache_matches_full_forward():
    """逐 token 增量解码（带缓存）与一次性全量前向的 logits 一致。"""
    for v in VARIANTS:
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            full = model(idx)["logits"]
            past = None
            inc = []
            for t in range(12):
                out = model(idx[:, t:t + 1], past_kv=past)
                past = out["past_kv"]
                inc.append(out["logits"])
            inc = torch.cat(inc, dim=1)
        assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4), \
            f"{v} KV 缓存不一致, max diff={(full-inc).abs().max().item()}"
        print(f"  [ok] {v}: KV 缓存一致 (max diff {(full-inc).abs().max().item():.2e})")


def test_chunked_prefill_matches_full_forward():
    """分块预填充（past_kv 非空且 T>1，触发掩码缓存的 past_len>0 分支）与全量前向一致。"""
    for v in VARIANTS:
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (2, 15))
        with torch.no_grad():
            full = model(idx)["logits"]
            past = None
            inc = []
            for lo in (0, 5, 10):
                out = model(idx[:, lo:lo + 5], past_kv=past)
                past = out["past_kv"]
                inc.append(out["logits"])
            inc = torch.cat(inc, dim=1)
        assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4), \
            f"{v} 分块预填充不一致, max diff={(full - inc).abs().max().item()}"
        print(f"  [ok] {v}: 分块预填充一致 (max diff {(full - inc).abs().max().item():.2e})")


def test_placebo_evidence_uses_absolute_positions():
    """lega_placebo 的伪证据必须按绝对位置生成，分块预填/逐 token 解码与全量一致。

    回归：旧实现用当前前向内的 0..T-1，T=1 的解码永远落在位置 0，分块预填
    每块也从 0 重新开始；默认 gate_a=gate_b=0 时该错误被完全掩盖。
    """
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2,
                n_heads=2, ffn_mult=4, variant="lega_placebo")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "gate_a" in name or "gate_b" in name:
                p.normal_(0, 0.5)
            elif "gate_w" in name:
                p.normal_(0, 0.1)
    model.eval()
    idx = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        full = model(idx)["logits"]
        # 分块预填 + KV 缓存
        past, chunks = None, []
        for lo in (0, 5):
            out = model(idx[:, lo:lo + 5], past_kv=past)
            past = out["past_kv"]
            chunks.append(out["logits"])
        chunked = torch.cat(chunks, dim=1)
        assert torch.allclose(full, chunked, atol=1e-4, rtol=1e-4), (
            "lega_placebo 分块预填与全量不一致, max diff="
            f"{(full - chunked).abs().max().item()}")
        # 静态 KV 缓存逐 token 解码
        kv = [(torch.empty(1, 2, 32, 16), torch.empty(1, 2, 32, 16))
              for _ in range(2)]
        outs = [model(idx[:, :5], kv_cache=kv, cache_pos=0)["logits"]]
        for pos in range(5, 10):
            outs.append(model(idx[:, pos:pos + 1], kv_cache=kv,
                              cache_pos=pos)["logits"])
        decoded = torch.cat(outs, dim=1)
        assert torch.allclose(full, decoded, atol=1e-4, rtol=1e-4), (
            "lega_placebo 逐 token 解码与全量不一致, max diff="
            f"{(full - decoded).abs().max().item()}")
    print("  [ok] lega_placebo 伪证据按绝对位置生成（分块/解码与全量一致）")


def test_variants_share_base_weights_same_seed():
    """公平对比前提：同 seed 下所有变体的共享基础参数必须逐元素一致。"""
    def build(v):
        torch.manual_seed(123)
        return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2,
                   n_heads=2, ffn_mult=4, variant=v)

    ref = build("standard").state_dict()
    for v in VARIANTS:
        sd = build(v).state_dict()
        for k, p in ref.items():
            assert torch.equal(p, sd[k]), f"{v}:{k} 基础权重与 standard 不一致"


def test_greedy_generation_deterministic():
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (1, 5))
    a = model.generate(idx, max_new_tokens=8)
    b = model.generate(idx, max_new_tokens=8)
    assert a.shape == (1, 13)
    assert torch.equal(a, b)


def test_generate_rejects_invalid_length_and_top_k():
    model = _tiny("standard")
    model.eval()
    idx = torch.randint(0, 64, (1, 5))
    for bad in (-1, 1.5, True):
        try:
            model.generate(idx, max_new_tokens=bad)
            raise AssertionError(f"max_new_tokens={bad!r} 应被拒绝")
        except ValueError as exc:
            assert "max_new_tokens" in str(exc)
    for bad in (0, -1, 1.5, True):
        try:
            model.generate(idx, max_new_tokens=2, greedy=False, top_k=bad)
            raise AssertionError(f"top_k={bad!r} 应被拒绝")
        except ValueError as exc:
            assert "top_k" in str(exc)
    assert torch.equal(model.generate(idx, max_new_tokens=0), idx)
    out = model.generate(idx, max_new_tokens=np.int64(2), greedy=False,
                         top_k=np.int64(4))
    assert out.shape == (1, 7), out.shape
    print("  [ok] generate：拒绝非法 max_new_tokens/top_k")


def test_generate_eos_stops_per_row_without_batch_coupling():
    """提供 eos_token_id 时各行的 EOS 判定必须独立：批内某行的命中/未命中
    不能改变其它行的生成内容，也不得把「各行不同步命中」误判成不停。"""
    torch.manual_seed(0)
    m = GPT(vocab_size=12, block_size=32, d_model=16, n_layers=1, n_heads=2,
            variant="standard")
    m.eval()
    T0, max_new, eos = 6, 10, 5
    # 以 prompt 首 token 标识行；值 = 产生 EOS 的生成步下标。行 1 第 1 步、
    # 行 2 第 3 步命中，行 3 从不命中（不同步，无法靠“全体同时命中”停止）。
    hit_at = {1: 1, 2: 3}
    state = {"rows": None}

    def fake_forward(cur, kv_cache=None, cache_pos=0, last_logits_only=True):
        if cur.shape[1] > 1:
            state["rows"] = cur[:, 0].tolist()
        step = cache_pos + cur.shape[1] - T0     # 本次将产生的 token 的步下标
        logits = torch.full((cur.shape[0], 1, m.vocab_size), -5.0)
        for r, key in enumerate(state["rows"]):
            t = eos if hit_at.get(key, -1) == step else (key + 6) % m.vocab_size
            logits[r, 0, t] = 5.0
        return {"logits": logits, "loss": None, "past_kv": None, "gate_stats": None}

    m.forward = fake_forward
    idx = torch.tensor([[k, 7, 7, 7, 7, 7] for k in (1, 2, 3)])
    with torch.no_grad():
        ref = m.generate(idx, max_new_tokens=max_new)
        ab = m.generate(idx[:2], max_new_tokens=max_new, eos_token_id=eos)
        ac = m.generate(idx[::2], max_new_tokens=max_new, eos_token_id=eos)
        single = m.generate(idx[:1], max_new_tokens=max_new, eos_token_id=eos)
    assert ref.shape == (3, T0 + max_new)
    # 所有行都命中后，截到「最后一行首次命中」的步（行 2 第 3 步 → 长度 10）
    assert ab.shape[1] == T0 + 3 + 1
    assert torch.equal(ab, ref[:2, :T0 + 3 + 1]), "停止只能裁剪，不能改变生成内容"
    # 有行从不命中时不得提前停止；行内容不随批内其它行改变
    assert ac.shape == (2, T0 + max_new)
    assert torch.equal(ac, ref[::2]), "行内容与批内其它行耦合"
    # 单行行为保持：首次命中即停（第 1 步 → 长度 8）
    assert torch.equal(single, ref[:1, :T0 + 1 + 1])
    print("  [ok] EOS 逐行独立停止：批量截断到最晚命中步，单行不受批内影响")


def test_last_logits_only_matches_full_logits():
    """快速路径必须与全量 logits 的最后位置完全一致。"""
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        full = model(idx)["logits"]
        last = model(idx, last_logits_only=True)["logits"]
    assert last.shape == (2, 1, model.vocab_size)
    assert torch.allclose(last[:, -1], full[:, -1], atol=1e-6)


def test_generate_restores_training_mode_on_error():
    """生成中途失败（如超过 block_size）不能把模型永久留在 eval 模式。"""
    model = GPT(vocab_size=64, block_size=8, d_model=32, n_layers=1, n_heads=2,
                ffn_mult=4, variant="lega", dropout=0.2)
    model.train()
    idx = torch.randint(0, 64, (1, 5))
    try:
        model.generate(idx, max_new_tokens=10)
        raise AssertionError("预期超过 block_size 时 forward 抛出异常")
    except AssertionError as exc:
        if str(exc) == "预期超过 block_size 时 forward 抛出异常":
            raise
    assert model.training


def test_checkpoint_roundtrip():
    model = _tiny("lega")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, model, extra={"variant": "lega", "step": 3, "val_ppl": 1.23})
        m2, ck = load_model(path)
        assert ck["step"] == 3 and ck["variant"] == "lega"
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            o1 = model(idx)["logits"]
            o2 = m2(idx)["logits"]
        assert torch.allclose(o1, o2, atol=1e-6)


def test_checkpoint_save_is_atomic_on_serialization_failure(monkeypatch):
    """torch.save 中途失败（磁盘满/进程被杀）不能破坏上一份完整 checkpoint。

    torch.save 默认先截断目标文件；若直接写目标路径，一次失败就会让
    model_best.pt 变成半截文件，而下游 resolve_ckpt 只检查文件是否存在，
    会把损坏权重当成有效产物。原子保存必须保证旧文件逐字节不变、临时文件
    被清理，并且新路径在失败时完全不存在。
    """
    model = _tiny("lega")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_best.pt")
        save_checkpoint(path, model, extra={"variant": "lega", "step": 1})
        before = open(path, "rb").read()

        def broken_save(payload, target, *args, **kwargs):
            # 模拟写到一半后失败：目标位置已经出现半截内容
            with open(target, "wb") as f:
                f.write(b"partial checkpoint bytes")
            raise OSError("No space left on device")

        monkeypatch.setattr(torch, "save", broken_save)
        for target, should_exist in ((path, True),
                                     (os.path.join(d, "model.pt"), False)):
            try:
                save_checkpoint(target, model, extra={"variant": "lega"})
            except OSError:
                pass
            else:
                raise AssertionError("torch.save 失败必须向上抛出")
            if should_exist:
                assert open(path, "rb").read() == before
            else:
                assert not os.path.exists(target)
        # 临时子目录/半截文件必须被清理，目录里只能有原来的 checkpoint
        assert os.listdir(d) == ["model_best.pt"]
        m2, _ = load_model(path)
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            assert torch.allclose(model(idx)["logits"], m2(idx)["logits"],
                                  atol=1e-6)


def test_checkpoint_roundtrip_preserves_attention_path():
    """评测恢复必须沿用训练时的注意力实现，不能被默认值静默改写。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1, n_heads=2,
                ffn_mult=4, variant="standard", use_sdpa_standard=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, model)
        m2, _ = load_model(path)
        assert m2.blocks[0].attn.use_sdpa is False
        # 显式覆盖仍应生效（latency 需要 standard/standard_sdpa 两条路径）
        m3, _ = load_model(path, use_sdpa_standard=True)
        assert m3.blocks[0].attn.use_sdpa is True


def test_load_model_parses_attention_switch_strictly():
    """checkpoint/显式覆盖里的 "false" 不能因 bool("false") 静默变成 True。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1, n_heads=2,
                ffn_mult=4, variant="standard", use_sdpa_standard=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, model)
        m2, _ = load_model(path, use_sdpa_standard="false")
        assert m2.blocks[0].attn.use_sdpa is False

        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["model_cfg"]["use_sdpa_standard"] = "false"
        torch.save(payload, path)
        m3, _ = load_model(path)
        assert m3.blocks[0].attn.use_sdpa is False

        payload["model_cfg"]["use_sdpa_standard"] = "not-a-bool"
        torch.save(payload, path)
        try:
            load_model(path)
            raise AssertionError("非法布尔配置应被拒绝")
        except ValueError as exc:
            assert "use_sdpa_standard" in str(exc)
    print("  [ok] load_model 严格解析 use_sdpa_standard 开关")


def test_gate_stats_shapes():
    """need_stats=True 时诊断统计形状正确（standard 也应返回集中度统计供分析用）。"""
    for v in ("lega", "qgate", "null", "standard"):
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            out = model(idx, need_stats=True)
        assert out["gate_stats"] is not None, v
        st = out["gate_stats"][0]
        assert st["max_p"].shape == (1, 2, 10)
        assert st["argmax"].shape == (1, 2, 10)
        assert (st["max_p"] <= 1.0 + 1e-5).all()
        if v in ("lega", "qgate"):
            assert st["g"] is not None
        else:
            assert st["g"] is None  # standard/null 无门控


def test_compiled_checkpoint_roundtrip():
    """torch.compile 模型的 state_dict 键带 _orig_mod. 前缀，保存时必须剥离，
    否则 load_model 按 GPT 原始键名加载会失败。"""
    model = _tiny("lega")
    try:
        compiled = torch.compile(model)
        compiled_state = compiled.state_dict()
    except Exception as exc:
        print(f"  [skip] torch.compile 不可用: {exc}")
        return
    assert any(k.startswith("_orig_mod.") for k in compiled_state), \
        "环境未产生 _orig_mod 前缀，测试无意义"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, compiled, extra={"variant": "lega"})
        m2, _ = load_model(path)
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            assert torch.allclose(model(idx)["logits"], m2(idx)["logits"], atol=1e-6)


def test_val_eval_indices_cover_dataset_without_repeats():
    """验证集评估窗口应覆盖首尾且不重复，避免循环复用造成 PPL 偏置。"""
    idx, n_windows = _val_eval_indices(1001, 10, 3)
    assert n_windows == 100
    assert idx.tolist() == [0, 49, 99]

    idx, n_windows = _val_eval_indices(1001, 10, 500)
    assert n_windows == 100
    assert len(idx) == 100
    assert len(set(idx.tolist())) == 100   # 请求窗口数超过可用数时不循环复用
    assert idx[0] == 0 and idx[-1] == 99

    idx, _ = _val_eval_indices(31, 10, 10)
    assert idx.tolist() == [0, 1, 2]       # 不足 iters 时评估全部可用窗口

    idx, _ = _val_eval_indices(1001, 10, 1)
    assert idx.tolist() == [49]            # 单窗估计取中间窗口，避免偏前


def test_evaluate_val_returns_coverage_and_gate_metrics():
    """val_loss 与 gate_mean 应来自同一组窗口，并记录实际覆盖范围。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=50, amp=False)
    assert metrics["eval_iters_requested"] == 50
    assert metrics["eval_iters"] == 9      # (100-1)//10 个完整窗口，不循环复用
    assert metrics["val_windows"] == 9
    assert metrics["val_tokens_evaluated"] == 90
    assert metrics["val_loss"] == metrics["val_loss"]  # not NaN
    assert metrics["gate_mean"] is not None
    assert 0.0 < metrics["gate_mean"] < 1.0
    # 门控健康不能只看均值：方差/低高饱和比例/逐层均值必须来自同一批窗口。
    assert metrics["gate_std"] is not None and metrics["gate_std"] >= 0
    assert 0.0 <= metrics["gate_frac_low"] <= 1.0
    assert 0.0 <= metrics["gate_frac_high"] <= 1.0
    assert len(metrics["gate_layer_mean"]) == 2
    assert all(0.0 <= x <= 1.0 for x in metrics["gate_layer_mean"])
    # 无门的 standard 必须显式为空，而不是伪造 0/1 或复用别变体的统计。
    std_model = _tiny("standard")
    std_metrics = evaluate_val(std_model, val, ctx=10, device="cpu",
                               iters=9, amp=False)
    assert std_metrics["gate_mean"] is None
    assert std_metrics["gate_std"] is None
    assert std_metrics["gate_layer_mean"] is None


def test_evaluate_val_reports_window_sampling_uncertainty():
    """有限个验证窗口的点估计必须附带窗口间抽样标准误与 CI。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=9, amp=False)
    assert metrics["val_loss_se"] is not None and metrics["val_loss_se"] > 0
    lo, hi = metrics["val_loss_ci95"]
    assert lo < metrics["val_loss"] < hi
    ppl_lo, ppl_hi = metrics["val_ppl_ci95"]
    assert 0 < ppl_lo <= ppl_hi
    assert metrics["val_loss_min"] <= metrics["val_loss"] <= metrics["val_loss_max"]
    # 单窗口时无法估计窗口间方差：SE/CI 必须为 None，而不是给出虚假精度。
    one = evaluate_val(model, val, ctx=10, device="cpu", iters=1, amp=False)
    assert one["eval_iters"] == 1
    assert one["val_loss_se"] is None and one["val_loss_ci95"] is None
    assert one["val_loss_min"] == one["val_loss"] == one["val_loss_max"]


def test_val_eval_indices_rejects_fractional_bool_specs():
    """eval_iters/ctx/val_len 是评测规格，bool/小数不得被 int() 静默截断。"""
    for bad in (True, 1.5, 0, "1_0"):
        try:
            _val_eval_indices(100, 10, bad)
            raise AssertionError(f"非法 eval_iters 应报错: {bad!r}")
        except ValueError:
            pass
    for val_len, ctx in ((True, 10), (100.5, 10), (101, 10.5),
                         (101, True), (101, 0)):
        try:
            _val_eval_indices(val_len, ctx, 2)
            raise AssertionError(f"非法窗口规格应报错: {val_len!r}, {ctx!r}")
        except ValueError:
            pass
    assert _val_eval_indices(1001, 10, 2.0)[0].tolist() == [0, 99]
    print("  [ok] _val_eval_indices：拒绝 bool/小数/非法 ctx")


def test_evaluate_val_restores_mode_on_error():
    """评测抛错（验证集过短/窗口读取失败）也必须恢复调用前的 train 模式。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    model.train()
    try:
        evaluate_val(model, np.arange(5, dtype=np.int64).astype(np.uint16),
                     ctx=10, device="cpu", iters=2, amp=False)
        raise AssertionError("过短验证集应报错")
    except ValueError:
        pass
    assert model.training, "评测异常后模型被留在 eval 模式"

    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=2, amp=False)
    assert metrics["val_loss"] == metrics["val_loss"]
    assert model.training, "成功评测后也应恢复 train 模式"

    model.eval()
    evaluate_val(model, val, ctx=10, device="cpu", iters=2, amp=False)
    assert not model.training, "原本处于 eval 的模型不得被切回 train"
    print("  [ok] evaluate_val 异常/成功均恢复调用前模式")


def test_per_window_losses_match_individual_forwards():
    """loss_per_window 每行必须等于单独评测该窗口的损失，且均值等于总损失。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (3, 10))
    y = torch.randint(0, 64, (3, 10))
    with torch.inference_mode():
        out = model(idx, targets=y, return_window_losses=True)
    assert out["loss_per_window"].shape == (3,)
    assert abs(float(out["loss"]) -
               float(out["loss_per_window"].mean())) < 1e-6
    for r in range(3):
        with torch.inference_mode():
            one = model(idx[r:r + 1], targets=y[r:r + 1])
        assert abs(float(one["loss"]) -
                   float(out["loss_per_window"][r])) < 1e-5
    # 训练/需要梯度的路径不得支持该选项（避免悄悄改变训练图）
    model.train()
    try:
        model(idx, targets=y, return_window_losses=True)
    except RuntimeError as e:
        assert "return_window_losses" in str(e)
    else:
        raise AssertionError("梯度路径必须拒绝 return_window_losses")


def test_inference_mode_then_training_forward_backward():
    """inference_mode 前向产生的缓存常量不能污染随后同形状的训练图。"""
    model = _tiny("lega")
    idx = torch.randint(0, 64, (2, 12))
    with torch.inference_mode():
        model(idx)
    model.train()
    loss = model(idx, targets=idx)["loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_targets_ignore_index_forward_backward():
    """targets=-100 应遵守 PyTorch CE 约定：忽略损失且梯度为 0。"""
    torch.manual_seed(0)
    model = _tiny("standard")
    idx = torch.randint(0, 64, (2, 9))
    targets = idx.clone()
    targets[0, :4] = -100
    targets[1, 6:] = -100

    chunked = model(idx, targets=targets)["loss"]
    chunked.backward()
    grads_chunked = {n: p.grad.detach().clone()
                     for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    logits = model(idx)["logits"]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    ref.backward()
    assert torch.allclose(chunked, ref, atol=1e-6, rtol=1e-6)
    for n, p in model.named_parameters():
        assert torch.allclose(grads_chunked[n], p.grad, atol=1e-5, rtol=1e-5), n

    model.zero_grad(set_to_none=True)
    ignored = model(idx, targets=torch.full_like(idx, -100))["loss"]
    assert ignored.item() == 0.0
    ignored.backward()
    assert all(torch.all(p.grad == 0) for p in model.parameters() if p.grad is not None)


def test_chunked_ce_preserves_float64_dtype_and_value():
    """fp64 调试路径不能在 loss 处静默降为 fp32。"""
    model = _tiny("standard").double()
    idx = torch.randint(0, 64, (1, 7))
    loss = model(idx, targets=idx)["loss"]
    logits = model(idx)["logits"]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    assert loss.dtype == torch.float64
    assert torch.allclose(loss, ref, atol=1e-12, rtol=1e-12)


def test_chunked_ce_repeated_backward_does_not_decay_grads():
    """retain_graph 下重复 backward：前向物化的 dx/gw 不得被原地重复缩放，
    否则第二次梯度会被再乘一次 1/n_valid（并不是可累加的等值梯度）。"""
    model = _tiny("standard")
    idx = torch.randint(0, 64, (2, 9))
    loss = model(idx, targets=idx)["loss"]
    loss.backward(retain_graph=True)
    first = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    loss.backward()
    for n, p in model.named_parameters():
        if n in first:
            assert torch.allclose(p.grad, 2 * first[n], atol=1e-6, rtol=1e-5), \
                f"{n}: 重复 backward 梯度不可累加"
    print("  [ok] ChunkedLinearCE 重复 backward 梯度可累加（无 1/n 衰减）")


def test_frozen_embedding_training_backward_under_grad():
    """冻结嵌入/detach 上游时梯度模式下不得走残差原地加法路径：
    LayerNorm 反向会保存 x（用于 weight/bias 梯度），原地改写 x 会让
    backward 报“变量被 inplace 修改”而直接失败。"""
    model = _tiny("lega")
    model.tok_emb.weight.requires_grad_(False)   # lm_head 权重与其共享
    model.train()
    idx = torch.randint(0, 64, (2, 10))
    loss = model(idx, targets=idx)["loss"]
    loss.backward()
    assert model.tok_emb.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters() if p.requires_grad)
    print("  [ok] 冻结嵌入时训练 backward 在梯度模式下正常完成")


def test_evaluate_val_optional_window_loss_detail_for_pairing():
    """独立评测可落盘逐窗口损失/序号，供跨变体窗口级配对；训练默认不落盘。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=9,
                           amp=False, collect_window_losses=True)
    assert len(metrics["window_losses"]) == metrics["eval_iters"] == 9
    assert len(metrics["window_indices"]) == 9
    assert len(set(metrics["window_indices"])) == 9
    assert metrics["window_indices"] == sorted(metrics["window_indices"])
    assert abs(sum(metrics["window_losses"]) / 9 - metrics["val_loss"]) < 1e-6
    plain = evaluate_val(model, val, ctx=10, device="cpu", iters=9, amp=False)
    assert "window_losses" not in plain and "window_indices" not in plain
    print("  [ok] evaluate_val 逐窗口明细可选落盘，且均值与 val_loss 一致")


def test_forward_zero_length_targets_returns_finite_zero_loss():
    """空查询/空 batch 的带标签前向不能因空张量除零而崩溃。"""
    m = _tiny("lega")
    for shape in ((2, 0), (0, 4), (0, 0)):
        x = torch.zeros(shape, dtype=torch.long)
        with torch.no_grad():
            out = m(x, targets=x, return_window_losses=True)
        assert out["logits"] is None
        assert out["loss"].dim() == 0 and float(out["loss"]) == 0.0, (shape, out["loss"])
        assert out["loss_per_window"].shape == (shape[0],), (shape, out["loss_per_window"].shape)
        assert torch.isfinite(out["loss_per_window"]).all()
    # 需要梯度的空输入也应能完成 backward（零梯度），而不是抛 TypeError
    x = torch.zeros(2, 0, dtype=torch.long)
    m.zero_grad(set_to_none=True)
    loss = m(x, targets=x)["loss"]
    loss.backward()
    assert float(loss.detach()) == 0.0
    print("  [ok] GPT.forward 空输入/空标签返回可反向的有限零损失")


def test_forward_rejects_misaligned_targets_shape():
    """targets 与 idx 形状不一致时必须报错，不能静默错配标签算损失。"""
    m = _tiny("lega")
    x = torch.randint(0, 64, (2, 8))
    for shape in ((2, 7), (2, 9), (16,), (2, 8, 1)):
        try:
            m(x, targets=torch.randint(0, 64, shape))
        except ValueError as e:
            assert "targets" in str(e), e
        else:
            raise AssertionError(f"targets shape {shape} 未被拒绝")
    # 正确形状仍可训练
    out = m(x, targets=torch.randint(0, 64, (2, 8)))
    assert out["loss"] is not None
