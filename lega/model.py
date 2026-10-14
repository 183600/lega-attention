"""GPT 风格因果 Transformer：RoPE + 预归一化 + 词嵌入/lm_head 权重共享。

注意力层可替换为 lega.attention 中的任意变体（standard/qgate/null/lega 及消融）。
支持增量解码（KV 缓存），供 RULER 生成式评测使用。
"""
import contextlib
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, VARIANTS


class _RoPERotate(torch.autograd.Function):
    """RoPE 旋转：前向把两个半宽结果直接写入预分配输出，反向同理写回梯度缓冲。"""

    @staticmethod
    def forward(ctx, x, cos_h, sin_h):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        o1, o2 = out[..., :half], out[..., half:]
        torch.mul(x1, cos_h, out=o1)            # o1 = x1·cos - x2·sin
        o1.addcmul_(x2, sin_h, value=-1.0)
        torch.mul(x2, cos_h, out=o2)            # o2 = x2·cos + x1·sin
        o2.addcmul_(x1, sin_h)
        ctx.save_for_backward(cos_h, sin_h)
        return out

    @staticmethod
    def backward(ctx, go):
        cos_h, sin_h = ctx.saved_tensors
        half = go.shape[-1] // 2
        go1, go2 = go[..., :half], go[..., half:]
        gx = torch.empty_like(go, memory_format=torch.contiguous_format)
        gx1, gx2 = gx[..., :half], gx[..., half:]
        torch.mul(go1, cos_h, out=gx1)          # gx1 = go1·cos + go2·sin
        gx1.addcmul_(go2, sin_h)
        torch.mul(go2, cos_h, out=gx2)          # gx2 = go2·cos - go1·sin
        gx2.addcmul_(go1, sin_h, value=-1.0)
        return gx, None, None


class RoPE:
    """旋转位置编码（rotate_half 风格），满足平移等变性：score(m, n) 只依赖 m - n。"""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        assert head_dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)              # (max_len, head_dim/2)
        # 旋转只用到前半（两个半宽共享同一份频率），只存半宽表。
        self.cos_half = freqs.cos()              # (max_len, head_dim/2)
        self.sin_half = freqs.sin()
        self.max_len = max_len
        # 按 (device, dtype) 缓存已迁移的表：前向时仅切片，且旋转直接在
        # x 的原生 dtype 上完成，避免 AMP 下每层每次前向的 fp32 升精度分配
        self._device_cache: dict = {}    # (device, dtype) -> (cos, sin)

    def rotate(self, x: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        """对 (B, H, T, D) 张量在绝对位置 [past_len, past_len+T) 上施加旋转。"""
        # 末两维固定为 (T, D)，前导维任意（__call__ 的推理融合路径会多一个
        # (q|k) 堆叠维）：广播视图按 x 的维数展开。
        T = x.shape[-2]
        assert past_len + T <= self.max_len, (
            f"序列长度 {past_len + T} 超过 RoPE 最大位置 {self.max_len}")
        key = (x.device, x.dtype)
        tables = self._device_cache.get(key)
        if tables is None:
            # 迁移后的 RoPE 表会被跨调用缓存；不能在 inference_mode 下创建，
            # 否则推理之后进入训练时，普通 autograd 计算会引用 inference tensor。
            with torch.inference_mode(False):
                tables = (self.cos_half.to(device=x.device, dtype=x.dtype),
                          self.sin_half.to(device=x.device, dtype=x.dtype))
            self._device_cache[key] = tables
        half = x.shape[-1] // 2
        bshape = (1,) * (x.dim() - 2) + (T, half)
        cos_h = tables[0][past_len:past_len + T].view(bshape)
        sin_h = tables[1][past_len:past_len + T].view(bshape)
        if torch.is_grad_enabled() and x.requires_grad:
            return _RoPERotate.apply(x, cos_h, sin_h)
        out = torch.empty_like(x)
        x1, x2 = x[..., :half], x[..., half:]
        o1, o2 = out[..., :half], out[..., half:]
        torch.mul(x1, cos_h, out=o1)
        o1.addcmul_(x2, sin_h, value=-1.0)
        torch.mul(x2, cos_h, out=o2)
        o2.addcmul_(x1, sin_h)
        return out

    def __call__(self, q: torch.Tensor, k: torch.Tensor, past_len: int = 0):
        # 推理态（评测/逐 token 解码）且 q/k 同形状同 dtype 时，把两份旋转融合为
        # 一次 stack 调用。
        # 输出两视图共享同一缓冲：下游对 q 的原地缩放只写 q 自己的那一半，安全。
        # 训练态保持分开走 _RoPERotate（各自独立的 autograd 节点，不引入共享
        # 存储的视图，避免与就地缩放/反向写回产生视图别名问题）。
        if (not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))
                and q.shape == k.shape and q.dtype == k.dtype):
            qk = self.rotate(torch.stack((q, k)), past_len)
            return qk[0], qk[1]
        return self.rotate(q, past_len), self.rotate(k, past_len)


class _ChunkedLinearCE(torch.autograd.Function):
    """分块 lm_head + 交叉熵（支持 PyTorch 默认的 ignore_index=-100）。

    前向分块累加 loss，反向再按块重算 logits 并累积 dL/dx、dL/dW。
    被忽略位置既不计入损失分母，也完全不产生梯度；全部位置被忽略时
    返回 0 损失（以及零梯度），与监督微调流程中的空有效标签约定一致。
    """

    @staticmethod
    def _loss_dtype(logits: torch.Tensor) -> torch.Tensor:
        # 高精度输入保持原精度，低精度/16 位输入升到 fp32，与反向严格同口径。
        return logits if logits.dtype in (torch.float32, torch.float64) else logits.float()

    @staticmethod
    def forward(ctx, x2d, weight, tgt, chunk: int, ignore_index: int = -100):
        ctx.save_for_backward(x2d, weight, tgt)
        ctx.chunk = chunk
        ctx.ignore_index = int(ignore_index)
        # 记录前向的 autocast 状态：反向重算 logits 必须与前向同精度，
        # 否则 AMP 下前向用 bf16 logits 报告 loss、反向却按 fp32 logits
        # 求梯度，二者不再是同一个损失函数。
        if x2d.is_cuda:
            ctx.amp_dtype = (torch.get_autocast_dtype("cuda")
                             if torch.is_autocast_enabled("cuda") else None)
        else:
            ctx.amp_dtype = (torch.get_autocast_dtype("cpu")
                             if torch.is_autocast_enabled("cpu") else None)
        # n_valid 以设备张量保存：前向/反向都不再 .item() 同步，
        # 消除每个训练步的一次 GPU->CPU 阻塞（后续 kernel 可继续流水提交）。
        # clamp_min(1) 同时吸收"全部位置被忽略"的除零情形（此时梯度因
        # keep 掩码全零而天然为 0，无需提前特判）。
        n_valid = (tgt != ctx.ignore_index).sum()
        ctx.inv_n = n_valid.clamp_min(1).to(
            x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
            else torch.float32).reciprocal()
        loss = torch.zeros((), dtype=x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
                           else torch.float32, device=x2d.device)
        with torch.no_grad():
            for s in range(0, tgt.numel(), chunk):
                lc = F.linear(x2d[s:s + chunk], weight)
                loss.add_(F.cross_entropy(_ChunkedLinearCE._loss_dtype(lc),
                                          tgt[s:s + chunk],
                                          ignore_index=ctx.ignore_index,
                                          reduction="sum"))
        return loss * ctx.inv_n

    @staticmethod
    def backward(ctx, gout):
        x2d, weight, tgt = ctx.saved_tensors
        # gx 的每一行都会被下方按块整段覆写，无需先零填充
        gx = torch.empty_like(x2d)
        # gw 直接以计算精度（低精度时为 fp32）预分配并用 addmm_ 原地累加：
        # 避免每个分块都物化一份 (V, D) 的 gw_chunk 临时张量（大词表下每块
        # 上百 MB，且每块各分配/释放一次）；同时也比逐块 cast 到 weight.dtype
        # 再累加更精确（全程高精度累加，只在返回前 cast 一次）。
        gw = None
        gscale = (gout if gout.dtype in (torch.float32, torch.float64)
                  else gout.float()) * ctx.inv_n
        # 闭式求梯度（cross-entropy 对 logits 的梯度 = softmax - onehot），
        # 不搭建逐块 autograd 子图，反向为逐块 3 次 GEMM。
        with torch.no_grad():
            # 与前向一致的 autocast 上下文，保证反向重算的 logits 精度相同
            if ctx.amp_dtype is not None:
                dev_type = "cuda" if x2d.is_cuda else "cpu"
                ac_ctx = torch.autocast(dev_type, dtype=ctx.amp_dtype)
            else:
                ac_ctx = contextlib.nullcontext()
            neg_one = None   # scatter_add_ 的 -1 源张量按块复用，避免每块一次分配
            for s in range(0, tgt.numel(), ctx.chunk):
                xs = x2d[s:s + ctx.chunk]
                ts = tgt[s:s + ctx.chunk]
                with ac_ctx:
                    lc = F.linear(xs, weight)                        # (m, V)
                lcf = _ChunkedLinearCE._loss_dtype(lc)
                d = torch.softmax(lcf, dim=-1)
                keep = ts != ctx.ignore_index
                # one-hot 减 1 用 scatter_add_ 一步完成：先把无效位置的索引
                # 压到 0（其减 1 随后会被 keep 掩码清零），替代原先的
                # keep.any() 同步 + nonzero 索引分配 + 花式索引写入。
                ts_safe = torch.where(keep, ts, torch.zeros_like(ts))
                if neg_one is None or neg_one.size(0) < ts_safe.numel():
                    neg_one = d.new_full((max(ts_safe.numel(), 4096), 1), -1.0)
                d.scatter_add_(1, ts_safe.unsqueeze(1),
                               neg_one[:ts_safe.numel()])
                d.mul_(keep.unsqueeze(1).to(d.dtype))
                d.mul_(gscale)
                gx[s:s + ctx.chunk] = torch.matmul(d, weight).to(gx.dtype)
                if gw is None:
                    gw = torch.zeros(weight.shape, dtype=d.dtype,
                                     device=weight.device)
                # 原地累加该分块的 dL/dW，不产生 (V, D) 中间结果
                gw.addmm_(d.transpose(0, 1), xs.to(d.dtype))
        if gw is not None and gw.dtype != weight.dtype:
            gw = gw.to(weight.dtype)
        return gx, gw, None, None, None


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float,
                 attn_dropout: float, variant: str, gate_bias_init: float,
                 use_sdpa_standard: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, variant=variant, attn_dropout=attn_dropout,
            gate_bias_init=gate_bias_init, use_sdpa_standard=use_sdpa_standard)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        # dropout=0（默认配置）时直接用 Identity 代替 F.dropout
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, rope: RoPE, past_kv=None, need_stats: bool = False,
                stats_argmax: bool = True, kv_cache=None, cache_pos: int = 0):
        a, cache, stats = self.attn(self.ln1(x), rope=rope, past_kv=past_kv,
                                    need_stats=need_stats, stats_argmax=stats_argmax,
                                    kv_cache=kv_cache, cache_pos=cache_pos)
        # 无梯度（推理/评测）时残差连接用原地加法：x 是本层独占的新张量
        # （来自上一层输出或嵌入层输出，不被其它子图引用），每层每次前向
        # 省掉两份 (B, T, C) 输出分配；有梯度时 autograd 需要加法输出做反向，
        # 必须保持 out-of-place。
        if torch.is_grad_enabled() and x.requires_grad:
            x = x + self.drop(a)
            x = x + self.drop(self.w2(self.act(self.w1(self.ln2(x)))))
        else:
            x.add_(self.drop(a))
            x.add_(self.drop(self.w2(self.act(self.w1(self.ln2(x))))))
        return x, cache, stats


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int = 384,
                 n_layers: int = 6, n_heads: int = 6, ffn_mult: int = 4,
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 variant: str = "lega", gate_bias_init: float = 2.0,
                 use_sdpa_standard: bool = True):
        super().__init__()
        assert variant in VARIANTS, f"未知变体: {variant}"
        self.variant = variant
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.dropout_p = dropout
        self.gate_bias_init = gate_bias_init

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope = RoPE(d_model // n_heads, block_size)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, ffn_mult, dropout, attn_dropout, variant,
                  gate_bias_init, use_sdpa_standard)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # 权重共享

        self.apply(self._init_weights)
        # 残差投影缩放初始化（GPT-2 技巧，稳定深层训练）
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        # 门控参数显式初始化：w=a=b=0, d=gate_bias_init → 初始 g≈sigmoid(2)≈0.88
        for m in self.modules():
            if isinstance(m, CausalSelfAttention):
                m.init_gate_params_(gate_bias_init)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                past_kv: Optional[List] = None, need_stats: bool = False,
                last_logits_only: bool = False, stats_argmax: bool = True,
                kv_cache: Optional[List] = None, cache_pos: int = 0,
                ignore_index: int = -100) -> Dict:
        """kv_cache/cache_pos：静态 KV 缓存模式（预分配缓冲 + 写入偏移），
        供 generate 逐 token 解码使用；与 past_kv 的动态拼接缓存互斥。"""
        B, T = idx.shape
        past_len = (0 if past_kv is None else past_kv[0][0].shape[2]) \
            if kv_cache is None else cache_pos
        assert T + past_len <= self.block_size, (
            f"输入长度 {past_len}+{T} 超过 block_size={self.block_size}")

        x = self.drop(self.tok_emb(idx))
        # 逐层缓存容器预分配为定长列表再按序号写入：逐 token 解码每步都调 forward
        new_cache: List = [None] * len(self.blocks)
        stats_all: List = []
        for i, blk in enumerate(self.blocks):
            pk = None if (past_kv is None or kv_cache is not None) else past_kv[i]
            kc = None if kv_cache is None else kv_cache[i]
            x, cache, stats = blk(x, self.rope, past_kv=pk, need_stats=need_stats,
                                  stats_argmax=stats_argmax, kv_cache=kc,
                                  cache_pos=cache_pos)
            new_cache[i] = cache
            if stats is not None:
                stats = dict(stats)
                stats["layer"] = i
                stats_all.append(stats)
        loss = None
        if targets is not None:
            x = self.ln_f(x)
            # 训练/评测只需 loss：分块过 lm_head 并立即做交叉熵；
            # 分块求和再平均与整批 mean 结果一致；此时不再返回 logits。
            w = self.tok_emb.weight                       # 权重共享
            flat = x.reshape(-1, x.size(-1))
            tgt = targets.reshape(-1)
            # 每块 logits 为目标数乘以分块大小，分块大小随词表自适应。
            CHUNK = max(256, min(4096, 33_554_432 // self.vocab_size))
            if torch.is_grad_enabled() and x.requires_grad:
                # 训练路径：分块前向 + 分块重算反向
                loss = _ChunkedLinearCE.apply(flat, w, tgt, CHUNK, int(ignore_index))
            else:
                total = None  # 设备端累加，避免 Python/张量混加与每步同步
                for s in range(0, flat.size(0), CHUNK):
                    lc = F.linear(flat[s:s + CHUNK], w)
                    lcd = _ChunkedLinearCE._loss_dtype(lc)
                    part = F.cross_entropy(lcd, tgt[s:s + CHUNK],
                                           ignore_index=int(ignore_index),
                                           reduction="sum")
                    total = part if total is None else total.add_(part)
                # n_valid 保留为设备张量：评测循环无需每窗口批一次 .item() 同步
                loss = total / (tgt != ignore_index).sum().clamp_min(1)
            logits = None
        else:
            # last_logits_only 时（如 generate）先裁到最后位置再过 ln_f/lm_head：
            # LayerNorm 逐位置独立，结果与先归一化再切片一致
            if last_logits_only:
                x = x[:, -1:, :]
            x = self.ln_f(x)
            logits = F.linear(x, self.tok_emb.weight)  # 权重共享
        return {"logits": logits, "loss": loss, "past_kv": new_cache,
                "gate_stats": stats_all if stats_all else None}

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, greedy: bool = True,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 kv_cache: Optional[List] = None) -> torch.Tensor:
        """贪心/采样生成（带 KV 缓存）。idx: (B, T) 提示词。

        kv_cache：可选的外部静态 KV 缓冲（与内部预分配同布局，
        (B, H, max_len, Dh)，容量只需 >= min(T + max_new_tokens, block_size)），
        供批量评测在样例间复用，避免每次生成都逐层重新分配缓冲。
        """
        was_training = self.training
        self.eval()
        try:
            B, T0 = idx.shape
            # 生成上限受 block_size（RoPE 位置表）约束：在分配缓冲/缓存前
            # 提前报出清晰错误，避免生成到一半才在 forward 中触发位置断言
            assert T0 + max_new_tokens <= self.block_size, (
                f"生成长度超限: 提示 {T0} + 新增 {max_new_tokens} > "
                f"block_size {self.block_size}，请先截断提示或减少 max_new_tokens")
            # 预分配输出缓冲，逐 token 就地写入
            out_ids = torch.empty(B, T0 + max_new_tokens, dtype=idx.dtype,
                                  device=idx.device)
            out_ids[:, :T0] = idx
            cur_len = T0
            # 静态 KV 缓存：每层一次性预分配 (B, H, T0+max_new, Dh) 缓冲，
            # 逐 token 就地写入。
            total = min(T0 + max_new_tokens, self.block_size)
            if kv_cache is not None:
                # 外部缓冲必须在批大小与长度容量上覆盖本次生成，
                # 否则就地写入会静默截断或报错
                assert len(kv_cache) == len(self.blocks)
                assert all(kc[0].shape[0] == B and kc[1].shape[0] == B
                           and kc[0].shape[2] >= total and kc[1].shape[2] >= total
                           for kc in kv_cache), "外部 kv_cache 容量不足"
            else:
                attn0 = self.blocks[0].attn
                kvdtype = attn0.wk.weight.dtype
                kv_cache = [
                    (torch.empty(B, attn0.n_heads, total, attn0.d_head,
                                 dtype=kvdtype, device=idx.device),
                     torch.empty(B, attn0.n_heads, total, attn0.d_head,
                                 dtype=kvdtype, device=idx.device))
                    for _ in self.blocks]
            for _ in range(max_new_tokens):
                cur = out_ids[:, :cur_len] if cur_len == T0 else out_ids[:, cur_len - 1:cur_len]
                out = self.forward(cur, kv_cache=kv_cache,
                                   cache_pos=cur_len - cur.shape[1],
                                   last_logits_only=True)
                logits = out["logits"][:, -1]
                if greedy:
                    nxt = logits.argmax(dim=-1)
                else:
                    logits = logits / max(temperature, 1e-6)
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)[:, 0]
                out_ids[:, cur_len] = nxt
                cur_len += 1
                if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                    break
            return out_ids[:, :cur_len]
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------- 工具函数

def model_cfg_dict(model: GPT) -> Dict:
    """保存/重建模型所需的最小配置。"""
    attn0 = model.blocks[0].attn
    return {
        "vocab_size": model.vocab_size,
        "block_size": model.block_size,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "ffn_mult": model.ffn_mult,
        "dropout": model.dropout_p,
        "variant": model.variant,
        "gate_bias_init": model.gate_bias_init,
        "attn_dropout": attn0.attn_dropout,
        "use_sdpa_standard": attn0.use_sdpa,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gate_extra_params(model: nn.Module) -> int:
    """门控/空注意力引入的额外参数量（相对 standard）。"""
    return sum(p.numel() for n, p in model.named_parameters()
               if ("gate_" in n or "_null" in n))


def expected_gate_extras(variant: str, d_head: int, n_heads: int, n_layers: int) -> int:
    """理论额外参数量，用于单元测试核对。"""
    per = {
        "standard": 0,
        "qgate": d_head + 1,          # w, d
        "null": 2 * d_head,           # k_null, v_null
        "lega": d_head + 3,           # w, a, b, d
        "lega_no_len": d_head + 2,    # w, a, d
        "lega_no_conc": d_head + 2,   # w, b, d
        "lega_no_query": 3,           # a, b, d
    }[variant]
    return per * n_heads * n_layers


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float,
                        betas=(0.9, 0.95), eps: float = 1e-8, fused: bool = False):
    """AdamW 参数分组：≥2 维矩阵（含门控权重 gate_w）权重衰减；偏置/标量门控 a/b/d/LayerNorm/null 键不衰减。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "_null" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)


def save_checkpoint(path: str, model: GPT, extra: Optional[Dict] = None) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state_dict = model.state_dict()
    # torch.compile 的 OptimizedModule 会给 state_dict 键加 "_orig_mod." 前缀，
    # 保存前统一剥掉，保持键名与 load_model 约定一致。
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v
                      for k, v in state_dict.items()}
    payload = {"model_cfg": model_cfg_dict(model), "state_dict": state_dict}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(ckpt_path: str, device: str = "cpu",
               use_sdpa_standard: bool = True):
    """从 checkpoint 恢复模型。返回 (model, ckpt_dict)。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = dict(ck["model_cfg"])
    cfg["use_sdpa_standard"] = use_sdpa_standard
    model = GPT(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model, ck
