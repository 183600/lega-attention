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
        # 按 (device, dtype) 缓存已迁移的表：前向时仅切片，旋转直接在
        # x 的原生 dtype 上完成
        self._device_cache: dict = {}    # (device, dtype) -> (cos, sin)

    def rotate(self, x: torch.Tensor, past_len: int = 0,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对 (B, H, T, D) 张量在绝对位置 [past_len, past_len+T) 上施加旋转。

        out 非空时把结果写入该预分配缓冲（须与 x 同形状同 dtype）。"""
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
        if out is None:
            out = torch.empty_like(x)
        x1, x2 = x[..., :half], x[..., half:]
        o1, o2 = out[..., :half], out[..., half:]
        torch.mul(x1, cos_h, out=o1)
        o1.addcmul_(x2, sin_h, value=-1.0)
        torch.mul(x2, cos_h, out=o2)
        o2.addcmul_(x1, sin_h)
        return out

    def __call__(self, q: torch.Tensor, k: torch.Tensor, past_len: int = 0):
        # 推理态（评测/逐 token 解码）且 q/k 同形状同 dtype 时，把两份旋转写入
        # 同一预分配缓冲的两份输出视图：下游对 q 的原地缩放只写 q 自己的那一半，安全。
        # 训练态保持分开走 _RoPERotate（各自独立的 autograd 节点，不引入共享
        # 存储的视图，避免与就地缩放/反向写回产生视图别名问题）。
        if (not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))
                and q.shape == k.shape and q.dtype == k.dtype):
            qk = torch.empty(2, *q.shape, dtype=q.dtype, device=q.device)
            self.rotate(q, past_len, out=qk[0])
            self.rotate(k, past_len, out=qk[1])
            return qk[0], qk[1]
        return self.rotate(q, past_len), self.rotate(k, past_len)


# 反向复用整段 logits 的体积上限（元素数×计算精度字节），按设备类型取值；
# 超出上限仍走分块重算。
_CE_CACHE_BYTES_CPU = 512 * 1024 * 1024
_CE_CACHE_BYTES_ACCEL = 64 * 1024 * 1024


class _ChunkedLinearCE(torch.autograd.Function):
    """分块 lm_head + 交叉熵（支持 PyTorch 默认的 ignore_index=-100）。

    前向分块累加 loss，反向再按块重算 logits 并累积 dL/dx、dL/dW。
    需要梯度且整段 logits 体积不超过设备上限时，logits 缓存在 ctx 上供
    反向复用；否则反向逐块重算。被忽略位置既不计入损失分母，也完全不
    产生梯度；全部位置被忽略时返回 0 损失（以及零梯度），与监督微调
    流程中的空有效标签约定一致。
    """

    @staticmethod
    def _loss_dtype(logits: torch.Tensor) -> torch.Tensor:
        # 高精度输入保持原精度，低精度/16 位输入升到 fp32，与反向严格同口径。
        return logits if logits.dtype in (torch.float32, torch.float64) else logits.float()

    @staticmethod
    def _autocast_dtype(t: torch.Tensor):
        """当前 autocast 的目标 dtype（未启用时为 None）。

        torch.is_autocast_enabled/get_autocast_dtype 的 device_type 参数
        自 torch>=2.4 起提供；旧版回退到无参形式。
        """
        if t.is_cuda:
            try:
                enabled = torch.is_autocast_enabled("cuda")
            except TypeError:
                enabled = torch.is_autocast_enabled()
        else:
            try:
                enabled = torch.is_autocast_enabled("cpu")
            except TypeError:
                enabled = torch.is_autocast_cpu_enabled()
        if not enabled:
            return None
        try:
            return torch.get_autocast_dtype("cuda" if t.is_cuda else "cpu")
        except TypeError:
            return (torch.get_autocast_gpu_dtype() if t.is_cuda
                    else torch.get_autocast_cpu_dtype())

    @staticmethod
    def _ce_sum(lcf: torch.Tensor, tgt: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """sum-CE 的闭式计算：log_softmax - 目标 logit，逐元素掩掉 ignore_index。

        与 F.cross_entropy(reduction="sum") 等价（log_softmax 内部 max 平移，
        数值稳定）。
        """
        ignore = tgt == ignore_index
        ts_safe = tgt.masked_fill(ignore, 0)
        logp = torch.log_softmax(lcf, dim=-1, out=lcf)
        nll = logp.gather(1, ts_safe.unsqueeze(1)).squeeze(1).neg_()
        nll.masked_fill_(ignore, 0)
        return nll.sum()

    @staticmethod
    def forward(ctx, x2d, weight, tgt, chunk: int, ignore_index: int = -100):
        ctx.save_for_backward(x2d, weight, tgt)
        ctx.chunk = chunk
        ctx.ignore_index = int(ignore_index)
        # 记录前向的 autocast 状态：反向重算 logits 必须与前向同精度，
        # 否则 AMP 下前向用 bf16 logits 报告 loss、反向却按 fp32 logits
        # 求梯度，二者不再是同一个损失函数。
        ctx.amp_dtype = _ChunkedLinearCE._autocast_dtype(x2d)
        # n_valid 以设备张量保存；clamp_min(1) 同时吸收"全部位置被忽略"的
        # 除零情形（此时梯度因 keep 掩码全零而天然为 0，无需提前特判）。
        n_valid = (tgt != ctx.ignore_index).sum()
        ctx.inv_n = n_valid.clamp_min(1).to(
            x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
            else torch.float32).reciprocal()
        loss = torch.zeros((), dtype=x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
                           else torch.float32, device=x2d.device)
        # 仅在需要 x 的梯度时尝试缓存；上限按首块实际产出的 dtype 计算，
        # 分配失败则退回重算。
        cache_cap = (_CE_CACHE_BYTES_CPU if x2d.device.type == "cpu"
                     else _CE_CACHE_BYTES_ACCEL)
        cache_try = bool(ctx.needs_input_grad[0])
        logits_cache = None
        ctx.logits_cache = None
        with torch.no_grad():
            # 分块循环共用一份目标精度权重：autocast 会在每次 F.linear 调用时
            # 重新转换整份权重，而各分块使用的转换结果完全相同，转换一次即可。
            # 前向与反向重算共用该副本，GEMM 的输入逐位不变。
            ctx.w_fwd = weight if (ctx.amp_dtype is None
                                   or weight.dtype == ctx.amp_dtype) \
                else weight.to(ctx.amp_dtype)
            for s in range(0, tgt.numel(), chunk):
                lc = F.linear(x2d[s:s + chunk], ctx.w_fwd)
                if cache_try:
                    cache_try = False
                    if tgt.numel() * weight.size(0) * lc.element_size() <= cache_cap:
                        try:
                            logits_cache = torch.empty(
                                (tgt.numel(), weight.size(0)),
                                dtype=lc.dtype, device=lc.device)
                        except RuntimeError:
                            logits_cache = None
                        else:
                            ctx.logits_cache = logits_cache
                if logits_cache is not None:
                    logits_cache[s:s + chunk].copy_(lc)
                lcf = _ChunkedLinearCE._loss_dtype(lc)
                # 升精度副本已生成（不升精度时 lcf 与 lc 为同一张量）。
                del lc
                loss.add_(_ChunkedLinearCE._ce_sum(
                    lcf, tgt[s:s + chunk], ctx.ignore_index))
        return loss * ctx.inv_n

    @staticmethod
    def backward(ctx, gout):
        x2d, weight, tgt = ctx.saved_tensors
        # gx 的每一行都会被下方按块整段覆写，无需先零填充
        gx = torch.empty_like(x2d)
        # gw 以计算精度（低精度时为 fp32）累加：
        # 全程高精度累加，只在返回前 cast 一次。
        gw = None
        gscale = (gout if gout.dtype in (torch.float32, torch.float64)
                  else gout.float()) * ctx.inv_n
        # 前向若缓存了整段 logits 则反向直接复用，无需重算。
        cached = ctx.logits_cache
        # 闭式求梯度（cross-entropy 对 logits 的梯度 = softmax - onehot）。
        with torch.no_grad():
            # 与前向一致的 autocast 上下文，保证反向重算的 logits 精度相同
            if ctx.amp_dtype is not None and cached is None:
                dev_type = "cuda" if x2d.is_cuda else "cpu"
                ac_ctx = torch.autocast(dev_type, dtype=ctx.amp_dtype)
            else:
                ac_ctx = contextlib.nullcontext()
            neg_one = None   # scatter_add_ 的 -1 源张量
            # d 的计算精度（低精度时为 fp32）在整个反向固定：权重到该精度的
            # 转换在每个分块中完全相同。
            w_f = None
            # 是否有被忽略的位置（全局判定一次）：没有时无需逐块把 d 的
            # 无效行清零。gscale 是标量，延迟到两个输出端缩放
            # （gx 逐块、gw 收尾一次）。
            has_ignore = not bool((tgt != ctx.ignore_index).all())
            for s in range(0, tgt.numel(), ctx.chunk):
                xs = x2d[s:s + ctx.chunk]
                ts = tgt[s:s + ctx.chunk]
                if cached is not None:
                    lcf = _ChunkedLinearCE._loss_dtype(cached[s:s + ctx.chunk])
                else:
                    with ac_ctx:
                        lc = F.linear(xs, ctx.w_fwd)                 # (m, V)
                    lcf = _ChunkedLinearCE._loss_dtype(lc)
                    del lc
                d = torch.softmax(lcf, dim=-1, out=lcf)
                keep = ts != ctx.ignore_index
                # one-hot 减 1 用 scatter_add_ 一步完成：先把无效位置的索引
                # 压到 0（其减 1 随后被 keep 掩码清零）。
                ts_safe = ts.masked_fill(~keep, 0)
                if neg_one is None or neg_one.size(0) < ts_safe.numel():
                    neg_one = d.new_full((max(ts_safe.numel(), 4096), 1), -1.0)
                d.scatter_add_(1, ts_safe.unsqueeze(1),
                               neg_one[:ts_safe.numel()])
                # keep∈{0,1}，无效行清零即可；gscale 合并进输出端缩放。
                if has_ignore:
                    d.mul_(keep.unsqueeze(1).to(d.dtype))
                # d 为计算精度（低精度时为 fp32），weight 可能是 bf16/fp16：
                # 必须先在 d 的精度上做 GEMM（dtype 需一致）。
                if w_f is None:
                    w_f = weight.to(d.dtype)
                gxs = gx[s:s + ctx.chunk]
                if gxs.dtype == d.dtype:
                    # 同精度（常见 fp32 训练）：GEMM 直接写入 gx 的行切片
                    torch.matmul(d, w_f, out=gxs)
                else:
                    gxs.copy_(torch.matmul(d, w_f))
                # dL/dx 逐块按 gscale 缩放（标量，行切片原地乘）
                gxs.mul_(gscale)
                # 首块用 mm 直接产出 gw，后续分块就地累加
                # （gscale 在循环外统一缩放）
                xs_f = xs.to(d.dtype)
                if gw is None:
                    gw = torch.mm(d.transpose(0, 1), xs_f)
                else:
                    gw.addmm_(d.transpose(0, 1), xs_f)
                del d, lcf
        if gw is not None:
            gw.mul_(gscale)
            if gw.dtype != weight.dtype:
                gw = gw.to(weight.dtype)
        return gx, gw, None, None, None


def _cast_weight_cached(owner, slot: str, weight: torch.Tensor) -> torch.Tensor:
    """线性层前向权重：无梯度时按 autocast 目标精度缓存。

    有梯度时直接返回参数本身，保持反向连通到参数；无梯度路径返回目标精度的
    缓存副本——autocast 会在每次 F.linear 调用时重新转换权重，而同一权重在
    多次调用（逐 token 解码、逐分块评测）中的转换结果相同。副本在
    inference_mode(False)+no_grad 下创建，不会留下会被 autograd 拒绝的
    inference tensor；权重被替换或原地更新（_version 变化）会使缓存失效。
    """
    if torch.is_grad_enabled():
        return weight
    amp = _ChunkedLinearCE._autocast_dtype(weight)
    if amp is None or amp == weight.dtype:
        return weight
    key = (weight, weight._version, weight.data_ptr(), weight.device, amp)
    ent = getattr(owner, slot)
    if ent is not None:
        o_key = ent[0]
        # 逐项（含对象身份）比较，避免元组 != 在 Tensor 上触发
        # "Boolean value of Tensor is ambiguous"
        if (len(o_key) == len(key) and all(
                (o is n) if isinstance(n, torch.Tensor) else o == n
                for o, n in zip(o_key, key))):
            return ent[1]
    with torch.inference_mode(False), torch.no_grad():
        casted = weight.detach().to(amp)
    setattr(owner, slot, (key, casted))
    return casted


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
        # 推理态（无梯度）MLP 权重的 autocast 精度缓存：(key, tensor)
        self._w1_cast = None
        self._w2_cast = None

    def forward(self, x, rope: RoPE, past_kv=None, need_stats: bool = False,
                stats_argmax: bool = True, kv_cache=None, cache_pos: int = 0):
        a, cache, stats = self.attn(self.ln1(x), rope=rope, past_kv=past_kv,
                                    need_stats=need_stats, stats_argmax=stats_argmax,
                                    kv_cache=kv_cache, cache_pos=cache_pos)
        # 无梯度（推理/评测）时残差连接用原地加法：x 是本层独占的新张量
        # （来自上一层输出或嵌入层输出，不被其它子图引用）；有梯度时 autograd
        # 需要加法输出做反向，必须保持 out-of-place。
        if torch.is_grad_enabled() and x.requires_grad:
            x = x + self.drop(a)
            x = x + self.drop(self.w2(self.act(self.w1(self.ln2(x)))))
        else:
            x.add_(self.drop(a))
            h = F.linear(self.ln2(x),
                         _cast_weight_cached(self, "_w1_cast", self.w1.weight))
            x.add_(self.drop(F.linear(
                self.act(h), _cast_weight_cached(self, "_w2_cast", self.w2.weight))))
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
        # 推理/评测态 lm_head 权重的 autocast 精度缓存：(key, tensor)；
        # 键含参数身份/_version/data_ptr/device/dtype，权重被替换或原地修改
        # 时缓存自动失效（与注意力推理态融合权重缓存同一口径）。
        self._lmhead_cast = None

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

    def _lmhead_weight(self) -> torch.Tensor:
        """lm_head 前向权重（与 tok_emb 共享参数）。

        有梯度时返回参数本身，保证反向经共享参数连回嵌入权重；无梯度
        （评测/生成，含 inference_mode）时返回目标精度的缓存副本，避免每次
        F.linear（逐 token 解码、逐分块评测）重复转换整份 (vocab, d_model)
        权重，见 _cast_weight_cached。
        """
        return _cast_weight_cached(self, "_lmhead_cast", self.tok_emb.weight)

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
            w = self._lmhead_weight()                     # 共享权重
            flat = x.reshape(-1, x.size(-1))
            tgt = targets.reshape(-1)
            # 每块 logits 为目标数乘以分块大小，分块大小随词表自适应。
            CHUNK = max(256, min(4096, 33_554_432 // self.vocab_size))
            if torch.is_grad_enabled() and x.requires_grad:
                # 训练路径：分块前向 + 分块重算反向
                loss = _ChunkedLinearCE.apply(flat, w, tgt, CHUNK, int(ignore_index))
            else:
                total = None  # 设备端累加
                for s in range(0, flat.size(0), CHUNK):
                    lc = F.linear(flat[s:s + CHUNK], w)
                    lcd = _ChunkedLinearCE._loss_dtype(lc)
                    del lc
                    part = _ChunkedLinearCE._ce_sum(lcd, tgt[s:s + CHUNK],
                                                    int(ignore_index))
                    total = part if total is None else total.add_(part)
                # n_valid 保留为设备张量
                loss = total / (tgt != ignore_index).sum().clamp_min(1)
            logits = None
        else:
            # last_logits_only 时（如 generate）先裁到最后位置再过 ln_f/lm_head：
            # LayerNorm 逐位置独立，结果与先归一化再切片一致
            if last_logits_only:
                x = x[:, -1:, :]
            x = self.ln_f(x)
            logits = F.linear(x, self._lmhead_weight())  # 权重共享
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
        供批量评测在样例间复用。

        eos_token_id：各行独立判定 EOS；当批内所有行都已各自命中过 EOS 时
        停止，返回长度截到「最后一行首次命中 EOS 的步」。提前命中的行在返回
        张量里仍可能带上其 EOS 之后的采样 token，消费方应按各行首个 EOS 截断。
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
            # EOS 判定标志留在设备端逐 token 记录，每 _EOS_EVERY 步检查一次；
            # 检查时用累计命中记录回推首达步，结果与逐步检查完全一致。
            _EOS_EVERY = 8
            # (max_new, B) 的逐行 EOS 命中记录：各行独立判定，互不影响，
            # 批内某行命中 EOS 不改变其它行的生成内容与停止判断。
            eos_flags = (torch.zeros((max_new_tokens, B), dtype=torch.bool,
                                     device=idx.device)
                         if eos_token_id is not None else None)
            for i in range(max_new_tokens):
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
                if eos_flags is not None:
                    eos_flags[i] = nxt == eos_token_id
                    if (i + 1) % _EOS_EVERY == 0 or i + 1 == max_new_tokens:
                        window = eos_flags[:i + 1]                 # (k, B)
                        if bool(window.any(dim=0).all()):
                            # 所有行都已命中过 EOS：截到「最后一行首次命中的步」。
                            # 首个命中步用「前缀全 False 的个数」计算，避免
                            # argmax 对多个 True 的并列索引依赖后端约定。
                            first = (window.long().cumsum(dim=0) == 0).sum(dim=0)
                            cur_len = T0 + int(first.max().item()) + 1
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
    # CUDA 用 fused；CPU 显式开 foreach（与单张量路径逐元素同序运算，
    # 结果一致）。
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused,
                             foreach=not fused)


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
    # 权重已拷贝进模型，返回的 ckpt 字典不再需要整份 state_dict；
    # 及时释放可避免每个下游进程多驻留一份权重副本（CPU 内存/页缓存）。
    ck.pop("state_dict", None)
    model.to(device)
    model.eval()
    return model, ck
