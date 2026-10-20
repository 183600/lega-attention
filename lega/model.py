"""GPT 风格因果 Transformer：RoPE + 预归一化 + 词嵌入/lm_head 权重共享。

注意力层可替换为 lega.attention 中的任意变体（standard/qgate/null/lega 及消融）。
支持增量解码（KV 缓存），供 RULER 生成式评测使用。
"""
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


class _ChunkedLinearCE(torch.autograd.Function):
    """分块 lm_head + 交叉熵（支持 PyTorch 默认的 ignore_index=-100）。

    前向逐块计算 logits 与 loss，并按 softmax - onehot 的闭式解在同一分块
    内求出 dL/dx、dL/dW；反向只合并上游标量梯度与有效标签数。被忽略位置
    既不计入损失分母，也完全不产生梯度；全部位置被忽略时返回 0 损失
    （以及零梯度），与监督微调流程中的空有效标签约定一致。
    """

    @staticmethod
    def _loss_dtype(logits: torch.Tensor) -> torch.Tensor:
        # 高精度输入保持原精度，低精度/16 位输入升到 fp32，与前向 loss 及
        # 梯度计算严格同口径。
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
    def _ce_nll(lcf: torch.Tensor, tgt: torch.Tensor, ignore_index: int,
                has_ignore: bool = True) -> torch.Tensor:
        """逐位置 CE 的闭式计算：softmax 就地写入 lcf，按目标位置取概率再取
        对数，逐元素掩掉 ignore_index（返回 1-D nll）。

        与 F.cross_entropy(reduction="none") 数值等价（softmax 内部 max 平移，
        数值稳定）。softmax 就地写入 lcf，调用方可直接把它用作 dL/dlogits
        的 softmax 项（cross-entropy 对 logits 的梯度 = softmax - onehot），
        无需再取一次指数。目标概率下溢为 0 时以 finfo(dtype).tiny 兜底取对数，
        避免 -inf 进入损失。
        has_ignore=False（调用方已确认本批不存在忽略位置，语言建模训练的常见
        情形）时跳过逐块掩码构造，直接按原目标取值；两条路径的取值结果完全
        一致（无忽略位置时掩码恒为 False）。
        """
        probs = torch.softmax(lcf, dim=-1, out=lcf)
        tiny = torch.finfo(probs.dtype).tiny
        if not has_ignore:
            return probs.gather(1, tgt.unsqueeze(1)).squeeze(1) \
                .clamp_min_(tiny).log_().neg_()
        ignore = tgt == ignore_index
        ts_safe = tgt.masked_fill(ignore, 0)
        nll = probs.gather(1, ts_safe.unsqueeze(1)).squeeze(1) \
            .clamp_min_(tiny).log_().neg_()
        nll.masked_fill_(ignore, 0)
        return nll

    @staticmethod
    def _ce_sum(lcf: torch.Tensor, tgt: torch.Tensor, ignore_index: int,
                has_ignore: bool = True) -> torch.Tensor:
        """sum-CE；语义同 ``_ce_nll(...).sum()``。"""
        return _ChunkedLinearCE._ce_nll(
            lcf, tgt, ignore_index, has_ignore).sum()

    @staticmethod
    def forward(ctx, x2d, weight, tgt, chunk: int, ignore_index: int = -100):
        # weight 参与反向的版本校验（参数被原地改写时能及时报错）；
        # x2d/tgt 只在本次前向使用，不在 ctx 上保留。
        ctx.save_for_backward(weight)
        ctx.chunk = chunk
        ctx.ignore_index = int(ignore_index)
        # 记录前向的 autocast 状态：logits、loss 与 dL/dlogits 全程同一精度，
        # 反向无需再重建 logits。
        ctx.amp_dtype = _ChunkedLinearCE._autocast_dtype(x2d)
        # n_valid 以设备张量保存；clamp_min(1) 同时吸收"全部位置被忽略"的
        # 除零情形（此时梯度因 keep 掩码全零而天然为 0，无需提前特判）。
        # 有忽略位置 ⇔ 有效数少于总数，tgt_has_ignore 由 n_valid 导出。
        n_valid = (tgt != ctx.ignore_index).sum()
        ctx.inv_n = n_valid.clamp_min(1).to(
            x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
            else torch.float32).reciprocal()
        loss = torch.zeros((), dtype=x2d.dtype if x2d.dtype in (torch.float32, torch.float64)
                           else torch.float32, device=x2d.device)
        # 只计算上游真正需要的梯度张量；dx 的每一行、gw 的每个元素都会在
        # 分块循环中被完整覆写或累加，无需预置零。
        need_dx = bool(ctx.needs_input_grad[0])
        need_gw = bool(ctx.needs_input_grad[1])
        dx = torch.empty_like(x2d) if need_dx else None
        gw = None
        w_f = None
        w_g = None
        # AMP 下梯度 GEMM 走 autocast 精度（与 nn.Linear 在 autocast 下的
        # 反向一致）：softmax/掩码仍在 fp32 上完成，进入 GEMM 前才降精度；
        # 权重梯度先按块在低精度下求出、再累加到 fp32 缓冲，避免跨块在
        # bf16/fp16 上累加。非 AMP（fp32/fp64）路径保持原精度。
        gemm_dtype = (ctx.amp_dtype if ctx.amp_dtype in
                      (torch.float16, torch.bfloat16) else None)
        # 是否有被忽略的位置（全局判定一次）：没有时损失与梯度两条路径
        # 都无需逐块构造掩码，直接按原目标取值/散加。CUDA 上不把有效数取回
        # 主机，按有掩码处理；CPU 上取回有效数并保留无掩码路径。
        tgt_has_ignore = (True if x2d.is_cuda
                          else int(n_valid) != tgt.numel())
        has_ignore = (need_dx or need_gw) and tgt_has_ignore
        # 非 AMP 时梯度 GEMM 必须与 logits 同精度：显式关闭 autocast，避免
        # fp32/fp64 的 GEMM 被本就处于 autocast 上下文的外层调用降到低精度。
        grad_ctx = torch.autocast("cuda" if x2d.is_cuda else "cpu", enabled=False)
        with torch.no_grad():
            # 分块循环共用一份目标精度权重，各块 GEMM 的输入逐位一致。
            w_fwd = weight if (ctx.amp_dtype is None
                               or weight.dtype == ctx.amp_dtype) \
                else weight.to(ctx.amp_dtype)
            neg_one = None   # scatter_add_ 的 -1 源张量
            for s in range(0, tgt.numel(), chunk):
                ts = tgt[s:s + chunk]
                lc = F.linear(x2d[s:s + chunk], w_fwd)               # (m, V)
                lcf = _ChunkedLinearCE._loss_dtype(lc)
                # 升精度副本已生成（不升精度时 lcf 与 lc 为同一张量）。
                del lc
                # _ce_sum 把 softmax 原地写在 lcf 上，它同时就是 dL/dlogits 的
                # softmax 项（cross-entropy 对 logits 的梯度 = softmax - onehot）。
                loss.add_(_ChunkedLinearCE._ce_sum(
                    lcf, ts, ctx.ignore_index, tgt_has_ignore))
                if not (need_dx or need_gw):
                    continue
                with grad_ctx:
                    d = lcf
                    idx_src = ts
                    if has_ignore:
                        keep = ts != ctx.ignore_index
                        # one-hot 减 1 用 scatter_add_ 一步完成：先把无效位置的
                        # 索引压到 0（其减 1 随后被 keep 掩码清零）。
                        idx_src = ts.masked_fill(~keep, 0)
                    if neg_one is None or neg_one.size(0) < idx_src.numel():
                        neg_one = d.new_full((max(idx_src.numel(), 4096), 1), -1.0)
                    d.scatter_add_(1, idx_src.unsqueeze(1),
                                   neg_one[:idx_src.numel()])
                    # keep∈{0,1}，无效行清零即可。
                    if has_ignore:
                        d.mul_(keep.unsqueeze(1).to(d.dtype))
                    if gemm_dtype is not None:
                        d = d.to(gemm_dtype)
                        # 低精度副本生成后立即释放 fp32 概率矩阵。
                        del lcf
                        if w_g is None:
                            w_g = (w_fwd if w_fwd.dtype == gemm_dtype
                                   else weight.to(gemm_dtype))
                        if need_dx:
                            dx[s:s + chunk].copy_(torch.matmul(d, w_g))
                        if need_gw:
                            xs_g = x2d[s:s + chunk].to(gemm_dtype)
                            part = torch.mm(d.transpose(0, 1), xs_g)
                            gw = part.float() if gw is None else gw.add_(part)
                        del d
                        continue
                    if need_dx:
                        # d 与权重副本同为计算精度，GEMM 结果与向上游要求的
                        # 梯度同精度（常见 fp32/fp64 下直接写入 dx 的行切片）。
                        if w_f is None:
                            w_f = weight.to(d.dtype)
                        gxs = dx[s:s + chunk]
                        if gxs.dtype == d.dtype:
                            torch.matmul(d, w_f, out=gxs)
                        else:
                            gxs.copy_(torch.matmul(d, w_f))
                    if need_gw:
                        xs_f = x2d[s:s + chunk].to(d.dtype)
                        # 首块用 mm 直接产出 gw，后续分块就地累加。
                        if gw is None:
                            gw = torch.mm(d.transpose(0, 1), xs_f)
                        else:
                            gw.addmm_(d.transpose(0, 1), xs_f)
                    del d
        ctx.dx = dx
        ctx.gw = gw
        # 反向缩放输出的缓冲，首次 backward 时按参数精度惰性分配。
        ctx.gw_buf = None
        return loss * ctx.inv_n

    @staticmethod
    def backward(ctx, gout):
        # 前向已按 dL/dlogits 求出 dx、gw（含 ignore 掩码），这里只合并
        # 上游标量梯度与 1/n_valid，并把权重梯度转到参数精度。
        weight, = ctx.saved_tensors
        gscale = (gout if gout.dtype in (torch.float32, torch.float64)
                  else gout.float()) * ctx.inv_n
        # 一律不原地缩放 ctx.dx/ctx.gw：它们是前向物化的中间量，同一计算图
        # 多次 backward（retain_graph=True）时必须保持不变；若在第一次调用中
        # 原地缩放，后续调用会再乘一次 1/n_valid，梯度随 backward 次数衰减。
        dx = None if ctx.dx is None else ctx.dx * gscale
        gw = None
        if ctx.gw is not None:
            if torch.is_grad_enabled():
                # create_graph（高阶梯度）下本次 backward 自身也建图，复用
                # 缓冲的原地写会与已记录的版本冲突，退回新建临时。
                gw = torch.mul(ctx.gw, gscale).to(weight.dtype)
            else:
                # out= 路径与「先乘后转参数精度」逐位一致。
                buf = ctx.gw_buf
                if (buf is None or buf.dtype != weight.dtype
                        or buf.shape != ctx.gw.shape
                        or buf.device != ctx.gw.device):
                    buf = torch.empty(ctx.gw.shape, dtype=weight.dtype,
                                      device=ctx.gw.device)
                    ctx.gw_buf = buf
                grad = weight.grad
                if (grad is not None and grad.device == buf.device
                        and grad.shape == buf.shape
                        and grad.data_ptr() == buf.data_ptr()):
                    # 缓冲已被 AccumulateGrad 采纳为 weight.grad：再原地
                    # 覆写会丢掉已累加的梯度，本次改用新建临时并换新缓冲。
                    buf = torch.empty(ctx.gw.shape, dtype=weight.dtype,
                                      device=ctx.gw.device)
                    ctx.gw_buf = buf
                gw = torch.mul(ctx.gw, gscale, out=buf)
        return dx, gw, None, None, None


def _ce_chunk_size(vocab_size: int, device: torch.device, *,
                   train: bool) -> int:
    """lm_head+交叉熵的分块大小：按设备与词表自适应。

    训练态与无梯度路径使用不同的元素上限；CUDA 上按设备规模估算，结果限制在
    固定上下限之间。
    """
    if not train and device.type == "cpu":
        elems = 33_554_432
    else:
        elems = 33_554_432 * 4
        if device.type == "cuda":
            try:
                total = torch.cuda.get_device_properties(device).total_memory
                elems = min(max(total // 64, 33_554_432), elems)
            except Exception:
                pass
    return max(256, min(4096, elems // max(1, int(vocab_size))))


def _cast_weight_cached(owner, slot: str, weight: torch.Tensor) -> torch.Tensor:
    """线性层前向权重：无梯度时按 autocast 目标精度缓存。

    有梯度时直接返回参数本身，保持反向连通到参数；无梯度路径返回目标精度的
    缓存副本。副本在 inference_mode(False)+no_grad 下创建，不会留下会被
    autograd 拒绝的 inference tensor；权重被替换或原地更新（_version 变化）
    会使缓存失效。
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
        # 无梯度（no_grad/inference_mode，推理/评测）时残差连接用原地加法：x 是
        # 本层独占的新张量（来自上一层输出或嵌入层输出，不被其它子图引用）。
        # 只要梯度模式开启就必须 out-of-place：即使 x 本身 requires_grad=False
        # （冻结嵌入/detach 上游），LayerNorm 的反向仍会保存 x（用于计算
        # weight/bias 梯度），原地 add_ 会破坏其版本计数并直接报错。
        if torch.is_grad_enabled():
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
        （评测/生成，含 inference_mode）时返回目标精度的缓存副本，
        见 _cast_weight_cached。
        """
        return _cast_weight_cached(self, "_lmhead_cast", self.tok_emb.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                past_kv: Optional[List] = None, need_stats: bool = False,
                last_logits_only: bool = False, stats_argmax: bool = True,
                kv_cache: Optional[List] = None, cache_pos: int = 0,
                ignore_index: int = -100,
                return_window_losses: bool = False) -> Dict:
        """kv_cache/cache_pos：静态 KV 缓存模式（预分配缓冲 + 写入偏移），
        供 generate 逐 token 解码使用；与 past_kv 的动态拼接缓存互斥。

        return_window_losses=True 时，在无梯度评测路径额外返回每行（每个
        输入窗口）的平均损失 ``loss_per_window``，供验证集抽样误差估计；
        训练/需要梯度的路径不支持该选项（避免改变训练图）。
        """
        B, T = idx.shape
        past_len = (0 if past_kv is None else past_kv[0][0].shape[2]) \
            if kv_cache is None else cache_pos
        assert T + past_len <= self.block_size, (
            f"输入长度 {past_len}+{T} 超过 block_size={self.block_size}")

        x = self.drop(self.tok_emb(idx))
        # 逐层缓存容器预分配为定长列表再按序号写入
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
        loss_per_window = None
        if targets is not None:
            x = self.ln_f(x)
            # 训练/评测只需 loss：分块过 lm_head 并立即做交叉熵；
            # 分块求和再平均与整批 mean 结果一致；此时不再返回 logits。
            w = self._lmhead_weight()                     # 共享权重
            flat = x.reshape(-1, x.size(-1))
            tgt = targets.reshape(-1)
            # 只要计算图需要梯度（输入或共享 lm_head 权重任一 requires_grad，
            # 如冻结主干只训分类头/detach 上游）就走闭式梯度路径；否则
            # softmax(out=) 在 grad 模式下反传会报“derivative not implemented”。
            needs_grad = (torch.is_grad_enabled()
                          and (x.requires_grad or w.requires_grad))
            # 每块 logits 为目标数乘以分块大小，分块大小随设备/词表自适应。
            CHUNK = _ce_chunk_size(self.vocab_size, flat.device,
                                   train=needs_grad)
            if return_window_losses and needs_grad:
                raise RuntimeError(
                    "return_window_losses 仅支持无梯度评测（inference_mode）")
            if needs_grad:
                # 训练路径：分块前向，交叉熵闭式梯度在前向内求出
                loss = _ChunkedLinearCE.apply(flat, w, tgt, CHUNK, int(ignore_index))
            else:
                total = None  # 设备端累加
                # 有忽略位置 ⇔ 有效数少于总数，tgt_has_ignore 由 n_valid
                # 导出；无忽略位置（常见情形）时逐块损失无需掩码。CUDA 上
                # 不把有效数取回主机，按有掩码处理。
                n_valid = (tgt != ignore_index).sum()
                tgt_has_ignore = (True if flat.is_cuda
                                  else int(n_valid) != tgt.numel())
                row_sums = None
                if return_window_losses:
                    # 每个输入窗口占一行（B 行、每行 T 个位置）；按行累加
                    # 逐位置 nll，最后除以每行有效位置数。这个统计只服务
                    # 于验证集抽样误差估计，不进入训练路径。
                    per_row_valid = (tgt != ignore_index).reshape(B, T) \
                        .sum(dim=1).clamp_min(1)
                    rows = torch.arange(flat.size(0), device=flat.device) // T
                for s in range(0, flat.size(0), CHUNK):
                    lc = F.linear(flat[s:s + CHUNK], w)
                    lcd = _ChunkedLinearCE._loss_dtype(lc)
                    del lc
                    if return_window_losses:
                        nll = _ChunkedLinearCE._ce_nll(
                            lcd, tgt[s:s + CHUNK], int(ignore_index),
                            tgt_has_ignore)
                        total = nll.sum() if total is None else total.add_(nll.sum())
                        if row_sums is None:
                            row_sums = torch.zeros(B, dtype=nll.dtype,
                                                   device=nll.device)
                        row_sums.index_add_(0, rows[s:s + nll.numel()], nll)
                    else:
                        part = _ChunkedLinearCE._ce_sum(
                            lcd, tgt[s:s + CHUNK], int(ignore_index),
                            tgt_has_ignore)
                        total = part if total is None else total.add_(part)
                # n_valid 保留为设备张量
                loss = total / n_valid.clamp_min(1)
                if return_window_losses:
                    loss_per_window = row_sums / per_row_valid.to(row_sums.dtype)
            logits = None
        else:
            # last_logits_only 时（如 generate）先裁到最后位置再过 ln_f/lm_head：
            # LayerNorm 逐位置独立，结果与先归一化再切片一致
            if last_logits_only:
                x = x[:, -1:, :]
            x = self.ln_f(x)
            logits = F.linear(x, self._lmhead_weight())  # 权重共享
        return {"logits": logits, "loss": loss,
                "loss_per_window": loss_per_window,
                "past_kv": new_cache,
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
            # 空提示（T0=0）无法生成：第一个 token 的 logits 必须来自至少一个
            # 输入位置；提前报出清晰错误，避免在注意力的 max/切片处抛出
            # 难以定位的 "amax(): Expected reduction dim -1 to have non-zero size"。
            assert T0 >= 1, (
                f"提示长度不能为 0（收到 shape {tuple(idx.shape)}）："
                "生成至少需要 1 个起始 token 才能产生第一个 token 的 logits")
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
                # k/v 在 autocast 区按目标精度计算：缓存同精度分配，
                # 写入与读取均为该精度张量，注意力消费的亦是缓存视图本身。
                kvdtype = (_ChunkedLinearCE._autocast_dtype(attn0.wk.weight)
                           or attn0.wk.weight.dtype)
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
        "gate_only": 1,               # d（仅门控偏置，无 A/B/C）
        "qgate": d_head + 1,          # w, d
        "null": 2 * d_head,           # k_null, v_null
        "lega": d_head + 3,           # w, a, b, d
        "lega_no_len": d_head + 2,    # w, a, d
        "lega_no_conc": d_head + 2,   # w, b, d
        "lega_no_query": 3,           # a, b, d
        "lega_conc_only": 2,          # a, d
        "lega_len_only": 2,           # b, d
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
               use_sdpa_standard: Optional[bool] = None):
    """从 checkpoint 恢复模型。返回 (model, ckpt_dict)。

    use_sdpa_standard=None 时沿用 checkpoint 训练时的注意力实现路径，避免
    评测/生成静默切到另一条实现（数值与内核不同，混用会污染对比）。显式
    传值仅用于需要强制路径的调用（如 latency 的 standard/standard_sdpa）。
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = dict(ck["model_cfg"])
    if use_sdpa_standard is None:
        use_sdpa_standard = bool(cfg.get("use_sdpa_standard", True))
    cfg["use_sdpa_standard"] = bool(use_sdpa_standard)
    model = GPT(**cfg)
    model.load_state_dict(ck["state_dict"])
    ck.pop("state_dict", None)
    model.to(device)
    model.eval()
    return model, ck
