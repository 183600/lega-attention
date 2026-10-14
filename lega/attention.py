"""核心模块：带"长度校准证据门控"的因果多头自注意力（LEGA）与全部对照变体。

对每个查询位置 t、每个头 h，注意力输出 o 与门控写回规则：

    standard      :  o' = o                                        （标准注意力）
    qgate         :  g = sigmoid(w·q + d)                          （已有工作：只看查询的输出门控）
    null          :  额外可学习 null 键值槽，注意力可整体"不读取"    （已有工作：显式空注意力）
    lega (ours)   :  c = log(n * max_j p_j)
                     g = sigmoid(w·q + a*c + b*log n + d)
                     o' = g * o
    lega_no_len   :  g = sigmoid(w·q + a*log max_j p_j + d)        （消融：去掉长度校准）
    lega_no_conc  :  g = sigmoid(w·q + b*log n + d)                （消融：去掉集中度证据）
    lega_no_query :  g = sigmoid(a*c + b*log n + d)                （消融：去掉查询项）

其中 p_j 为因果注意力权重，n 为该查询可读取的真实 token 数（null 槽不计入 n）。
q 特指 RoPE 旋转前的查询内容向量；注意力打分仍使用旋转后的 q/k。

关键性质：均匀注意力时 max_j p_j = 1/n，因此 c = log(n * 1/n) = 0，与上下文长度无关。
c 度量的是"超出均匀分布的证据集中度"；b*log n 项让门显式感知长度，二者解耦。
初始时 w=a=b=0、d=2.0，g ≈ 0.88，接近标准注意力，保证训练稳定启动。
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 全部变体名（配置/命令行使用）
VARIANTS = ("standard", "qgate", "null", "lega", "lega_no_len", "lega_no_conc", "lega_no_query")

# 供图表与汇总表使用的中文名
VARIANT_CN = {
    "standard": "标准Attention",
    "qgate": "查询门控(已有)",
    "null": "空注意力(已有)",
    "lega": "LEGA(ours)",
    "lega_no_len": "LEGA去长度项",
    "lega_no_conc": "LEGA去集中度项",
    "lega_no_query": "LEGA去查询项",
}

# 各公式项出现的变体
_USE_QUERY = ("qgate", "lega", "lega_no_len", "lega_no_conc")      # w·q 项
_USE_CONC = ("lega", "lega_no_len", "lega_no_query")               # log max_j p_j 项
_USE_LEN = ("lega", "lega_no_conc", "lega_no_query")               # log n 项
_GATED = ("qgate", "lega", "lega_no_len", "lega_no_conc", "lega_no_query")  # 有输出门
_LEGA_FAMILY = ("lega", "lega_no_len", "lega_no_conc", "lega_no_query")     # 门依赖注意力统计

# 显式注意力按查询维分块计算 (B,H,T,S) 的打分/概率，每块约 _CHUNK_TARGET_ELEMS
# 个元素；训练态配合逐块 checkpoint。
_CHUNK_TARGET_ELEMS = 8_000_000
_CHUNK_MIN_T = 64

# 进程级共享缓存：各层（乃至各实例）在同一设备/长度下的因果掩码完全相同，
# log n 由设备级前缀表统一切片，按模块共享。
# 键内已含 device，跨设备天然隔离；长度变化场景（解码）由容量上限兜底。
# 只构造 not_allowed（显式路径直接用）；allowed 仅 SDPA 带掩码的少见分支需要，
# 用时再按位取反。
_MASK_CACHE: dict = {}   # (past_len, T, S, device) -> [not_allowed, ones_col, allowed]
                         # allowed 惰性填充并与 not_allowed 一并保存，供 SDPA
                         # 带掩码分支直接使用
# 设备端 log 前缀表：table[n] = log(n+1)，任意 (past_len, T) 的 log n 恒为其
# [past_len, past_len+T) 切片。按需倍增扩展：避免按 (past_len, T) 逐键缓存
# 导致的重复驻留，以及单查询解码步每层每步从主机端新建标量（H2D + 分配）。
_LOGN_TABLE: dict = {}   # device -> (N,) float
_LOGN_TABLE_INIT = 4096
# 查询分块路径使用的三角对角带掩码：块 (r, c) 内掩码只依赖相对偏移（c > r），
# 与 past 偏移无关，全层/全块共享一份即可；分块步常见 chunk 形状只有一两种。
_TRI_CACHE: dict = {}    # (r, c, device) -> (1, 1, r, c) bool，严格上三角为 True
_CACHE_CAP = 256


def _log_n_table(dev, need: int) -> torch.Tensor:
    """设备端 log 前缀表（table[i] = log(i+1)），不足 need 时倍增重建。"""
    table = _LOGN_TABLE.get(dev)
    if table is None or table.numel() < need:
        with torch.inference_mode(False):  # 缓存常量不能是 inference tensor
            n = max(_LOGN_TABLE_INIT, 1 << (need - 1).bit_length())
            table = torch.log((torch.arange(n, device=dev) + 1).float())
        _LOGN_TABLE[dev] = table
    return table


def _log_n(past_len: int, T: int, dev) -> torch.Tensor:
    """log(n) = log(可读取的真实 token 数) = log(past_len + t + 1)，为前缀表切片视图。

    通过设备端 log 前缀表切片得到：与原逐次 arange+log 的结果逐项一致（同一
    查表公式 log(i+1)），但任意 (past_len, T) 组合共享一份表：
      * 逐 token 解码（T==1，past_len 每步 +1）不再每层每步新建主机端标量并
        触发 H2D 拷贝，同一步内全部层返回同一视图的连续切片；
      * 训练/预填充的 (past_len, T) 组合增多时也不再各自驻留一份 (T,) 缓存。
    表以 inference_mode(False) 创建，不会被推理期现场污染 autograd。
    """
    return _log_n_table(dev, past_len + T)[past_len:past_len + T]


def _tri_band(r: int, c: int, dev) -> torch.Tensor:
    """(1, 1, r, c) 严格上三角布尔掩码：[row, col] = (col > row)。"""
    key = (r, c, dev)
    m = _TRI_CACHE.get(key)
    if m is None:
        with torch.inference_mode(False):  # 缓存常量不能是 inference tensor
            m = torch.ones(1, 1, r, c, dtype=torch.bool, device=dev).triu_(1)
        if len(_TRI_CACHE) < _CACHE_CAP:
            _TRI_CACHE[key] = m
    return m


class CausalSelfAttention(nn.Module):
    """因果多头自注意力，按 variant 切换标准/门控/空注意力行为。

    forward 返回 (out, cache, stats)：
      out    : (B, T, C) 写回残差流之前的注意力输出（已经过门控 o' = g*o）
      cache  : (k, v) 供增量解码的 KV 缓存（不含 null 槽）
      stats  : 仅当 need_stats=True 时返回诊断统计字典，否则为 None
    """

    def __init__(self, d_model: int, n_heads: int, variant: str = "lega",
                 attn_dropout: float = 0.0, gate_bias_init: float = 2.0,
                 use_sdpa_standard: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert variant in VARIANTS, f"未知变体: {variant}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.variant = variant
        self.attn_dropout = attn_dropout
        self.use_sdpa = use_sdpa_standard
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.use_query = variant in _USE_QUERY
        self.use_conc = variant in _USE_CONC
        self.use_len = variant in _USE_LEN

        Dh = self.d_head
        if self.use_query:
            self.gate_w = nn.Parameter(torch.zeros(n_heads, Dh))      # w
        if self.use_conc:
            self.gate_a = nn.Parameter(torch.zeros(n_heads))          # a
        if self.use_len:
            self.gate_b = nn.Parameter(torch.zeros(n_heads))          # b
        if variant in _GATED:
            self.gate_d = nn.Parameter(torch.full((n_heads,), float(gate_bias_init)))  # d
        if variant == "null":
            # 构造期初始化不能消耗全局 RNG：否则 apply(_init_weights) 重初始化全部
            # 共享参数时 RNG 状态已被 null 变体的 randn 挪动，导致 null 与
            # 其它变体在同 seed 下基础权重不一致，破坏"同一 seed 公平对比"。
            # GPT.__init__ 随后会调 init_gate_params_ 用全局 RNG 重新逐层随机化。
            gen = torch.Generator().manual_seed(0)
            self.k_null = nn.Parameter(
                torch.randn(n_heads, Dh, generator=gen) * 0.02)
            self.v_null = nn.Parameter(torch.zeros(n_heads, Dh))

        # 因果掩码 / log_n 分别由模块级 _MASK_CACHE / 设备级 _LOGN_TABLE 提供，
        # 全层共享，避免每层各重复驻留一份。
        # 推理态融合 QKV 权重缓存：(key, tensor)；key 为三个权重参数对象及其
        # _version。推理/评测/逐 token 解码时权重不变，每层每次前向不再重新
        # cat 一份 (3C, C)；训练态不走缓存（cat 必须参与 autograd 图），
        # 且优化器/测试对权重的原地修改会递增 _version 使缓存自动失效。
        self._qkv_fused = None

    def init_gate_params_(self, gate_bias_init: float = 2.0) -> None:
        """显式初始化门控参数（在 GPT 的整体初始化之后调用）。"""
        if self.use_query:
            nn.init.zeros_(self.gate_w)
        if self.use_conc:
            nn.init.zeros_(self.gate_a)
        if self.use_len:
            nn.init.zeros_(self.gate_b)
        if self.variant in _GATED:
            nn.init.constant_(self.gate_d, float(gate_bias_init))
        if self.variant == "null":
            nn.init.normal_(self.k_null, mean=0.0, std=0.02)
            nn.init.zeros_(self.v_null)

    def _gate_logit(self, q_content: torch.Tensor, max_p: Optional[torch.Tensor],
                    log_n: Optional[torch.Tensor], keep_terms: bool = False):
        """按 variant 组合门控 logit；返回 (g, logit, qterm)，不适用项为 None。

        查询项在（可能的 bf16）内容向量上直接归约后再升 fp32。
        keep_terms=False（训练/普通推理）时各项原地累加进 logit；
        keep_terms=True 时 qterm 需原样返回给统计，logit 在其副本上累加。
        """
        H = self.n_heads
        logit = qterm = None
        if self.use_query:
            # 每头契约向量用批量 GEMV（matmul 广播批维）实现（结果与
            # mul+sum/einsum 归约一致，见注释末）。
            qterm = torch.matmul(q_content, self.gate_w.to(q_content.dtype)
                                 .unsqueeze(-1)).squeeze(-1).float()      # (B,H,T)
            logit = qterm.clone() if keep_terms else qterm
        # logit.addcmul_(a, c) 融合"标量门控系数 × 统计项"的乘加，避免先物化
        # a*c / b*log_n 的 (B,H,T) 临时张量再逐项 add_（分块/checkpoint 路径下
        # 每个块、每次反向重算都要重复分配）。
        if self.use_conc:
            # 无统计需求且 max_p 不参与反向（推理/统计仅前向消费）时，
            # clamp/log 原地完成，省一份 (B,H,T) fp32 临时张量；max_p 为
            # amax 的反向保存值，需要梯度时必须保持原值。
            c = torch.log(max_p.clamp_min(1e-30)) if (keep_terms or max_p.requires_grad) \
                else max_p.clamp_min_(1e-30).log_()
            if self.use_len:
                c += log_n                                                # c = log(n * max_j p_j)
            a = self.gate_a.float().view(1, H, 1)
            logit = a * c if logit is None else logit.addcmul_(a, c)
        if self.use_len:
            b = self.gate_b.float().view(1, H, 1)
            logit = b * log_n if logit is None else logit.addcmul_(b, log_n)
        dterm = self.gate_d.float().view(1, H, 1)
        logit = dterm.expand(q_content.shape[0], H, q_content.shape[2]).clone() \
            if logit is None else logit.add_(dterm)
        # keep_terms=False（训练/普通推理）时调用方不消费 logit：
        # sigmoid 原地写入，省掉一份 (B,H,T) fp32 输出张量
        # （分块/checkpoint 路径下每块、每次反向重算各一次分配）。
        g = torch.sigmoid(logit) if keep_terms else logit.sigmoid_()
        return g, logit, qterm

    def _attend_q_chunk(self, q_rot: torch.Tensor, q_content: torch.Tensor,
                        kT: torch.Tensor, v: torch.Tensor,
                        band_start: int, band_mask: Optional[torch.Tensor],
                        log_n: Optional[torch.Tensor], collect_stats: bool):
        """单查询块的显式注意力 + 门控；collect_stats=False 时只返回输出。

        q_rot 是 RoPE 旋转后且已预乘 1/sqrt(Dh) 的查询（打分的 q·k）；
        q_content 是 RoPE 旋转前且未乘 1/sqrt(Dh) 的查询内容向量，门控查询项
        必须用它（长度/绝对位置只通过显式 log n 进入门控）。

        kT/v 已被调用方裁到本块可见前缀（末列即块内最末查询可见的最后一键），
        全块不可见的尾列既不参与打分也不参与 softmax，数学上与先算全量再做整列
        -inf 屏蔽完全等价；块内只需对对角带
        [band_start, band_start+cq) 用平移不变的 (1,1,cq,cq) 三角掩码做
        masked_fill_，面积只有 cq² 而非 cq*S。

        null 变体不把 null 槽拼接进 K/V（避免每层每步复制整个缓存），也不把空槽
        得分 cat 到打分矩阵末尾（避免整块复制 (B,H,C,S)），而是解析合并：
          p_null = e^{a_n} / (Z + e^{a_n}) = sigmoid(a_n - logsumexp(att))
          o = (softmax(att) @ v) * (1 - p_null) + p_null * v_null
        与先 cat 再做联合 softmax 的语义逐项等价（softmax/cat 的尺度因子恰好
        为 1 - p_null），省掉一份 (B,H,C,S) 的 cat 副本与拷贝。
        """
        S = kT.size(-1)
        att = torch.matmul(q_rot, kT)                               # (B,H,C,S)，S=可见前缀长
        if band_mask is not None:
            att[..., band_start:band_start + band_mask.shape[-1]].masked_fill_(
                band_mask, float("-inf"))
        is_null = self.variant == "null"
        if is_null:
            att_n = torch.matmul(q_rot, self.k_null.to(q_rot.dtype).unsqueeze(-1))
            # p_null = 1 / (1 + Z/e^{a_n})，logsumexp 内部做 max 平移，数值稳定；
            # 注意是 a_n - lse（sigmoid(lse - a_n) 得到的是 1 - p_null）
            p_null = torch.sigmoid(att_n - att.logsumexp(dim=-1, keepdim=True))
            del att_n
        p = F.softmax(att, dim=-1)
        del att

        max_p = None
        if self.variant in _LEGA_FAMILY and self.use_conc:
            max_p = p.amax(dim=-1).float()                            # (B,H,C)
        if self.attn_dropout > 0 and self.training:
            p = F.dropout(p, p=self.attn_dropout)
        o = torch.matmul(p, v)                                        # (B,H,C,Dh)
        del p
        if is_null:
            # 真实键部分整体乘 (1 - p_null)（联合 softmax 的尺度因子），
            # addcmul_ 再融合 p_null(×)v_null 的乘加（训练态逐块 checkpoint
            # 反向重算时逐块生效）
            o.mul_(1 - p_null)
            o.addcmul_(p_null, self.v_null.to(v.dtype).view(1, self.n_heads, 1, -1))

        g = logit = qterm = None
        if self.variant in _GATED:
            g, logit, qterm = self._gate_logit(q_content, max_p, log_n,
                                               keep_terms=collect_stats)
            o.mul_(g.to(o.dtype).unsqueeze(-1))
        if not collect_stats:
            return o
        return o, max_p, qterm, logit, g

    def _attend_chunked(self, q_rot: torch.Tensor, q_content: torch.Tensor,
                        k: torch.Tensor, v: torch.Tensor,
                        past_len: int, log_n: Optional[torch.Tensor],
                        B: int, H: int, T: int, collect_stats: bool):
        """按查询块执行显式注意力；训练态逐块 checkpoint。

        q_rot 必须已预乘 1/sqrt(Dh)：在进入分块循环前统一缩放一次。
        """
        S = k.shape[2]
        # 目标：每块打分矩阵约 <= _CHUNK_TARGET_ELEMS 元素
        cq = max(_CHUNK_MIN_T, min(T, _CHUNK_TARGET_ELEMS // max(1, B * H * S)))
        cq = min(T, max(cq, 1))
        kT = k.transpose(-2, -1)
        # checkpoint 的前向以 collect_stats=False 运行且返回单个张量，
        # 需要统计时必须走普通调用，否则统计会被静默丢弃（g/logit 全为 None）。
        use_ckpt = torch.is_grad_enabled() and q_rot.requires_grad and not collect_stats
        # 输出直接按 (B,T,H,Dh) 连续布局写入预分配缓冲（逐块 copy_ 是 autograd
        # 安全的）：块输出到多头合并的转置在写块时一次完成
        Dh = v.shape[-1]
        o = v.new_empty(B, T, H, Dh)
        max_ps, qterms, logits, gs = [], [], [], []
        for q0 in range(0, T, cq):
            q1 = min(T, q0 + cq)
            qr = q_rot[:, :, q0:q1]  # 已预乘 1/sqrt(Dh) 的 RoPE 旋转后查询块（打分用）
            qc = q_content[:, :, q0:q1]  # 未缩放的 RoPE 前查询内容向量块（门控用）
            # 因果掩码两段式：
            #   * [fill_from, S) 列对本块全部查询不可见：不置 -inf，直接把 kT/v
            #     切片到可见前缀，打分矩阵从 (B,H,Cq,S) 缩到 (B,H,Cq,fill_from)；
            #   * 对角带 [band_start, band_start + (q1-q0))：用全局共享的
            #     严格上三角掩码逐块置 -inf（只依赖块内相对偏移，与 past 无关）。
            fill_from = min(past_len + q1, S)
            if fill_from < S:
                kT_c = kT[..., :fill_from]          # kT 布局 (B,H,Dh,S)，键维在末维
                v_c = v[:, :, :fill_from]           # v 布局 (B,H,S,Dh)
            else:
                kT_c, v_c = kT, v
            band_start = past_len + q0
            # 单行块（逐 token 解码）的对角带掩码为全 False（triu(1)），
            # 逐元素 masked_fill_ 不会对任何元素置 -inf，整块跳过。
            band_mask = None if q1 - q0 == 1 else _tri_band(q1 - q0, q1 - q0, q_rot.device)
            logn_c = None if log_n is None else log_n[q0:q1]
            if use_ckpt:
                # preserve_rng_state=False：分块路径仅在 attn_dropout==0 时启用
                # （use_chunk 条件已排除训练态 dropout），块内无 RNG 消耗
                oc = torch.utils.checkpoint.checkpoint(
                    self._attend_q_chunk, qr, qc, kT_c, v_c, band_start,
                    band_mask, logn_c, False,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                if collect_stats:
                    oc, mp, qt, lg, gg = self._attend_q_chunk(
                        qr, qc, kT_c, v_c, band_start, band_mask,
                        logn_c, True)
                    if mp is not None:
                        max_ps.append(mp)
                    if qt is not None:
                        qterms.append(qt)
                    if lg is not None:
                        logits.append(lg)
                    if gg is not None:
                        gs.append(gg)
                else:
                    oc = self._attend_q_chunk(qr, qc, kT_c, v_c, band_start,
                                              band_mask, logn_c, False)
            # oc 为 (B,H,C,Dh)，目标切片 (B,C,H,Dh)：copy_ 自转置视图，
            # 布局变换在此一次完成，取代末端整表 transpose().contiguous()
            o[:, q0:q1].copy_(oc.transpose(1, 2))
        # 送出的 o 已按 (B,T,H) 布局写满，直接 view 成多头合并的 (B,T,C)
        o = o.view(B, T, H * Dh)
        if not collect_stats:
            return o, None, None, None, None
        cat = lambda xs: torch.cat(xs, dim=2) if xs else None
        return o, cat(max_ps), cat(qterms), cat(logits), cat(gs)

    def forward(self, x: torch.Tensor, rope=None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                need_stats: bool = False, stats_argmax: bool = True,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache_pos: int = 0):
        """past_kv 为逐次拼接的动态缓存（每次前向 torch.cat）。

        生成式逐 token 解码请改用 kv_cache + cache_pos 的静态缓存：
        预分配 (B, H, max_len, Dh) 缓冲一次，本层把新 k/v 就地写入
        [cache_pos, cache_pos+T)，注意力只读取已写入的前缀视图。
        """
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head
        dev = x.device
        past_len = (0 if past_kv is None else past_kv[0].shape[2]) \
            if kv_cache is None else cache_pos

        # 三个投影合并为一次 GEMM（参数仍分存于 wq/wk/wv），再按 (q|k|v, H, Dh) 切分
        if torch.is_grad_enabled():
            # 训练态：cat 必须参与 autograd 图，每次前向重建
            w_qkv = torch.cat([self.wq.weight, self.wk.weight, self.wv.weight], dim=0)
        else:
            # 推理态（eval/验证/生成/延迟基准，含 inference_mode）：权重不变，
            # 融合权重按 (参数对象, _version) 缓存复用，省掉每层每次前向一份
            # (3C, C) 的 cat 分配与拷贝。缓存张量在 inference_mode(False)+no_grad
            # 下创建：既不在 inference 期现场留下 inference tensor（训练前向若
            # 误用会被 autograd 拒绝），也不搭建多余计算图。
            wq_w, wk_w, wv_w = self.wq.weight, self.wk.weight, self.wv.weight
            key = (wq_w, wk_w, wv_w, wq_w._version, wk_w._version, wv_w._version)
            ent = self._qkv_fused
            if ent is None or ent[0] != key:
                with torch.inference_mode(False), torch.no_grad():
                    fused = torch.cat([wq_w.detach(), wk_w.detach(), wv_w.detach()],
                                      dim=0).contiguous()
                ent = (key, fused)
                self._qkv_fused = ent
            w_qkv = ent[1]
        q, k, v = (F.linear(x, w_qkv)
                   .view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4).unbind(0))  # (B, H, T, Dh)
        # 门控的“查询内容”必须使用 RoPE 之前的 q：RoPE 会把绝对位置混入 q，
        # 若用旋转后的 q 计算 w·q，会让本应只负责长度校准的 log n 与位置相互混淆。
        q_content = q

        if rope is not None:
            q, k = rope(q, k, past_len=past_len)
        if kv_cache is not None:
            # 静态缓存：就地写入，attention 在已写入的前缀视图上计算
            end = cache_pos + T
            kv_cache[0][:, :, cache_pos:end] = k
            kv_cache[1][:, :, cache_pos:end] = v
            k = kv_cache[0][:, :, :end]
            v = kv_cache[1][:, :, :end]
            cache = kv_cache
        else:
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
            cache = (k, v)  # KV 缓存不含 null 槽（null 每次前向临时追加）
        n_real = k.shape[2]
        S = n_real   # null 槽不再拼接进 K/V：显式路径把空槽得分追加到打分矩阵
                     # 末尾单独合成输出；SDPA 路径在分支内临时拼接（见下）。

        # 集中度门控（lega/lega_no_len/lega_no_query）必须消费注意力概率统计
        # max_j p_j，无法走 SDPA；qgate/lega_no_conc 的门只依赖查询与 log n，
        # 即便 need_stats=True（轻量门控诊断）也可走 SDPA——统计项 qterm/g/logit
        # 由 q_content 与 log n 单独重组，argmax 等完整诊断仍需显式路径。
        full_stats = bool(need_stats) and stats_argmax
        single_q = (T == 1)
        use_sdpa = (self.use_sdpa and not full_stats
                    and not (self.variant in _USE_CONC)
                    and not (self.variant == "null" and single_q))
        # is_causal=True 快路（无掩码全量前向）由 SDPA 内部保证因果性，
        # 无需构建显式掩码。
        # 静态 KV 缓存的整段预填充（cache_pos=0）同样满足"无任何历史"，
        # 注意力只在已写入的 T 个前缀键上做全因果，可同走 is_causal 快路。
        causal_only = (use_sdpa and past_kv is None
                       and (kv_cache is None or cache_pos == 0)
                       and self.variant in ("standard", "qgate", "lega_no_conc"))

        # 显式分块路径只使用逐块对角带掩码（全局共享的严格上三角小掩码），
        # 完全不需要 (1,1,T,S) 整表掩码：此时无需构建并缓存这份 T*S 布尔表
        # （每个新的 (past_len, T, S) 组合各驻留一份，直至 _CACHE_CAP）。
        elems = B * H * T * S
        will_chunk = (not use_sdpa and T >= _CHUNK_MIN_T
                      and elems >= _CHUNK_TARGET_ELEMS
                      and not full_stats
                      and not (self.training and self.attn_dropout > 0))

        # 因果掩码：真实键 j 仅当 j <= 查询绝对位置时可见；null 槽对全部查询可见。
        # 单查询步（T==1 的增量解码）：键数 n_real = past_len + 1，查询绝对位置
        # 恰为 past_len，全部键（含 null 槽）必然可见，无需构建掩码。
        allowed = not_allowed = mask_entry = None
        if not single_q and not causal_only and not will_chunk:
            mkey = (past_len, T, S, dev)
            entry = _MASK_CACHE.get(mkey)
            if entry is None:
                # 缓存中的常数张量必须创建于普通模式。若在 inference_mode
                # （验证/生成）下创建并被随后的训练前向复用，autograd 会拒绝保存
                # inference tensor。这里显式退出 inference_mode，仅用于创建不依赖
                # 当前输入的缓存常量。
                with torch.inference_mode(False):
                    qpos = torch.arange(past_len, past_len + T, device=dev)
                    not_allowed = qpos.view(1, 1, T, 1) < torch.arange(n_real, device=dev).view(1, 1, 1, n_real)
                    # null 槽常数可见列随掩码一起缓存，避免每层每步 ones 列的重复分配
                    entry = [not_allowed,
                             torch.ones(1, 1, T, 1, dtype=torch.bool, device=dev),
                             None, None]  # allowed / 拼 null 槽的 allowed，惰性填充
                if len(_MASK_CACHE) < _CACHE_CAP:   # 解码场景键持续变化，限制缓存规模
                    _MASK_CACHE[mkey] = entry
            not_allowed, ones_col = entry[0], entry[1]
            mask_entry = entry
            # 不 expand：masked_fill / SDPA 均支持 (1, 1, T, S) 广播

        stats: Optional[dict] = None
        g = qterm = logit = max_p = log_n = None

        if use_sdpa:
            # 无诊断需求的路径：SDPA（含 null/qgate/lega_no_conc 变体）
            if causal_only:
                o = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=True)
            else:
                # allowed 只在此分支需要：惰性取反一次后随 not_allowed 一起缓存，
                # 避免每层每次前向对同一张 (1,1,T,S) 掩码重复按位取反/拼接。
                # 单查询步（T==1）不建掩码，此处 mask_entry 为 None，attn_mask 传 None。
                if mask_entry is None:
                    allowed = None
                else:
                    if self.variant == "null":
                        if mask_entry[3] is None:
                            with torch.inference_mode(False):
                                mask_entry[3] = torch.cat([~mask_entry[0], ones_col], dim=3)
                        allowed = mask_entry[3]
                    else:
                        if mask_entry[2] is None:
                            with torch.inference_mode(False):
                                mask_entry[2] = ~mask_entry[0]
                        allowed = mask_entry[2]
                if self.variant == "null":
                    # SDPA 只接受单张量 K/V，这里一次性拼接（每前向仅一次，
                    # 逐 token 解码已在上面分流到显式路径）
                    kn = self.k_null.view(1, H, 1, Dh).to(k.dtype).expand(B, H, 1, Dh)
                    vn = self.v_null.view(1, H, 1, Dh).to(v.dtype).expand(B, H, 1, Dh)
                    k = torch.cat([k, kn], dim=2)
                    v = torch.cat([v, vn], dim=2)
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed,
                    dropout_p=self.attn_dropout if self.training else 0.0)
            if self.variant in _GATED:
                # 与显式路径相同的门控公式（qgate: w·q+d；lega_no_conc: w·q+b·log n+d）
                if self.use_len:
                    log_n = _log_n(past_len, T, dev)
                g, logit, qterm = self._gate_logit(q_content, None, log_n,
                                                   keep_terms=bool(need_stats))
                if torch.is_grad_enabled():
                    # SDPA 输出的反向依赖该张量本身，不能原地乘
                    o = o * g.to(o.dtype).unsqueeze(-1)
                else:
                    # 推理路径：o 是本次前向新产出张量，可安全原地乘
                    o.mul_(g.to(o.dtype).unsqueeze(-1))
            if need_stats:
                # 轻量门控诊断：g/qterm/logit 均可从查询与 log n 重组，
                # max_p/argmax 等依赖注意力概率的完整统计为 None
                stats = {"max_p": None, "log_n": log_n, "argmax": None,
                         "qterm": qterm, "g": g, "logit": logit}
        else:
            # 显式注意力路径：gated / 需要诊断统计时。
            # 长上下文且不需要 argmax/p_null 完整诊断时，按查询块计算：
            # 前向保持 (B,H,Cq,S) 块级中间量，训练态配合逐块 checkpoint。
            # 逐块 softmax 覆盖完整键行，数学不变。
            # attn_dropout>0 的训练步保持原来的单块路径：分块会改变 dropout 的
            # RNG 消耗粒度，虽无系统性偏差，但不逐位一致。
            use_chunk = will_chunk
            # 一次性预缩放 q：分块与单块共用。rope 启用时其输出是本层独占的新
            # 张量（旋转不依赖原地值反向），可直接 mul_；
            # rope 未启用时 q 与门控内容向量 q_content 共享存储，必须 out-of-place。
            if q is q_content:
                q = q * self.scale
            else:
                q.mul_(self.scale)
            if use_chunk:
                need_logn = self.variant in _LEGA_FAMILY or bool(need_stats)
                log_n = None
                if self.use_len or (need_logn and self.variant in _LEGA_FAMILY):
                    log_n = _log_n(past_len, T, dev)
                collect = bool(need_stats)
                o, max_p, qterm, logit, g = self._attend_chunked(
                    q, q_content, k, v, past_len, log_n, B, H, T, collect)
            else:
                # q 已在分块循环外统一预乘 1/sqrt(Dh)；argmax/softmax 对正缩放不敏感。
                att = torch.matmul(q, k.transpose(-2, -1))             # (B, H, T, S)
                if not_allowed is not None:
                    att.masked_fill_(not_allowed, float("-inf"))
                # null 槽的两种合成方式（与 _attend_q_chunk 相同的解析合并）：
                #   * 需要完整联合 softmax 的场景（null 的 argmax/p_null 诊断、
                #     训练态 dropout 的 RNG 消耗口径随列数变化）仍 cat 出 S+1 列；
                #   * 无完整统计需求的训练/推理前向：p_null = sigmoid(a_n - lse)，
                #     softmax 只覆盖真实键，省掉逐次前向的 (B,H,T,S) cat 拷贝——
                #     逐 token 解码（T==1 走此路径）时省掉每层每步的整行复制。
                p_null = None
                if self.variant == "null":
                    att_n = torch.matmul(q, self.k_null.to(q.dtype).unsqueeze(-1))
                    if full_stats or (self.training and self.attn_dropout > 0):
                        att = torch.cat([att, att_n], dim=-1)          # (B,H,T,S+1)
                    else:
                        # logsumexp 内部做 max 平移，数值稳定；
                        # 注意是 a_n - lse（sigmoid(lse - a_n) = 1 - p_null）
                        p_null = torch.sigmoid(
                            att_n - att.logsumexp(dim=-1, keepdim=True))
                        del att_n
                # 只有完整诊断统计（非 null 变体的 argmax）还需要原始 logit。
                # argmax 落点是 softmax 不变量（softmax 的反向只依赖输出 p），
                # 故在 softmax 前从 att 提取即可，随后 att 可释放。
                argmax_stat = None
                if bool(need_stats) and stats_argmax and self.variant != "null":
                    argmax_stat = att[..., :n_real].argmax(dim=-1)
                p = F.softmax(att, dim=-1)
                del att

                # 门控所需的注意力统计（LEGA 家族训练时也必须计算）。
                # max_j p_j 直接取 softmax 输出的行最大值（在 dropout 之前，与原基于
                # softmax logit 的实现语义一致）。stats_argmax=False（轻量诊断，如
                # 验证评测只需要 g）时不计算 argmax；不含集中度项的变体（qgate 等）
                # 此时也不必计算 max_p/log_n。
                full_stats = bool(need_stats) and stats_argmax
                # log n 仅在门控需要长度项(use_len)或完整诊断(full_stats)时需要；
                # 如 lega_no_len 的训练前向（use_len=False 且无统计需求），不计算。
                need_logn = self.use_len or full_stats
                if need_logn:
                    log_n = _log_n(past_len, T, dev)
                # null 变体虽无 null 槽时 p 恰好覆盖全部真实键；
                # null 变体的完整统计在下方 need_stats 分支内单独处理。
                # max_p 是集中度门控（lega/lega_no_len）训练前向的必需输入，
                # 与 log_n 是否被需要（use_len/full_stats）无关，不能随 need_logn 跳过。
                if self.variant != "null" and (self.use_conc or full_stats):
                    max_p = p.amax(dim=-1).float()                    # max_j p_j ∈ (0, 1]

                # 诊断统计（null 变体的 max_p/argmax/p_null 等）必须基于 dropout 之前的
                # softmax 分布，否则会因 dropout 的 1/(1-p) 重标定与置零而失真；
                # dropout 只作用于实际输出路径，因此保留一份 dropout 前的引用。
                p_pre = p
                if self.attn_dropout > 0 and self.training:
                    p = F.dropout(p, p=self.attn_dropout)
                if self.variant == "null" and p_null is None:
                    # 联合 softmax 全列路径：o = p_real @ v + p_null * v_null
                    o = torch.matmul(p[..., :n_real], v)               # (B, H, T, Dh)
                    o.addcmul_(p[..., n_real:], self.v_null.to(v.dtype).view(1, H, 1, Dh))
                else:
                    o = torch.matmul(p, v)                             # (B, H, T, Dh)
                if p_null is not None:
                    # 解析合并路径：真实键部分乘联合尺度因子 (1-p_null)，
                    # 再加 p_null * v_null
                    o.mul_(1 - p_null)
                    o.addcmul_(p_null, self.v_null.to(v.dtype).view(1, H, 1, Dh))
                del p

                # 输出门控（惰性累加：仅偏置项 (1,H,1) 参与广播）
                if self.variant in _GATED:
                    g, logit, qterm = self._gate_logit(q_content, max_p, log_n,
                                                       keep_terms=bool(need_stats))
                    # 原地乘法：o 的全部反向依赖 p/v，mul_ 自身被 autograd 正确处理
                    o.mul_(g.to(o.dtype).unsqueeze(-1))                   # o' = g * o

            if need_stats:
                stats = {
                    "max_p": max_p,
                    "log_n": log_n,     # need_stats=True 时上方必然已计算（轻量模式下未用项为 None）
                    "argmax": None,     # 轻量模式不计算 argmax
                    "qterm": qterm,
                    "g": g,
                    "logit": logit,
                }
                if stats_argmax:
                    if self.variant == "null":
                        # null 变体的统计必须基于含 null 槽的完整 softmax，
                        # 否则会虚增集中度并错误地把 null 槽落点记为真实键
                        stats["max_p"] = p_pre[..., :n_real].amax(dim=-1).float()
                        amax_full = p_pre.argmax(dim=-1)
                        stats["argmax"] = torch.where(amax_full < n_real, amax_full,
                                                      torch.full_like(amax_full, -1))
                        stats["p_null"] = p_pre[..., -1].float()          # 分给 null 槽的概率质量
                    else:
                        # argmax 已在 softmax 前提取（softmax 不变量，真实键内相对位置）
                        stats["argmax"] = argmax_stat

        # 合并多头: (B, H, T, Dh) -> (B, T, C)；分块路径已按 (B,T,H,Dh) 布局
        # 直接写出，其 o 本身就是合并后的 (B,T,C)
        if o.dim() == 4:
            o = o.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(o)
        return out, cache, stats
