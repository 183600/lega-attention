"""核心模块：带"长度校准证据门控"的因果多头自注意力（LEGA）与全部对照变体。

对每个查询位置 t、每个头 h，注意力输出 o 与门控写回规则：

    standard      :  o' = o                                        （标准注意力）
    qgate         :  g = sigmoid(w·q + d)                          （已有工作：只看查询的输出门控）
    null          :  额外可学习 null 键值槽，注意力可整体"不读取"    （已有工作：显式空注意力）
    lega (ours)   :  c = log(n * max_j p_j)
                     g = sigmoid(w·q + a*c + b*log n + d)
                     o' = g * o
    lega_no_len   :  g = sigmoid(w·q + a*c + d)                    （消融：去掉长度项 b*log n）
    lega_no_conc  :  g = sigmoid(w·q + b*log n + d)                （消融：去掉集中度项 a*c）
    lega_no_query :  g = sigmoid(a*c + b*log n + d)                （消融：去掉查询项）
    lega_conc_only:  g = sigmoid(a*c + d)                          （全因子消融：仅集中度项 B）
    lega_len_only :  g = sigmoid(b*log n + d)                      （全因子消融：仅长度项 C）
    lega_placebo  :  g = sigmoid(w·q + a*u + b*v + d)              （对照：u/v 为确定性
                     伪随机特征，参数个数/门控 FLOPs 与 lega 相同，但不含任何
                     查询/注意力/长度信息；用于隔离“多几个门控参数/一次门控
                     运算”的训练期优化红利）
    lega_raw      :  g = sigmoid(w·q + a*log max_p + b*log n + d)  （补充对照：
                     原始最大概率对数，不做 c 的长度归一化；与 lega 的函数
                     表达能力相同，用于区分“重参数化”与“额外容量”）
    lega_raw_no_len: g = sigmoid(w·q + a*log max_p + d)            （补充对照：
                     原始 log max_p 且无显式长度项）
    lega_raw_conc_only: g = sigmoid(a*log max_p + d)               （原始参数化
                     全因子 B 格：仅原始集中度、无查询/长度项）
    lega_raw_no_query : g = sigmoid(a*log max_p + b*log n + d)     （原始参数化
                     全因子 B+C 格：集中度与长度，无查询项）
    lega_detach_conc: g = sigmoid(w·q + a*detach(c) + b*log n + d) （补充对照：
                     集中度统计只作为前向特征、不向 Q/K 反传）
    null_zero     :  与 null 相同，但 value 槽固定为 0、不参与训练（补充对照：
                     把“选择 null”与“写入学习到的内容”分开）

全因子消融矩阵（门控内 2^3：无=gate_only, A=查询, B=集中度, C=长度）：
    无   -> gate_only        A    -> qgate          B   -> lega_conc_only
    C    -> lega_len_only    A+B  -> lega_no_len    A+C -> lega_no_conc
    B+C  -> lega_no_query    A+B+C-> lega
standard 是全因子范围外的无门控基线（用于“LEGA vs 标准注意力”的整体对比），
不能当作 2^3 的 none 格，否则组件主效应会混入“加不加输出门”。
去一法（leave-one-out）：Full=lega, Full-A=lega_no_query,
Full-B=lega_no_conc, Full-C=lega_no_len。两套消融均被实现，可互相印证。

其中 p_j 为因果注意力权重，n 为该查询可读取的真实 token 数（null 槽不计入 n）。
q 特指 RoPE 旋转前的查询内容向量；注意力打分仍使用旋转后的 q/k。

重要边界（不要过度解读 c）：均匀注意力时 max_j p_j = 1/n，因此
c = log(n * 1/n) = 0，**只证明均匀分布这一特例与长度无关**。一般情况下
c = log n + log max_p，若 max_p 不随 n 按 1/n 收缩，c 仍会随长度增长。
因此“c 已经消除长度混淆”不是可仅靠单元测试成立的命题；必须用随机上下文、
重复上下文和不同打分方差下的 c 零假设分布来验证（见 lega.length_null）。

重参数化：完整门控可展开为 w·q + a·log max_p + (a+b)·log n + d，与
“查询 + 原始最大概率对数 + 长度对数”的线性门控同属一个函数族。归一化 c
改变的是参数化/优化几何与解释，并不增加表达能力；lega_no_len 保留 a·c，
所以 b=0 检验的是“集中度系数与长度系数相等”，不是“去掉长度信息”。
lega_raw / lega_raw_no_len 是为此增加的直接对照；补齐
lega_raw_conc_only / lega_raw_no_query 后，原始参数化也构成同构的完整
2^3 网格（none=gate_only, A=qgate, B=raw_conc_only, C=len_only,
AB=raw_no_len, AC=no_conc, BC=raw_no_query, ABC=raw），用于检验
A/B/C 的主效应、交互与 Shapley 归因是否依赖 c 的归一化选择。

初始时 w=a=b=0、d=2.0，g ≈ 0.88，接近标准注意力，保证训练稳定启动。
"""
import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils import config_finite_float, config_probability

# 主实验变体（配置/命令行默认使用；与预注册的 2^3 门控矩阵一致）
VARIANTS = ("standard", "gate_only", "qgate", "null", "lega", "lega_no_len",
            "lega_no_conc", "lega_no_query", "lega_conc_only", "lega_len_only")

# 额外的对照变体：不进入主矩阵/默认 run，但可被训练、评测与汇总。
# lega_placebo 是训练期的**参数匹配安慰剂**：门控结构与 LEGA 完全相同
# （w·q + a·u + b·v + d，u/v 为逐位置/逐头的确定性伪随机特征），
# 参数个数、门控 FLOPs、初始化与优化器分组都与 LEGA 一致，但 u/v 不携带
# 任何查询/注意力/长度信息。它回答“LEGA 的收益是否只是多了几个门控参数/
# 一次额外门控运算带来的优化红利”，而不是“是否用到了 B/C 证据”。
CONTROL_VARIANTS = ("lega_placebo",)
# 补充对照：不进入主矩阵/默认 run，但用于回答审稿人要求的“重参数化、
# detach 反传路径、固定零 value null”三类直接对照。其中 raw_* 用于构成
# 原始集中度参数化的完整 2^3（与归一化网格逐格配对），但它们不属于
# 归一化 A/B/C 因子设计，因此不进入 GATE_FACTORS。
SUPPLEMENTAL_VARIANTS = ("lega_raw", "lega_raw_no_len",
                         "lega_raw_conc_only", "lega_raw_no_query",
                         "lega_detach_conc", "null_zero")
RUNNABLE_VARIANTS = VARIANTS + CONTROL_VARIANTS + SUPPLEMENTAL_VARIANTS

# 供图表与汇总表使用的中文名
VARIANT_CN = {
    "standard": "标准Attention",
    "gate_only": "仅门控偏置(无A/B/C)",
    "qgate": "查询门控(已有)",
    "null": "空注意力(已有)",
    "lega": "LEGA(ours)",
    "lega_no_len": "LEGA去长度项",
    "lega_no_conc": "LEGA去集中度项",
    "lega_no_query": "LEGA去查询项",
    "lega_conc_only": "仅集中度门控",
    "lega_len_only": "仅长度门控",
    "lega_placebo": "LEGA参数匹配安慰剂(证据=噪声)",
    "lega_raw": "LEGA原始log max_p(不归一化)",
    "lega_raw_no_len": "原始log max_p且无长度项",
    "lega_raw_conc_only": "仅原始log max_p门控",
    "lega_raw_no_query": "原始集中度+长度(无查询)",
    "lega_detach_conc": "集中度统计不反传到Q/K",
    "null_zero": "空注意力(固定零value)",
}

# 各公式项出现的变体
_USE_QUERY = ("qgate", "lega", "lega_no_len", "lega_no_conc",
              "lega_placebo", "lega_raw", "lega_raw_no_len",
              "lega_detach_conc")                                  # w·q 项
_USE_CONC = ("lega", "lega_no_len", "lega_no_query",
             "lega_conc_only",
             "lega_placebo", "lega_detach_conc")                   # a*c, c=log(n*max_j p_j)
_USE_RAW_CONC = ("lega_raw", "lega_raw_no_len",
                 "lega_raw_conc_only", "lega_raw_no_query")        # a*log max_p（不归一化）
_USE_LEN = ("lega", "lega_no_conc", "lega_no_query",
            "lega_len_only",
            "lega_placebo", "lega_raw", "lega_raw_no_query",
            "lega_detach_conc")                                    # log n 项
# 有输出门
_GATED = ("gate_only", "qgate", "lega", "lega_no_len", "lega_no_conc",
          "lega_no_query", "lega_conc_only", "lega_len_only",
          "lega_placebo", "lega_raw", "lega_raw_no_len",
          "lega_raw_conc_only", "lega_raw_no_query",
          "lega_detach_conc")
# 门依赖注意力统计（需要 max_j p_j）
_LEGA_FAMILY = ("lega", "lega_no_len", "lega_no_query", "lega_conc_only",
                "lega_placebo", "lega_raw", "lega_raw_no_len",
                "lega_raw_conc_only", "lega_raw_no_query",
                "lega_detach_conc")
# 固定零 value 的 null 对照：语义上仍是“空注意力”，但 value 槽不学习。
_IS_NULL = ("null", "null_zero")

# 门控内三因子 A/B/C 的激活表（单一事实来源）：直接从上面的
# _USE_QUERY/_USE_CONC/_USE_LEN 派生，避免“实验设计元数据”与实现漂移。
# 消融对比是否只改变一个因子，由 run_all 的单因素审计消费；standard/null
# 不属于门控内 2^3 消融，故不出现在此表中。参数匹配安慰剂虽然形式上带
# A/B/C 三个槽位，但不是任何一个真实证据因子，必须排除在组件归因之外。
GATE_FACTORS = {
    v: frozenset(f for f, enabled in (("A", v in _USE_QUERY),
                                      ("B", v in _USE_CONC),
                                      ("C", v in _USE_LEN)) if enabled)
    for v in _GATED
    if v not in CONTROL_VARIANTS and v not in SUPPLEMENTAL_VARIANTS
}

# 显式注意力按查询维分块计算 (B,H,T,S) 的打分/概率；训练态配合逐块 checkpoint。
_CHUNK_TARGET_ELEMS = 8_000_000
_CHUNK_MIN_T = 64

# 训练态逐块 checkpoint 的使用下限：低于该值时改走不重算的分块路径。
_CKPT_MIN_BYTES = 128 * 1024 * 1024


def _chunk_elems(dev: torch.device) -> int:
    """当前设备每块打分矩阵的目标元素数（CPU 与 GPU 取值不同）。"""
    if dev.type == "cpu":
        return max(1, _CHUNK_TARGET_ELEMS // 4)
    return _CHUNK_TARGET_ELEMS


def _ckpt_worthwhile(B: int, H: int, T: int, S: int, dtype: torch.dtype) -> bool:
    """按反向实际保留的软概率矩阵估算逐块 checkpoint 是否达到启用下限。

    训练态显式路径的 softmax 反向依赖其输出，保留的是输出概率本身：AMP
    （autocast）与 CPU 下 softmax 强制升 fp32，因此即使输入为 fp16/bf16，
    反向也按 fp32 保留完整 (B,H,T,S)（4 字节/元素）。估算与保存口径一致。
    """
    elem = max(4, torch.empty((), dtype=dtype).element_size())
    return B * H * T * S * elem >= _CKPT_MIN_BYTES


# 进程级共享缓存：各层（乃至各实例）在同一设备/长度下的因果掩码完全相同，
# log n 由设备级前缀表统一切片，按模块共享。
# 键内已含 device，跨设备天然隔离；长度变化场景（解码）由容量上限兜底。
# 只构造 not_allowed（显式路径直接用）；allowed 仅 SDPA 带掩码的少见分支需要，
# 用时再按位取反。
_MASK_CACHE: OrderedDict = OrderedDict()   # (past_len, T, S, device) ->
                                           # [not_allowed, ones_col, allowed, allowed+null]
# allowed/allowed+null 惰性填充并与 not_allowed 一并保存，供 SDPA 带掩码分支直接使用。
# 缓存条目按条目数与总字节数双重上限做 LRU 驱逐。
_MASK_CACHE_BYTES = 0
_MASK_CACHE_MAX_BYTES = 128 * 1024 * 1024
# 设备端 log 前缀表：table[n] = log(n+1)，任意 (past_len, T) 的 log n 恒为其
# [past_len, past_len+T) 切片。按需倍增扩展。
_LOGN_TABLE: dict = {}   # device -> (N,) float
_LOGN_TABLE_INIT = 4096
# 查询分块路径使用的三角对角带掩码：块 (r, c) 内掩码只依赖相对偏移（c > r），
# 与 past 偏移无关。
_TRI_CACHE: OrderedDict = OrderedDict()  # (r, c, device) -> (1, 1, r, c) bool
_TRI_CACHE_BYTES = 0
_TRI_CACHE_MAX_BYTES = 64 * 1024 * 1024
_CACHE_CAP = 256


def _mask_entry_bytes(entry) -> int:
    return sum(t.numel() * t.element_size() for t in entry if t is not None)


def _enforce_mask_cache_bounds() -> None:
    global _MASK_CACHE_BYTES
    while _MASK_CACHE and (len(_MASK_CACHE) > _CACHE_CAP
                           or _MASK_CACHE_BYTES > _MASK_CACHE_MAX_BYTES):
        _, old = _MASK_CACHE.popitem(last=False)
        _MASK_CACHE_BYTES -= _mask_entry_bytes(old)


def _mask_cache_store(key, entry) -> None:
    """按 LRU 插入掩码；超上限的超大掩码仅返回本次使用。"""
    global _MASK_CACHE_BYTES
    if _MASK_CACHE.get(key) is not None:
        return
    _MASK_CACHE[key] = entry
    _MASK_CACHE_BYTES += _mask_entry_bytes(entry)
    _enforce_mask_cache_bounds()


def _mask_cache_grow(key, entry, tensor) -> None:
    """惰性掩码加入已缓存条目时同步记账；条目已被驱逐则跳过。"""
    global _MASK_CACHE_BYTES
    if _MASK_CACHE.get(key) is not entry:
        return
    _MASK_CACHE_BYTES += tensor.numel() * tensor.element_size()
    _MASK_CACHE.move_to_end(key)
    _enforce_mask_cache_bounds()


def _enforce_tri_cache_bounds() -> None:
    global _TRI_CACHE_BYTES
    while _TRI_CACHE and (len(_TRI_CACHE) > _CACHE_CAP
                          or _TRI_CACHE_BYTES > _TRI_CACHE_MAX_BYTES):
        _, old = _TRI_CACHE.popitem(last=False)
        _TRI_CACHE_BYTES -= old.numel() * old.element_size()


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

    通过设备端 log 前缀表切片得到，与原逐次 arange+log 的结果逐项一致（同一
    查表公式 log(i+1)），任意 (past_len, T) 组合共享同一份表。
    表以 inference_mode(False) 创建，不会被推理期现场污染 autograd。
    """
    return _log_n_table(dev, past_len + T)[past_len:past_len + T]


def _placebo_evidence(x: torch.Tensor, n_heads: int, salt: float,
                      pos_offset: int = 0) -> torch.Tensor:
    """参数匹配安慰剂使用的确定性“伪证据”特征。

    返回与 ``x`` 同形状、取值在 [-1, 1] 的高频伪随机数（由 **绝对** token
    位置与头索引哈希得到）。它不是任何真实查询/注意力/长度的函数，也不
    消耗全局 RNG，因此：

    * 门控参数个数、前向 FLOPs、初始化与优化器分组均与 LEGA 相同；
    * 训练期不会因为随机流被额外扰动而破坏“同 seed、同数据顺序”的公平性；
    * 推理期全序列/逐 token 解码都不依赖 RNG，同形状调用可复现。

    单个标量系数 ``a``/``b`` 无法用这种高频特征逼近 ``log n`` 或集中度，
    因此该变体隔离的是“多出的门控容量/优化自由度”，而不是真实的 B/C 证据。
    """
    if x.numel() == 0:
        return x
    T = int(x.shape[-1])
    dev = x.device
    pos_offset = int(pos_offset)
    with torch.no_grad():
        # 伪证据必须是绝对位置的函数：全量前向的 0..T-1 与 KV 缓存逐 token
        # 解码/分块预填的局部 0..T-1 不能混为一谈，否则同一 checkpoint 在
        # 不同 prefill/decode 切分下会得到不同的门控与 logits。
        pos = torch.arange(pos_offset, pos_offset + T, device=dev,
                           dtype=torch.float32).view(1, 1, T)
        head = torch.arange(int(n_heads), device=dev,
                            dtype=torch.float32).view(1, -1, 1)
        z = torch.sin((pos + 1.0) * (12.9898 + salt)
                      + (head + 1.0) * (78.233 + 0.37 * salt)) * 43758.5453
        z = z - torch.floor(z)
        z = z * 2.0 - 1.0
    if x.dim() == 1:
        # 与真实 log n 一致：逐位置标量（b 再把它变成逐头项）。
        z = z.view(int(n_heads), T)[0].view(T)
    elif x.dim() == 3 and int(x.shape[-2]) == int(n_heads):
        # c: (B,H,T)，保留头间差异。
        z = z.view(1, int(n_heads), T)
    else:
        # 任意广播形状（如 (1,1,T)/(B,T)）：逐位置标量，避免把头维错配。
        z = z.view(int(n_heads), T)[0].view(
            *([1] * (x.dim() - 1)), T)
    return z.to(dtype=x.dtype).expand_as(x)


def _ensure_mm_input(t: torch.Tensor) -> torch.Tensor:
    """仅当批量 matmul 无法直接消费时才连续化。

    判定标准：每个 (m, n) 矩阵行主序（末维 stride 1、次末维 stride 恰为行宽），
    批量维步长任意——strided batched GEMM 直接接受。permute().unbind() 之类
    在矩阵维上留下跨步的视图才需要复制；(B,H,S,Dh) 连续张量或其序列维前缀
    切片（如静态 KV 缓存 [:, :, :end]）已满足条件。
    """
    if t.stride(-1) == 1 and (t.size(-1) <= 1 or t.stride(-2) == t.size(-1)):
        return t
    return t.contiguous()


def _tri_band(r: int, c: int, dev) -> torch.Tensor:
    """(1, 1, r, c) 严格上三角布尔掩码：[row, col] = (col > row)。"""
    global _TRI_CACHE_BYTES
    key = (r, c, dev)
    m = _TRI_CACHE.get(key)
    if m is not None:
        _TRI_CACHE.move_to_end(key)
        return m
    with torch.inference_mode(False):  # 缓存常量不能是 inference tensor
        m = torch.ones(1, 1, r, c, dtype=torch.bool, device=dev).triu_(1)
    _TRI_CACHE[key] = m
    _TRI_CACHE_BYTES += m.numel() * m.element_size()
    _enforce_tri_cache_bounds()
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
        assert variant in RUNNABLE_VARIANTS, f"未知变体: {variant}"
        attn_dropout = config_probability(attn_dropout, "attn_dropout")
        gate_bias_init = config_finite_float(gate_bias_init, "gate_bias_init")
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
        self.use_raw_conc = variant in _USE_RAW_CONC
        self.use_len = variant in _USE_LEN
        self.is_null = variant in _IS_NULL
        self.detach_conc = variant == "lega_detach_conc"

        Dh = self.d_head
        if self.use_query:
            self.gate_w = nn.Parameter(torch.zeros(n_heads, Dh))      # w
        if self.use_conc or self.use_raw_conc:
            self.gate_a = nn.Parameter(torch.zeros(n_heads))          # a
        if self.use_len:
            self.gate_b = nn.Parameter(torch.zeros(n_heads))          # b
        if variant in _GATED:
            self.gate_d = nn.Parameter(torch.full((n_heads,), float(gate_bias_init)))  # d
        if self.is_null:
            # 构造期初始化不能消耗全局 RNG：否则 apply(_init_weights) 重初始化全部
            # 共享参数时 RNG 状态已被 null 变体的 randn 挪动，导致 null 与
            # 其它变体在同 seed 下基础权重不一致，破坏"同一 seed 公平对比"。
            # GPT.__init__ 随后会调 init_gate_params_ 用全局 RNG 重新逐层随机化。
            gen = torch.Generator().manual_seed(0)
            self.k_null = nn.Parameter(
                torch.randn(n_heads, Dh, generator=gen) * 0.02)
            if variant == "null_zero":
                # 固定零 value：注册为 buffer，优化器不会更新；null 键仍可
                # 学习（“何时读空槽”与“空槽写入什么”由此解耦）。
                self.register_buffer("v_null", torch.zeros(n_heads, Dh))
            else:
                self.v_null = nn.Parameter(torch.zeros(n_heads, Dh))

        # 因果掩码 / log_n 分别由模块级 _MASK_CACHE / 设备级 _LOGN_TABLE 提供，
        # 全层共享。
        # 推理态融合 QKV 权重缓存：(key, tensor)；key 为三个权重参数对象及其
        # _version。训练态不走缓存（cat 必须参与 autograd 图），且优化器/测试
        # 对权重的原地修改会递增 _version 使缓存自动失效。
        self._qkv_fused = None
        # 只在推理期病灶分析中临时打开的分布匹配置换：把 c 在序列维
        # roll 一个非零位移，保持逐值分布不变、破坏位置对应关系。
        # 训练/普通推理恒为 None，不进入 state_dict。
        self.conc_permute_shift = None
        # 推理态参数 dtype 转换缓存（与 _qkv_fused 同失效键口径）：
        # gate_w / (k_null, v_null)。
        self._gate_w_cast = None
        self._null_cast = None

    def _cast_cached(self, slot: str, params: tuple, dtype):
        """推理态缓存参数的 dtype 转换结果；训练态每次即时 .to 以参与 autograd。

        键含（参数对象身份, _version, data_ptr, 目标 dtype），与 _qkv_fused
        相同：优化器/测试对权重的原地修改、.to()/.double()/param.data= 替换
        都会使缓存自动失效，不会复用陈旧张量。
        """
        if torch.is_grad_enabled():
            return tuple(p.to(dtype) for p in params)
        key = tuple(v for p in params
                    for v in (p, p._version, p.data_ptr(), dtype))
        ent = getattr(self, slot)
        if ent is not None:
            o_key = ent[0]
            # 逐项（含对象身份）比较，避免元组 != 在 Tensor 上触发
            # "Boolean value of Tensor is ambiguous"
            if (len(o_key) == len(key) and all(
                    (o is n) if isinstance(o, torch.Tensor) else o == n
                    for o, n in zip(o_key, key))):
                return ent[1]
        with torch.inference_mode(False), torch.no_grad():
            casted = tuple(p.detach().to(dtype).contiguous() for p in params)
        setattr(self, slot, (key, casted))
        return casted

    def init_gate_params_(self, gate_bias_init: float = 2.0) -> None:
        """显式初始化门控参数（在 GPT 的整体初始化之后调用）。"""
        gate_bias_init = config_finite_float(gate_bias_init, "gate_bias_init")
        if self.use_query:
            nn.init.zeros_(self.gate_w)
        if self.use_conc or self.use_raw_conc:
            nn.init.zeros_(self.gate_a)
        if self.use_len:
            nn.init.zeros_(self.gate_b)
        if self.variant in _GATED:
            nn.init.constant_(self.gate_d, float(gate_bias_init))
        if self.is_null:
            nn.init.normal_(self.k_null, mean=0.0, std=0.02)
            if self.variant == "null_zero":
                self.v_null.zero_()
            else:
                nn.init.zeros_(self.v_null)

    def _gate_logit(self, qterm: Optional[torch.Tensor],
                    max_p: Optional[torch.Tensor],
                    log_n: Optional[torch.Tensor], bht: Tuple[int, int, int],
                    keep_terms: bool = False, pos_offset: int = 0):
        """按 variant 组合门控 logit；返回 (g, logit, qterm)，不适用项为 None。

        qterm 为 RoPE 旋转前的查询内容向量与门控权重 w 的归约结果
        （(B,H,T) 或 None），由调用方在对应路径上计算。
        keep_terms=False（训练/普通推理/轻量诊断）时各项累加进 logit，
        qterm/logit 不在返回中保留（返回 None）。
        keep_terms=True（完整机制分析）时 qterm 需原样返回给统计，logit 在其
        副本上累加。
        """
        H = self.n_heads
        logit = None
        if self.use_query:
            # qterm 已由调用方预归约并升到 fp32。
            logit = qterm.clone() if keep_terms else qterm
        # logit 累加"标量门控系数 × 统计项"（addcmul_）。
        if self.use_conc or self.use_raw_conc:
            # detach 对照：集中度只作为前向特征，不让 max_p 的梯度回流到
            # Q/K；其余参数仍正常训练。
            src = max_p.detach() if self.detach_conc else max_p
            # 无统计需求且 src 不参与反向（推理/统计仅前向消费）时，
            # clamp/log 原地完成；max_p 为 max/amax 的反向保存值，需要梯度
            # 时必须保持原值。clamp_min 的输出是一次性中间张量，在其上原地
            # 取 log 不影响反向（反向比较的是 clamp 的输入 max_p）。
            # detach 时 src 与 max_p 共享存储，绝不能原地改写，否则后续
            # 统计拿到的 max_p 已被 clamp/log 破坏。
            c = src.clamp_min(1e-30).log() \
                if (keep_terms or src.requires_grad or self.detach_conc) \
                else src.clamp_min_(1e-30).log_()
            # 归一化集中度项按定义是 c=log(n*max_j p_j)：即使 b*log n 被
            # 消融（lega_no_len / lega_conc_only），a*c 里的长度归一化也
            # 不能丢，否则 Full−C 实际多去掉了 a·log n，不再是单组件消融。
            # raw 对照则刻意保留 log max_p、不并入 log n。
            if self.use_conc:
                c = c + log_n
            shift = getattr(self, "conc_permute_shift", None)
            if shift is not None and c.shape[-1] > 1:
                # 病灶 conc_permute：只在序列维循环移位。逐位置 c 的多重集
                # 完全不变（均值/方差/长度分布匹配），但“哪个位置拿到哪个
                # 证据”被破坏；用于排除容量/尺度解释，而不是训练期变体。
                c = torch.roll(c, shifts=int(shift), dims=-1)
            if self.variant in CONTROL_VARIANTS:
                # 参数匹配安慰剂：保留 a 的形状与梯度路径，但把它乘的
                # “集中度证据”换成与真实统计无关的确定性噪声。
                c = _placebo_evidence(c, H, salt=1.0,
                                      pos_offset=pos_offset)
            a = self.gate_a.float().view(1, H, 1)
            logit = a * c if logit is None else logit.addcmul_(a, c)
        if self.use_len:
            len_term = log_n
            if self.variant in CONTROL_VARIANTS:
                # 与 c 相同：保留 b 的参数/计算，只把长度证据替换为噪声。
                len_term = _placebo_evidence(
                    log_n, H, salt=2.0, pos_offset=pos_offset)
            b = self.gate_b.float().view(1, H, 1)
            logit = b * len_term if logit is None else logit.addcmul_(b, len_term)
        dterm = self.gate_d.float().view(1, H, 1)
        if logit is None:
            # 防御分支：当前全部变体至少含一个门控项，logit 不会为 None。
            logit = dterm.expand(bht[0], H, bht[2]).clone()
        else:
            logit.add_(dterm)
            if logit.shape[0] != bht[0] or logit.shape[2] != bht[2]:
                # 仅含 log n 的项（lega_len_only）与 (1,H,1) 参数广播后批次维
                # 为 1；统计消费与原地 sigmoid 都要求与输入批次一致，物化补齐。
                logit = logit.expand(bht[0], H, bht[2]).clone()
        # keep_terms=False（训练/普通推理/轻量诊断）时调用方不消费 logit：
        # sigmoid 原地写入；此时 qterm/logit 已被原地累加/覆盖，
        # 返回 None 表示未保留。
        if keep_terms:
            return torch.sigmoid(logit), logit, qterm
        return logit.sigmoid_(), None, None

    def _gate_qterm(self, q_content: torch.Tensor,
                    gw: torch.Tensor) -> torch.Tensor:
        """门控查询项 w·q 的 (B,H,T) 归约，返回 fp32。

        统一用 einsum 归约（逐头点积）：对训练与推理都走同一条归约路径，
        保证两种模式数值一致。
        """
        return torch.einsum("bhtd,hd->bht", q_content, gw).float()

    def _softmax_reuse(self, att: torch.Tensor) -> torch.Tensor:
        if ((torch.is_grad_enabled() and att.requires_grad)
                or torch.compiler.is_compiling()):
            return F.softmax(att, dim=-1)
        return torch.softmax(att, dim=-1, out=att)

    def _attend_q_chunk(self, q_rot: torch.Tensor, kT: torch.Tensor, v: torch.Tensor,
                        band_start: int, band_mask: Optional[torch.Tensor],
                        collect_stats: bool, keep_gate_terms: bool = True,
                        null_c: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """单查询块的显式注意力；门控由调用方在整段上统一计算后施加。

        各查询位置的门控值只依赖该位置的 qterm/max_j p_j/log n，逐块与整段
        计算结果逐位一致，因此门控不在此处按块计算。

        collect_stats=False 返回 (o, max_p)；collect_stats=True 返回
        (o, max_p, argmax_stat, p_null_stat)。max_p 为 (B,H,C) fp32 的
        max_j p_j（null 变体已按联合 softmax 口径并入空槽），供调用方汇总后
        计算门控或作为统计返回。

        q_rot 是 RoPE 旋转后且已预乘 1/sqrt(Dh) 的查询（打分的 q·k）。

        kT/v 已被调用方裁到本块可见前缀（末列即块内最末查询可见的最后一键），
        全块不可见的尾列既不参与打分也不参与 softmax，数学上与先算全量再做整列
        -inf 屏蔽完全等价；块内只需对对角带
        [band_start, band_start+cq) 用平移不变的 (1,1,cq,cq) 三角掩码做
        masked_fill_。

        null 变体不把 null 槽拼接进 K/V，也不把空槽得分 cat 到打分矩阵末尾，
        而是解析合并：
          p_null = e^{a_n} / (Z + e^{a_n}) = sigmoid(a_n - logsumexp(att))
          o = (softmax(att) @ v) * (1 - p_null) + p_null * v_null
        与先 cat 再做联合 softmax 的语义逐项等价（softmax/cat 的尺度因子恰好
        为 1 - p_null）。
        """
        att = torch.matmul(q_rot, kT)                               # (B,H,C,S)，S=可见前缀长
        if band_mask is not None:
            att[..., band_start:band_start + band_mask.shape[-1]].masked_fill_(
                band_mask, float("-inf"))
        is_null = self.is_null
        k_null_c = v_null_c = None
        lse = None
        if is_null:
            # null 权重按块共享：调用方（分块路径）在循环外转换一次并传入。
            if null_c is None:
                null_c = self._cast_cached(
                    '_null_cast', (self.k_null, self.v_null), v.dtype)
            k_null_c, v_null_c = null_c
            att_n = torch.matmul(q_rot, k_null_c.unsqueeze(-1))
            # p_null = 1 / (1 + Z/e^{a_n})，logsumexp 内部做 max 平移，数值稳定；
            # 注意是 a_n - lse（sigmoid(lse - a_n) 得到的是 1 - p_null）
            lse = att.logsumexp(dim=-1, keepdim=True)
            p_null = torch.sigmoid(att_n - lse)
            del att_n
        # null 变体复用同一次 logsumexp 求 softmax：p = exp(att - lse)。
        if lse is None:
            p = self._softmax_reuse(att)
        elif (torch.is_grad_enabled() and att.requires_grad) \
                or torch.compiler.is_compiling():
            # 减法结果是本块独占的临时张量：exp_ 原地复用；
            # exp 的反向只依赖输出，对 autograd 安全。
            p = (att - lse).exp_()
        else:
            p = att.sub_(lse).exp_()
        del att

        # 完整统计也按查询块收集。
        max_p = argmax_stat = p_null_stat = None
        # keep_gate_terms=False 的轻量统计只消费 g：不为 standard/null 计算
        # max_p，也不为任何变体计算 argmax/p_null 诊断。
        full_diag = collect_stats and keep_gate_terms
        need_p_stats = full_diag or (self.variant in _LEGA_FAMILY and (self.use_conc or self.use_raw_conc))
        if need_p_stats:
            if full_diag:
                max_v, argmax_real = torch.max(p, dim=-1)
                max_p = max_v.float()                                  # (B,H,C)
            elif p.requires_grad:
                # 训练态用 max(dim)：反向只保存 argmax 索引并 scatter，
                # 而 amax 的反向需要在整张 p 上逐元素比较。
                max_p = torch.max(p, dim=-1).values.float()            # (B,H,C)
            else:
                max_p = p.amax(dim=-1).float()                         # (B,H,C)
        if full_diag:
            if is_null:
                # 解析合并的 null 槽参与联合归一化：真实键概率需乘 (1-p_null)，
                # argmax 也需与 p_null 比较，结果与 cat 出 S+1 列完全一致。
                p_null_stat = p_null.squeeze(-1).float()
                max_p.mul_(1 - p_null_stat)
                argmax_stat = torch.where(
                    p_null_stat > max_p,
                    torch.full_like(argmax_real, -1), argmax_real)
                # 联合 softmax 的最大概率必须包含 null 槽：否则 null 主导时
                # argmax=-1 而 max_p 仍只是真实键的最大值，两个统计自相矛盾。
                max_p = torch.maximum(max_p, p_null_stat)
            else:
                argmax_stat = argmax_real
        if self.attn_dropout > 0 and self.training:
            p = F.dropout(p, p=self.attn_dropout)
        o = torch.matmul(p, v)                                        # (B,H,C,Dh)
        del p
        if is_null:
            # 真实键部分整体乘 (1 - p_null)（联合 softmax 的尺度因子），
            # addcmul_ 再累加 p_null(×)v_null。
            o.mul_(1 - p_null)
            o.addcmul_(p_null, v_null_c.view(1, self.n_heads, 1, -1))
        if not collect_stats:
            return o, max_p
        return o, max_p, argmax_stat, p_null_stat

    def _attend_chunked(self, q_rot: torch.Tensor, q_content: torch.Tensor,
                        k: torch.Tensor, v: torch.Tensor,
                        past_len: int, log_n: Optional[torch.Tensor],
                        B: int, H: int, T: int, collect_stats: bool,
                        keep_gate_terms: bool = True):
        """按查询块执行显式注意力；训练态按层包一层 checkpoint。

        q_rot 必须已预乘 1/sqrt(Dh)：在进入分块循环前统一缩放一次。
        q_content 为 RoPE 前的查询内容视图 (B,H,T,Dh)。
        门控在全部查询块算完后按整段 T 计算（各块 max_p 先按查询维汇总）。
        """
        S = k.shape[2]
        # q/k/v 来自 QKV 融合 GEMM 的 permute().unbind() 视图：批量维步长
        # 不可展平。进入循环前统一连续化：之后每块切片均为批量维可展平的
        # 行主序视图。
        # 序列本身的 Scale 已在调用方按块外完成，此处只做布局规整。
        q_rot = _ensure_mm_input(q_rot)
        # 门控权重按内容 dtype 转换（只读，可被闭包安全捕获）。
        gw = self._cast_cached(
            '_gate_w_cast', (self.gate_w,), q_content.dtype)[0] \
            if self.use_query else None
        # 门控查询项 w·q 在进入分块循环前统一归约（整段 T）；
        # 训练态 checkpoint 反向重算直接复用该结果。
        qterm_pre = None if gw is None else self._gate_qterm(q_content, gw)
        k = _ensure_mm_input(k)
        v = _ensure_mm_input(v)
        cq = max(_CHUNK_MIN_T, min(T, _chunk_elems(q_rot.device) // max(1, B * H * S)))
        cq = min(T, max(cq, 1))
        # null 权重在整个分块循环内共享：循环外转换一次，作为 checkpoint
        # 输入传入（而非闭包捕获），反向梯度经输入路径回传。
        null_weights = (self._cast_cached(
            '_null_cast', (self.k_null, self.v_null), v.dtype)
            if self.is_null else None)
        kT = k.transpose(-2, -1)
        # checkpoint 的前向以 collect_stats=False 运行且返回单个张量，
        # 需要统计时必须走普通调用，否则统计会被静默丢弃（g/logit 全为 None）。
        use_ckpt = (torch.is_grad_enabled() and q_rot.requires_grad
                    and not collect_stats
                    and _ckpt_worthwhile(B, H, T, S, q_rot.dtype))
        # 输出按 (B,T,H,Dh) 连续布局写入预分配缓冲（逐块 copy_ 是 autograd
        # 安全的），块输出到多头合并的转置在此完成
        Dh = v.shape[-1]

        def _run_chunks(qt_full, null_c):
            # max_p 逐块汇总表必须是本次调用的局部变量：训练态 checkpoint
            # 反向会重算整个函数，闭包列表会在重算时被重复追加。
            max_ps = []
            o = v.new_empty(B, T, H, Dh)
            for q0 in range(0, T, cq):
                q1 = min(T, q0 + cq)
                qr = q_rot[:, :, q0:q1]  # 已预乘 1/sqrt(Dh) 的 RoPE 旋转后查询块（打分用）
                # 因果掩码两段式：
                #   * [fill_from, S) 列对本块全部查询不可见：不置 -inf，直接把 kT/v
                #     切片到可见前缀，打分矩阵从 (B,H,Cq,S) 缩到 (B,H,Cq,fill_from)；
                #   * 对角带 [band_start, band_start + (q1-q0))：用全局共享的
                #     严格上三角掩码逐块置 -inf（只依赖块内相对偏移，与 past 无关）。
                fill_from = min(past_len + q1, S)
                if fill_from < S:
                    kT_c = kT[..., :fill_from]      # kT 布局 (B,H,Dh,S)，键维在末维
                    v_c = v[:, :, :fill_from]       # v 布局 (B,H,S,Dh)
                else:
                    kT_c, v_c = kT, v
                band_start = past_len + q0
                # 单行块（逐 token 解码）的对角带掩码为全 False（triu(1)），
                # 逐元素 masked_fill_ 不会对任何元素置 -inf，整块跳过。
                band_mask = None if q1 - q0 == 1 else _tri_band(q1 - q0, q1 - q0, q_rot.device)
                if collect_stats:
                    oc, mp, am, pnl = self._attend_q_chunk(
                        qr, kT_c, v_c, band_start, band_mask, True,
                        keep_gate_terms, null_c)
                    if mp is not None:
                        max_ps.append(mp)
                    if am is not None:
                        argmaxes.append(am)
                    if pnl is not None:
                        p_nulls.append(pnl)
                else:
                    oc, mp = self._attend_q_chunk(
                        qr, kT_c, v_c, band_start, band_mask, False,
                        null_c=null_c)
                    if mp is not None:
                        max_ps.append(mp)
                # oc 为 (B,H,C,Dh)，目标切片 (B,C,H,Dh)：copy_ 自转置视图，
                # 布局变换在此完成
                o[:, q0:q1].copy_(oc.transpose(1, 2))
            # 各块 max_p 按查询维汇成整段：门控值逐位置只依赖
            # (qterm, max_p, log n)，逐块与整段结果逐位一致。
            max_p_all = torch.cat(max_ps, dim=2) if max_ps else None
            if self.variant in _GATED:
                # keep_terms=False 时 _gate_logit 原地改写 qterm：checkpoint
                # 反向会重算本函数并再次读取其输入，必须传副本；非 checkpoint
                # 路径中 qterm 是本次前向的独占中间量，可直接原地使用。
                qterm_in = qt_full
                if qt_full is not None and use_ckpt:
                    qterm_in = qt_full.clone()
                gg, lg, qt = self._gate_logit(
                    qterm_in, max_p_all, log_n, (B, H, T),
                    keep_terms=collect_stats and keep_gate_terms,
                    pos_offset=past_len)
                chunk_out[0], chunk_out[1], chunk_out[2] = max_p_all, gg, lg
                chunk_out[3] = qt
                o.mul_(gg.to(o.dtype).transpose(1, 2).unsqueeze(-1))
            else:
                chunk_out[0] = max_p_all
            # 送出的 o 已按 (B,T,H) 布局写满，直接 view 成多头合并的 (B,T,C)
            return o.view(B, T, H * Dh)

        # [max_p, g, logit, qterm]：仅统计路径（collect_stats=True，不启用
        # checkpoint）消费，闭包写入后由下方按需返回。
        chunk_out = [None, None, None, None]
        argmaxes, p_nulls = [], []
        if use_ckpt:
            # 训练态按层（而非按查询块）包一层 checkpoint；门控查询项与
            # null 权重作为输入传入（而非闭包捕获），其梯度经输入路径正常回传。
            # preserve_rng_state=False：分块路径仅在 attn_dropout==0 时启用
            # （use_chunk 条件已排除训练态 dropout），块内无 RNG 消耗。
            o = torch.utils.checkpoint.checkpoint(
                _run_chunks, qterm_pre, null_weights, use_reentrant=False,
                preserve_rng_state=False)
        else:
            o = _run_chunks(qterm_pre, null_weights)
        if not collect_stats:
            return o, None, None, None, None, None, None
        cat = lambda xs: torch.cat(xs, dim=2) if xs else None
        # 轻量统计（keep_gate_terms=False，契约上只消费 g）：集中度门控在
        # _gate_logit 内对 max_p 原地 clamp_min_/log_，此后 max_p 装的已是
        # log(max_p) 而不再是概率，不能作为统计返回，
        # 与 SDPA 轻量分支的 max_p=None 口径保持一致。
        return o, (chunk_out[0] if keep_gate_terms else None), \
            chunk_out[3], chunk_out[2], chunk_out[1], \
            cat(argmaxes), cat(p_nulls)

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

        if T == 0:
            # 空查询序列没有注意力可算。standard/qgate/null 原本会返回空
            # 张量，但集中度门控变体会在 max/amax 的空维上抛 IndexError。
            # 统一契约：输出与 cache 均为空（已有缓存原样返回），统计为
            # 空值，不执行投影/RoPE/softmax。
            if kv_cache is not None:
                cache = kv_cache
            elif past_kv is not None:
                cache = past_kv
            else:
                empty_kv = x.new_empty((B, H, 0, Dh))
                cache = (empty_kv, empty_kv)
            stats = ({"max_p": None, "log_n": None, "argmax": None,
                      "qterm": None, "g": None, "logit": None}
                     if need_stats else None)
            return x.new_empty((B, 0, C)), cache, stats

        # 训练态使用三个独立投影（各自独立的 autograd 路径）；
        # 推理态权重不变，走下方融合权重缓存。
        if torch.is_grad_enabled():
            q = F.linear(x, self.wq.weight).view(B, T, H, Dh).transpose(1, 2)
            k = F.linear(x, self.wk.weight).view(B, T, H, Dh).transpose(1, 2)
            v = F.linear(x, self.wv.weight).view(B, T, H, Dh).transpose(1, 2)
        else:
            # 推理态（eval/验证/生成，含 inference_mode）：权重不变，
            # 融合权重按 (参数对象身份, _version, data_ptr, device, dtype) 缓存复用。
            # .to()/.double()/param.data= 替换可能不更新 _version，故必须额外
            # 核对 data_ptr/device/dtype，否则缓存陈旧会崩溃或静默出错。
            # 缓存张量在 inference_mode(False)+no_grad 下创建：既不在 inference
            # 期现场留下 inference tensor（训练前向若误用会被 autograd 拒绝），
            # 也不搭建多余计算图。逐项身份比较可避免元组 != 在 Tensor 上触发
            # "Boolean value of Tensor is ambiguous"。
            wq_w, wk_w, wv_w = self.wq.weight, self.wk.weight, self.wv.weight
            key = tuple(
                v for w in (wq_w, wk_w, wv_w)
                for v in (w, w._version, w.data_ptr(), w.device, w.dtype))
            ent = self._qkv_fused
            stale = ent is None
            if not stale:
                o_key = ent[0]
                # 逐项（含对象身份）比较，避免元组不等在 Tensor 上退化为
                # 逐元素 == 而触发 "Boolean value of Tensor is ambiguous"；
                # 任一项不一致（not 匹配）才判为缓存失效
                stale = len(o_key) != len(key) or any(
                    not (o is n if isinstance(n, torch.Tensor)
                         else o == n) for o, n in zip(o_key, key))
            if stale:
                with torch.inference_mode(False), torch.no_grad():
                    fused = torch.cat([wq_w.detach(), wk_w.detach(), wv_w.detach()],
                                      dim=0).contiguous()
                ent = (key, fused)
                self._qkv_fused = ent
            w_qkv = ent[1]
            qkv = F.linear(x, w_qkv).view(B, T, 3, H, Dh)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)                 # (B, H, T, Dh)
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
        S = n_real   # null 槽以 lse(a·w_null) 解析地计入 softmax 归一化
                     # （非分块完整统计/dropout 分支除外，见下）。

        # 集中度门控（lega/lega_no_len/lega_no_query/lega_conc_only）必须消费
        # 注意力概率统计 max_j p_j，无法走 SDPA；qgate/lega_no_conc/lega_len_only
        # 的门只依赖查询与 log n，即便 need_stats=True（轻量门控诊断）也可走
        # SDPA——统计项 qterm/g/logit 由预计算的 qterm 与 log n 单独重组，
        # argmax 等完整诊断仍需显式路径。
        full_stats = bool(need_stats) and stats_argmax
        single_q = (T == 1)
        use_sdpa = (self.use_sdpa and not full_stats
                    and not (self.variant in _USE_CONC)
                    and not (self.variant in _USE_RAW_CONC)
                    and not (self.is_null and single_q))
        # is_causal=True 快路（无掩码全量前向）由 SDPA 内部保证因果性，
        # 无需构建显式掩码。
        # 静态 KV 缓存的整段预填充（cache_pos=0）同样满足"无任何历史"，
        # 注意力只在已写入的 T 个前缀键上做全因果，可同走 is_causal 快路。
        # null/null_zero 必须把空槽并入 is_causal 之外的掩码分支：is_causal
        # 快路只对真实键做 softmax，会丢掉 p_null 对输出的归一化/写回。
        causal_only = (use_sdpa and past_kv is None
                       and (kv_cache is None or cache_pos == 0)
                       and self.variant in ("standard", "gate_only", "qgate",
                                            "lega_no_conc", "lega_len_only"))

        # 显式分块路径只使用逐块对角带掩码（全局共享的严格上三角小掩码）。
        elems = B * H * T * S
        will_chunk = (not use_sdpa and T >= _CHUNK_MIN_T
                      and elems >= _chunk_elems(dev)
                      and not (self.training and self.attn_dropout > 0))

        # 门控查询项 w·q 预计算；分块路径在 _attend_chunked 内整段归约一次
        # （checkpoint 重算安全），此处只为 SDPA 与非分块显式路径服务。
        qterm_pre = None
        if self.use_query and not will_chunk:
            gw_pre = self._cast_cached(
                '_gate_w_cast', (self.gate_w,), q_content.dtype)[0]        # (H, Dh)
            # (B, H, T, Dh) @ (H, Dh, 1) -> (B, H, T) fp32
            qterm_pre = self._gate_qterm(q_content, gw_pre)

        # 因果掩码：真实键 j 仅当 j <= 查询绝对位置时可见；null 槽对全部查询可见。
        # 单查询步（T==1 的增量解码）：键数 n_real = past_len + 1，查询绝对位置
        # 恰为 past_len，全部键（含 null 槽）必然可见，无需构建掩码。
        allowed = not_allowed = mask_entry = None
        if not single_q and not causal_only and not will_chunk:
            mkey = (past_len, T, S, dev)
            entry = _MASK_CACHE.get(mkey)
            if entry is not None:
                _MASK_CACHE.move_to_end(mkey)
            else:
                # 缓存中的常数张量必须创建于普通模式。若在 inference_mode
                # （验证/生成）下创建并被随后的训练前向复用，autograd 会拒绝保存
                # inference tensor。这里显式退出 inference_mode，仅用于创建不依赖
                # 当前输入的缓存常量。
                with torch.inference_mode(False):
                    qpos = torch.arange(past_len, past_len + T, device=dev)
                    not_allowed = qpos.view(1, 1, T, 1) < torch.arange(n_real, device=dev).view(1, 1, 1, n_real)
                    # null 槽常数可见列随掩码一起缓存
                    entry = [not_allowed,
                             torch.ones(1, 1, T, 1, dtype=torch.bool, device=dev),
                             None, None]  # allowed / 拼 null 槽的 allowed，惰性填充
                _mask_cache_store(mkey, entry)
            not_allowed, ones_col = entry[0], entry[1]
            mask_entry = entry
            # 不 expand：masked_fill / SDPA 均支持 (1, 1, T, S) 广播

        stats: Optional[dict] = None
        g = qterm = logit = max_p = log_n = None
        argmax_stat = p_null_stat = None

        if use_sdpa:
            # 无诊断需求的路径：SDPA（含 null/qgate/lega_no_conc 变体）
            if causal_only:
                o = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=True)
            else:
                # allowed 只在此分支需要：惰性取反一次后随 not_allowed 一起缓存。
                # 单查询步（T==1）不建掩码，此处 mask_entry 为 None，attn_mask 传 None。
                if mask_entry is None:
                    allowed = None
                else:
                    if self.is_null:
                        if mask_entry[3] is None:
                            with torch.inference_mode(False):
                                mask_entry[3] = torch.cat([~mask_entry[0], ones_col], dim=3)
                            _mask_cache_grow(mkey, mask_entry, mask_entry[3])
                        allowed = mask_entry[3]
                    else:
                        if mask_entry[2] is None:
                            with torch.inference_mode(False):
                                mask_entry[2] = ~mask_entry[0]
                            _mask_cache_grow(mkey, mask_entry, mask_entry[2])
                        allowed = mask_entry[2]
                if self.is_null:
                    # SDPA 只接受单张量 K/V，这里拼接 null 槽；
                    # 逐 token 解码已在上面分流到显式路径
                    kn, vn = self._cast_cached(
                        '_null_cast', (self.k_null, self.v_null), k.dtype)
                    kn = kn.view(1, H, 1, Dh).expand(B, H, 1, Dh)
                    vn = vn.view(1, H, 1, Dh).expand(B, H, 1, Dh)
                    k = torch.cat([k, kn], dim=2)
                    v = torch.cat([v, vn], dim=2)
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed,
                    dropout_p=self.attn_dropout if self.training else 0.0)
            # 与显式路径一致：need_stats 时即使该变体不用 log n（qgate/standard）
            # 也计算并记录，保证 SDPA/显式两条路径返回的轻量 stats 口径完全相同。
            if self.use_conc or self.use_len or bool(need_stats):
                log_n = _log_n(past_len, T, dev)
            if self.variant in _GATED:
                # 与显式路径相同的门控公式（qgate: w·q+d；lega_no_conc: w·q+b·log n+d）
                # 轻量诊断（stats_argmax=False）只消费 g：不保留 qterm/logit，
                # sigmoid 原地完成
                g, logit, qterm = self._gate_logit(
                    qterm_pre, None, log_n, (B, H, T), keep_terms=full_stats,
                    pos_offset=past_len)
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
                         "qterm": qterm, "g": g, "logit": logit,
                         "gate_bias": (self.gate_d.float().view(1, H, 1)
                                       if self.variant in _GATED else None)}
        else:
            # 显式注意力路径：gated / 需要诊断统计时。长上下文按查询块计算并
            # 逐块汇总完整统计，训练态配合逐块 checkpoint。
            # 逐块 softmax 覆盖完整可见键行，数学不变。
            # attn_dropout>0 的训练步保持单块路径：分块会改变 dropout 的
            # RNG 消耗粒度，虽无系统性偏差，但不逐位一致。
            use_chunk = will_chunk
            # 预缩放 q：分块与单块共用。rope 启用时其输出是本层独占的新
            # 张量（旋转不依赖原地值反向），可直接 mul_；
            # rope 未启用时 q 与门控内容向量 q_content 共享存储，必须 out-of-place。
            if q is q_content:
                q = q * self.scale
            else:
                q.mul_(self.scale)
            if use_chunk:
                log_n = None
                if self.use_conc or self.use_len or bool(need_stats):
                    log_n = _log_n(past_len, T, dev)
                collect = bool(need_stats)
                o, max_p, qterm, logit, g, argmax_stat, p_null_stat = \
                    self._attend_chunked(q, q_content, k, v, past_len,
                                         log_n, B, H, T, collect,
                                         keep_gate_terms=bool(need_stats)
                                         and stats_argmax)
            else:
                # q 已在分块循环外统一预乘 1/sqrt(Dh)；argmax/softmax 对正缩放不敏感。
                att = torch.matmul(q, k.transpose(-2, -1))             # (B, H, T, S)
                if not_allowed is not None:
                    att.masked_fill_(not_allowed, float("-inf"))
                # null 槽的两种合成方式（与 _attend_q_chunk 相同的解析合并）：
                #   * 需要完整联合 softmax 的场景（null 的 argmax/p_null 诊断、
                #     训练态 dropout 的 RNG 消耗口径随列数变化）仍 cat 出 S+1 列；
                #   * 无完整统计需求的训练/推理前向：p_null = sigmoid(a_n - lse)，
                #     softmax 只覆盖真实键；逐 token 解码（T==1）走此路径。
                p_null = None
                lse = None
                if self.is_null:
                    k_null_c2, v_null_c2 = self._cast_cached(
                        '_null_cast', (self.k_null, self.v_null), v.dtype)
                    att_n = torch.matmul(q, k_null_c2.unsqueeze(-1))
                    if full_stats or (self.training and self.attn_dropout > 0):
                        att = torch.cat([att, att_n], dim=-1)          # (B,H,T,S+1)
                    else:
                        # logsumexp 内部做 max 平移，数值稳定；
                        # 注意是 a_n - lse（sigmoid(lse - a_n) = 1 - p_null）
                        lse = att.logsumexp(dim=-1, keepdim=True)
                        p_null = torch.sigmoid(att_n - lse)
                        del att_n
                argmax_stat = None
                if lse is None:
                    p = self._softmax_reuse(att)
                elif (torch.is_grad_enabled() and att.requires_grad) \
                        or torch.compiler.is_compiling():
                    # 同上：梯度路径不能原地 sub_（需保留 att），exp_ 仍可原地
                    p = (att - lse).exp_()
                else:
                    p = att.sub_(lse).exp_()
                del att

                # 门控所需的注意力统计（LEGA 家族训练时也必须计算）。
                # max_j p_j 直接取 softmax 输出的行最大值（在 dropout 之前）。
                # stats_argmax=False（轻量诊断，如验证评测只需要 g）时不计算
                # argmax；不含集中度项的变体（qgate 等）此时也不必计算 max_p/log_n。
                full_stats = bool(need_stats) and stats_argmax
                # log n 在门控含集中度项（c 的定义需要 n）或长度项时需要；
                # 完整诊断也计算。注意 lega_no_len/lega_conc_only 虽无 b*log n，
                # 但 a*c 仍需要 log n。
                need_logn = self.use_conc or self.use_len or bool(need_stats)
                if need_logn:
                    log_n = _log_n(past_len, T, dev)
                # max_p 与完整诊断的 argmax 都必须基于 dropout 前的 softmax 分布。
                # argmax 落点是 softmax 不变量。
                need_argmax_real = (bool(need_stats) and stats_argmax
                                    and not self.is_null)
                need_max_real = (not self.is_null
                                 and (self.use_conc or self.use_raw_conc
                                      or full_stats))
                if need_max_real and need_argmax_real:
                    max_v, argmax_stat = torch.max(p, dim=-1)
                    max_p = max_v.float()                    # max_j p_j ∈ (0, 1]
                elif need_max_real:
                    # 有梯度时 max(dim) 的反向只需 argmax 索引（见
                    # _attend_q_chunk）；无梯度时用 amax。
                    max_p = (torch.max(p, dim=-1).values if p.requires_grad
                             else p.amax(dim=-1)).float()
                elif need_argmax_real:
                    argmax_stat = p.argmax(dim=-1)

                # 诊断统计（null 变体的 max_p/argmax/p_null 等）必须基于 dropout 之前的
                # softmax 分布，否则会因 dropout 的 1/(1-p) 重标定与置零而失真；
                # dropout 只作用于实际输出路径，因此保留一份 dropout 前的引用。
                # 无完整统计需求时不保留该引用。
                p_pre = p if full_stats else None
                if self.attn_dropout > 0 and self.training:
                    p = F.dropout(p, p=self.attn_dropout)
                if self.is_null and p_null is None:
                    # 联合 softmax 全列路径：o = p_real @ v + p_null * v_null
                    o = torch.matmul(p[..., :n_real], v)               # (B, H, T, Dh)
                    o.addcmul_(p[..., n_real:], v_null_c2.view(1, H, 1, Dh))
                else:
                    o = torch.matmul(p, v)                             # (B, H, T, Dh)
                if p_null is not None:
                    # 解析合并路径：真实键部分乘联合尺度因子 (1-p_null)，
                    # 再加 p_null * v_null
                    o.mul_(1 - p_null)
                    o.addcmul_(p_null, v_null_c2.view(1, H, 1, Dh))
                del p

                # 输出门控（惰性累加：仅偏置项 (1,H,1) 参与广播）
                if self.variant in _GATED:
                    g, logit, qterm = self._gate_logit(
                        qterm_pre, max_p, log_n, (B, H, T),
                        keep_terms=full_stats, pos_offset=past_len)
                    # 原地乘法：o 的全部反向依赖 p/v，mul_ 自身被 autograd 正确处理
                    o.mul_(g.to(o.dtype).unsqueeze(-1))                   # o' = g * o

            if need_stats:
                # 轻量统计（stats_argmax=False）只消费 g：LEGA 家族的门控在
                # _gate_logit 内对 max_p 原地 clamp_min_/log_，此后 max_p 不再是
                # 概率值，按约定不落进 stats（与上方 SDPA 轻量分支的 max_p=None 一致）。
                stats = {
                    "max_p": max_p if full_stats else None,
                    "log_n": log_n,     # need_stats=True 时上方必然已计算（轻量模式下未用项为 None）
                    "argmax": None,     # 轻量模式不计算 argmax
                    "qterm": qterm,
                    "g": g,
                    "logit": logit,
                    "gate_bias": (self.gate_d.float().view(1, H, 1)
                                  if self.variant in _GATED else None),
                }
                if stats_argmax:
                    if use_chunk:
                        # 完整统计逐块汇总
                        stats["argmax"] = argmax_stat
                        if self.is_null:
                            stats["p_null"] = p_null_stat.float()
                    elif self.is_null:
                        # null 变体的统计必须基于含 null 槽的完整 softmax，
                        # 否则会虚增集中度并错误地把 null 槽落点记为真实键。
                        # 真实键 max 给出概率与落点，再与 null 槽比较；
                        # 与对 S+1 列整体 argmax 后判是否落在 null 槽完全一致。
                        max_real, amax_real = torch.max(p_pre[..., :n_real], dim=-1)
                        null_p = p_pre[..., -1]
                        stats["argmax"] = torch.where(null_p > max_real,
                                                      torch.full_like(amax_real, -1),
                                                      amax_real)
                        # max_p 是联合 softmax 的最大值：null 槽主导时必须取 null 概率。
                        stats["max_p"] = torch.maximum(max_real, null_p).float()
                        stats["p_null"] = null_p.float()                  # 分给 null 槽的概率质量
                    else:
                        # argmax 已在 softmax 前提取（softmax 不变量，真实键内相对位置）
                        stats["argmax"] = argmax_stat

        # 合并多头: (B, H, T, Dh) -> (B, T, C)；分块路径已按 (B,T,H,Dh) 布局
        # 直接写出，其 o 本身就是合并后的 (B,T,C)
        if o.dim() == 4 and T == 1:
            # 单查询步（逐 token 解码）：(B, H, 1, Dh) 的 H×Dh 子块步长连续，
            # 可直接折成 (B, C) 进输出投影。
            out = self.wo(o.reshape(B, C)).unsqueeze(1)
        elif o.dim() == 4:
            o = o.transpose(1, 2).contiguous().view(B, T, C)
            out = self.wo(o)
        else:
            out = self.wo(o)
        return out, cache, stats
