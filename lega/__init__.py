"""LEGA: 长度校准的证据门控 Attention (Length-calibrated Evidence-Gated Attention)

研究问题：当一个注意力头在当前上下文中找不到可靠信息时，能否减少向残差流写入无关内容？
创新点：门控同时参考 当前查询(q)、本次注意力分布的集中度(max_j p_j) 与 上下文长度(n)。
"""
import os as _os

# 包级 CUDA 分配器配置：须在任何 CUDA 初始化之前生效（所有入口脚本都经此导入）。
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 确定性算法（torch.use_deterministic_algorithms）要求 cuBLAS 使用固定工作区；
# 该变量必须在第一个 cuBLAS 句柄创建前设置，放在包导入处而不是 set_seed 内，
# 以免个别入口（后台评测/冒烟脚本）先做了 CUDA 计算再调 set_seed 而失效。
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
