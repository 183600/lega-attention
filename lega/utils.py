"""通用工具：随机种子、JSON 读写、指标计算（AUROC/PR-AUC/F1/Pearson/统计检验）。

指标选择口径：类别不平衡时不能只看准确率。AUROC 在负样本占绝对多数时
（FPR 增长缓慢）会过于乐观，须与 PR-AUC（average_precision）并报；正类率
低于 5% 时以 PR-AUC 为主，其“随机猜”基线就是正类率。F1 必须显式注明
macro / micro / weighted（三者在不平衡数据上可能差异巨大，macro 对稀有类
最诚实）。以上均已实现且无 scipy 依赖。

统计检验的设计口径：正式实验的种子数通常只有 3~5，配对 t 检验的正态假设在此
样本量下并不可靠；因此本模块同时提供
  * paired_t_test            —— 参数检验（主判定依据）；
  * paired_sign_flip_test    —— 符号翻转置换检验（H0 下精确/分布无关，n<=16 时
                                全枚举精确，否则固定种子 Monte-Carlo）；
  * paired_sign_test         —— 逐对差值符号的精确二项检验（忽略幅度，对离群
                                种子稳健；零差按标准做法从符号检验中剔除）；
  * bootstrap_mean_ci        —— 配对差的百分位 bootstrap 置信区间；
  * min_detectable_effect    —— 给定种子数与配对差标准差的最小可检测效应，
                                用于区分"没有效应"与"功效不足"；
  * wilson_score_ci          —— 二值指标（RULER EM）的小样本比例置信区间。
汇总同时报告以上全部统计量；t 与置换检验结论冲突时会显式标注，而不是静默
采信任何一方。
"""
import contextlib
import hashlib
import json
import math
import operator
import os
import random
import re
import statistics
import time

import numpy as np
import torch

# 解释器启动时的环境快照：PYTHONHASHSEED 只在解释器启动前设置才影响**当前**
# 进程；set_seed 之后再写 os.environ 只对之后启动的子进程生效。审计字段若直接
# 读取当前 env，会被 set_seed 自己写成 seed，变成“永远为真”的自证字段——必须
# 记录导入时（任何 set_seed 之前）看到的值，才能区分两种情况。
_ENV_AT_IMPORT = {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}

# 最近一次 set_seed 的随机源状态指纹：(seed, fingerprints)，供 seed_report
# 给出“种子确实生效”的可核对证据，而不是复述传入参数。同时记录确定性算法
# 的实际启用模式（严格/warn_only），避免把“只警告”审计成“严格确定性”。
_LAST_SEED = {"seed": None, "fingerprints": None,
              "deterministic_warn_only": None}


def deterministic_algorithms_warn_only():
    """当前确定性算法的实际模式：True=仅告警，False=严格，None=未启用/未知。

    PyTorch 公开 API 只暴露 ``are_deterministic_algorithms_enabled()``，
    无法区分严格与 warn_only；本函数据 set_seed 时记录的实际模式如实汇报，
    保证 run_manifest/seed_report 中的“确定性已启用”不被夸大。
    """
    if not torch.are_deterministic_algorithms_enabled():
        return None
    return _LAST_SEED.get("deterministic_warn_only")


CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")


def ensure_cublas_workspace_config(env=None) -> str:
    """确保 CUDA 确定性工作区配置有效；无效/缺失时改为标准值。"""
    target = os.environ if env is None else env
    value = target.get("CUBLAS_WORKSPACE_CONFIG")
    if value not in CUBLAS_WORKSPACE_CONFIGS:
        value = ":4096:8"
        target["CUBLAS_WORKSPACE_CONFIG"] = value
    return value


def data_rng(seed: int) -> np.random.Generator:
    """项目唯一的“dataloader”随机源：显式种子的 NumPy Generator。

    train/sanity 的批采样必须经此构造，而不是各自 ``default_rng(seed)``。
    这样“数据顺序由 seed 唯一确定”不仅是约定，还可用
    ``data_rng_state_fingerprint(seed)`` 复算同一初始状态作为审计证据。
    若将来引入 torch DataLoader，必须同样用该 seed 构造 ``torch.Generator``
    并设置 ``worker_init_fn``，否则 worker 会从默认随机源取种子。
    """
    return np.random.default_rng(seed_int(seed, "data rng seed"))


def data_rng_state_fingerprint(seed: int) -> str:
    """由 seed 可复算的数据顺序 RNG 初始状态短指纹（不消耗全局 RNG）。

    直接构造 ``data_rng(seed)`` 并哈希其 bit generator 状态，因此第三方
    重跑同一函数即可核对数据顺序随机源的种子身份；它与全局 NumPy RNG
    相互独立，不受模型初始化/评测消耗影响。
    """
    state = data_rng(seed).bit_generator.state
    raw = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _python_random_state_fingerprint(state) -> str:
    """Python ``random`` 状态对象的短指纹（供全局/本地构造器共用）。"""
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()[:16]


def _numpy_state_fingerprint(state) -> str:
    """NumPy 旧式 RandomState 状态元组的短指纹。"""
    ns = state
    h = hashlib.sha256()
    h.update(str(ns[0]).encode("ascii"))
    h.update(np.asarray(ns[1], dtype="<u4").tobytes())
    h.update(int(ns[2]).to_bytes(8, "little"))
    h.update(repr(ns[3:]).encode("ascii"))
    return h.hexdigest()[:16]


def _torch_state_fingerprint(state) -> str:
    """Torch CPU RNG 字节状态的短指纹。"""
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()[:16]


def rng_state_fingerprint() -> dict:
    """各全局 RNG **当前状态**的短指纹（只读：get_state 不消耗随机数）。

    与“记录了 seed 字面量”不同，指纹来自 RNG 真实内部状态：同 seed 的
    set_seed 必然得到相同指纹，不同 seed 得到不同指纹，任何人重跑
    ``set_seed(seed) + rng_state_fingerprint()`` 即可验证运行时的种子是否
    真的生效。Python random / NumPy MT19937 的状态跨平台一致；PyTorch CPU
    为 MT19937 字节状态，同版本跨平台一致；CUDA 状态哈希在无 GPU 时不出现。
    """
    fp = {"python_random": _python_random_state_fingerprint(
        random.getstate())}
    fp["numpy_global"] = _numpy_state_fingerprint(np.random.get_state())
    fp["torch_cpu"] = _torch_state_fingerprint(torch.get_rng_state())
    if torch.cuda.is_available():
        h = hashlib.sha256()
        for st in torch.cuda.get_rng_state_all():
            h.update(st.cpu().numpy().tobytes())
        fp["torch_cuda_all"] = h.hexdigest()[:16]
    return fp


def expected_rng_state_fingerprint(seed: int) -> dict:
    """由 seed 独立复算的 RNG **初始状态**短指纹（不触碰全局 RNG）。

    ``rng_state_fingerprint`` 记录的是进程内全局 RNG 的当前状态，适合作为
    “这个 seed 真的被 set_seed 执行过”的运行时证据；但审计方不能在只读
    台账时调用它（会消耗/改写自己的全局随机流）。本函数用与 set_seed 等价的
    **本地**构造器（random.Random / np.random.RandomState / torch.Generator）
    复算同一初始状态，因此第三方可在不污染自己实验随机源的前提下核对
    ``seed_report.rng_fingerprints`` 与 ``seed_plan.rng_fingerprints``。

    只覆盖 CPU 三个随机源：Python random 与 NumPy MT19937 的初始状态跨版本
    稳定，Torch CPU 状态同版本一致；CUDA 生成器需要初始化设备，且其状态
    跨版本/硬件不保证一致，不在此处伪造期望值。
    """
    seed = seed_int(seed, "seed")
    return {
        "python_random": _python_random_state_fingerprint(
            random.Random(seed).getstate()),
        "numpy_global": _numpy_state_fingerprint(
            np.random.RandomState(seed).get_state()),
        "torch_cpu": _torch_state_fingerprint(
            torch.Generator().manual_seed(seed).get_state()),
    }


def state_dict_fingerprint(state_dict) -> dict:
    """对参数/缓冲区名称、dtype、形状与原始字节做稳定 SHA-256。

    与只记录 seed 不同，指纹来自初始化后的真实张量：同 seed/同版本应得到
    同一指纹，初始化顺序、默认 dtype 或模型结构变化会改变指纹。名称排序，
    不依赖 state_dict 的插入顺序；非浮点/低精度张量统一转成可哈希字节。
    """
    h = hashlib.sha256()
    n_tensors = 0
    n_elements = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"state_dict[{name!r}] 不是张量")
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(t.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(b"\0")
        try:
            raw = t.numpy().tobytes()
        except TypeError:
            raw = t.to(torch.float64).numpy().tobytes()
        h.update(raw)
        n_tensors += 1
        n_elements += int(t.numel())
    return {"sha256": h.hexdigest(), "tensors": n_tensors,
            "elements": n_elements}


def strict_int(value, name: str = "值") -> int:
    """严格整数化：只接受整数（含整数值浮点与整数字符串），否则 ValueError。

    为什么不能直接用 ``int()``：``int(True) is 1``、``int(1.5) is 1``——
    bool 是 int 的子类、小数被静默截断。种子/步数这类“身份与规格”字段一旦
    被截断，记录下来的值（台账、checkpoint、报告）就会与真正使用的值错位：
    伪造的 1.5 或 True 也能与种子 1 “对上”，核对彻底失效。整数值浮点
    （1337.0）与整数字符串（"1337"）语义明确，规范化为 int；其余（bool、
    小数、NaN/Inf、"1_0"、非数值类型）一律显式拒绝。
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是整数，收到 bool {value!r}")
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"[+-]?\d+", s) is None:
            raise ValueError(f"{name} 必须是整数，收到 {value!r}")
        return int(s)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} 必须是有限整数，收到 {value!r}")
        return int(value)
    try:
        # numpy 整数（np.int64 等）不是 int 子类，但定义了 __index__。
        return int(operator.index(value))
    except TypeError as e:
        raise ValueError(f"{name} 必须是整数，收到 {value!r}") from e


MAX_SEED = 2 ** 32 - 1


def seed_int(value, name: str = "种子") -> int:
    """严格种子化：整数且在 NumPy/PYTHONHASHSEED 共同支持的 0..2^32-1 内。

    ``set_seed`` 会先写 PYTHONHASHSEED、再调用 ``random.seed``，最后才由
    ``np.random.seed`` 做范围检查；负种子会在前两步已改写全局状态后才失败，
    留下“尝试播种但只成功一半”的进程。run_all/tuning 的种子解析若放行
    负数/超上限，也会让错误推迟到子进程启动后才暴露。这里统一在入口拒绝。
    """
    seed = strict_int(value, name)
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"{name} 必须在 0..{MAX_SEED} 内，收到 {seed}")
    return seed


def config_finite_float(value, name: str = "值") -> float:
    """配置浮点数严格化：拒绝 bool 与非有限值（NaN/Inf），其余转 float。

    ``float(True) == 1.0`` 会把开关值静默当成数值；NaN/Inf 则会污染参数
    （例如 gate_bias_init 直接写入 Parameter）并让后续训练/评测失去意义。
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限数值，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}") from e
    if not math.isfinite(out):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return out


def config_probability(value, name: str = "概率") -> float:
    """配置概率严格化：在 config_finite_float 基础上要求落在 [0, 1]。"""
    out = config_finite_float(value, name)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 内，收到 {value!r}")
    return out


def config_bool(value, name: str = "开关") -> bool:
    """配置布尔严格化：只接受 bool/0/1/true/false，拒绝任意真值转换。

    ``bool("false") is True`` 会让旧产物/手写配置里的关闭开关静默变成开启，
    从而切换注意力后端或 AMP；必须显式解析或拒绝，而不是按非空字符串/非零
    数值猜测。
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0"):
            return False
        raise ValueError(f"{name} 必须是布尔值，收到 {value!r}")
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        value = int(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return bool(value)
    raise ValueError(f"{name} 必须是布尔值，收到 {value!r}")


def set_seed(seed: int) -> None:
    """固定当前进程内所有随机源，使同一台机器上的同一实验可复现。

    覆盖：Python random、NumPy 全局 RNG、PyTorch（CPU/CUDA）、cuDNN 确定性。
    本项目的“dataloader”是 data.py 中显式接收 ``np.random.default_rng(seed)`` 的
    批采样函数（无 torch DataLoader/worker），因此数据顺序由同一 seed 唯一确定；
    若将来引入 DataLoader，须同时传入该 seed 构造的 ``torch.Generator`` 并设置
    ``worker_init_fn``（参考 PyTorch 官方可复现性说明）。
    TensorFlow 不在依赖内，无需 TF seed。

    PYTHONHASHSEED 仅对之后启动的子进程生效（当前进程的哈希盐在解释器启动时已
    固定）；本函数写入本次 seed（覆盖旧值），保证之后由本进程启动的子进程与
    本次 seed 对齐，而不是继承一个不同的旧值。run_all.py 另在每条子命令的
    环境中显式设置。本函数结束时记录各全局 RNG 的状态指纹（_LAST_SEED），
    seed_report 据此提供种子生效的可核对证据。
    跨硬件架构（不同 GPU 型号、CPU 指令集、cuDNN 版本）无法保证逐位一致，
    本函数只保证同机相对稳定。

    种子先经 seed_int 规范化：bool/小数/非整数字符串直接 ValueError，
    负值/超过 2^32-1 也在写入任何全局状态前拒绝；不允许 True 被当成 1、
    1.5 被截断成 1 后“看起来播种成功”，也不允许半途播种。
    """
    seed = seed_int(seed, "seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 训练结论依赖同 seed 的变体间对比；禁用 cudnn 自动调优并启用确定性算法
    # （warn_only 避免因个别无确定实现的 CUDA 算子直接中断，同时给出警告）。
    ensure_cublas_workspace_config()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        _LAST_SEED["deterministic_warn_only"] = True
    except TypeError:  # 兼容没有 warn_only 参数的旧版 PyTorch
        torch.use_deterministic_algorithms(True)
        _LAST_SEED["deterministic_warn_only"] = False
    _LAST_SEED["seed"] = seed
    _LAST_SEED["fingerprints"] = rng_state_fingerprint()


def seed_report(seed: int) -> dict:
    """逐随机源的种子审计记录（写入 run_manifest/实验台账，供事后核对）。

    证明“这个 run 到底固定了哪些种子”，而不是口头声称：set_seed 覆盖的每个
    随机源都以实际传给它的 seed 记录在案；同时记录确定性开关、数据顺序机制
    与各 RNG **真实状态**的短指纹（见 rng_state_fingerprint）——指纹只在
    最近一次 set_seed 的 seed 与本次请求一致时给出，否则为 None，避免拿
    后来被消耗过的状态冒充任意 seed 的证据。
    字段全部为 JSON 原生类型。

    PYTHONHASHSEED 用三个字段分开记录，防止自证陷阱：
      * ``pythonhashseed_at_start``：解释器启动时（导入本模块时）环境里的值，
        None 表示当时未设置——当前进程的哈希随机化仍在生效，set_seed 写入
        的 env 只对子进程有效；
      * ``pythonhashseed_env``：当前环境值（只影响之后启动的子进程）；
      * ``pythonhashseed_effective``：仅当启动时已带数值且等于 seed 才为
        True；启动时未设置时为 False（等价于“本进程未固定，已显式标注”），
        启动时的值不可解析为整数时为 None。

    seed 与 set_seed 同口径先经 seed_int 规范化：bool/小数/非整数字符串
    直接 ValueError，负数/超上限同样拒绝，报告里的 "seed" 永远等于真正
    播种的那个整数。
    """
    seed = seed_int(seed, "seed")

    def _as_int(value):
        """环境值 → int；bool/小数/不可解析字符串返回 None（不截断、不当 1）。"""
        if value is None:
            return None
        try:
            return strict_int(value, "PYTHONHASHSEED")
        except ValueError:
            return None

    at_start = _ENV_AT_IMPORT.get("PYTHONHASHSEED")
    start_int = _as_int(at_start)
    fp = (_LAST_SEED["fingerprints"]
          if _LAST_SEED["seed"] == seed else None)
    # 逐随机源的 seed 不是“复述参数”：只有最近一次 set_seed 的种子与请求
    # 一致时，这些随机源才真的被该 seed 固定。否则记录 None，避免请求旧/
    # 未设置种子的报告冒充实际 RNG 证据（与 rng_fingerprints 的口径一致）。
    verified_seed = seed if fp is not None else None
    # 期望指纹由 seed 独立复算（不触碰全局 RNG）：把“实际状态指纹”与
    # “该 seed 应产生的初始状态”并排记录，审计方即使不重跑 set_seed 也能
    # 发现被复制/手改的指纹字段。CUDA 状态不在期望值内（跨版本/硬件不稳定）。
    expected_fp = expected_rng_state_fingerprint(seed) if fp is not None else None
    match_expected = (None if (fp is None or expected_fp is None)
                      else all(fp.get(key) == value
                               for key, value in expected_fp.items()))
    return {
        "seed": seed,
        "seed_verified": fp is not None,
        "python_random": verified_seed,
        "numpy_global": verified_seed,
        "torch_cpu": verified_seed,
        "torch_cuda": (verified_seed if torch.cuda.is_available() else None),
        "pythonhashseed_at_start": at_start,
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "pythonhashseed_effective": (
            None if (at_start is not None and start_int is None)
            else start_int == seed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        # 严格 / 仅告警：只报 deterministic_algorithms=True 会夸大确定性保证
        "deterministic_algorithms_warn_only":
            deterministic_algorithms_warn_only(),
        "data_order": "data_rng(seed)=np.random.default_rng(seed) 显式批采样（无 DataLoader worker）",
        # 项目约定：批采样与本审计共用同一 seed（train/sanity/evaluate 均如此），
        # 机器可读字段便于台账/清单自动核对数据顺序来源
        "data_order_seed": verified_seed,
        # 数据顺序 RNG 的初始状态指纹：第三方可用
        # data_rng_state_fingerprint(seed) 独立复算；未验证时同样不给证据。
        "data_order_rng_fingerprint": (
            data_rng_state_fingerprint(seed) if fp is not None else None),
        # 实际全局 RNG 状态指纹；expected 为由 seed 本地复算的初始状态，
        # 二者一致才说明这些指纹确实来自该 seed，而不是被复制/改写。
        "rng_fingerprints": dict(fp) if fp else None,
        "rng_fingerprints_expected": expected_fp,
        "rng_fingerprints_match_expected": match_expected,
    }


def seed_plan(seed: int) -> dict:
    """声明式种子计划：不修改任何全局 RNG，供流水线启动时先落盘审计。

    ``set_seed`` 的实际生效证据保存在各 run 的 ``run_manifest.seed_report``
    （含真实 RNG 状态指纹）。但 run_all 在启动时（训练子进程尚未运行、台账
    刚写入 started）也需要回答“这轮准备覆盖哪些 seed、数据顺序 RNG 会是
    什么”。本函数只做纯查询：
      * 逐随机源声明将使用的 seed（Python/NumPy/PyTorch/CUDA/PYTHONHASHSEED）；
      * 由 seed 独立复算的 data_rng 初始状态短指纹，第三方可核对；
      * 由 seed 本地复算的全局 RNG 初始状态指纹（``rng_fingerprints``），
        供没有 run_manifest 的 started 记录也能核对“计划播种的随机源”；
      * ``is_plan=True`` 明确这是计划而非“已生效”证据，实际指纹见
        ``run_manifest.seed_report``。
    不调用 set_seed，因此不会消耗/改变调用方的任何随机源。
    """
    seed = seed_int(seed, "seed")
    return {
        "seed": seed,
        "is_plan": True,
        "python_random": seed,
        "numpy_global": seed,
        "torch_cpu": seed,
        "torch_cuda": seed if torch.cuda.is_available() else None,
        # 仅对之后启动的子进程生效；父进程自身哈希盐在解释器启动时已固定
        "pythonhashseed_child_env": str(seed),
        "data_order_seed": seed,
        "data_order_rng_fingerprint": data_rng_state_fingerprint(seed),
        "rng_fingerprints": expected_rng_state_fingerprint(seed),
        "verification": "run_manifest.seed_report（实际 RNG 状态指纹）",
    }


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(obj):
    """递归把非有限浮点（NaN/±Inf）转成 None。

    非有限浮点统一记为 null（缺失值语义），保证输出是严格合法 JSON，
    与 evaluate.json_safe_eval_metrics / pearson 等既有口径一致。
    """
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
        # np.longdouble/float128 的 .item() 仍返回 numpy 标量（不是 Python
        # float），json.dump 会直接报不可序列化；显式降为内置 float。
        if isinstance(obj, np.generic):
            if isinstance(obj, np.floating):
                obj = float(obj)
            elif isinstance(obj, np.integer):
                obj = int(obj)
            elif isinstance(obj, np.bool_):
                obj = bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_json(obj, path: str) -> None:
    """原子写入 JSON（临时文件 + os.replace），非有限浮点记为 null。

    读者看到的要么是旧完整文件、要么是新完整文件；临时名带 pid。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def get_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def enable_tf32(device: str) -> None:
    """CUDA 上启用 TF32 矩阵乘/卷积；bf16/AMP 路径不受影响，非 CUDA 设备为空操作。"""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def cpu_bf16_available() -> bool:
    """CPU 是否具备可用的 bf16 内核（oneDNN/MKLDNN 后端）。

    PyTorch 的 CPU autocast 在所有 x86 机器上都能运行 bf16，但只有带
    AVX512-BF16（或等价的硬件支持）的内核才会启用。这里以 oneDNN 自报的
    bf16 支持为准，缺失该查询接口的构建按“不支持”处理。
    """
    try:
        return bool(torch.backends.mkldnn.is_available()
                    and torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except (AttributeError, RuntimeError, TypeError):
        return False


def amp_dtype_for(device: str, enabled: bool = True):
    """当前设备/开关下的 autocast dtype；不可用或未启用时返回 None。

    CUDA 与支持 bf16 内核的 CPU 都使用 bfloat16（动态范围与 fp32 相同，
    无需 GradScaler）；其它情况返回 None，调用方回退 fp32。
    """
    if not enabled:
        return None
    if str(device).startswith("cuda") or cpu_bf16_available():
        return torch.bfloat16
    return None


def amp_autocast(device: str, enabled: bool = True):
    """bf16 autocast 上下文（CUDA 或支持 bf16 内核的 CPU）；否则空上下文。"""
    if not enabled:
        return contextlib.nullcontext()
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if cpu_bf16_available():
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def pearson(x, y):
    """Pearson 相关系数；样本不足/零方差/结果非有限时返回 None（调用方以 None 表示不可计算，
    避免 nan 写入 JSON 产生 NaN 字面量——那并非合法 JSON）。

    先按各自最大绝对值缩放再计算：相关系数对正缩放不变，但朴素
    ``np.corrcoef`` 对接近 float 上限的合法观测会在中心化平方/求和时
    溢出成 inf，把完全相关的两组大数误判成不可计算。缩放后的数组落在
    [-1, 1]，不会溢出。
    """
    x = _obs_array(x)
    y = _obs_array(y)
    if x.size != y.size:
        raise ValueError(f"x/y 长度不一致: {x.size} vs {y.size}")
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    scale_x = float(np.max(np.abs(x)))
    scale_y = float(np.max(np.abs(y)))
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    xs, ys = x / scale_x, y / scale_y
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return r if np.isfinite(r) else None


def _finite_observation(value):
    """统计观测值 → 有限 float；None/bool/非数值/NaN/Inf → None。

    bool 是 int 子类，float(True)=1.0；若当作观测，外部 JSON 的
    ``"em": true`` 或 ``"seed": true`` 会伪造出 1.0 的均值/配对差。
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    # float("1_0")==10.0：下划线数字串不是规范 JSON 数值，不能伪装成观测。
    if isinstance(value, str) and "_" in value:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _obs_array(values) -> np.ndarray:
    """任意可迭代/数组 → 有限 float 观测数组；bool/非数值/非有限记 NaN。

    列表/元组里 bool 与 float 混合时 np.asarray 会先统一成 float 并丢掉
    bool 身份，因此非数值 ndarray 必须逐元素走 _finite_observation。
    """
    arr = np.asarray(values)
    # 只有真正的数值 dtype 才能走 astype 快路。字符串 ndarray 也会被
    # NumPy 解析（"1_0" → 10.0），绕过 _finite_observation 对下划线数字串
    # 的显式拒绝；必须逐元素走同一套严格解析。
    if (isinstance(values, np.ndarray) and arr.dtype.kind in "iuf"):
        out = arr.astype(np.float64, copy=False).ravel()
        if out.dtype.kind == "f" and not np.isfinite(out).all():
            # 数值 ndarray 的快速路径也必须遵守“非有限值记 NaN”的契约：
            # 否则 np.inf 会绕过 pearson 的成对缺失剔除，让整条相关系数
            # 变成 None，而不是忽略该观测后计算其余点。
            out = np.where(np.isfinite(out), out, np.nan)
        return out
    vals = [_finite_observation(v)
            for v in np.asarray(values, dtype=object).ravel()]
    return np.array([np.nan if v is None else v for v in vals],
                    dtype=np.float64)


def _open_unit_interval(value, name: str) -> float:
    """(0,1) 内的有限配置值（置信水平/显著性/功效）；bool/越界显式拒绝。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是 (0,1) 内的有限数值，收到 bool {value!r}")
    f = _finite_observation(value)
    if f is None or not 0.0 < f < 1.0:
        raise ValueError(f"{name} 必须是 (0,1) 内的有限数值，收到 {value!r}")
    return f


def _stable_mean_std(values):
    """有限观测的均值/样本标准差，先按最大绝对值缩放以避免溢出。

    ``np.mean``/``np.std`` 对接近 float 上限的合法观测会在求和/平方时
    溢出成 inf（例如 [1e308, 1e308] 的均值本应是 1e308，却得到 nan/inf）。
    统计量是尺度等变的：先除以 max|x| 再还原，可在不改变结果的前提下把
    中间量限制在 [-1,1]。仍不可表示时返回 None（缺失值语义），不把 inf
    写进报告。
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return None, None
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0, (0.0 if n >= 2 else None)
    scaled = arr / scale
    mean_scaled = float(scaled.mean())
    # 缩放后均值理论上在 [-1,1]；夹一下防止舍入让它略越界。
    mean_scaled = min(1.0, max(-1.0, mean_scaled))
    mean = mean_scaled * scale
    if n < 2:
        return (mean if math.isfinite(mean) else None), None
    sd = float(scaled.std(ddof=1)) * scale
    return (mean if math.isfinite(mean) else None,
            sd if math.isfinite(sd) else None)


def _stable_median(values):
    """有限观测的中位数（偶数个时用 a/2+b/2，避免 a+b 溢出）。

    numpy/Python 的常规中位数对偶数样本会先相加再除以 2；[1e308, 1e308]
    因此得到 inf，和均值/方差的溢出是同一类问题。
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = int(arr.size)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(arr[mid])
    return float(arr[mid - 1] / 2.0 + arr[mid] / 2.0)


def mean_std(values):
    """样本均值与样本标准差（ddof=1，无偏）。n<2 时 std 为 None。

    统计学约定：跨随机种子的结果必须报告 Mean ± Std（样本标准差），
    单种子结果无法评估方差，调用方应显式标注 n=1。使用缩放后的稳定
    计算，避免巨大但合法的观测在求和/平方时溢出。
    """
    vals = []
    for v in values:
        f = _finite_observation(v)
        if f is not None:
            vals.append(f)
    if not vals:
        return None, None
    return _stable_mean_std(vals)


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数的连分式（Lentz 法，Numerical Recipes 口径）。"""
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# 大自由度下 t 分布与标准正态的差异 < O(1/df)；df≥1e7 时临界值差异远小于
# 报告精度。直接走正态尾概率还能避开 x=df/(df+t²) 在 df 很大时舍入成 1.0、
# 使 p 值错误地变成 1 的灾难性抵消。
_T_NORMAL_DF = 1e7


def _normal_two_sided_p(t: float) -> float:
    """标准正态双侧 p 值（无 scipy）：erfc(|t|/√2)。

    合法但超出 float 上限的整数 t 视为 ±∞（双侧 p=0），而不是让
    ``float(t)`` 的 OverflowError 穿透；非数值输入返回 NaN。
    """
    try:
        at = abs(float(t))
    except OverflowError:
        return 0.0
    except (TypeError, ValueError):
        return float("nan")
    if math.isinf(at):
        return 0.0
    if math.isnan(at):
        return float("nan")
    return float(math.erfc(at / math.sqrt(2.0)))


def student_t_two_sided_p(t: float, df: float) -> float:
    """Student-t 分布双侧 p 值（无 scipy 依赖）。

    NaN 输入返回 NaN（不能当作最显著）；±Inf 返回 0。df 无效时返回 NaN。
    大自由度使用正态极限，避免 ``df/(df+t²)`` 在 float 中舍入到 1 后把
    p 错算成 1（df≈1e18 时该错误会把 0.0455 变成 1.0）。
    """
    try:
        t = float(t)
    except OverflowError:
        # 合法但超出 float 上限的整数 t：|t|→∞，双侧 p→0。
        return 0.0
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    try:
        df = float(df)
    except (TypeError, ValueError, OverflowError):
        if isinstance(df, (int, np.integer)) and not isinstance(
                df, (bool, np.bool_)) and df > 0:
            # 合法但超出 float 上限的整数自由度：等价于无穷自由度。
            return _normal_two_sided_p(t)
        return float("nan")
    if not math.isfinite(df):
        return (_normal_two_sided_p(t) if df > 0 else float("nan"))
    if df <= 0:
        return float("nan")
    if df >= _T_NORMAL_DF:
        return _normal_two_sided_p(t)
    x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def t_critical_95(df: float) -> float:
    """95% 双侧 Student-t 临界值（用现有 t 分布 p 值二分求根，无 scipy）。

    小样本（如验证窗口仅 eval_iters=20 个）用正态 1.96 会低估区间宽度；
    df<=0/无效返回 None。大自由度（含超出 float 上限的整数 df）走 t 的
    正态极限，df=1e18 时不会再把临界值错算成 8 或 90。
    """
    if df is None or isinstance(df, (bool, np.bool_)):
        return None
    try:
        df = float(df)
    except (TypeError, ValueError, OverflowError):
        if (isinstance(df, (int, np.integer))
                and not isinstance(df, (bool, np.bool_)) and df > 0):
            # 合法但超出 float 上限的整数自由度：t 已收敛到正态，用阈值
            # 自由度求根即可；显式 float("inf") 仍按“无效自由度”返回 None。
            df = _T_NORMAL_DF
        else:
            return None
    if not math.isfinite(df) or df <= 0:
        return None
    # 不要用 df>200 就退化为正态近似：df=201 的 t 临界值仍有 1.972，
    # 与 1.960 的差异会系统性收窄验证区间。现有二分代价很低，直接求根。
    lo, hi = 0.0, 1e3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _paired_diffs(a, b) -> np.ndarray:
    """按位置提取配对差 a_i − b_i，供 t 检验/置换检验/bootstrap 共用。

    配对按位置严格一一对应：两侧长度必须一致，None 也必须成对出现，否则抛
    ValueError（None 位置错开意味着种子集合没有对齐，静默过滤会把不同种子
    的观测配到一起）。任一观测非有限（NaN/Inf）或为 bool 的配对整体剔除：
    bool 不是可比较的数值观测（float(True)=1.0 会伪造配对差）。
    """
    a_list, b_list = list(a), list(b)
    if len(a_list) != len(b_list):
        raise ValueError(f"配对样本长度不一致: {len(a_list)} vs {len(b_list)}")
    diffs = []
    for x, y in zip(a_list, b_list):
        if (x is None) != (y is None):
            raise ValueError("配对样本 None 位置不一致，无法按位置一一对应")
        xf, yf = _finite_observation(x), _finite_observation(y)
        if xf is None or yf is None:
            continue
        d = xf - yf
        # 两个有限观测的差仍可能溢出（1e308 - (-1e308)）：该配对没有可表示
        # 的差值，与 TOST 的 _paired_diffs 同口径按缺失剔除，而不是把 inf
        # 送进均值/方差/置换检验。
        if math.isfinite(d):
            diffs.append(d)
    return np.asarray(diffs, dtype=np.float64)


def paired_sign_test(a, b) -> dict:
    """逐对差值符号的精确双侧二项检验（符号检验）。

    只使用 d_i = a_i − b_i 的符号，不使用幅度：少数离群种子可以把 t 检验
    的均值/方差拉向显著或掩盖稳定的小效应，符号检验则回答“多数配对方向
    是否一致”。零差按经典符号检验口径剔除（不提供方向信息），有效样本量
    为 nz = n_pos + n_neg；nz=0 时 p=1。H0 下正负概率各 1/2，双侧精确
    p = min(1, 2·P(Bin(nz, 1/2) ≤ min(n_pos, n_neg)))。返回
    {n, median_diff, n_pos, n_neg, n_zero, p_sign, exact}。
    """
    d = _paired_diffs(a, b)
    n = int(d.size)
    out = {"n": n, "median_diff": None, "n_pos": None, "n_neg": None,
           "n_zero": None, "p_sign": None, "exact": True}
    if n == 0:
        return out
    out["median_diff"] = _stable_median(d)
    n_pos = int(np.count_nonzero(d > 0))
    n_neg = int(np.count_nonzero(d < 0))
    n_zero = int(np.count_nonzero(d == 0))
    out.update({"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero})
    nz = n_pos + n_neg
    if nz == 0:
        out["p_sign"] = 1.0
        return out
    k = min(n_pos, n_neg)
    tail = sum(math.comb(nz, i) for i in range(k + 1)) / (2 ** nz)
    out["p_sign"] = min(1.0, 2.0 * tail)
    return out


def paired_t_test(a, b) -> dict:
    """同种子配对样本的双侧 t 检验：比较 a 与 b 两组一一对应的观测。

    实验协议中所有变体使用同一组随机种子训练，因此变体间比较天然配对：
    d_i = a_i - b_i 消除“种子运气”的公共方差，统计功效高于独立样本检验。
    返回 {n, mean_diff, std_diff, t, df, p, d_z}；n<2 时 t/df/p 为 None。
    d_z = mean_diff / std_diff 为配对 Cohen's d 效应量（方差为 0 时为 None）。
    标准差为 0（所有配对差完全相同）时：均值非 0 记 p=0，否则 p=1。
    """
    d = _paired_diffs(a, b)
    n = int(d.size)
    if n == 0:
        return {"n": 0, "mean_diff": None, "std_diff": None,
                "t": None, "df": None, "p": None, "d_z": None}
    mean_diff, std_diff = _stable_mean_std(d)
    if n < 2:
        return {"n": n, "mean_diff": mean_diff, "std_diff": None,
                "t": None, "df": None, "p": None, "d_z": None}
    if mean_diff is None or std_diff is None:
        # 差值本身有限，但均值/方差超出 float 表示范围：无法给出可靠检验。
        return {"n": n, "mean_diff": mean_diff, "std_diff": std_diff,
                "t": None, "df": n - 1, "p": None, "d_z": None}
    df = n - 1
    if std_diff == 0.0:
        p = 1.0 if mean_diff == 0.0 else 0.0
        # 零方差时 t 无有限值，方向必须跟随 mean_diff 的符号：
        # 常数负差（方法一致更差）也记 t=+inf 会把方向说反。
        t_inf = (math.copysign(float("inf"), mean_diff)
                 if mean_diff != 0.0 else 0.0)
        return {"n": n, "mean_diff": mean_diff, "std_diff": 0.0,
                "t": t_inf, "df": df, "p": p, "d_z": None}
    t = mean_diff / (std_diff / math.sqrt(n))
    return {"n": n, "mean_diff": mean_diff, "std_diff": std_diff,
            "t": float(t), "df": df, "p": student_t_two_sided_p(t, df),
            "d_z": float(mean_diff / std_diff)}


# 符号翻转置换检验的精确全枚举上限：2^16 = 65536 次枚举。
_SIGNFLIP_EXACT_MAX = 16


def paired_sign_flip_test(a, b, n_resamples: int = 20000,
                          seed: int = 0) -> dict:
    """配对样本的符号翻转置换检验（双侧，分布无关）。

    H0：配对差 d_i 关于 0 对称。检验统计量 T = |mean(d)|；H0 下对任意
    d_i 翻转符号得同样合法的样本，p = P(|mean(翻转样本)| >= |mean(d)|)。
    该检验不依赖正态假设，对 3~5 个种子的小样本推断比 t 检验更稳健：
    n<=16 时 2^n 次全枚举给出**精确** p 值；更大 n 用固定种子 Monte-Carlo
    （同一输入恒定复现）。零差值对任何翻转不变，天然计入分布。

    粒度边界（必须知情）：双侧 p 值的理论下限为 2/2^n——n=3 时 0.25、
    n=5 时 0.0625，n<6 时分布无关证据**本来就不可能**达到 α=0.05。因此
    小种子下"置换不显著"是粒度所限，不能单独推翻 t 检验结论；此时 t 的
    显著性完全依赖正态假设，应报告两者并补种子，而不是静默采信任何一方。

    配对规则与 paired_t_test 完全一致（_paired_diffs）。返回
    {n, mean_diff, p_perm, exact, n_resamples}；n<2 时不检验（p_perm=None，
    与 t 检验口径一致，避免 n=1 的"精确 p=1"被误读为证据）。
    """
    n_resamples = strict_int(n_resamples, "n_resamples")
    if n_resamples < 1:
        raise ValueError(f"n_resamples 必须 >= 1，收到 {n_resamples!r}")
    seed = strict_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed 必须 >= 0，收到 {seed!r}")
    d = _paired_diffs(a, b)
    n = int(d.size)
    out = {"n": n, "mean_diff": None, "p_perm": None,
           "exact": None, "n_resamples": None}
    if n == 0:
        return out
    mean_diff, _ = _stable_mean_std(d)
    out["mean_diff"] = mean_diff
    if n < 2:
        return out
    # 置换统计量只依赖差值的相对大小：先按 max|d| 缩放再做翻转/均值，
    # 避免 [1e308, 1e308] 这类合法观测在求和时溢出成 inf，使 p 值失真。
    scale = float(np.max(np.abs(d)))
    scaled = d / scale if scale > 0.0 else d
    obs = abs(float(scaled.mean()))
    if obs == 0.0:
        # 全零差：任何翻转都不改变均值，p=1（无可检效应）。
        out.update({"p_perm": 1.0, "exact": True, "n_resamples": 2 ** n})
        return out
    if n <= _SIGNFLIP_EXACT_MAX:
        # 精确全枚举：2^n 个翻转样本（含观测自身与其全反号）。
        signs = (np.arange(2 ** n, dtype=np.int64)[:, None]
                 >> np.arange(n, dtype=np.int64)[None, :]) & 1
        flipped = scaled[None, :] * (1.0 - 2.0 * signs)
        ge = int((np.abs(flipped.mean(axis=1)) >= obs - 1e-15).sum())
        out.update({"p_perm": ge / float(2 ** n), "exact": True,
                    "n_resamples": int(2 ** n)})
        return out
    # Monte-Carlo：固定种子可复现。随机翻转每个 d_i 的符号（±1），
    # 双侧口径把观测自身与其全反号一并计入（+2），与精确枚举一致。
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(int(n_resamples), n)) * 2.0 - 1.0
    stat = np.abs((scaled[None, :] * flips).mean(axis=1))
    ge = int((stat >= obs - 1e-15).sum())
    out.update({"p_perm": (ge + 2) / float(n_resamples + 2),
                "exact": False, "n_resamples": int(n_resamples)})
    return out


def bootstrap_mean_ci(values, n_resamples: int = 10000, ci: float = 0.95,
                      seed: int = 0):
    """样本均值的百分位 bootstrap 置信区间（配对差/任意标量观测通用）。

    None/bool/非有限值剔除后 n<2 时返回 None（单点无区间可言）。
    固定随机种子，同一输入恒定复现。返回 (lo, hi)。
    """
    n_resamples = strict_int(n_resamples, "n_resamples")
    if n_resamples < 1:
        raise ValueError(f"n_resamples 必须 >= 1，收到 {n_resamples!r}")
    ci = _open_unit_interval(ci, "ci")
    seed = strict_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed 必须 >= 0，收到 {seed!r}")
    vals = []
    for v in values:
        f = _finite_observation(v)
        if f is not None:
            vals.append(f)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        means = np.zeros(n_resamples, dtype=np.float64)
    else:
        # 缩放后重采样求均值，避免 [1e308, 1e308] 的 numpy 求和溢出；
        # 均值是尺度等变的，还原后与未缩放口径一致。
        means = (arr / scale)[idx].mean(axis=1) * scale
    if not np.all(np.isfinite(means)):
        return None
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
        return None
    return float(lo), float(hi)


def min_detectable_effect(std_diff, n: int, alpha: float = 0.05,
                          power: float = 0.8):
    """配对设计在给定 (配对差标准差, 种子数) 下的最小可检测效应（MDE）。

    正态近似（大样本 z 口径）：MDE = (z_{1-α/2} + z_{power}) · std/√n。
    用途：区分“不显著 = 无效应”与“不显著 = 功效不足”。报告的 MDE 远大于
    声称的效应量时，应先补种子而不是下结论；n>=3 时该近似已足够指引决策
    （严格 t 口径的临界值略大，近似值偏乐观，作解释性参考而非判定门槛）。
    标准差不可估计（None/0）或 n<2 时返回 None。
    """
    if std_diff is None or n is None:
        return None
    std = _finite_observation(std_diff)
    if std is None or std <= 0:
        return None
    n = strict_int(n, "n")
    if n < 2:
        return None
    alpha = _open_unit_interval(alpha, "alpha")
    power = _open_unit_interval(power, "power")
    if not (math.isfinite(std) and std > 0.0):
        return None
    nd = statistics.NormalDist()
    z = nd.inv_cdf(1.0 - float(alpha) / 2.0) + nd.inv_cdf(float(power))
    # n 是合法的 Python 大整数时 math.sqrt(n) 会先尝试转 float 并抛
    # OverflowError；但 √n 本身完全可能仍可表示（如 n=10**400 → 1e200）。
    # 用对数求根，避免把“n 很大 → MDE 很小”误当成不可计算。
    try:
        root_n = math.sqrt(n)
    except OverflowError:
        try:
            root_n = math.exp(0.5 * math.log(n))
        except (OverflowError, ValueError):
            return None
    if not math.isfinite(root_n) or root_n <= 0.0:
        return None
    out = float(z * std / root_n)
    return out if math.isfinite(out) else None


def wilson_score_ci(k: int, n: int, ci: float = 0.95):
    """二值比例的 Wilson 得分置信区间（小样本下远优于正态近似区间）。

    k 为成功数、n 为试验数；n<=0 返回 None。k=0 时下界恰为 0、k=n 时上界
    恰为 1（不会像 Wald 区间那样越界）。返回 (lo, hi)。
    """
    if n is None or k is None:
        return None
    n = strict_int(n, "n")
    k = strict_int(k, "k")
    if n <= 0:
        return None
    ci = _open_unit_interval(ci, "ci")
    k = int(min(max(k, 0), n))
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - ci) / 2.0)
    z2 = z * z
    # 用整数除法先求 1/n 与 1/n²：n 是合法大整数（如 10**400）时
    # ``z2 / n`` 会先把 n 转成 float 并抛 OverflowError，而 ``1 / n``
    # 走 Python 的整数真除法，可安全下溢到 0.0。公式与原来逐项等价。
    n_inv = 1 / n
    n2_inv = 1 / (n * n)
    z2_n = z2 * n_inv
    if k == 0:
        # 端点闭式：p=0 时区间为 [0, z²/(n+z²)]（避免浮点残差 1e-17 进 JSON）
        return 0.0, z2_n / (1.0 + z2_n)
    if k == n:
        # p=1 时区间为 [n/(n+z²), 1]
        return 1.0 / (1.0 + z2_n), 1.0
    p = k / n
    den = 1.0 + z2_n
    center = (p + z2_n / 2.0) / den
    half = z * math.sqrt(p * (1.0 - p) * n_inv
                         + z2 * n2_inv / 4.0) / den
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact_p(candidate_only: int, reference_only: int) -> float:
    """配对二分类的精确 McNemar 检验（双侧，等价于 discordant 上的符号检验）。

    只使用两侧结果不同的配对（candidate_only / reference_only），相同结果
    不提供方向信息。H0 下每个 discordant 对等概率偏向任一侧，故
    p = 2·P(Bin(n, 0.5) ≤ min(b, c))（截断到 1）。n 可能达数千，用
    log-gamma + log-sum-exp 稳定求和，避免组合数溢出。
    """
    # discordant 计数必须是非负整数：bool/小数/负数会被 int() 静默截断或
    # 让后续组合数求和退化为空序列（max() 抛 ValueError），必须显式拒绝。
    candidate_only = strict_int(candidate_only, "candidate_only")
    reference_only = strict_int(reference_only, "reference_only")
    if candidate_only < 0 or reference_only < 0:
        raise ValueError(
            "candidate_only/reference_only 必须是 >= 0 的整数，收到 "
            f"{candidate_only}/{reference_only}")
    n = candidate_only + reference_only
    if n <= 0:
        return 1.0
    k = min(candidate_only, reference_only)
    log2 = math.log(2.0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1)
            - math.lgamma(n - i + 1) - n * log2 for i in range(k + 1)]
    m = max(logs)
    log_tail = m + math.log(math.fsum(math.exp(x - m) for x in logs))
    return min(1.0, 2.0 * math.exp(log_tail))


def holm_bonferroni(pvals) -> list:
    """Holm–Bonferroni 逐步下降法（step-down）校正的单步族错误率 p 值。

    同一次汇总会对多个 (候选, 参照) 配对做显著性检验；不校正时 α=0.05 下
    检验越多、至少一次假阳性的概率越大（family-wise error）。Holm 法在
    任意依赖结构下都控制 FWER，且比 Bonferroni 保守度更低。

    输入为 p 值列表（允许 None 表示无法检验，不占家族大小；NaN 与 None 同义），
    返回同序列表：
    p_adj_(i) = max_{j<=i} min(1, (m-j+1) * p_(j))（p 升序的第 j 个观测），
    保证校正后 p 随原始 p 单调不减且 >= 原始 p。
    NaN 与 None 一样不参与排序，也不计入家族大小。
    """
    # p 必须落在 [0,1] 且为有限数值：bool 是 int 子类（True→1），字符串/
    # 容器/NaN/Inf/负数/大于 1 的值都不是可检验的 p，按缺失处理而不参与
    # 家族大小；否则排序会混入不可比较类型或把损坏值当成显著/不显著。
    norm = []
    for p in pvals:
        if p is None or isinstance(p, (bool, np.bool_)):
            norm.append(None)
            continue
        try:
            f = float(p)
        except (TypeError, ValueError, OverflowError):
            norm.append(None)
            continue
        norm.append(f if math.isfinite(f) and 0.0 <= f <= 1.0 else None)
    idx = [i for i, p in enumerate(norm) if p is not None]
    m = len(idx)
    adj = [None] * len(pvals)
    if m == 0:
        return adj
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: norm[i])):
        val = min(1.0, norm[i] * (m - rank))
        running = max(running, val)
        adj[i] = running
    return adj


def _score_label_arrays(scores, labels, label_dtype=None):
    """分数/标签按位置对齐；长度不一致显式拒绝，避免静默截断或错位。

    非数值 dtype（字符串/对象/bool）走与 ``_obs_array`` 相同的严格解析：
    ``float("1_0") == 10.0`` 会让坏分数冒充远高于其他样本的观测，而
    ``True`` 也不能凭 ``float(True)==1.0`` 变成合法分数/标签。纯数字
    ndarray 仍走 astype 快路，保持原有性能与逐位结果。
    """
    scores_arr = np.asarray(scores)
    if scores_arr.dtype.kind in "iuf":
        scores = scores_arr.astype(np.float64, copy=False)
    else:
        scores = _obs_array(scores)
    labels_arr = np.asarray(labels)
    if labels_arr.dtype.kind in "iuf":
        labels = (labels_arr.astype(label_dtype)
                  if label_dtype is not None else labels_arr)
    else:
        labels = _obs_array(labels)
    if scores.size != labels.size:
        raise ValueError(
            f"scores/labels 长度不一致: {scores.size} vs {labels.size}")
    return scores.ravel(), labels.ravel()


def auroc(scores, labels) -> float:
    """秩和法 AUROC（Mann-Whitney U），含并列取平均秩；labels: 1=正类。

    用于检验“门控分数能否区分这次读取是否找对了信息”。

    类别极度不平衡时 AUROC 会被巨大的负样本基数抬高（FPR 增长缓慢），
    必须与 `average_precision` 并报；正类率 <5% 时应以 PR-AUC 为主。
    """
    scores, labels = _score_label_arrays(scores, labels)
    # 与 average_precision 同一口径：先剔除非有限分数。
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    # 并列取平均秩：等值样本按等值分组获得同一平均秩，与组内先后次序无关。
    order = np.argsort(allv)
    # 有序数组的等值组起点用相邻差分直接定位（与 np.unique 的
    # return_index 结果一致）。
    # 组内平均秩（1-based）= 首名次（0-based 首个位置）+ (组大小+1)/2
    sorted_v = allv[order]
    first = np.concatenate([[0], np.flatnonzero(np.diff(sorted_v)) + 1])
    counts = np.diff(np.append(first, [sorted_v.size]))
    avg_ranks = first.astype(np.float64) + (counts + 1.0) / 2.0
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.repeat(avg_ranks, counts)
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def average_precision(scores, labels) -> float:
    """PR-AUC（average precision，逐步积分，含并列分数分组）；labels: 1=正类。

    为什么需要它：类别不平衡时 ROC-AUC 会因负样本基数巨大、FPR 增长缓慢
    而给出过于乐观的分数（正类率 <5% 即须警惕，<0.1% 时优先看 PR-AUC）。
    PR-AUC 的“随机猜”基线恰为正类率（prevalence），因此可直接对照判断
    分数是否真的有判别力。分组口径：同分样本整体进入当前阈值，取
    P=TP/(TP+FP) 与 R=TP/n_pos，AP = Σ (R_k − R_{k−1}) · P_k，
    与并列取平均秩的 AUROC 一样对并列稳健。单一类别或空输入返回 NaN。
    """
    scores, labels = _score_label_arrays(scores, labels, np.float64)
    # 与 auroc 同口径：只承认 0/1 标签。非二值标签若被计入累计正类数，
    # TP 可超过 n_pos，AP 会算出 >1 的非法值（例如 label=2 会把 AP 抬到
    # 2.67），直接污染“优于随机基线”的结论。
    m = np.isfinite(scores) & ((labels == 0.0) | (labels == 1.0))
    scores, labels = scores[m], labels[m]
    n_pos = int((labels == 1).sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    # 等值样本在分组边界处整体计入，结果与组内先后次序无关。
    order = np.argsort(-scores)
    s_sorted, l_sorted = scores[order], labels[order]
    # 分数降序后按等值组切边界（合并并列，避免排序稳定性影响结果）
    cuts = np.flatnonzero(np.diff(s_sorted)) + 1
    ends = np.concatenate([cuts, [len(s_sorted)]])
    # 逐组累计：tp 为按分数降序的累计正类数（组末位置取值），
    # 组末累计样本数 b 即 tp + fp，故 precision = tp / b。
    tp = np.cumsum(l_sorted)[ends - 1]
    recalled = tp / n_pos
    precision = tp / ends
    # 逐步积分与逐组循环同序累加，保持逐位一致。
    prev_recall = np.concatenate([[0.0], recalled[:-1]])
    return float(np.cumsum((recalled - prev_recall) * precision)[-1])


def roc_pr_aucs(scores, labels):
    """同时计算 AUROC 与 PR-AUC（average precision）。

    与 `auroc` / `average_precision` 在二值标签下逐位一致：AUROC 的并列
    平均秩与 AP 的等值整组累计都以“按分数排序后的等值分组”为基础，分组
    结果与组内次序无关。两个指标共享同一份升序排序的等值分组与秩映射。

    约定与两函数一致：先按分数有限性过滤；AUROC 需正负两类齐备，AP 需
    正类率既不等于 0 也不等于 1，否则对应值为 NaN（另一指标仍照常计算）。
    返回 (auroc, ap)。
    """
    scores, labels = _score_label_arrays(scores, labels)
    m = np.isfinite(scores)
    sc, lb = scores[m], labels[m]
    # 非二值标签（调用方不应出现）语义含糊：average_precision 会把非 0/1
    # 标签计入累计，与按 0/1 计数推导的快速路径不一致；此时退化为直接
    # 调用两个原函数，保证任何输入下都逐位一致。
    is_pos = lb == 1
    is_neg = lb == 0
    if not bool((is_pos | is_neg).all()):
        return auroc(sc, lb), average_precision(sc, lb)
    n_pos = int(is_pos.sum())
    n_neg = int(is_neg.sum())
    need_auc = n_pos > 0 and n_neg > 0
    need_ap = n_pos > 0 and n_pos != len(lb)
    if not (need_auc or need_ap):
        return float("nan"), float("nan")
    order = np.argsort(sc)
    sv = sc[order]
    lv = lb[order].astype(np.float64)
    n = sv.size
    auc = float("nan")
    ap = float("nan")
    # 无并列（浮点分数几乎总如此）的路径：组大小恒为 1，用“正类在升序
    # 位置上的前缀和”这一等价形式。整数计数在 float64 下精确，结果与等值
    # 分组通用路径逐位一致（tests/test_utils.py 有逐位对照）。
    if n < 2 or bool((sv[1:] == sv[:-1]).any()):
        # 等值组首位置（升序）；组内平均秩（1-based）= 首位置 + (组大小+1)/2
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sv)) + 1])
        counts = np.diff(np.append(starts, [n]))
        # 每组正类数：AUROC 与 AP 共用（整数计数在 float64 下精确）。
        g_pos = np.add.reduceat(lv, starts)
        if need_auc:
            avg_ranks = starts.astype(np.float64) + (counts + 1.0) / 2.0
            # 正类平均秩之和 = Σ_组 组平均秩×组内正类数。avg_ranks 是
            # 半整数、g_pos 是小整数，乘积与求和全程在 float64 下精确，
            # 与散射回各位置后逐个求和的旧口径逐位一致。
            r_pos = float((avg_ranks * g_pos).sum())
            u = r_pos - n_pos * (n_pos + 1) / 2.0
            auc = float(u / (n_pos * n_neg))
        if need_ap:
            # AP 需要按分数降序的分组；降序组末的累计样本数 = n - 组首位
            # 置、累计正类数 = n_pos - 升序前缀正类数，与
            # average_precision 中 cumsum 在组末取到的值逐位一致。
            tp_desc = (n_pos - np.concatenate([[0.0], np.cumsum(g_pos)[:-1]]))[::-1]
            ends_desc = (n - starts)[::-1]
            recalled = tp_desc / n_pos
            precision = tp_desc / ends_desc
            prev_recall = np.concatenate([[0.0], recalled[:-1]])
            ap = float(np.cumsum((recalled - prev_recall) * precision)[-1])
        return auc, ap
    if need_auc:
        # 正类平均秩之和 = Σ(升序位置+1) = n_pos + Σ位置；两项都是不超过
        # n(n+1)/2 的精确整数（float64/ int64 均无舍入），与散射路径同值。
        r_pos = float(n_pos + int(np.flatnonzero(lv).sum()))
        u = r_pos - n_pos * (n_pos + 1) / 2.0
        auc = float(u / (n_pos * n_neg))
    if need_ap:
        # 逐位置前缀正类数（升序）→ 段末累计口径，与通用路径的
        # reduceat(组大小=1)=identity 完全一致。
        tp_desc = (n_pos - np.concatenate([[0.0], np.cumsum(lv)[:-1]]))[::-1]
        ends_desc = np.arange(1, n + 1)
        recalled = tp_desc / n_pos
        precision = tp_desc / ends_desc
        prev_recall = np.concatenate([[0.0], recalled[:-1]])
        ap = float(np.cumsum((recalled - prev_recall) * precision)[-1])
    return auc, ap


def calibration_metrics(scores, labels, n_bins: int = 10) -> dict:
    """概率校准指标：Brier、ECE、MCE 与可靠性分箱。

    为什么需要：门控 g 是 sigmoid 输出，被当作“该不该写入”的概率/软权重
    使用；AUROC/PR-AUC 只衡量排序，无法回答“g=0.8 时命中率是否真的约
    80%”。Brier 是严格评分规则（越小越好），ECE 是分箱后的 |置信−准确率|
    加权平均；两者与常数基率预测（Brier 基线）并报，才能区分“排序好”
    与“校准好”。ECE 依赖等宽分箱，样本少时噪声大，只作描述性证据。

    返回 {n, n_pos, base_rate, brier, brier_constant, brier_skill,
    ece, mce, n_bins, bins, ece_adaptive, mce_adaptive,
    brier_reliability, brier_resolution, brier_uncertainty,
    brier_within_score, brier_within_cov, brier_binning,
    brier_decomposition_bins}；分数先 clip 到 [0,1]，非有限值与非法标签剔除；
    单一类别时 brier 仍可算，但 ECE 的“校准”语义不完整，返回 None。

    `ece`/`mce` 用等宽分箱（与 `bins` 一致）；`ece_adaptive`/`mce_adaptive`
    改用等频分箱（按分数排序后均分样本）。等宽分箱在分数聚集或样本少时
    可能大量空箱，单个样本独占一箱还会让 ECE 机械地趋近 0；等频口径给出
    对分箱选择不敏感的第二组证据。两者都只作描述性证据，不替代 Brier。
    Brier 的 Murphy 分解用等频箱计算。标准三项
    `reliability - resolution + uncertainty` 对应“箱均值预测”的 Brier；
    原始分数在箱内不为常数时，还须加上箱内项
    `brier_binning = within_score - 2*within_cov`，恒等式为
    `brier = reliability - resolution + uncertainty + brier_binning`。
    其中 `within_score` 是箱内分数方差，`within_cov` 是箱内分数与标签
    偏离箱均值的协方差；箱内分数恒定时两者均为 0。
    """
    n_bins = strict_int(n_bins, "n_bins")
    scores = _obs_array(scores)
    labels = _obs_array(labels)
    if scores.size != labels.size:
        raise ValueError(
            f"scores/labels 长度不一致: {scores.size} vs {labels.size}")
    m = np.isfinite(scores) & ((labels == 0.0) | (labels == 1.0))
    scores, labels = scores[m], labels[m]
    n = int(scores.size)
    if n == 0:
        return {"n": 0, "n_pos": 0, "base_rate": None, "brier": None,
                "brier_constant": None, "brier_skill": None, "ece": None,
                "mce": None, "n_bins": int(n_bins), "bins": [],
                "ece_adaptive": None, "mce_adaptive": None,
                "brier_reliability": None, "brier_resolution": None,
                "brier_uncertainty": None,
                "brier_within_score": None, "brier_within_cov": None,
                "brier_binning": None,
                "brier_decomposition_bins": "equal_frequency"}
    scores = np.clip(scores, 0.0, 1.0)
    n_pos = int((labels == 1.0).sum())
    base_rate = float(n_pos / n)
    brier = float(np.mean((scores - labels) ** 2))
    brier_constant = float(np.mean((base_rate - labels) ** 2))
    skill = (1.0 - brier / brier_constant
             if brier_constant > 0 else None)
    out = {"n": n, "n_pos": n_pos, "base_rate": base_rate,
           "brier": brier, "brier_constant": brier_constant,
           "brier_skill": skill, "ece": None, "mce": None,
           "n_bins": int(n_bins), "bins": [],
           "ece_adaptive": None, "mce_adaptive": None,
           "brier_reliability": None, "brier_resolution": None,
           "brier_uncertainty": None,
           "brier_within_score": None, "brier_within_cov": None,
           "brier_binning": None,
           "brier_decomposition_bins": "equal_frequency"}
    if len(set(labels.tolist())) != 2 or n_bins < 1:
        return out
    idx = np.clip((scores * n_bins).astype(np.int64), 0, n_bins - 1)
    ece, mce, bins = 0.0, 0.0, []
    for b in range(n_bins):
        mb = idx == b
        nb = int(mb.sum())
        if nb == 0:
            bins.append({"bin": b, "n": 0, "mean_score": None,
                         "mean_label": None, "gap": None})
            continue
        mean_score = float(scores[mb].mean())
        mean_label = float(labels[mb].mean())
        gap = abs(mean_score - mean_label)
        ece += (nb / n) * gap
        mce = max(mce, gap)
        bins.append({"bin": b, "n": nb, "mean_score": mean_score,
                     "mean_label": mean_label, "gap": gap,
                     "lo": b / n_bins, "hi": (b + 1) / n_bins})
    out["ece"] = float(ece)
    out["mce"] = float(mce)
    out["bins"] = bins
    # 等频（adaptive）分箱：按分数排序后均分，避免等宽箱在分数聚集/小样本
    # 下大量为空，或每个样本独占一箱使 ECE 被机械压低。箱内仍用样本加权，
    # 因此 ECE 的含义与等宽口径完全一致，只是划分方式不同。
    groups = np.array_split(np.argsort(scores, kind="mergesort"),
                            int(n_bins))
    ece_adapt, mce_adapt = 0.0, 0.0
    reliability, resolution = 0.0, 0.0
    within_score, within_cov = 0.0, 0.0
    for g in groups:
        nb = int(g.size)
        if nb == 0:
            continue
        gs, gl = scores[g], labels[g]
        mean_score = float(gs.mean())
        mean_label = float(gl.mean())
        gap = abs(mean_score - mean_label)
        ece_adapt += (nb / n) * gap
        mce_adapt = max(mce_adapt, gap)
        reliability += (nb / n) * (mean_score - mean_label) ** 2
        resolution += (nb / n) * (mean_label - base_rate) ** 2
        within_score += (nb / n) * float(np.mean((gs - mean_score) ** 2))
        within_cov += (nb / n) * float(np.mean((gs - mean_score)
                                                * (gl - mean_label)))
    out["ece_adaptive"] = float(ece_adapt)
    out["mce_adaptive"] = float(mce_adapt)
    out["brier_reliability"] = float(reliability)
    out["brier_resolution"] = float(resolution)
    out["brier_uncertainty"] = float(base_rate * (1.0 - base_rate))
    out["brier_within_score"] = float(within_score)
    out["brier_within_cov"] = float(within_cov)
    out["brier_binning"] = float(within_score - 2.0 * within_cov)
    return out


def f1_scores(y_true, y_pred, labels=None) -> dict:
    """二/多分类 F1：同时给出 macro / micro / weighted 三种平均口径与逐类明细。

    为什么必须注明平均口径：类别不平衡时三者可能差异巨大——macro 对每个
    类等权（对稀有类敏感，是不平衡场景的诚实口径）、micro 汇总全局
    TP/FP/FN（多分类下偏向多数类）、weighted 按类支持度加权（同样被多数类
    主导）。任何只写“F1”的表格都不完整，本函数一次给全并显式命名。

    返回 {macro, micro, weighted, per_class: {label: {precision, recall, f1,
    support}}, labels}；单一类别或无样本时值为 None（表示不可计算）。
    """
    def _class_ids(values, name):
        # 类别是整数身份：int(1.5)=1、int(True)=1 会把不同类静默合并，
        # 同一批预测的 macro/micro/weighted 也就全部失真。
        i64 = np.iinfo(np.int64)
        arr = np.asarray(values)
        if isinstance(values, np.ndarray) and arr.dtype.kind in "iu":
            # uint64 的值可能超过 int64 上限；直接 astype 会回绕成负数，
            # 把一个大类别静默合并到另一个合法标签上。
            if arr.size:
                lo, hi = int(arr.min()), int(arr.max())
                if lo < int(i64.min) or hi > int(i64.max):
                    raise ValueError(
                        f"{name} 类别超出 int64 可表示范围: [{lo}, {hi}]")
            return arr.astype(np.int64, copy=False).ravel()
        vals = []
        for i, v in enumerate(np.asarray(values, dtype=object).ravel()):
            iv = strict_int(v, f"{name}[{i}]")
            # Python 大整数无法作为 int64 标签参与逐元素比较；显式拒绝，
            # 而不是让 NumPy 在 astype 时抛裸 OverflowError。
            if iv < int(i64.min) or iv > int(i64.max):
                raise ValueError(
                    f"{name}[{i}] 类别超出 int64 可表示范围: {iv}")
            vals.append(iv)
        return np.array(vals, dtype=np.int64)

    y_true = _class_ids(y_true, "y_true")
    y_pred = _class_ids(y_pred, "y_pred")
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true/y_pred 长度不一致: {y_true.size} vs {y_pred.size}")
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [strict_int(x, "labels") for x in labels]
    i64 = np.iinfo(np.int64)
    for lb in labels:
        if lb < int(i64.min) or lb > int(i64.max):
            raise ValueError(f"labels 中的类别超出 int64 可表示范围: {lb}")
    if len(set(labels)) != len(labels):
        # per_class 以标签为键：重复标签会覆盖明细，同时让 macro/weighted
        # 对同一类重复计权，三种平均口径之间的差异失真。
        raise ValueError(f"labels 不得重复: {labels!r}")
    per_class = {}
    f1s, supports = [], []
    total = int(y_true.size)
    tp_all = fp_all = fn_all = 0
    for lb in labels:
        tp = int(((y_true == lb) & (y_pred == lb)).sum())
        fp = int(((y_true != lb) & (y_pred == lb)).sum())
        fn = int(((y_true == lb) & (y_pred != lb)).sum())
        support = int((y_true == lb).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[lb] = {"precision": prec, "recall": rec, "f1": f1,
                         "support": support}
        f1s.append(f1)
        supports.append(support)
        tp_all += tp
        fp_all += fp
        fn_all += fn
    # 单一类别（观测到的 y_true ∪ y_pred 只有一个类）没有可区分的正/负
    # 类：此时 micro/macro/weighted 都机械地等于 1，会让“全判同一类”的
    # 退化预测看起来完美。与空输入一样按不可计算返回 None，逐类明细保留。
    observed = set(y_true.tolist()) | set(y_pred.tolist())
    if not labels or total == 0 or len(observed) < 2:
        return {"macro": None, "micro": None, "weighted": None,
                "per_class": per_class, "labels": labels, "n": total}
    macro = float(np.mean(f1s)) if f1s else None
    micro = (2 * tp_all / (2 * tp_all + fp_all + fn_all)
             if (2 * tp_all + fp_all + fn_all) else None)
    weighted = (float(np.average(f1s, weights=supports))
                if sum(supports) else None)
    return {"macro": macro, "micro": float(micro) if micro is not None else None,
            "weighted": weighted, "per_class": per_class, "labels": labels,
            "n": total}
