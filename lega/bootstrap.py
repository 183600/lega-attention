"""进程级 PYTHONHASHSEED 引导：让编排进程自身的哈希盐固定下来。

背景
----
``PYTHONHASHSEED`` 只在**解释器启动前**设置才影响当前进程；运行期写
``os.environ`` 只对之后启动的子进程有效。种子审计（``lega.utils.seed_report``）
已经把“解释器启动值 / 子进程 env / 是否对当前进程生效”分成三个字段如实
记录，但编排进程（``run_all``）此前始终没有固定哈希盐：任何依赖
``set``/``frozenset`` 哈希顺序的路径（集合遍历、由集合构造输出顺序）在不同
进程启动下可能给出不同顺序，同一 commit 的两次运行因而可能产出顺序不同的
产物或报告。这类差异不会被“种子相同”的声明覆盖，却会被读者当作结果不稳定。

本模块提供
----------
``hash_seed_state()``
    只读地报告当前进程的哈希盐状态（是否固定、取值、来源），供审计字段使用。
``ensure_hash_seed()``
    必要时以固定哈希盐重新 ``execve`` 自身：幂等（已固定则什么都不做）、
    可关闭（``LEGA_HASH_SEED_BOOTSTRAP=0``）、失败时安全退回（不阻断实验）。

环境变量
--------
``LEGA_HASH_SEED``
    引导时使用的哈希盐（默认 ``0``，即关闭哈希随机化）。
``LEGA_HASH_SEED_BOOTSTRAP``
    设为 ``0``/``off``/``false`` 时完全关闭引导（用于排障或自定义入口）。

边界
----
哈希盐固定只影响 ``str``/``bytes`` 的哈希值，因此**不影响**任何随机数流
（Python ``random``、NumPy、PyTorch 的随机性由各自的 seed 决定）；它只保证
集合/字典的迭代顺序稳定。跨硬件的逐位一致性仍由
``lega.utils.set_seed`` 与环境锁负责，本模块不承诺这一点。
"""

import os
import sys

HASH_SEED_ENV = "PYTHONHASHSEED"
HASH_SEED_VALUE_ENV = "LEGA_HASH_SEED"
HASH_SEED_BOOTSTRAP_ENV = "LEGA_HASH_SEED_BOOTSTRAP"
BOOTSTRAP_GUARD_ENV = "LEGA_HASH_SEED_BOOTSTRAPPED"

MAX_HASH_SEED = 2**32 - 1
DEFAULT_HASH_SEED = 0

_FALSE_VALUES = ("0", "off", "false", "no", "none")


def parse_hash_seed(value):
    """环境值 → 合法哈希盐整数；未设置/非法/``random`` 返回 None。

    ``PYTHONHASHSEED`` 允许 ``random``（显式随机）与 0..2^32-1 的整数；
    其它取值（如 ``1.5``、``abc``、负数）不是合法哈希盐，不能截断后使用。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "random":
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    if number < 0 or number > MAX_HASH_SEED:
        return None
    return number


def hash_seed_state(env=None) -> dict:
    """当前进程哈希盐状态（只读，不改变任何状态）。

    返回 ``env``（原始环境值）、``value``（解析出的整数或 None）、``fixed``
    （是否已固定）、``randomized``（是否处于随机化状态）与 ``source``
    （``env``/``random``/``unset``/``invalid``）。
    """
    env = os.environ if env is None else env
    raw = env.get(HASH_SEED_ENV)
    text = None if raw is None else str(raw).strip()
    value = parse_hash_seed(text)
    randomized = text is None or text.lower() == "random"
    if value is not None:
        source = "env"
    elif raw is None:
        source = "unset"
    elif randomized:
        source = "random"
    else:
        source = "invalid"
    return {
        "env": raw,
        "value": value,
        "fixed": value is not None,
        "randomized": randomized,
        "source": source,
    }


def desired_hash_seed(env=None) -> int:
    """引导时使用的哈希盐：``LEGA_HASH_SEED``（默认 0），非法值退回默认。"""
    env = os.environ if env is None else env
    raw = env.get(HASH_SEED_VALUE_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_HASH_SEED
    value = parse_hash_seed(raw)
    return DEFAULT_HASH_SEED if value is None else value


def bootstrap_enabled(env=None) -> bool:
    """引导是否启用（``LEGA_HASH_SEED_BOOTSTRAP=0/off/false`` 关闭）。"""
    env = os.environ if env is None else env
    raw = str(env.get(HASH_SEED_BOOTSTRAP_ENV, "") or "").strip().lower()
    return raw not in _FALSE_VALUES


def package_root() -> str:
    """``lega`` 包所在目录的父目录（源码仓库根或 site-packages）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def module_name_for_script(path) -> str:
    """脚本路径 → 对应的 ``-m`` 模块名；不在 ``lega`` 包内时返回 ""。

    包内脚本必须以 ``-m`` 方式重新执行：``lega/train.py`` 使用相对导入
    （``from .data import ...``），直接当脚本执行会因“attempted relative
    import with no known parent package”失败。
    """
    if not path:
        return ""
    absolute = os.path.abspath(path)
    root = package_root()
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:
        return ""
    if relative.startswith(".."):
        return ""
    parts = relative.split(os.sep)
    if not parts or not parts[-1].endswith(".py"):
        return ""
    parts[-1] = parts[-1][: -len(".py")]
    if parts[0] != "lega":
        return ""
    return ".".join(parts)


def console_script_module(path) -> str:
    """控制台脚本 → 其调用的顶层模块名；无法判定时返回 ""。

    安装后的 ``lega-run`` 是一个生成的包装脚本（内容里写着
    ``run_all:main_entry``），重新执行时必须用 ``-m run_all``，否则会把
    包装脚本当 Python 源码执行。
    """
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(4096)
    except OSError:
        return ""
    if "run_all" in content:
        return "run_all"
    return ""


def reexec_command(argv=None) -> list:
    """构造重新执行自身的命令；无法安全重建时返回空列表。

    支持三种入口：``python -m lega.xxx``（包内脚本 → ``-m`` 形式）、
    ``python run_all.py``（顶层脚本 → 按路径执行）与安装后的控制台脚本
    （``lega-run`` → ``-m run_all``）。
    """
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        return []
    script = argv[0]
    args = [str(a) for a in argv[1:]]
    if script.endswith(".py"):
        module = module_name_for_script(script)
        if module:
            return [sys.executable, "-m", module, *args]
        return [sys.executable, os.path.abspath(script), *args]
    module = console_script_module(script)
    if module:
        return [sys.executable, "-m", module, *args]
    return []


def ensure_hash_seed(argv=None, env=None, quiet: bool = False) -> bool:
    """必要时重新执行当前进程，使哈希盐在**解释器启动前**就已固定。

    返回 True 表示当前进程的哈希盐已经固定（无需重启）；返回 False 表示
    未固定且无法安全重启——调用方应继续运行并如实记录状态，不得假装已固定。
    发生重新执行时本函数不会返回：进程映像被替换。

    重新执行是幂等的：环境里已有合法 ``PYTHONHASHSEED`` 时直接返回；
    已经尝试过一次（``LEGA_HASH_SEED_BOOTSTRAPPED=1``）时不再重试，避免
    入口无法重建时的死循环。

    测试运行器（pytest 已在 sys.modules 中）不做引导：替换进程会破坏测试
    框架的输出捕获与退出码，测试环境不是实验环境，固定哈希盐的收益为零。
    """
    env = os.environ if env is None else env
    if hash_seed_state(env)["fixed"]:
        return True
    if "pytest" in sys.modules:
        return False
    if not bootstrap_enabled(env):
        return False
    if str(env.get(BOOTSTRAP_GUARD_ENV, "")) == "1":
        return False
    if os.name == "nt" or not hasattr(os, "execve"):
        return False
    command = reexec_command(argv)
    if not command:
        return False
    new_env = dict(env)
    new_env[HASH_SEED_ENV] = str(desired_hash_seed(env))
    new_env[BOOTSTRAP_GUARD_ENV] = "1"
    root = package_root()
    existing = [p for p in str(new_env.get("PYTHONPATH", "")).split(os.pathsep) if p]
    if root not in existing:
        existing.insert(0, root)
    new_env["PYTHONPATH"] = os.pathsep.join(existing)
    if not quiet:
        print(
            f"[bootstrap] PYTHONHASHSEED 未固定：以 {new_env[HASH_SEED_ENV]} "
            "重新执行当前命令",
            flush=True,
        )
    try:
        os.execve(sys.executable, command, new_env)
    except OSError as e:
        if not quiet:
            print(
                f"[bootstrap] ⚠ 重新执行失败（{e}）；本进程哈希盐仍未固定，"
                "该轮结果不得声称与哈希随机化无关",
                flush=True,
            )
        return False
    return False
