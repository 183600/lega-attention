"""确定性机制自检：进程级哈希盐引导、确定性档位审计与线程配额。

这些机制回答的是“同 seed 重跑为什么会/不会给出同样的数字”：

  * **哈希盐**（``lega.bootstrap``）：``PYTHONHASHSEED`` 只在解释器启动前
    设置才对当前进程生效。引导逻辑负责在必要时以固定哈希盐重启自身，
    让集合/字典的迭代顺序在两次运行之间保持一致；解析与命令重建是纯函数，
    可独立核对。
  * **确定性档位**（``lega.utils``）：``use_deterministic_algorithms`` 只说明
    “开关打开了”，不能说明“本轮没有踩到无确定性实现的算子”。审计字段必须
    同时记录档位（strict/warn_only）与实际触发的非确定性算子告警数。
  * **线程配额**（``LEGA_NUM_THREADS``）：CPU 归约顺序随线程数变化，跨机
    比较必须知道线程数是否被固定。
"""

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lega import bootstrap  # noqa: E402
from lega.utils import (  # noqa: E402
    determinism_mode,
    non_deterministic_op_warnings,
    seed_report,
    set_seed,
)

NONDET_MESSAGE = (
    "aten::some_op does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'."
)


def test_hash_seed_state_classifies_every_environment_value():
    """哈希盐状态必须区分：已固定 / 显式 random / 未设置 / 非法取值。"""
    fixed = bootstrap.hash_seed_state({"PYTHONHASHSEED": "0"})
    assert fixed["fixed"] is True and fixed["value"] == 0
    assert fixed["source"] == "env" and fixed["randomized"] is False

    randomized = bootstrap.hash_seed_state({"PYTHONHASHSEED": "random"})
    assert randomized["fixed"] is False and randomized["randomized"] is True
    assert randomized["source"] == "random"

    unset = bootstrap.hash_seed_state({})
    assert unset["fixed"] is False and unset["source"] == "unset"

    invalid = bootstrap.hash_seed_state({"PYTHONHASHSEED": "1.5"})
    assert invalid["fixed"] is False and invalid["source"] == "invalid"

    # 越界与非法字符串都不许被截断成合法盐
    assert bootstrap.parse_hash_seed(str(bootstrap.MAX_HASH_SEED + 1)) is None
    assert bootstrap.parse_hash_seed("-1") is None
    assert bootstrap.parse_hash_seed("abc") is None
    assert bootstrap.parse_hash_seed(str(bootstrap.MAX_HASH_SEED)) is not None
    print("  [ok] 哈希盐状态解析覆盖固定/随机/未设置/非法")


def test_ensure_hash_seed_is_idempotent_and_never_loops():
    """已固定时直接返回；关闭开关或已尝试过时不得重新执行自身。"""
    fixed_env = {"PYTHONHASHSEED": "0"}
    assert bootstrap.ensure_hash_seed(argv=[], env=fixed_env) is True

    disabled_env = {"LEGA_HASH_SEED_BOOTSTRAP": "0"}
    assert bootstrap.ensure_hash_seed(argv=[], env=disabled_env) is False

    guarded_env = {"LEGA_HASH_SEED_BOOTSTRAPPED": "1"}
    assert bootstrap.ensure_hash_seed(argv=[], env=guarded_env) is False

    # 无可重建的命令行时安全退回（不 exec、不抛错）
    assert bootstrap.ensure_hash_seed(argv=[], env={}) is False
    print("  [ok] 哈希盐引导幂等、可关闭、不会自激重启")


def test_reexec_command_preserves_module_semantics():
    """包内脚本必须以 ``-m`` 重新执行（相对导入在脚本模式下会失败）。"""
    train_py = str(ROOT / "lega" / "train.py")
    assert bootstrap.module_name_for_script(train_py) == "lega.train"
    nested = str(ROOT / "lega" / "pipeline" / "train.py")
    assert bootstrap.module_name_for_script(nested) == "lega.pipeline.train"
    assert bootstrap.module_name_for_script(str(ROOT / "run_all.py")) == ""

    cmd = bootstrap.reexec_command([train_py, "--config", "configs/tiny.json"])
    assert cmd[1:4] == ["-m", "lega.train", "--config"], cmd
    assert cmd[4] == "configs/tiny.json"

    top = bootstrap.reexec_command([str(ROOT / "run_all.py"), "--smoke"])
    assert top[1].endswith("run_all.py") and top[2] == "--smoke", top
    print("  [ok] 重新执行命令保持模块语义与参数顺序")


def test_determinism_mode_parses_and_rejects_typos():
    """确定性档位显式开关：写错必须报错，不能静默退回默认。"""
    assert determinism_mode({"LEGA_DETERMINISM": ""}) == "warn_only"
    assert determinism_mode({"LEGA_DETERMINISM": "warn_only"}) == "warn_only"
    assert determinism_mode({"LEGA_DETERMINISM": "strict"}) == "strict"
    try:
        determinism_mode({"LEGA_DETERMINISM": "strick"})
    except ValueError:
        pass
    else:
        raise AssertionError("非法确定性档位必须显式报错")
    assert determinism_mode({"LEGA_DETERMINISM": "strick"}, strict=False) == "invalid"
    print("  [ok] 确定性档位解析严格（拼错即报错）")


def test_seed_report_exposes_the_determinism_audit():
    """seed_report 必须给出档位、非确定性算子计数与线程数。"""
    import lega.utils as utils

    old_env = os.environ.get("PYTHONHASHSEED")
    try:
        set_seed(1234)
        rep = seed_report(1234)
        assert rep["seed_verified"] is True
        assert rep["deterministic_mode"] in ("strict", "warn_only")
        assert isinstance(rep["deterministic_algorithms_warn_only"], bool)
        assert isinstance(rep["non_deterministic_op_warnings"], int)
        assert rep["non_deterministic_op_warnings"] == 0
        assert isinstance(rep["torch_threads"], int) and rep["torch_threads"] >= 1
        assert isinstance(rep["pythonhashseed_fixed"], bool)
        # 启动值取自模块导入时的快照（其他测试可能在此之前改过环境变量），
        # 不能拿测试当下的 env 去断言，否则与被审计的语义无关。
        assert rep["pythonhashseed_at_start"] == utils._ENV_AT_IMPORT["PYTHONHASHSEED"]
        expected_fixed = bootstrap.parse_hash_seed(rep["pythonhashseed_at_start"])
        assert rep["pythonhashseed_fixed"] is (expected_fixed is not None)
    finally:
        if old_env is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_env
    print("  [ok] seed_report 暴露确定性档位/非确定性算子/线程数")


def test_non_deterministic_op_warnings_are_counted_and_reset():
    """无确定性实现的算子告警必须被计数，并在下次 set_seed 时清零。"""
    import torch

    set_seed(99)
    assert non_deterministic_op_warnings()["count"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(NONDET_MESSAGE)
    counted = non_deterministic_op_warnings()
    assert counted["count"] == 1, counted
    assert counted["samples"] and "deterministic" in counted["samples"][0]
    set_seed(99)
    assert non_deterministic_op_warnings()["count"] == 0
    assert torch.are_deterministic_algorithms_enabled() is True
    print("  [ok] 非确定性算子告警可计数且随 set_seed 清零")


def test_thread_limit_is_applied_and_recorded():
    """``LEGA_NUM_THREADS`` 固定线程数后必须写进审计字段。"""
    import torch

    original_threads = torch.get_num_threads()
    old = os.environ.get("LEGA_NUM_THREADS")
    try:
        os.environ["LEGA_NUM_THREADS"] = "1"
        set_seed(7)
        rep = seed_report(7)
        assert torch.get_num_threads() == 1
        assert rep["torch_threads"] == 1
        assert rep["torch_threads_pinned"] is True
    finally:
        if old is None:
            os.environ.pop("LEGA_NUM_THREADS", None)
        else:
            os.environ["LEGA_NUM_THREADS"] = old
        try:
            torch.set_num_threads(original_threads)
        except RuntimeError:
            pass
    print("  [ok] 线程配额生效并进入审计字段")


def test_strict_mode_is_recorded_and_restores_warn_only():
    """strict 档位必须如实记录；退出后恢复默认档位。"""
    old = os.environ.get("LEGA_DETERMINISM")
    try:
        os.environ["LEGA_DETERMINISM"] = "strict"
        set_seed(11)
        rep = seed_report(11)
        assert rep["deterministic_mode"] == "strict"
        assert rep["deterministic_algorithms_warn_only"] is False
    finally:
        if old is None:
            os.environ.pop("LEGA_DETERMINISM", None)
        else:
            os.environ["LEGA_DETERMINISM"] = old
        set_seed(11)
    assert seed_report(11)["deterministic_mode"] == "warn_only"
    print("  [ok] strict 档位如实记录并可恢复默认")
