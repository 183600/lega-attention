"""测试入口 `tests.run_tests` 的自检：无 pytest 时必须能注入 monkeypatch/capsys。

回归：run_tests 以 `fn()` 直接调用测试函数，遇到带 `monkeypatch`/`capsys`
形参的测试会抛 “missing 1 required positional argument” 而假失败；兼容
实现必须能注入、捕获输出并自动回滚。
"""
import os
import sys

from tests.run_tests import _MonkeyPatch, _run_with_fixtures


def test_monkeypatch_fixture_setattr_roundtrip():
    """setattr 后自动回滚到原值；新增属性在 undo 后被移除。"""

    class _Box:
        pass

    box = _Box()
    box.x = 1

    def use(monkeypatch):
        monkeypatch.setattr(box, "x", 2)
        assert box.x == 2
        monkeypatch.setattr(box, "y", 3, raising=False)
        assert box.y == 3
        monkeypatch.setattr(os, "sep", "!")

    _run_with_fixtures(use)
    assert box.x == 1
    assert not hasattr(box, "y")
    assert os.sep != "!"
    print("  [ok] _MonkeyPatch.setattr 生效并自动回滚")


def test_monkeypatch_fixture_setenv_roundtrip(monkeypatch):
    """setenv 在当前测试内生效；本测试本身也验证 fixture 注入可运行。"""
    key = "LEGA_RUN_TESTS_FIXTURE_PROBE"
    assert key not in os.environ
    monkeypatch.setenv(key, "1")
    assert os.environ[key] == "1"
    print("  [ok] _MonkeyPatch.setenv 生效（fixture 注入可运行）")


def test_monkeypatch_fixture_delenv_roundtrip():
    """delenv 生效并自动恢复原环境变量；缺失时 raising=False 不报错。"""
    key = "LEGA_RUN_TESTS_DELENV_PROBE"
    os.environ[key] = "orig"

    def use(monkeypatch):
        monkeypatch.delenv(key)
        assert key not in os.environ
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(key, "new")

    _run_with_fixtures(use)
    assert os.environ.get(key) == "orig"
    os.environ.pop(key, None)
    print("  [ok] _MonkeyPatch.delenv 生效并自动恢复")


def test_capsys_fixture_captures_and_clears():
    """capsys 捕获 stdout/stderr，readouterr 后清空，测试输出仍可见。"""

    def use(capsys):
        print("probe-out")
        print("probe-err", file=sys.stderr)
        got = capsys.readouterr()
        assert got.out == "probe-out\n", got
        assert got.err == "probe-err\n", got
        got2 = capsys.readouterr()
        assert got2.out == "" and got2.err == ""
        print("  [ok] _CapSys 捕获 stdout/stderr 并可清空")

    _run_with_fixtures(use)
    print("  [ok] capsys fixture 注入可运行")


def test_runner_module_list_covers_every_test_file():
    """新增 tests/test_*.py 必须进入 MODULES，否则官方入口会静默漏跑。"""
    from tests import run_tests

    tests_dir = os.path.dirname(os.path.abspath(run_tests.__file__))
    files = {os.path.basename(p)[:-3] for p in os.listdir(tests_dir)
             if p.startswith("test_") and p.endswith(".py")}
    listed = {m.rsplit(".", 1)[-1] for m in run_tests.MODULES}
    assert files <= listed, sorted(files - listed)
    print("  [ok] run_tests.MODULES 覆盖全部测试文件")


def test_run_with_fixtures_rejects_unknown_fixture():
    """未知 fixture 必须显式报错，而不是静默用 None 掩盖依赖。"""

    def bad(pytestconfig):
        pass

    try:
        _run_with_fixtures(bad)
    except TypeError as e:
        assert "pytestconfig" in str(e)
    else:
        raise AssertionError("未知 fixture 应报 TypeError")
    print("  [ok] 未知 fixture 显式报错")
