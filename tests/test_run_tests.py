"""测试入口 `tests.run_tests` 的自检：无 pytest 时必须能注入 monkeypatch。

回归：run_tests 以 `fn()` 直接调用测试函数，遇到带 `monkeypatch` 形参的
测试会抛 “missing 1 required positional argument” 而假失败；此处的
_MonkeyPatch 兼容实现必须能 setattr/setenv 且自动回滚。
"""
import os

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
