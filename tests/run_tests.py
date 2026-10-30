"""极简测试入口（不依赖 pytest）: python -m tests.run_tests（或 python tests/run_tests.py）"""
import contextlib
import importlib
import inspect
import io
import os
import sys
import traceback
from types import SimpleNamespace

# 允许以脚本方式直接运行（python tests/run_tests.py）：此时 sys.path[0] 是
# tests/ 而非仓库根，需要显式把仓库根加入路径才能 import tests.* / lega.*。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["tests.test_attention", "tests.test_model", "tests.test_ruler",
           "tests.test_latency", "tests.test_length_effect",
           "tests.test_analysis", "tests.test_utils",
           "tests.test_uncertainty",
           "tests.test_sanity", "tests.test_repro",
           "tests.test_datacheck_protocol", "tests.test_datacheck_bins",
           "tests.test_datacheck_exact_ngram",
           "tests.test_datacheck_provenance",
           "tests.test_pipeline_datacheck", "tests.test_pipeline_audit",
           "tests.test_pipeline_report", "tests.test_pipeline_baselines",
           "tests.test_pipeline_ruler", "tests.test_pipeline_train",
           "tests.test_pipeline_factorial", "tests.test_pipeline_sampling",
           "tests.test_reparameterization", "tests.test_budget_sensitivity",
           "tests.test_curves", "tests.test_badcase",
           "tests.test_nan_checklist", "tests.test_evaluate",
           "tests.test_experiment", "tests.test_hypotheses",
           "tests.test_tuning", "tests.test_ngram", "tests.test_early_stop",
           "tests.test_postrun", "tests.test_plots", "tests.test_probe",
           "tests.test_ablation_methods", "tests.test_dose_response",
           "tests.test_output_ablation",
           "tests.test_equivalence", "tests.test_replication",
           "tests.test_bayes", "tests.test_multiverse",
           "tests.test_design", "tests.test_nested", "tests.test_run_tests",
           "tests.test_lengthcalib"]


class _MonkeyPatch:
    """最小 pytest `monkeypatch` 兼容实现（setattr / delattr / setenv）。

    该测试入口不依赖 pytest，但个别测试用 `monkeypatch` fixture 做局部猴子
    补丁；此实现让它们在无 pytest 环境下也能运行并自动回滚，避免“仅在
    pytest 下通过、在 run_tests 下报 missing argument”的假失败。
    """

    def __init__(self):
        self._setattrs = []
        self._dels = {}
        self._env = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str):  # "module.attr" 形式：由 setattr 代理给 resolve
            raise TypeError("monkeypatch.setattr 需要对象而非字符串目标")
        if not hasattr(target, name) and raising:
            raise AttributeError(f"{target!r} 无属性 {name!r}")
        old = getattr(target, name, None)
        had = hasattr(target, name)
        self._setattrs.append((target, name, old, had))
        setattr(target, name, value)

    def delattr(self, target, name, raising=True):
        had = hasattr(target, name)
        if not had and raising:
            raise AttributeError(f"{target!r} 无属性 {name!r}")
        self._dels.setdefault((id(target), name), (target, name,
                                                    getattr(target, name, None)))
        if had:
            delattr(target, name)

    def setenv(self, name, value, prepend=None):
        old = os.environ.get(name)
        self._env.append((name, old))
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        if name not in os.environ and raising:
            raise KeyError(name)
        old = os.environ.pop(name, None)
        self._env.append((name, old))

    def undo(self):
        for target, name, old, had in reversed(self._setattrs):
            if had:
                setattr(target, name, old)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        for target, name, old in reversed(list(self._dels.values())):
            setattr(target, name, old)
        for name, old in reversed(self._env):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


class _CapSys:
    """最小 pytest `capsys` 兼容：捕获 stdout/stderr，readouterr 后清空。"""

    def __init__(self):
        self._out = io.StringIO()
        self._err = io.StringIO()

    def readouterr(self):
        out, err = self._out.getvalue(), self._err.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return SimpleNamespace(out=out, err=err)


def _run_with_fixtures(fn):
    """按签名注入支持的 fixture（monkeypatch/capsys）；未知参数直接报错。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    names = [p.name for p in params.values()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    kwargs = {}
    mp = None
    cap = None
    for n in names:
        if n == "monkeypatch":
            if mp is None:
                mp = _MonkeyPatch()
            kwargs[n] = mp
        elif n == "capsys":
            if cap is None:
                cap = _CapSys()
            kwargs[n] = cap
        else:
            raise TypeError(f"测试函数 {fn.__name__} 需要未知 fixture {n!r}")
    try:
        if cap is not None:
            with contextlib.redirect_stdout(cap._out), \
                    contextlib.redirect_stderr(cap._err):
                fn(**kwargs)
        else:
            fn(**kwargs)
    finally:
        if mp is not None:
            mp.undo()
        if cap is not None:
            # 测试内部已 readouterr 的输出不重复；剩余输出照常可见。
            sys.stdout.write(cap._out.getvalue())
            sys.stderr.write(cap._err.getvalue())


def main() -> int:
    total, fails = 0, 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            print(f"[RUN ] {mod_name}.{name}")
            try:
                _run_with_fixtures(fn)
                print(f"[PASS] {mod_name}.{name}")
            except Exception:
                fails += 1
                print(f"[FAIL] {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n=== 共 {total} 个测试，失败 {fails} 个 ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
