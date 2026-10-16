"""极简测试入口（不依赖 pytest）: python -m tests.run_tests（或 python tests/run_tests.py）"""
import importlib
import os
import sys
import traceback

# 允许以脚本方式直接运行（python tests/run_tests.py）：此时 sys.path[0] 是
# tests/ 而非仓库根，需要显式把仓库根加入路径才能 import tests.* / lega.*。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["tests.test_attention", "tests.test_model", "tests.test_ruler",
           "tests.test_latency", "tests.test_analysis", "tests.test_utils",
           "tests.test_sanity", "tests.test_repro", "tests.test_datacheck",
           "tests.test_run_all"]


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
                fn()
                print(f"[PASS] {mod_name}.{name}")
            except Exception:
                fails += 1
                print(f"[FAIL] {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n=== 共 {total} 个测试，失败 {fails} 个 ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
