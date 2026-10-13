"""极简测试入口（不依赖 pytest）: python -m tests.run_tests"""
import importlib
import sys
import traceback

MODULES = ["tests.test_attention", "tests.test_model", "tests.test_ruler"]


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
