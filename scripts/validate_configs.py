#!/usr/bin/env python3
"""Validate every JSON experiment config against the project config schema.

配置契约（未知键、类型/范围、跨字段冲突）由 ``lega.config_schema`` 声明，
正式入口在开跑前 fail closed。本脚本把同一套校验前移到提交阶段：新增或
修改 configs/*.json 时立刻得到反馈，而不是等到几十分钟的训练开始后才报错。

用法：
    python scripts/validate_configs.py [配置目录]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lega.utils import load_config_json  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_dir = Path(argv[0]) if argv else ROOT / "configs"
    if not config_dir.is_dir():
        print(f"ERROR: 配置目录不存在: {config_dir}")
        return 1
    files = sorted(config_dir.glob("*.json"))
    if not files:
        print(f"ERROR: 配置目录内没有 JSON 配置: {config_dir}")
        return 1
    failed = 0
    for path in files:
        try:
            load_config_json(str(path), validate=True)
        except Exception as e:
            failed += 1
            print(f"ERROR: {path}: {e}")
        else:
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            print(f"OK: {shown}")
    if failed:
        print(f"ERROR: {failed}/{len(files)} 个配置未通过契约校验")
        return 1
    print(f"OK: {len(files)} 个配置全部通过契约校验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
