#!/usr/bin/env python3
"""一键实验流水线（薄入口；实现见 lega/pipeline/）。

参数、默认值与各阶段行为保持与拆分前完全一致。完整协议说明见 README
以及 lega/pipeline 各子模块。

进程启动即固定哈希盐：PYTHONHASHSEED 只在解释器启动前设置才对当前进程
生效，运行期写 os.environ 只能影响之后启动的子进程。编排进程自身若不固定，
集合/字典的迭代顺序会随进程启动而变，同一 commit 的两次运行可能产出顺序
不同的产物。引导逻辑见 lega/bootstrap.py（幂等、可关闭、失败不阻断实验）。
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.bootstrap import ensure_hash_seed  # noqa: E402

from lega.pipeline import *  # noqa: E402,F401,F403
from lega.pipeline import main  # noqa: E402


def main_entry():
    """CLI 入口函数，用于 pyproject.toml 中的脚本注册。"""
    ensure_hash_seed()
    main()


if __name__ == "__main__":
    ensure_hash_seed()
    main()
