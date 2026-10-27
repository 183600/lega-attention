#!/usr/bin/env python3
"""一键实验流水线（薄入口；实现见 lega/pipeline/）。

参数、默认值与各阶段行为保持与拆分前完全一致。完整协议说明见 README
以及 lega/pipeline 各子模块。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.pipeline import *  # noqa: E402,F401,F403
from lega.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
