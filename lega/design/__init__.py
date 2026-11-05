"""实验设计与准备（拆分模块，对外接口与命令行保持不变）。"""

from . import _shared
from . import preparation
from . import power
from . import scheduling
from . import seeds
from . import calibration
from . import pilot
from . import budget
from . import planning
from . import conformance

# 顺序有意义：conformance 必须排在最后，它提供 `python -m lega.design` 的
# 命令行入口（main）；pilot/budget 也各自带 main，但只能通过
# `python -m lega.design.pilot` / `python -m lega.design.budget` 调用。
_MODULES = [
    _shared,
    preparation,
    power,
    scheduling,
    seeds,
    calibration,
    pilot,
    budget,
    planning,
    conformance,
]
_merged = {}
for _m in _MODULES:
    _merged.update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
for _m in _MODULES:
    _ns = vars(_m)
    for _k, _v in _merged.items():
        _ns.setdefault(_k, _v)
globals().update(_merged)
__all__ = sorted(k for k in _merged if not k.startswith("__"))

if __name__ == "__main__":
    raise SystemExit(main())
