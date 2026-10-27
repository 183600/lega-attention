"""实验流水线包：由原 run_all.py 拆分，保持对外名称与行为不变。"""

from . import _shared
from . import common
from . import datacheck
from .eval import stats
from .eval import windows
from .eval import comparisons
from .eval import factorial
from .eval import baselines
from .eval import curves
from .report import summary
from .report import sections
from .report import tables
from . import train

_MODULES = [
    _shared, common, datacheck, stats, windows, comparisons, factorial,
    baselines, curves, summary, sections, tables, train,
]

_merged = {}
for _m in _MODULES:
    _merged.update({k: v for k, v in vars(_m).items()
                    if not k.startswith("__")})

for _m in _MODULES:
    _ns = vars(_m)
    for _k, _v in _merged.items():
        _ns.setdefault(_k, _v)

globals().update(_merged)
__all__ = sorted(k for k in _merged if not k.startswith("__"))
