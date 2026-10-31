"""实验设计与准备（拆分模块，对外接口与命令行保持不变）。"""

from . import _shared
from . import preparation
from . import power
from . import scheduling
from . import planning
from . import conformance

_MODULES = [_shared, preparation, power, scheduling, planning, conformance]
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
