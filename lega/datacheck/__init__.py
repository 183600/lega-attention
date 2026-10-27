"""数据卫生与防泄漏检查（拆分模块，对外接口保持不变）。"""

from . import _shared
from . import text
from . import tokens
from . import reports

_MODULES = [_shared, text, tokens, reports]
_merged = {}
for _m in _MODULES:
    _merged.update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
for _m in _MODULES:
    _ns = vars(_m)
    for _k, _v in _merged.items():
        _ns.setdefault(_k, _v)
globals().update(_merged)
__all__ = sorted(k for k in _merged if not k.startswith("__"))
