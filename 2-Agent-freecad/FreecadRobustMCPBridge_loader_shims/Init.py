"""FreeCAD loader shim for Robust MCP Bridge.

FreeCAD discovers Python workbenches through an Init.py file in the Mod folder.
The installed bridge keeps its startup implementation in __init__.py, so this
shim delegates to that file while preserving the module directory on sys.path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import FreeCAD


_CANDIDATE_DIRS = [
    Path(FreeCAD.getUserAppDataDir()) / "Mod" / "FreecadRobustMCPBridge",
    Path(FreeCAD.getHomePath()) / "Mod" / "FreecadRobustMCPBridge",
]
_HERE = next(
    (path for path in _CANDIDATE_DIRS if (path / "__init__.py").exists()),
    _CANDIDATE_DIRS[0],
)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_INIT_FILE = _HERE / "__init__.py"
_SPEC = importlib.util.spec_from_file_location(
    "FreecadRobustMCPBridge_init_impl", _INIT_FILE
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load Robust MCP Bridge init from {_INIT_FILE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
