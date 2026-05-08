"""FreeCAD GUI loader shim for Robust MCP Bridge.

FreeCAD discovers GUI workbenches through InitGui.py. The installed bridge keeps
its workbench implementation in init_gui.py, so this shim imports that module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import FreeCAD


_CANDIDATE_DIRS = [
    Path(FreeCAD.getUserAppDataDir()) / "Mod" / "FreecadRobustMCPBridge",
    Path(FreeCAD.getHomePath()) / "Mod" / "FreecadRobustMCPBridge",
]
_HERE = next(
    (path for path in _CANDIDATE_DIRS if (path / "init_gui.py").exists()),
    _CANDIDATE_DIRS[0],
)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from init_gui import *  # noqa: F401,F403,E402
