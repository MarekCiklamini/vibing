# FreeCAD MCP Setup

This project uses FreeCAD through the Robust MCP Bridge. Codex connects to a local MCP server, and that server talks to FreeCAD through the bridge.

## Current Codex MCP Binding

The project-local config is `.codex/config.toml`:

```toml
[mcp_servers.freecad]
command = "C:\\Users\\marek\\Projects\\Vibing\\abaqus-agent\\.venv\\Scripts\\python.exe"
args = [
  "-m",
  "freecad_mcp.server",
  "--mode",
  "xmlrpc",
  "--host",
  "localhost",
  "--port",
  "9875",
]
```

This means Codex uses the existing `abaqus-agent` virtual environment to run `freecad_mcp.server`. That is intentional for the current local machine, but it is a machine-local dependency.

## Bridge Installation Shape

FreeCAD discovers Python workbenches from its `Mod` directory. The Robust MCP Bridge implementation keeps its startup code under module-style file names, while FreeCAD expects loader files with these names:

- `Init.py`
- `InitGui.py`

This repo keeps loader shims in `FreecadRobustMCPBridge_loader_shims/`:

- `Init.py` delegates to the bridge package's `__init__.py`.
- `InitGui.py` delegates to the bridge package's `init_gui.py`.

The shims check both the user FreeCAD app data folder and the FreeCAD home `Mod` folder, then insert the bridge directory into `sys.path`.

## Known Working State

A working GUI bridge should report:

```text
connected=true
mode=xmlrpc
freecad_version=1.0.x
gui_available=true
error=null
```

Port expectations:

- `9875`: XML-RPC bridge used by this repo's Codex MCP config.
- `9876`: JSON-RPC socket bridge exposed by the Robust MCP Bridge.

Basic PowerShell port checks:

```powershell
Test-NetConnection localhost -Port 9875
Test-NetConnection localhost -Port 9876
```

The most direct Codex-side check is the MCP tool:

```text
get_connection_status
```

## Project Skill

The project-owned Codex skill lives at:

```text
.agents/skills/freecad-mcp-workflow/
```

It documents the repeatable agent workflow for FreeCAD MCP setup, bridge health checks, and troubleshooting. The installed global mirror is:

```text
%USERPROFILE%\.codex\skills\freecad-mcp-workflow\
```

Keep the project copy as the source of truth. When the skill changes, validate the project copy, copy it globally, then validate the global copy:

```powershell
python %USERPROFILE%\.codex\skills\.system\skill-creator\scripts\quick_validate.py .agents\skills\freecad-mcp-workflow
Copy-Item -LiteralPath .agents\skills\freecad-mcp-workflow -Destination %USERPROFILE%\.codex\skills\freecad-mcp-workflow -Recurse -Force
python %USERPROFILE%\.codex\skills\.system\skill-creator\scripts\quick_validate.py %USERPROFILE%\.codex\skills\freecad-mcp-workflow
```

This skill does not replace `.codex/config.toml`; it tells future agents how to use and troubleshoot the FreeCAD MCP workflow.

## Repair Notes

`FreecadRobustMCPBridge_repaired/` is an experimental repaired copy, not the active source of truth. Keep it out of commits unless the repair is finished, reviewed, and intentionally promoted.

If FreeCAD starts but the bridge is missing, reinstall or copy the bridge into FreeCAD's user `Mod` folder and include the loader shims:

```text
%APPDATA%\FreeCAD\Mod\FreecadRobustMCPBridge\
```

The bridge folder should contain the implementation files plus `Init.py` and `InitGui.py`.

## Common Restart Case

If startup reports `9876 already in use`, FreeCAD or the bridge may have stale state from a previous run. Close FreeCAD completely, make sure no old FreeCAD process remains, then start FreeCAD again and reload the bridge.


