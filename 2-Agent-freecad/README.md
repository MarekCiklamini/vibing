# freecad-agent

Local FreeCAD MCP workbench for testing Codex against FreeCAD through the Robust MCP Bridge.

This repository is not a general Python package. It is a small local workspace that keeps the MCP configuration, bridge loader shims, fallback scripts, and example FreeCAD models in one place so another agent can quickly restore and verify the setup.

## Quick Start

1. Start FreeCAD 1.0.x with the GUI.
2. Enable or load the Robust MCP Bridge workbench in FreeCAD.
3. Confirm the bridge ports are available:
   - `9875`: XML-RPC bridge used by this repo's Codex MCP config.
   - `9876`: JSON-RPC socket bridge exposed by the Robust MCP Bridge.
4. In Codex, call the FreeCAD MCP tool `get_connection_status`.

A healthy local state looks like:

```text
connected=true
mode=xmlrpc
gui_available=true
error=null
```

The current project-scoped MCP configuration is in `.codex/config.toml`. It uses:

```text
.\.venv\Scripts\python.exe
```

as the Python executable for `python -m freecad_mcp.server --mode xmlrpc --host localhost --port 9875`.

## Layout

```text
.codex/config.toml                         Codex MCP server entry for FreeCAD
.agents/skills/freecad-mcp-workflow/       Project-owned Codex skill source
FreecadRobustMCPBridge_loader_shims/       FreeCAD workbench loader shims
docs/                                      Setup, catalog, and troubleshooting notes
examples/                                  Versioned .FCStd and SVG example outputs
scripts/create_simple_box_5mm.py           Fallback FreeCAD Python smoke test
```

`FreecadRobustMCPBridge_repaired/` is an experimental repair copy and is ignored by default unless it is intentionally promoted into a real patch.

## Codex Skill

This repo owns a reusable FreeCAD MCP workflow skill at:

```text
.agents/skills/freecad-mcp-workflow/
```

The matching global installed copy is:

```text
%USERPROFILE%\.codex\skills\freecad-mcp-workflow\
```

Use the project copy as the source of truth. After editing it, validate and sync it globally:

```powershell
python %USERPROFILE%\.codex\skills\.system\skill-creator\scripts\quick_validate.py .agents\skills\freecad-mcp-workflow
Copy-Item -LiteralPath .agents\skills\freecad-mcp-workflow -Destination %USERPROFILE%\.codex\skills\freecad-mcp-workflow -Recurse -Force
python %USERPROFILE%\.codex\skills\.system\skill-creator\scripts\quick_validate.py %USERPROFILE%\.codex\skills\freecad-mcp-workflow
```

The skill is for agent behavior and operational memory; `.codex/config.toml` remains the MCP server registration.

## Examples

The main example artifacts live in `examples/`:

- `mcp_box_5mm.FCStd`
- `rectangular_pyramid.FCStd`
- `simple_bicycle.FCStd`
- `simple_bicycle_drawing.FCStd`
- `simple_bicycle_drawing.svg`
- `stol_pusher_plane.FCStd`
- `simple_building.FCStd`
- `avatar_mannequin.FCStd`
- `motor_boat.FCStd`

Additional retained smoke/demo files include `simple_box_5mm.FCStd` and `simple_human_body.FCStd`.

See `docs/model_catalog.md` for object counts and purpose notes.

## Fallback Smoke Test

If the MCP bridge is not available but FreeCAD's Python environment works, run:

```powershell
freecadcmd scripts\create_simple_box_5mm.py
```

The script writes `examples/simple_box_5mm.FCStd`.

## Verification Checklist

Use this repo as ready only when all of these are true:

- FreeCAD GUI is running.
- Robust MCP Bridge is loaded.
- Port `9875` responds through the MCP XML-RPC server.
- `get_connection_status` returns `connected=true` and `gui_available=true`.
- A simple model can be created and saved through MCP.

For common failure modes, see `docs/troubleshooting.md`.


