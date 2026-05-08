# Troubleshooting

Use this file to classify the common FreeCAD MCP bridge failures without reading the chat history.

## Healthy State

`get_connection_status` should return:

```text
connected=true
mode=xmlrpc
gui_available=true
error=null
```

`gui_available=true` matters because screenshots, selection, workbench activation, and visible view operations need the FreeCAD GUI.

## Bridge Ports

- `9875`: XML-RPC bridge. This is the port used by `.codex/config.toml`.
- `9876`: JSON-RPC socket bridge exposed by the Robust MCP Bridge.

PowerShell checks:

```powershell
Test-NetConnection localhost -Port 9875
Test-NetConnection localhost -Port 9876
```

## State Diagnosis

| Symptom | Likely meaning | Action |
| --- | --- | --- |
| Port closed | Bridge is not running, FreeCAD is closed, or the workbench did not load. | Start FreeCAD GUI, load Robust MCP Bridge, then check `9875` again. |
| Port open but MCP call times out | Bridge process/thread is present but the request queue may be stuck. | Restart FreeCAD completely and retry `get_connection_status`. |
| `connected=false` | Codex MCP server cannot reach the FreeCAD bridge. | Check `.codex/config.toml`, confirm FreeCAD is open, and test port `9875`. |
| `gui_available=false` or `0` | FreeCAD is running headless or the server is not connected to a GUI session. | Start the GUI FreeCAD application, not only `freecadcmd`. |
| `gui_available=true` or `1` | Live GUI bridge is available. | GUI-dependent MCP tools can be used. |
| `9876 already in use` | Stale bridge socket or previous FreeCAD process is still holding the port. | Close FreeCAD, kill any leftover FreeCAD process if needed, then start again. |

## XML-RPC Ping

For the XML-RPC bridge, the practical test is a Codex MCP `get_connection_status` call. If you need a raw check, use a Python environment with XML-RPC support and call the bridge `ping` method on `http://localhost:9875`.

## Qt Geometry Warning

Warnings about Qt window geometry during FreeCAD startup are not, by themselves, MCP bridge failures. Treat them as unrelated unless they coincide with a missing GUI, closed port, or failed MCP call.

## When to Use the Fallback Script

Use `scripts/create_simple_box_5mm.py` only to validate FreeCAD Python itself. It does not prove that the live MCP bridge is working. The output is `examples/simple_box_5mm.FCStd`.
