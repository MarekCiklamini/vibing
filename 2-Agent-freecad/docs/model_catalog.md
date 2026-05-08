# Model Catalog

The `.FCStd` files are first-class examples for this workspace and should stay versioned unless repository size becomes a real problem. They are stored under `examples/`.

Object counts were read from each `.FCStd` archive's `Document.xml`.

| File | Objects | Source | Purpose and main elements |
| --- | ---: | --- | --- |
| `examples/mcp_box_5mm.FCStd` | 1 | Live MCP smoke output | Minimal 5 mm box created through the MCP path. Useful for confirming model creation and save behavior. |
| `examples/rectangular_pyramid.FCStd` | 1 | Live MCP example | Simple pyramid/wedge-style solid for primitive geometry checks. |
| `examples/simple_bicycle.FCStd` | 52 | Live MCP example | Multi-part bicycle model with wheels, frame, fork/handlebar, saddle, and small detail components. |
| `examples/simple_bicycle_drawing.FCStd` | 12 | Live MCP/example drawing | Drawing-oriented bicycle representation used with the SVG export. |
| `examples/simple_bicycle_drawing.svg` | n/a | Exported example | SVG drawing export associated with `simple_bicycle_drawing.FCStd`. |
| `examples/stol_pusher_plane.FCStd` | 41 | Live MCP example | STOL pusher aircraft concept with wing, fuselage, tail, landing gear, and propulsion elements. |
| `examples/simple_building.FCStd` | 89 | Live MCP example | Building massing/demo model with repeated architectural elements. Useful for larger object-count behavior. |
| `examples/avatar_mannequin.FCStd` | 33 | Live MCP example | Simple mannequin/avatar figure built from body primitives. |
| `examples/motor_boat.FCStd` | 44 | Live MCP example | Motor boat concept with hull, cabin/console, seating, and propulsion/detail parts. |
| `examples/simple_box_5mm.FCStd` | 1 | Fallback script output | Box produced by `scripts/create_simple_box_5mm.py` without relying on the live MCP bridge. |
| `examples/simple_human_body.FCStd` | 30 | Retained earlier example | Human body/mannequin-style model retained as an additional geometry example. |

## Practical Use

Use `mcp_box_5mm.FCStd` when validating live MCP save behavior. Use `simple_box_5mm.FCStd` when validating only FreeCAD's local Python execution. Use the larger bicycle, building, plane, avatar, and boat files when checking object listing, display, export, or screenshot behavior.
