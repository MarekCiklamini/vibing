from __future__ import annotations

from pathlib import Path

import FreeCAD
import Part


OUT_PATH = Path(__file__).resolve().parents[1] / "examples" / "simple_box_5mm.FCStd"
OUT_PATH.parent.mkdir(exist_ok=True)

doc = FreeCAD.newDocument("SimpleBox5mm")
shape = Part.makeBox(5.0, 5.0, 5.0)

box = doc.addObject("Part::Feature", "Box_5mm")
box.Label = "Box 5 mm"
box.Shape = shape

doc.recompute()
doc.saveAs(str(OUT_PATH))

print(f"Saved {OUT_PATH}")
