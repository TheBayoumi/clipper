"""Render front, right, back, left of the *actual* GLB for manual comparison.

No concept-image compositing or stock model fallbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.glb.is_file():
        parser.error("GLB missing")

    import vtk
    from PIL import Image, ImageDraw

    importer = vtk.vtkGLTFImporter()
    importer.SetFileName(str(args.glb.resolve()))
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(True)
    window.SetSize(600, 720)
    render = vtk.vtkRenderer()
    window.AddRenderer(render)
    importer.SetRenderWindow(window)
    importer.Update()
    if render.GetActors().GetNumberOfItems() == 0:
        raise RuntimeError("GLB not renderable")
    render.SetBackground(.91, .91, .92)
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(2, 4, 5)
    light.SetFocalPoint(0, 0, 0)
    light.SetIntensity(1.5)
    render.AddLight(light)
    for index in range(render.GetActors().GetNumberOfItems()):
        actor = render.GetActors().GetItemAsObject(index)
        actor.GetProperty().SetAmbient(.37)
        actor.GetProperty().SetDiffuse(.7)
    sheet = Image.new("RGB", (1200, 1524), "white")
    for index, (label, position) in enumerate((
        ("Front", (0, 0, 2.6)), ("Right", (2.6, 0, 0)),
        ("Back", (0, 0, -2.6)), ("Left", (-2.6, 0, 0))
    )):
        camera = render.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(0, -.02, 0)
        camera.SetViewUp(0, 1, 0)
        camera.SetClippingRange(.01, 100)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(.7)
        window.Render()
        pixels = vtk.vtkWindowToImageFilter()
        pixels.SetInput(window)
        pixels.Update()
        writer = vtk.vtkPNGWriter()
        writer.WriteToMemoryOn()
        writer.SetInputConnection(pixels.GetOutputPort())
        writer.Write()
        frame = Image.open(BytesIO(memoryview(writer.GetResult()))).convert("RGB")
        cell = Image.new("RGB", (600, 762), "white")
        cell.paste(frame, (0, 42))
        ImageDraw.Draw(cell).text((20, 15), label, fill="black")
        sheet.paste(cell, ((index % 2) * 600, (index // 2) * 762))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    proof = {
        "glb_sha256": hashlib.sha256(args.glb.read_bytes()).hexdigest(),
        "four_actual_mesh_views": True,
        "creative_identity_review": "PENDING",
        "publish_eligible": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
