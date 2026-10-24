"""Render and verify *real skinned-body motion* from a rigged GLB in Blender.

This is an early cinematic character-performance probe, not a publishable video.
Never treat virtual-camera zoom as skeletal motion or claim phoneme lip-sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from viseme_cues import frame_angles, read_cues


def args() -> argparse.Namespace:
    cli = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--mouth-cues", type=Path)
    return p.parse_args(cli)


def light(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = 3.0
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector((0, 0, 1.2)) - obj.location).to_track_quat("-Z", "Y").to_euler()


def prop(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> None:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat = bpy.data.materials.new(name + "_material")
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def genuine_wave_motion(rig: bpy.types.Object) -> tuple[bpy.types.Action, dict[str, object]]:
    actions = [a for a in bpy.data.actions if "wave" in a.name.lower()]
    if not actions:
        raise RuntimeError(
            "No actual rigged Wave animation found. Actions="
            + repr([a.name for a in bpy.data.actions])
        )
    action = sorted(actions, key=lambda a: len(a.name))[0]
    if rig.animation_data is None:
        rig.animation_data_create()
    for track in rig.animation_data.nla_tracks:
        track.mute = True
    rig.animation_data.action = action
    scene = bpy.context.scene
    start = math.floor(action.frame_range[0])
    end = math.ceil(action.frame_range[1])
    if end - start < 12:
        raise RuntimeError("Wave animation has fewer than 12 frames")
    bones = [
        p
        for p in rig.pose.bones
        if any(word in p.name.lower() for word in ("arm", "hand", "shoulder"))
    ]
    if len(bones) < 2:
        raise RuntimeError("Expected genuine arm and hand skeletal bones")
    snapshots: list[dict[str, tuple[float, ...]]] = []
    for frame in [start, start + 8, start + 18]:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        snapshots.append({p.name: tuple(float(v) for row in p.matrix for v in row) for p in bones})
    measured = max(
        abs(a - b)
        for name, orig in snapshots[0].items()
        for alt in snapshots[1:]
        for a, b in zip(orig, alt[name], strict=True)
    )
    if measured < 0.015:
        raise RuntimeError(f"Character remained static: maximum arm bone delta {measured:.6f}")
    return action, {
        "wave_action": action.name,
        "wave_start_frame": start,
        "max_arm_bone_matrix_delta": round(measured, 6),
        "arm_joint_count": len(bones),
    }



def timed_jaw_animation(
    rig: bpy.types.Object, wave: bpy.types.Action, cues_file: Path,
    *, frame_start: int, frames: int,
) -> dict[str, object]:
    """Add a voice-derived jaw NLA track above the actual body performance."""
    jaw = rig.pose.bones.get("Jaw")
    if jaw is None:
        raise RuntimeError("CC0 skinned character has no jaw bone for spoken cues")
    cues = read_cues(cues_file)
    values = frame_angles(cues, fps=12, frames=frames)
    if len({round(value, 3) for value in values}) < 3:
        raise RuntimeError("audio-derived jaw animation has insufficient shape changes")

    animation = rig.animation_data
    if animation is None:
        raise RuntimeError("imported human armature has no animation data")
    animation.action = None
    body_track = animation.nla_tracks.new()
    body_track.name = "Actual_Skinned_Wave"
    body_track.strips.new("Wave", frame_start, wave)

    jaw_action = bpy.data.actions.new("Kokoro_Rhubarb_Jaw")
    animation.action = jaw_action
    jaw.rotation_mode = "XYZ"
    for index, angle in enumerate(values):
        jaw.rotation_euler.x = -angle
        jaw.keyframe_insert(
            data_path="rotation_euler", index=0, frame=frame_start + index,
            group="Speech_Jaw",
        )
    animation.action = None
    jaw_track = animation.nla_tracks.new()
    jaw_track.name = "Voice_Derived_Viseme_Jaw"
    jaw_strip = jaw_track.strips.new("Jaw_Cues", frame_start, jaw_action)
    jaw_strip.blend_type = "REPLACE"

    scene = bpy.context.scene
    samples = []
    for index in range(frames):
        scene.frame_set(frame_start + index)
        bpy.context.view_layer.update()
        samples.append(tuple(float(v) for row in jaw.matrix for v in row))
    max_change = max(
        abs(a - b)
        for sample in samples[1:]
        for a, b in zip(samples[0], sample, strict=True)
    )
    if max_change < 0.015:
        raise RuntimeError(
            "acoustic speech cues did not drive visible jaw-bone motion"
        )
    return {
        "recognizer": "Rhubarb-Lip-Sync-1.14.0",
        "voice_origin": "archived_kokoro_tts",
        "mouth_cues": len(cues),
        "distinct_sampled_jaw_positions": len({round(v, 3) for v in values}),
        "jaw_bone": jaw.name,
        "max_jaw_bone_matrix_delta": round(max_change, 6),
        "full_lip_visemes": False,
        "face_review": "PENDING",
    }


def main() -> None:
    options = args()
    if not 12 <= options.frames <= 72:
        raise ValueError("Render 12-72 genuine motion frames for a proof")
    options.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(options.glb.resolve()))
    rigs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(rigs) != 1:
        raise RuntimeError(f"Expected exactly one imported skinned rig; found {len(rigs)}")
    rig = rigs[0]
    skinned = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and any(mod.type == "ARMATURE" for mod in obj.modifiers)
    ]
    if not skinned:
        raise RuntimeError("The GLB has no mesh rigged to a real armature")
    action, evidence = genuine_wave_motion(rig)
    jaw_evidence = (
        timed_jaw_animation(
            rig, action, options.mouth_cues,
            frame_start=int(evidence["wave_start_frame"]), frames=options.frames,
        )
        if options.mouth_cues is not None else None
    )
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = options.samples
    scene.render.resolution_x = 360
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 12
    scene.world.color = (0.035, 0.054, 0.083)
    if scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Color"].default_value = (0.035, 0.054, 0.083, 1)
            background.inputs["Strength"].default_value = 0.4
    prop("Studio_floor", (0, 0, -0.08), (10, 10, 0.12), (0.028, 0.048, 0.070, 1))
    prop("Studio_backdrop", (0, 1.7, 2), (10, 0.10, 6), (0.045, 0.070, 0.11, 1))
    prop("Stage_book_stand", (1.10, 0.25, 0.45), (0.65, 0.7, 0.90), (0.08, 0.25, 0.28, 1))
    prop("Original_book_prop", (1.1, -0.15, 0.93), (0.27, 0.20, 0.045), (0.9, 0.67, 0.29, 1))
    light("Warm_key", (-2.8, -3.0, 4.1), (1.0, 0.76, 0.56), 650)
    light("Cool_fill", (2.7, -2.0, 2.8), (0.45, 0.73, 1), 440)
    light("Teal_rim", (0.5, 2.3, 4.1), (0.26, 0.86, 0.78), 800)
    camera_data = bpy.data.cameras.new("Vertical_cinema_camera")
    camera = bpy.data.objects.new("Vertical_cinema_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.15, -4.7, 1.37)
    camera.rotation_euler = (
        (Vector((0, 0, 1.22)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.9
    scene.camera = camera
    scene.view_settings.view_transform = "Standard"
    frames_dir = options.output / "frames"
    frames_dir.mkdir(exist_ok=True)
    for i in range(options.frames):
        scene.frame_set(int(evidence["wave_start_frame"]) + i)
        scene.render.filepath = str(frames_dir / f"frame_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    result = {
        "source_glb_sha256": hashlib.sha256(options.glb.read_bytes()).hexdigest(),
        "skinned_mesh_count": len(skinned),
        "skeletal_bone_count": len(rig.pose.bones),
        "action": action.name,
        "frames_rendered": options.frames,
        "fps": 12,
        "resolution": [360, 640],
        "motion": evidence,
        "creative_status": "REAL_RIG_MOTION_NOT_EDITORIALLY_APPROVED",
        "publish_eligible": False,
        "lip_sync": (
            "RHUBARB_TIMED_JAW_ONLY" if jaw_evidence is not None else "NOT_IMPLEMENTED"
        ),
        "jaw_evidence": jaw_evidence,
    }
    (options.output / "rig-proof.json").write_text(json.dumps(result, indent=2))
    print("RIG_MOTION_VERIFIED:", json.dumps(result))


if __name__ == "__main__":
    main()
