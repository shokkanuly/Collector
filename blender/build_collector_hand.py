"""
Collector Hand: Blender model builder.

Builds a 1:1-scale humanoid robotic right hand (palm ~8 cm wide) with a forearm that
holds 8 servos, an Arduino Uno and a PCA9685 driver. It also keyframes the full
ASL alphabet (A-Z), one letter every 20 frames, with timeline markers.

Usage (Blender 3.6 - 5.x):
    Scripting tab → Open this file → Run Script
or headless:
    blender --background --python build_collector_hand.py -- --out collector_hand.blend

Every joint is an Empty named like `index.mcp`, `thumb.ip`, `wrist`. Select one and
rotate it (R X) to pose by hand. Rotation X = flexion, Y = spread.
Poses come from the same table as the web preview; keep them in sync with hand/poses.json.
"""
import math
import sys

import bpy
import bmesh

S = 0.09  # 1 model unit = 9 cm (palm width 0.92 u ≈ 8.3 cm)
SKIN = False  # True = skin-tone "glove" material instead of white robot shell

# ---------------------------------------------------------------- poses
OPEN = (0, 0, 0); FIST = (1.55, 1.75, 1.1); CURVE = (0.5, 0.7, 0.45)
OCUR = (0.85, 1.05, 0.7); CLAW = (0.7, 1.9, 1.25); HOOK = (0.35, 1.65, 1.1)
MN = (1.45, 1.3, 0.6)
def T(opp, abd, mcp, ip, tw): return dict(opp=opp, abd=abd, mcp=mcp, ip=ip, tw=tw)
T_ACROSS = T(-2.05, 1.05, 0, 1.4, -1.2); T_TUCK = T(-2.05, 1.05, 0.45, 0.6, -0.6)
T_UP = T(0, 0.12, 0, 0, -1.2); T_OUT = T(0, 1.25, 0, 0, -1.2); T_TOUCH = T(-1.6, 0.75, 0.15, 0, -1.8)
T_G = T(0, 0.3, 0, 0, -1.2); T_K = T(-1.9, 0.3, 0.15, 0, -1.2)
def P(f, t, s=(0, 0, 0, 0), w=(0, 0, 0)): return dict(f=f, t=t, s=s, w=w)
POSES = {
    "A": P([FIST]*4, T_UP),
    "B": P([OPEN]*4, T(-1, 0.75, 1.95, 0.4, -0.6)),
    "C": P([CURVE]*4, T(-1.45, 0.6, 0, 0, -0.6), w=(0, -1.2, 0)),
    "D": P([OPEN, OCUR, OCUR, OCUR], T(-1.9, 0.75, 0, 0, -0.6)),
    "E": P([CLAW]*4, T(-0.85, 0.6, 1.2, 1, -0.6)),
    "F": P([OCUR, OPEN, OPEN, OPEN], T_TOUCH, s=(0, 0, 0.12, 0.22)),
    "G": P([OPEN, FIST, FIST, FIST], T_G, w=(0, -0.3, -1.5)),
    "H": P([OPEN, OPEN, FIST, FIST], T_TUCK, w=(0, -0.3, -1.5)),
    "I": P([FIST, FIST, FIST, OPEN], T_ACROSS),
    "J": P([FIST, FIST, FIST, OPEN], T_ACROSS, w=(0.3, -1.2, 0)),
    "K": P([OPEN, (0.75, 0, 0), FIST, FIST], T_K, s=(0.18, 0, 0, 0)),
    "L": P([OPEN, FIST, FIST, FIST], T_OUT),
    "M": P([MN, MN, MN, FIST], T(-2.2, 0.9, 0.75, 0, -0.6)),
    "N": P([MN, MN, FIST, FIST], T(-1, 0.45, 1.2, 0, -0.6)),
    "O": P([OCUR]*4, T_TOUCH, w=(0, -0.9, 0)),
    "P": P([OPEN, (0.75, 0, 0), FIST, FIST], T_K, s=(0.18, 0, 0, 0), w=(1.5, 0, -0.3)),
    "Q": P([OPEN, FIST, FIST, FIST], T_G, w=(1.9, 0, -0.2)),
    "R": P([OPEN, (0.12, 0, 0), FIST, FIST], T_TUCK, s=(-0.13, 0.22, 0, 0)),
    "S": P([FIST]*4, T_ACROSS),
    "T": P([(1.3, 1.6, 1.0), FIST, FIST, FIST], T(-1.6, 0.6, 0.6, 0, -1.2)),
    "U": P([OPEN, OPEN, FIST, FIST], T_TUCK),
    "V": P([OPEN, OPEN, FIST, FIST], T_TUCK, s=(0.26, 0.14, 0, 0)),
    "W": P([OPEN, OPEN, OPEN, FIST], T_TUCK, s=(0.26, 0, 0.2, 0)),
    "X": P([HOOK, FIST, FIST, FIST], T_ACROSS),
    "Y": P([FIST, FIST, FIST, OPEN], T_OUT, s=(0, 0, 0, 0.3)),
    "Z": P([OPEN, FIST, FIST, FIST], T_ACROSS, w=(0, 0, -0.3)),
}

# ---------------------------------------------------------------- helpers
def reset_scene():
    """Remove everything from the current file (safe to run from the Text Editor)."""
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for d in list(coll):
            coll.remove(d)
    bpy.context.scene.timeline_markers.clear()

def material(name, rgb, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*rgb, 1)
    return m

def empty(name, parent=None, loc=(0, 0, 0), mode="XYZ"):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "SPHERE"
    e.empty_display_size = 0.012
    bpy.context.collection.objects.link(e)
    e.parent = parent
    e.location = [v * S for v in loc]
    e.rotation_mode = mode
    return e

def _obj(name, bm, mat, parent, loc, bevel=0.0):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    o.parent = parent
    o.location = [v * S for v in loc]
    if bevel:
        m = o.modifiers.new("Bevel", "BEVEL"); m.width = bevel * S; m.segments = 3
        m.limit_method = "ANGLE"
    for p in o.data.polygons:
        p.use_smooth = True
    return o

def box(name, dims, mat, parent=None, loc=(0, 0, 0), bevel=0.05):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=[d * S for d in dims], verts=bm.verts)
    return _obj(name, bm, mat, parent, loc, bevel)

def cyl(name, r, depth, mat, parent=None, loc=(0, 0, 0), axis="Z", r2=None, seg=24):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r * S,
                          radius2=(r2 if r2 is not None else r) * S, depth=depth * S)
    if axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                         matrix=__import__("mathutils").Matrix.Rotation(math.pi / 2, 3, "Y"))
    return _obj(name, bm, mat, parent, loc, bevel=min(r, r2 or r) * 0.35)

def sphere(name, r, mat, parent=None, loc=(0, 0, 0), scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=14, radius=r * S)
    bmesh.ops.scale(bm, vec=scale, verts=bm.verts)
    return _obj(name, bm, mat, parent, loc)

def text(label, parent, loc, size=0.11, mat=None):
    cu = bpy.data.curves.new(label, "FONT"); cu.body = label; cu.size = size * S; cu.align_x = "CENTER"
    o = bpy.data.objects.new("label." + label, cu)
    bpy.context.collection.objects.link(o)
    o.parent = parent; o.location = [v * S for v in loc]; o.rotation_euler = (math.pi / 2, 0, 0)
    if mat: o.data.materials.append(mat)
    return o

# ---------------------------------------------------------------- build
def build():
    reset_scene()
    shell = material("Shell", (0.93, 0.78, 0.68) if SKIN else (0.92, 0.93, 0.94), 0.55)
    dark = material("Joint", (0.13, 0.15, 0.18), 0.6, 0.3)
    metal = material("Metal", (0.62, 0.66, 0.7), 0.3, 0.9)
    horn = material("ServoHorn", (0.88, 0.63, 0.0), 0.45)
    pad = material("GripPad", (0.2, 0.22, 0.25), 0.9)
    blue = material("Servo", (0.17, 0.37, 0.82), 0.5)
    pcb_ard = material("PCB_Arduino", (0.0, 0.45, 0.55), 0.6)
    pcb_pca = material("PCB_PCA9685", (0.1, 0.1, 0.12), 0.6)
    white = material("Silkscreen", (0.95, 0.95, 0.95), 0.5)

    root = empty("CollectorHand")
    # forearm, base (static)
    box("forearm", (0.78, 0.42, 1.5), shell, root, (0, 0.02, -0.95), 0.1)
    box("servo_bay", (0.66, 0.06, 1.1), dark, root, (0, -0.2, -0.95), 0.02)
    cyl("base", 0.66, 0.1, dark, root, (0, 0, -1.67), r2=0.62, seg=40)
    cyl("wrist_bearing", 0.2, 0.62, metal, root, (0, 0, -0.13), axis="X")
    names = ["S0 thumb flex", "S1 thumb swing", "S2 index", "S3 middle",
             "S4 ring", "S5 pinky", "S6 spread", "S7 wrist"]
    for i, n in enumerate(names):
        col, row = i % 2, i // 2
        x, z = -0.14 + col * 0.28, -0.55 - row * 0.25
        box("servo." + n, (0.2, 0.06, 0.1), blue, root, (x, -0.24, z), 0.01)
        box("horn." + n, (0.14, 0.015, 0.025), horn, root, (x + 0.09, -0.28, z), 0.0)
    # electronics on the back of the forearm
    ard = box("Arduino_Uno", (0.6, 0.02, 0.76), pcb_ard, root, (0, 0.25, -0.62), 0.0)
    box("Arduino_USB_B", (0.14, 0.12, 0.1), metal, ard, (0.1, 0.07, 0.36), 0.0)
    box("Arduino_ATmega328P", (0.08, 0.04, 0.4), dark, ard, (-0.08, 0.03, -0.1), 0.0)
    pca = box("PCA9685", (0.28, 0.02, 0.5), pcb_pca, root, (0, 0.25, -1.32), 0.0)
    for k in range(8):
        box(f"PCA_header_{k}", (0.03, 0.05, 0.06), dark, pca, (0.08, 0.035, 0.2 - k * 0.055), 0.0)
    text("ARDUINO UNO", ard, (0, 0.02, -0.25), 0.07, white)
    text("PCA9685", pca, (-0.04, 0.02, -0.2), 0.06, white)

    wrist = empty("wrist", root, (0, 0, -0.1), mode="YXZ")
    palm = empty("palm", wrist)
    box("palm_shell", (0.92, 0.3, 1.0), shell, palm, (0, 0, 0.55), 0.1)
    box("palm_pad", (0.7, 0.04, 0.62), pad, palm, (-0.04, -0.16, 0.52), 0.02)

    fingers = {}
    spec = {"index": (0.31, (0.44, 0.27, 0.2), 0.19), "middle": (0.1, (0.48, 0.3, 0.21), 0.2),
            "ring": (-0.11, (0.45, 0.28, 0.2), 0.19), "pinky": (-0.31, (0.35, 0.22, 0.18), 0.17)}
    for name, (x, lens, w) in spec.items():
        parent, chain = palm, []
        for i, jn in enumerate(("mcp", "pip", "dip")):
            loc = (x, 0, 1.02) if i == 0 else (0, 0, lens[i - 1])
            j = empty(f"{name}.{jn}", parent, loc, mode="YXZ")
            cyl(f"{name}.{jn}.pin", w * 0.42, w * 1.08, horn, j, axis="X")
            ww = w * (1 - i * 0.08)
            cyl(f"{name}.{jn}.seg", ww * 0.5, lens[i] - 0.03, shell, j, (0, 0, lens[i] / 2 + 0.005), r2=ww * 0.46)
            if i == 2:
                sphere(f"{name}.tip", ww * 0.46, pad, j, (0, -ww * 0.18, lens[i] - 0.05), (1, 0.7, 0.9))
            chain.append(j); parent = j
        fingers[name] = chain

    t_base = empty("thumb.base", palm, (0.3, -0.1, 0.2))
    t_opp = empty("thumb.opposition", t_base)
    t_abd = empty("thumb.abduction", t_opp)
    t_tw = empty("thumb.twist", t_abd)
    sphere("thumb.cmc_ball", 0.12, metal, t_tw)
    cyl("thumb.metacarpal", 0.1, 0.36, shell, t_tw, (0, 0, 0.18), r2=0.095)
    t_mcp = empty("thumb.mcp", t_tw, (0, 0, 0.36))
    cyl("thumb.mcp.pin", 0.085, 0.21, horn, t_mcp, axis="X")
    cyl("thumb.mcp.seg", 0.095, 0.25, shell, t_mcp, (0, 0, 0.14), r2=0.09)
    t_ip = empty("thumb.ip", t_mcp, (0, 0, 0.28))
    cyl("thumb.ip.pin", 0.085, 0.21, horn, t_ip, axis="X")
    cyl("thumb.ip.seg", 0.09, 0.21, shell, t_ip, (0, 0, 0.12), r2=0.08)
    sphere("thumb.tip", 0.085, pad, t_ip, (0, -0.04, 0.18), (1, 0.7, 0.9))

    return dict(wrist=wrist, fingers=fingers, t_opp=t_opp, t_abd=t_abd, t_tw=t_tw, t_mcp=t_mcp, t_ip=t_ip)

SPREAD_SIGN = {"index": -1, "middle": 0, "ring": 1, "pinky": 1}

def apply_pose(rig, p):
    """Axis conversion from the web model (y-up, z toward viewer) to Blender (z-up, -y toward viewer):
    rot about web X → Blender X; web Y → Blender Z; web Z(θ) → Blender Y(-θ)."""
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        mcp, pip, dip = rig["fingers"][name]
        a = p["f"][i]
        spread_web = -p["s"][1] if name == "middle" else SPREAD_SIGN[name] * p["s"][i]
        mcp.rotation_euler = (a[0], -spread_web, 0)
        pip.rotation_euler = (a[1], 0, 0)
        dip.rotation_euler = (a[2], 0, 0)
    t = p["t"]
    rig["t_opp"].rotation_euler = (0, 0, t["opp"])
    rig["t_abd"].rotation_euler = (0, t["abd"], 0)
    rig["t_tw"].rotation_euler = (0, 0, t["tw"])
    rig["t_mcp"].rotation_euler = (t["mcp"], 0, 0)
    rig["t_ip"].rotation_euler = (t["ip"], 0, 0)
    wx, wy, wz = p["w"]
    rig["wrist"].rotation_euler = (wx, -wz, wy)

def all_joints(rig):
    js = [rig["wrist"], rig["t_opp"], rig["t_abd"], rig["t_tw"], rig["t_mcp"], rig["t_ip"]]
    for c in rig["fingers"].values(): js += c
    return js

def animate(rig, step=20, hold=10):
    scene = bpy.context.scene
    for i, (letter, pose) in enumerate(POSES.items()):
        f = 1 + i * step
        apply_pose(rig, pose)
        for j in all_joints(rig):
            j.keyframe_insert("rotation_euler", frame=f)
            j.keyframe_insert("rotation_euler", frame=f + hold)
        scene.timeline_markers.new(letter, frame=f)
    scene.frame_start, scene.frame_end = 1, 1 + len(POSES) * step
    scene.render.fps = 24

def stage():
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera"); cam_data.lens = 50
    cam = bpy.data.objects.new("Camera", cam_data); bpy.context.collection.objects.link(cam)
    cam.location = (0.2, -0.56, 0.05); cam.rotation_euler = (math.radians(88), 0, math.radians(19))
    scene.camera = cam
    for name, loc, energy, size in (("Key", (0.4, -0.5, 0.5), 35, 0.3), ("Fill", (-0.5, -0.3, 0.2), 12, 0.5),
                                    ("Rim", (0, 0.5, 0.4), 25, 0.3)):
        ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size
        lo = bpy.data.objects.new(name, ld); bpy.context.collection.objects.link(lo); lo.location = loc
        d = lo.location.copy(); lo.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("World"); scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.85, 0.87, 0.9, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.35

def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    rig = build()
    animate(rig)
    stage()
    bpy.context.scene.frame_set(1)
    if "--out" in argv:
        bpy.ops.wm.save_as_mainfile(filepath=argv[argv.index("--out") + 1])

if __name__ == "__main__":
    main()
