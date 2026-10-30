import bpy, math
from pathlib import Path
from mathutils import Vector

root = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
for name, file, offset in [("12 steps", "pixal_profile_12.glb", -0.55),
                           ("10 steps", "pixal_steps_10.glb", 0.55)]:
    bpy.ops.import_scene.gltf(filepath=str(root / file))
    imported = list(bpy.context.selected_objects)
    for obj in imported:
        obj.location.x += offset
    text_data = bpy.data.curves.new(name, "FONT"); text_data.body = name; text_data.size = 0.09
    text = bpy.data.objects.new(name, text_data); bpy.context.scene.collection.objects.link(text)
    text.location = (offset - 0.25, -0.55, -0.48)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 1400, 700
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world = bpy.data.worlds.new("World"); scene.world.color = (0.16, 0.16, 0.16)
camd = bpy.data.cameras.new("Camera"); cam = bpy.data.objects.new("Camera", camd); scene.collection.objects.link(cam)
scene.camera = cam; camd.type = "ORTHO"; camd.ortho_scale = 3.0
cam.location = (2.2, -3.8, 2.2)
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
for loc, power, size in [((2,-3,4), 100, 4),((-3,-1,2),50,3)]:
    d=bpy.data.lights.new("Area", "AREA"); d.energy=power; d.shape="DISK"; d.size=size
    o=bpy.data.objects.new("Area",d);scene.collection.objects.link(o);o.location=loc
    o.rotation_euler = (Vector((0,0,0))-o.location).to_track_quat("-Z","Y").to_euler()
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.render.filepath = str(root / "pixal_steps_texture_compare.png")
bpy.ops.render.render(write_still=True)
