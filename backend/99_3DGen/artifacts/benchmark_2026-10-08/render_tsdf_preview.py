import bpy, math
from mathutils import Vector
from pathlib import Path

root = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(root / "triposplat_tsdf.glb"))
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
mesh.rotation_euler[0] = math.radians(90)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 700
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("World")
scene.world.color = (0.8, 0.8, 0.8)
# Show the GLB COLOR_0 data in this preview, independent of importer material settings.
mat = bpy.data.materials.new("vertex-color preview")
mat.use_nodes = True
nodes = mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")
vc = nodes.new("ShaderNodeVertexColor")
vc.layer_name = "Color"
mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.85
mesh.data.materials.clear(); mesh.data.materials.append(mat)
center = Vector(mesh.location)
camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data); scene.collection.objects.link(camera)
scene.camera = camera; camera_data.type = "ORTHO"; camera_data.ortho_scale = 1.25
def aim(obj, target): obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
camera.location = center + Vector((1.8, -2.4, 1.4)); aim(camera, center)
for loc, power, size in [((2,-3,3), 500, 4),((-3,-1,1),300,3)]:
    d=bpy.data.lights.new("Area", "AREA"); d.energy=power; d.shape="DISK"; d.size=size
    o=bpy.data.objects.new("Area",d);scene.collection.objects.link(o);o.location=loc;aim(o,center)
scene.view_settings.view_transform = "Standard"
scene.display.shading.light = "STUDIO"
scene.render.filepath = str(root / "triposplat_tsdf_preview.png")
bpy.ops.render.render(write_still=True)
