#!/usr/bin/env python3
"""Open the structural source samples, regularized shell, and correspondence lines."""

from pathlib import Path
import argparse
import colorsys
import json
import math
import os
import time

import numpy as np

# Open3D 0.19's legacy Visualizer fails to initialize GLEW on this Wayland path;
# use the available XWayland display before GLFW is imported.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY")

import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


BACKEND = Path(__file__).resolve().parents[1]
SCENE = BACKEND / "current_scene"
INPUT = SCENE / "06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
ALIGNMENT_OUTPUT = OUTPUT / "segmentation_alignment"
ROLE_COLORS = {"wall": (0.2, 0.55, 1.0), "floor": (0.2, 0.9, 0.3), "ceiling": (1.0, 0.65, 0.1)}


def write_instance_translation_delta(shell):
    baseline_path = OUTPUT / "rigid_baseline_objects.ply"
    translated_path = OUTPUT / "instance_translated_objects.ply"
    baseline = o3d.io.read_triangle_mesh(str(baseline_path))
    translated = o3d.io.read_triangle_mesh(str(translated_path))
    baseline_ids = np.load(OUTPUT / "rigid_baseline_provenance.npz")["source_face_ids"]
    translated_ids = np.load(OUTPUT / "instance_translated_provenance.npz")["source_face_ids"]
    before = np.asarray(baseline.vertices)[np.asarray(baseline.triangles)].mean(axis=1)
    after = np.asarray(translated.vertices)[np.asarray(translated.triangles)].mean(axis=1)
    order = np.argsort(baseline_ids)
    sorted_ids, sorted_before = baseline_ids[order], before[order]
    report = json.loads((OUTPUT / "instance_translation_report.json").read_text())
    shell_vertices, shell_faces = np.asarray(shell.vertices), np.asarray(shell.triangles)
    stride = max(1, math.ceil(len(shell_faces) / 25_000))

    starts, ends = [], []
    offset = 0
    for group in report["groups"]:
        count = group["source_faces"]
        ids = translated_ids[offset:offset + count]
        locations = np.searchsorted(sorted_ids, ids)
        starts.append(sorted_before[locations].mean(axis=0))
        ends.append(after[offset:offset + count].mean(axis=0))
        offset += count
    starts, ends = np.asarray(starts), np.asarray(ends)
    moved = np.linalg.norm(ends - starts, axis=1) > 1e-8
    views = ((0, 2, "Plan · X/Z", "X (m)", "Z (m)"),
             (0, 1, "Elevation · X/Y", "X (m)", "Y (m)"),
             (2, 1, "Elevation · Z/Y", "Z (m)", "Y (m)"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, (x_axis, y_axis, title, xlabel, ylabel) in zip(axes, views):
        edges = shell_vertices[shell_faces[::stride]][:, :, [x_axis, y_axis]]
        ax.add_collection(LineCollection(edges, colors="#aaaaaa", linewidths=0.3, alpha=0.45))
        for start, end in zip(starts[moved], ends[moved]):
            ax.annotate("", xy=end[[x_axis, y_axis]], xytext=start[[x_axis, y_axis]],
                        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 0.65,
                                    "shrinkA": 0, "shrinkB": 0})
        ax.scatter(starts[moved, x_axis], starts[moved, y_axis], s=5, c="#2878b5",
                   alpha=0.8, label="Before")
        ax.scatter(ends[moved, x_axis], ends[moved, y_axis], s=5, c="#d62728",
                   alpha=0.8, label="After")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    path = OUTPUT / "instance_translation_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_contact_group_meshes(manifest_path, alignment_output, unresolved_group_ids=()):
    manifest = json.loads(manifest_path.read_text())
    meshes = []
    for group in manifest["groups"]:
        hue = (int(group["group_id"]) * 0.618033988749895) % 1.0
        color = ((1.0, 0.04, 0.04) if int(group["group_id"]) in unresolved_group_ids
                 else colorsys.hsv_to_rgb(hue, 0.78, 0.95))
        mesh_path = alignment_output / group["path"]
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not len(mesh.triangles):
            raise ValueError(f"Contact group mesh is empty or missing: {mesh_path}")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        meshes.append(mesh)
    return manifest, meshes


def back_shell_triangles(mesh, camera_position):
    centers = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)
    return np.einsum("ij,ij->i", normals, centers - camera_position) >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts-only", action="store_true", help="show provisional object-to-shell contact vectors")
    parser.add_argument("--rigid-baseline", action="store_true", help="show touching object blocks around the see-through shell")
    parser.add_argument("--structural-fit", action="store_true", help="show translated contact blocks inside the see-through shell")
    parser.add_argument("--instance-translated", action="store_true", help="show minimum per-instance translations inside the shell")
    args = parser.parse_args()
    shell = o3d.io.read_triangle_mesh(str(INPUT / "all_structural_regularized.ply"))
    shell.compute_vertex_normals()
    shell.paint_uniform_color((0.72, 0.72, 0.72))
    culling_shell = None
    if args.rigid_baseline or args.structural_fit or args.instance_translated:
        mode = "instance_translated" if args.instance_translated else ("structural_fit" if args.structural_fit else "rigid_baseline")
        if mode in {"rigid_baseline", "structural_fit"}:
            report_path = (ALIGNMENT_OUTPUT / "rigid_baseline_report.json" if mode == "rigid_baseline"
                           else ALIGNMENT_OUTPUT / "structural_fit/structural_fit_report.json")
            if not report_path.is_file():
                package = SCENE / "05_semantic_segmentation/mesh_objects_step3/segmentation_output"
                if mode == "structural_fit":
                    raise SystemExit("Fitted blocks are missing. Build them first with:\n"
                                     "  backend/.venv/bin/python backend/07_objecs_alignment/fit_contact_groups_inside.py")
                raise SystemExit("Alignment output is missing. Build it first with:\n"
                                 "  backend/.venv/bin/python backend/07_objecs_alignment/build_rigid_baseline.py\n"
                                 f"Required Stage 05 instance package: {package}\n"
                                 "Generate that package with the 3D object extraction step, then rerun this viewer.")
            report = json.loads(report_path.read_text())
            shell_path = (json.loads((ALIGNMENT_OUTPUT / "rigid_baseline_report.json").read_text())["inputs"]["regularized_shell"]
                          if mode == "structural_fit" else report["inputs"]["regularized_shell"])
            shell = o3d.io.read_triangle_mesh(shell_path)
            shell.compute_triangle_normals()
            shell.compute_vertex_normals()
            shell.paint_uniform_color((0.72, 0.72, 0.72))
            culling_shell = o3d.geometry.TriangleMesh(shell)
            group_report_path = (ALIGNMENT_OUTPUT / "contact_groups.json" if mode == "rigid_baseline"
                                 else ALIGNMENT_OUTPUT / "structural_fit/contact_groups.json")
            if not group_report_path.is_file():
                raise SystemExit(
                    "Contact groups are missing. Build them with:\n"
                    "  backend/.venv/bin/python backend/07_objecs_alignment/group_touching_instances.py"
                )
            unresolved = ({item["group_id"] for item in report["groups"]
                           if item.get("surface_intersection_edge_hits_after", 0) > 0
                           or item.get("minimum_instance_contained_surface_percent",
                                       item.get("contained_surface_percent", 100.0)) < 95.0}
                          if mode == "structural_fit" else ())
            group_report, group_meshes = load_contact_group_meshes(group_report_path, ALIGNMENT_OUTPUT, unresolved)
            geometries = (shell, *group_meshes)
            if mode == "rigid_baseline":
                title = "Touching object blocks around the regularized shell"
                transform = report["transform"]
                print(f"Translation: {transform['translation_m']}; rotation: "
                      f"{transform['rotation_angle_deg']:.3f} degrees; "
                      f"structural fit inliers: {report['fit']['inliers']}/{report['fit']['anchors']}.")
                print(f"Contact blocks: {group_report['group_count']} from {group_report['instance_count']} instances; "
                      f"{group_report['contact_pair_count']} touching pairs at "
                      f"{group_report['contact_distance_m']:.3f} m tolerance.")
            else:
                title = "Support fitting of contact blocks inside the regularized shell"
                print(f"Fitted blocks: {report['group_count']}; transformed: {report['transformed_groups']}; "
                      f"unresolved intersections: {report['unresolved_intersection_groups']}; "
                      f"partial: {report['partial_groups']}; mean contained surface: "
                      f"{report['mean_contained_surface_percent']:.2f}%.")
                print("Nearby floor and wall contacts are fitted by translation only. Blocks with more than 5% of any instance outside are red.")
            print("Each color is one rigid block. Near-side shell triangles are hidden from the current view.")
            fallback = report_path
        if mode == "instance_translated":
            report_path = OUTPUT / "instance_translation_report.json"
            objects = o3d.io.read_triangle_mesh(str(OUTPUT / f"{mode}_objects.ply"))
            report = json.loads(report_path.read_text())
            objects.compute_vertex_normals()
            geometries = (shell, objects)
            fallback = report_path
            print("Gray: regularized shell. Brown: translated objects.")
            title = "Minimum per-instance translations inside the shell"
            print(f"Translated groups: {report['translated_groups']}; unresolved groups: {report['unresolved_groups']}.")
            print("Each group is rigidly translated; no triangles are removed.")
            print(f"Before/after displacement view: {write_instance_translation_delta(shell)}")
    elif args.contacts_only:
        lines = o3d.io.read_line_set(str(OUTPUT / "instance_contact_preview.ply"))
        objects = o3d.io.read_triangle_mesh(str(OUTPUT / "candidate_instances_for_contact_review.ply"))
        geometries = (shell, objects, lines)
        title = "Provisional instance-to-shell contact candidates"
        print("Object surfaces and their contact lines share a color; instance IDs and target roles are in instance_contact_candidates.json")
        print("Lines show view-supported, nearby, parallel surfaces; they are not confirmed attachments.")
        fallback = OUTPUT / "instance_contact_candidates.json"
    else:
        lines = o3d.io.read_line_set(str(OUTPUT / "structural_correspondence.ply"))
        points = np.asarray(lines.points)
        samples = o3d.geometry.PointCloud()
        samples.points = o3d.utility.Vector3dVector(points[0::2])
        archive = np.load(OUTPUT / "structural_correspondence.npz")
        point_colors = []
        for role, color in ROLE_COLORS.items():
            key = role + "_source_face"
            if key in archive:
                point_colors.extend([color] * len(np.arange(0, len(archive[key]), 100)))
        samples.colors = o3d.utility.Vector3dVector(np.asarray(point_colors))
        targets = o3d.geometry.PointCloud()
        targets.points = o3d.utility.Vector3dVector(points[1::2])
        targets.colors = o3d.utility.Vector3dVector(np.asarray(lines.colors))
        geometries = (shell, samples, targets, lines)
        title = "New-step structural correspondences"
        print("Source samples: wall blue, floor green, ceiling orange")
        print("Target samples and lines: red low confidence, green high confidence, gray rejected, magenta ambiguous")
        print("Orbit/zoom to check whether each line endpoint lands on its intended shell surface.")
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(title, width=1400, height=900):
        print(f"Open3D could not create a graphics window. Review artifact: {fallback if args.contacts_only or args.rigid_baseline or args.structural_fit or args.instance_translated else OUTPUT / 'structural_correspondence_overview.png'}")
        return
    for geometry in geometries:
        viewer.add_geometry(geometry)
    options = viewer.get_render_option()
    if options is None:
        viewer.destroy_window()
        print(f"Open3D graphics initialization failed. Review artifact: {fallback if args.contacts_only or args.rigid_baseline or args.structural_fit or args.instance_translated else OUTPUT / 'structural_correspondence_overview.png'}")
        return
    options.mesh_color_option = o3d.visualization.MeshColorOption.Color
    options.light_on = False
    options.mesh_show_back_face = True
    options.point_size = 4.0
    while viewer.poll_events():
        if culling_shell is not None:
            parameters = viewer.get_view_control().convert_to_pinhole_camera_parameters()
            camera_position = np.linalg.inv(parameters.extrinsic)[:3, 3]
            keep = back_shell_triangles(culling_shell, camera_position)
            current = np.asarray(shell.triangles)
            if len(current) != int(keep.sum()) or not np.array_equal(current, np.asarray(culling_shell.triangles)[keep]):
                shell.triangles = o3d.utility.Vector3iVector(np.asarray(culling_shell.triangles)[keep])
                shell.compute_triangle_normals()
                shell.compute_vertex_normals()
                viewer.update_geometry(shell)
        viewer.update_renderer()
        # Cap software-rendered previews at 10 fps so an idle review window stays usable.
        time.sleep(0.1)
    viewer.destroy_window()


if __name__ == "__main__":
    main()
