#!/usr/bin/env python3
"""Open the structural source samples, regularized shell, and correspondence lines."""

from pathlib import Path
import argparse
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
from matplotlib.lines import Line2D


BACKEND = Path(__file__).resolve().parents[1]
SCENE = BACKEND / "current_scene"
INPUT = SCENE / "06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
ROLE_COLORS = {"wall": (0.2, 0.55, 1.0), "floor": (0.2, 0.9, 0.3), "ceiling": (1.0, 0.65, 0.1)}
BASELINE_VIEWS = ((0, 2, "Plan · X/Z", "X (m)", "Z (m)"),
                  (0, 1, "Elevation · X/Y", "X (m)", "Y (m)"),
                  (2, 1, "Elevation · Z/Y", "Z (m)", "Y (m)"))


def write_baseline_overview(shell, objects, provenance_path, output_path, source_face_ids=None,
                            point_size=0.5):
    sv, sf = np.asarray(shell.vertices), np.asarray(shell.triangles)
    ov, of = np.asarray(objects.vertices), np.asarray(objects.triangles)
    source_faces = (np.load(provenance_path)["source_face_ids"] if source_face_ids is None
                    else source_face_ids)
    regions = np.load(SCENE / "06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy",
                      mmap_mode="r")[source_faces]
    centers = ov[of].mean(axis=1)
    class189 = regions == 189
    ordinary = ~class189
    shell_stride = max(1, math.ceil(len(sf) / 25_000))
    object_stride = max(1, math.ceil(len(of) / 35_000))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, (x_axis, y_axis, title, xlabel, ylabel) in zip(axes, BASELINE_VIEWS):
        edges = sv[sf[::shell_stride]][:, :, [x_axis, y_axis]]
        ax.add_collection(LineCollection(edges, colors="#777777", linewidths=0.35,
                                         alpha=0.45, zorder=1))
        ax.scatter(centers[ordinary][::object_stride, x_axis],
                   centers[ordinary][::object_stride, y_axis], s=point_size,
                   c="#8f5937", alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(centers[class189][::object_stride, x_axis],
                   centers[class189][::object_stride, y_axis], s=point_size * 1.6,
                   c="#ff7f0e", alpha=0.85, linewidths=0, zorder=3)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(handles=[
        Line2D([], [], color="#777777", label="Regularized shell"),
        Line2D([], [], marker="o", linestyle="", color="#8f5937", label="Other objects"),
        Line2D([], [], marker="o", linestyle="", color="#ff7f0e", label="Region class 189")],
        loc="upper right", fontsize=8)
    path = output_path
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts-only", action="store_true", help="show provisional object-to-shell contact vectors")
    parser.add_argument("--rigid-baseline", action="store_true", help="compare rigid candidate objects with the unchanged shell")
    parser.add_argument("--instance-translated", action="store_true", help="show minimum per-instance translations inside the shell")
    args = parser.parse_args()
    shell = o3d.io.read_triangle_mesh(str(INPUT / "all_structural_regularized.ply"))
    shell.compute_vertex_normals()
    shell.paint_uniform_color((0.72, 0.72, 0.72))
    if args.rigid_baseline or args.instance_translated:
        mode = "instance_translated" if args.instance_translated else "rigid_baseline"
        provenance_path = OUTPUT / f"{mode}_provenance.npz"
        report_path = OUTPUT / ("instance_translation_report.json" if args.instance_translated
                                else "rigid_baseline_report.json")
        objects = o3d.io.read_triangle_mesh(str(OUTPUT / f"{mode}_objects.ply"))
        report = json.loads(report_path.read_text())
        objects.compute_vertex_normals()
        regions = np.load(SCENE / "06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy",
                          mmap_mode="r")
        colors = np.tile((0.58, 0.34, 0.22), (len(objects.vertices), 1))
        source_faces = np.load(provenance_path)["source_face_ids"]
        class_vertices = np.unique(np.asarray(objects.triangles)[regions[source_faces] == 189])
        colors[class_vertices] = (1.0, 0.38, 0.04)
        objects.vertex_colors = o3d.utility.Vector3dVector(colors)
        instance_manifest_path = OUTPUT / "rigid_baseline_clusters.json"
        instance_meshes = []
        instance_points = []
        preview_objects, preview_source_faces = objects, None
        preview_shell = shell
        if mode == "rigid_baseline" and instance_manifest_path.exists():
            cluster_data = json.loads(instance_manifest_path.read_text())
            source_ids = np.load(OUTPUT / "rigid_baseline_instance_provenance.npz")["source_face_ids"]
            assignments = np.load(OUTPUT / "panoptic3d_workspace/06c_semantic_segmentation/"
                                  "panoptic3d_current_source_candidate/face_assignments.npz")
            all_instance_ids = assignments["instance_ids"]
            regions = np.load(SCENE / "06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy",
                              mmap_mode="r")
            structural = json.loads((SCENE / "06_mesh_elaboration/generic_graph_regions/"
                                     "structural_classes.json").read_text())["classifications"]
            structural_ids = {int(class_id) for class_id, info in structural.items()
                              if info.get("structural_role") in {"wall", "floor", "ceiling"}}
            first = None
            for item in cluster_data["instances"]:
                instance_faces = regions[(all_instance_ids == item["instance_id"]) &
                                         ~assignments["removed"] &
                                         (regions > 0)]
                if not len(instance_faces):
                    continue
                class_ids, class_counts = np.unique(instance_faces, return_counts=True)
                dominant_class = int(class_ids[np.argmax(class_counts)])
                if dominant_class not in structural_ids:
                    first = item
                    break
            if first is None:
                raise ValueError("No instance with a known non-structural dominant class was found")
            cluster_count = max(1, len(cluster_data["clusters"]))
            cluster_colors = plt.get_cmap("hsv")
            item = first
            mesh = o3d.io.read_triangle_mesh(str(OUTPUT / "rigid_baseline_instances" / item["path"]))
            mesh.compute_vertex_normals()
            color = cluster_colors(item["cluster_id"] / cluster_count)
            colors = np.tile(color[:3], (len(mesh.vertices), 1))
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            instance_meshes.append(mesh)
            points = o3d.geometry.PointCloud()
            points.points = mesh.vertices
            points.paint_uniform_color((1.0, 0.1, 0.1))
            instance_points.append(points)
            start, count = item["provenance_offset"], item["provenance_count"]
            preview_objects = mesh
            preview_source_faces = source_ids[start:start + count]
            geometries = (instance_meshes[0], instance_points[0])
            preview_shell = o3d.geometry.TriangleMesh()
            color = matplotlib.colors.to_hex(cluster_colors(first["cluster_id"] / cluster_count)[:3])
            _, component_counts, _ = instance_meshes[0].cluster_connected_triangles()
            print(f"Displaying first non-structural instance {first['path']} "
                  f"(dominant class {dominant_class}, {first['faces_written']} faces), "
                  f"cluster {first['cluster_id']} color {color}; "
                  f"{len(component_counts)} disconnected pieces. Red points mark its vertices.")
        else:
            geometries = (shell, objects)
        if mode == "instance_translated":
            print("Gray: regularized shell. Brown: translated objects. Orange: region class 189.")
        else:
            print("Gray: regularized shell. Each border-connected cluster has its own color; untracked geometry is gray.")
        if args.instance_translated:
            title = "Minimum per-instance translations inside the shell"
            print(f"Translated groups: {report['translated_groups']}; unresolved groups: {report['unresolved_groups']}.")
            print("Each group is rigidly translated; no triangles are removed.")
            print(f"Before/after displacement view: {write_instance_translation_delta(shell)}")
        else:
            title = ("First aligned object instance" if mode == "rigid_baseline" and instance_meshes
                     else "Global structural alignment applied to original objects")
            alignment = report["global_alignment"]
            role_rms = ", ".join(f"{role} {alignment['roles'][role]['after']['area_confidence_weighted_rms_m']:.3f}"
                                  for role in ("wall", "floor", "ceiling"))
            print(f"One rigid transform applied to all objects: translation {alignment['translation_m']}, "
                  f"rotation {alignment['rotation_angle_deg']:.3f}°. Fit RMS (m): {role_rms}.")
            print("No per-instance transforms or boundary-based triangle removals were applied.")
        fallback = report_path
        overview_path = OUTPUT / f"{mode}_overview.png"
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
        if args.rigid_baseline or args.instance_translated:
            fallback = write_baseline_overview(preview_shell, preview_objects, provenance_path, overview_path,
                                               preview_source_faces, point_size=8.0)
        print(f"Open3D could not create a graphics window. Review artifact: {fallback if args.contacts_only or args.rigid_baseline or args.instance_translated else OUTPUT / 'structural_correspondence_overview.png'}")
        return
    for geometry in geometries:
        viewer.add_geometry(geometry)
    options = viewer.get_render_option()
    if options is None:
        viewer.destroy_window()
        if args.rigid_baseline or args.instance_translated:
            fallback = write_baseline_overview(preview_shell, preview_objects, provenance_path, overview_path,
                                               preview_source_faces, point_size=8.0)
        print(f"Open3D graphics initialization failed. Review artifact: {fallback if args.contacts_only or args.rigid_baseline or args.instance_translated else OUTPUT / 'structural_correspondence_overview.png'}")
        return
    options.mesh_color_option = o3d.visualization.MeshColorOption.Color
    options.light_on = False
    options.mesh_show_back_face = True
    options.point_size = 8.0 if args.rigid_baseline else 4.0
    if args.rigid_baseline:
        options.mesh_show_wireframe = True
    while viewer.poll_events():
        viewer.update_renderer()
        # Cap software-rendered previews at 10 fps so an idle review window stays usable.
        time.sleep(0.1)
    viewer.destroy_window()


if __name__ == "__main__":
    main()
