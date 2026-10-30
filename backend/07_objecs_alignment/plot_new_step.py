#!/usr/bin/env python3
"""Write static plan/elevation overviews of correspondences or contact candidates."""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import open3d as o3d


BACKEND = Path(__file__).resolve().parents[1]
SCENE = BACKEND / "current_scene"
INPUT = SCENE / "06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
ROLES = {"wall": "#2878b5", "floor": "#28a745", "ceiling": "#e68a00"}
PLOT_STRIDE = 1000
VIEWS = ((0, 2, "Plan · X/Z", "X (m)", "Z (m)"),
         (0, 1, "Elevation · X/Y", "X (m)", "Y (m)"),
         (2, 1, "Elevation · Z/Y", "Z (m)", "Y (m)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts-only", action="store_true", help="plot provisional object-to-shell contacts")
    args = parser.parse_args()
    if args.contacts_only:
        plot_contacts()
        return

    mesh = o3d.io.read_triangle_mesh(str(INPUT / "original_mesh.ply"))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    shell = o3d.io.read_triangle_mesh(str(INPUT / "all_structural_regularized.ply"))
    shell_vertices = np.asarray(shell.vertices)
    shell_triangles = np.asarray(shell.triangles)
    archive = np.load(OUTPUT / "structural_correspondence.npz")
    records = {}
    for role in ROLES:
        face_ids = archive[role + "_source_face"]
        sampled = np.arange(0, len(face_ids), PLOT_STRIDE)
        source = vertices[triangles[face_ids[sampled]]].mean(axis=1)
        target = archive[role + "_target_point"][sampled]
        confidence = archive[role + "_confidence"][sampled]
        accepted = archive[role + "_accepted"][sampled]
        ambiguous = archive[role + "_ambiguous"][sampled]
        records[role] = source, target, confidence, accepted, ambiguous

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (x_axis, y_axis, title, xlabel, ylabel) in zip(axes, VIEWS):
        shell_edges = shell_vertices[shell_triangles][:, :, [x_axis, y_axis]]
        ax.add_collection(LineCollection(shell_edges, colors="#444444", linewidths=0.35, alpha=0.75, zorder=1))
        for role, (source, target, confidence, accepted, ambiguous) in records.items():
            source_2d = source[:, [x_axis, y_axis]]
            target_2d = target[:, [x_axis, y_axis]]
            segments = np.stack((source_2d, target_2d), axis=1)
            colors = plt.get_cmap("RdYlGn")(confidence)
            colors[~accepted] = (0.65, 0.65, 0.65, 0.45)
            colors[ambiguous] = (0.8, 0.0, 0.8, 0.9)
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.3, alpha=0.45, zorder=2))
            ax.scatter(source_2d[:, 0], source_2d[:, 1], s=1, c=ROLES[role], label=role, alpha=0.55, zorder=3)
            ax.scatter(target_2d[:, 0], target_2d[:, 1], s=1, c=colors, alpha=0.7, zorder=4)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    legend = [Line2D([], [], marker="o", linestyle="", color=color, label=role)
              for role, color in ROLES.items()]
    legend.extend(Line2D([], [], color=color, label=label) for color, label in (
        ("#d7191c", "low confidence"), ("#1a9641", "high confidence"),
        ("#8c8c8c", "rejected"), ("#cc00cc", "ambiguous")))
    axes[0].legend(handles=legend, markerscale=1.3, fontsize=8, loc="upper right")
    fig.suptitle(f"Source structural samples → regularized-shell correspondences (every {PLOT_STRIDE}th sample)\n"
                 "gray shell edges · role-colored source samples · green high confidence · magenta ambiguous")
    path = OUTPUT / "structural_correspondence_overview.png"
    fig.savefig(path, dpi=160)
    print(path)


def plot_contacts():
    shell = o3d.io.read_triangle_mesh(str(INPUT / "all_structural_regularized.ply"))
    shell_vertices, shell_triangles = np.asarray(shell.vertices), np.asarray(shell.triangles)
    objects = o3d.io.read_triangle_mesh(str(OUTPUT / "candidate_instances_for_contact_review.ply"))
    object_vertices, object_triangles = np.asarray(objects.vertices), np.asarray(objects.triangles)
    object_centers = object_vertices[object_triangles].mean(axis=1)
    object_colors = np.asarray(objects.vertex_colors).reshape(-1, 3, 3).mean(axis=1)
    lines = o3d.io.read_line_set(str(OUTPUT / "instance_contact_preview.ply"))
    line_points, line_indices = np.asarray(lines.points), np.asarray(lines.lines)
    line_colors = np.asarray(lines.colors)
    report = json.loads((OUTPUT / "instance_contact_candidates.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(19, 9))
    for ax, (x_axis, y_axis, title, xlabel, ylabel) in zip(axes, VIEWS):
        shell_edges = shell_vertices[shell_triangles][:, :, [x_axis, y_axis]]
        ax.add_collection(LineCollection(shell_edges, colors="#444444", linewidths=0.3, alpha=0.6, zorder=1))
        ax.scatter(object_centers[:, x_axis], object_centers[:, y_axis], s=1.1,
                   c=object_colors, alpha=0.55, linewidths=0, zorder=2)
        segments = line_points[line_indices][:, :, [x_axis, y_axis]]
        ax.add_collection(LineCollection(segments, colors=line_colors, linewidths=1.0, alpha=0.95, zorder=3))
        ax.scatter(line_points[line_indices[:, 1], x_axis], line_points[line_indices[:, 1], y_axis],
                   s=8, c=line_colors, marker="x", linewidths=0.8, zorder=4)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    handles, labels = [], []
    for row in report["instances"]:
        contacts = row["contact_candidates"]
        if not contacts:
            continue
        roles = ", ".join(dict.fromkeys(contact["role"] for contact in contacts))
        color = row["preview_color_rgb"]
        handles.append(Line2D([], [], color=color, linewidth=2))
        labels.append(f"ID {row['instance_id']}: {roles}")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, fontsize=8,
               title="Object ID and candidate target role (line color matches object)")
    fig.suptitle("Provisional object-to-shell contacts · x marks shell endpoints\n"
                 "Colored dots show sampled candidate object surfaces; gray lines show the regularized shell")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.89, bottom=0.28, wspace=0.2)
    path = OUTPUT / "instance_contact_overview.png"
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
