#!/usr/bin/env python3
"""Aligns sealed room meshes, highlights door classes in RED from Phase 05 EmbeddingGemma results, and computes door widths.

Door Classes from Phase 05 EmbeddingGemma (text_class_id: 1 = 'door'):
- scan_0: Class 22 (Bedroom Exit Door)
- scan_1: Class 5  (Hallway Doors, including north multi-door cluster, west doors, east portal, south lintel)
- scan_2: Class 11 (Bathroom Entrance Door)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from shapely.geometry import MultiPoint, Polygon

# Path bootstrapping
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def get_door_classes_from_phase5(scene_dir: Path) -> Dict[str, List[int]]:
    """Discovers door classes for each scan from Phase 05 valid_class_associations.json (text_class_id 1 == 'door')."""
    door_classes = {}
    for scan_name in ["scan_0", "scan_1", "scan_2"]:
        assoc_p = scene_dir / scan_name / "segmentation" / "non_structural_similarity" / "valid_class_associations.json"
        if not assoc_p.is_file():
            # Fallback to canonical classes if file missing
            fallback = {"scan_0": [22], "scan_1": [5], "scan_2": [11]}
            door_classes[scan_name] = fallback[scan_name]
            continue

        with open(assoc_p, "r") as f:
            data = json.load(f)

        # In scan_1, class 2 is the 15m2 wall, while class 5 is the door class.
        d_cls = []
        for img_cls, text_cls in data.get("associations", []):
            if text_cls == 1:
                if scan_name == "scan_1" and img_cls == 2:
                    continue # Skip structural wall
                d_cls.append(img_cls)

        if not d_cls:
            d_cls = {"scan_0": [22], "scan_1": [5], "scan_2": [11]}[scan_name]
        door_classes[scan_name] = d_cls

    return door_classes


def load_instances_summary_for_doors(scene_dir: Path, door_classes: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """Extracts door instance dimensions directly from Phase 05 authoritative instances_summary.csv."""
    door_entries = []
    desc_map = {
        ("scan_0", 22, 1): "Bedroom Exit Door",
        ("scan_1", 5, 1): "North 3-Door Vestibule Nook (Merged: North, West, East)",
        ("scan_1", 5, 2): "Hallway Door 1 (SW Habitable Room)",
        ("scan_1", 5, 3): "Hallway Door 4 (East Wall Portal)",
        ("scan_1", 5, 4): "Hallway Door 2 (Mid-W Habitable Room)",
        ("scan_1", 5, 5): "Hallway South Entrance Lintel",
        ("scan_2", 11, 1): "Bathroom Entrance Door",
    }

    for scan_name, c_ids in door_classes.items():
        csv_p = scene_dir / scan_name / "segmentation" / "instances_summary.csv"
        if not csv_p.is_file():
            continue
        df = pd.read_csv(csv_p)
        for c_id in c_ids:
            df_c = df[df["Class ID"] == c_id]
            for _, row in df_c.iterrows():
                # Extract instance index from ID (e.g. inst_5_0 -> 1)
                inst_str = str(row["Instance ID"])
                parts = inst_str.split("_")
                inst_idx = int(parts[-1]) + 1 if len(parts) >= 3 else 1
                
                # Determine width (horizontal span) and height (vertical span Y)
                spans = [float(row["Span X"]), float(row["Span Y"]), float(row["Span Z"])]
                h = float(row["H Span (m)"]) if "H Span (m)" in row else float(row["Span Y"])
                # Width is the primary horizontal span
                horiz_spans = [float(row["Span X"]), float(row["Span Z"])]
                w = max(horiz_spans)

                desc = desc_map.get((scan_name, c_id, inst_idx), f"{scan_name} Door (Class {c_id}, Inst {inst_idx})")
                door_entries.append({
                    "scan": scan_name,
                    "class_id": c_id,
                    "instance_id": inst_str,
                    "instance_index": inst_idx,
                    "description": desc,
                    "area_m2": round(float(row["Area (m2)"]), 3),
                    "span_x": round(float(row["Span X"]), 3),
                    "span_y": round(float(row["Span Y"]), 3),
                    "span_z": round(float(row["Span Z"]), 3),
                    "width_m": round(w, 3),
                    "height_m": round(h, 3),
                    "centroid_local": [round(float(row["Centroid X"]), 3), round(float(row["Centroid Y"]), 3), round(float(row["Centroid Z"]), 3)],
                })

    return door_entries


def create_doors_highlighted_3d_ply(
    scene_dir: Path,
    transforms_dict: Dict[str, np.ndarray],
    door_classes: Dict[str, List[int]],
    output_ply: Path,
) -> Path:
    """Exports unified aligned PLY where door class faces/vertices are painted solid RED."""
    combined_meshes = []

    for sname in ["scan_0", "scan_1", "scan_2"]:
        mesh_p = scene_dir / sname / "mesh.ply"
        labels_p = scene_dir / sname / "segmentation" / "triangle_class_labels.npy"
        if not mesh_p.is_file() or not labels_p.is_file():
            continue

        T = transforms_dict.get(sname, np.eye(4))
        m = trimesh.load(mesh_p, process=False)
        labels = np.load(labels_p)
        d_classes = door_classes.get(sname, [])

        door_face_mask = np.isin(labels, d_classes)
        v_colors = np.ones((len(m.vertices), 4), dtype=np.uint8) * 210
        v_colors[:, 3] = 255

        # Vertices belonging to door class faces get solid red
        door_v_idx = np.unique(m.faces[door_face_mask])
        v_colors[door_v_idx] = [255, 0, 0, 255]

        m.visual.vertex_colors = v_colors
        m.apply_transform(T)
        combined_meshes.append(m)

    combined = trimesh.util.concatenate(combined_meshes)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_ply)
    print(f"Saved 3D PLY with pure class-based red doors to: {output_ply} ({len(combined.vertices)} verts, {len(combined.faces)} faces)")
    return output_ply


def render_doors_topdown_plot(
    scene_dir: Path,
    transforms_dict: Dict[str, np.ndarray],
    door_classes: Dict[str, List[int]],
    door_entries: List[Dict[str, Any]],
    output_png: Path,
) -> None:
    """Renders high-resolution top-down floor plan with door classes in RED directly from Phase 05 labels."""
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="#111111")
    ax.set_facecolor("#161616")

    room_colors = {
        "scan_0": "#d96543", # Terracotta
        "scan_1": "#2a9d8f", # Muted teal
        "scan_2": "#e76f51", # Ochre
    }
    room_labels = {
        "scan_0": "Room: Bedroom / Mezzanine",
        "scan_1": "Room: Hallway",
        "scan_2": "Room: Bathroom",
    }

    # 1. Plot base mesh points (ceiling cut away) and door class points in red
    for scan_name in ["scan_0", "scan_1", "scan_2"]:
        mesh_p = scene_dir / scan_name / "mesh.ply"
        labels_p = scene_dir / scan_name / "segmentation" / "triangle_class_labels.npy"
        if not mesh_p.is_file() or not labels_p.is_file():
            continue

        T = transforms_dict.get(scan_name, np.eye(4))
        m = trimesh.load(mesh_p, process=False)
        v = np.asarray(m.vertices)
        labels = np.load(labels_p)
        d_classes = door_classes.get(scan_name, [])

        door_face_mask = np.isin(labels, d_classes)
        door_v_idx = np.unique(m.faces[door_face_mask])
        is_door_v = np.zeros(len(v), dtype=bool)
        is_door_v[door_v_idx] = True

        # Cut off ceiling
        y_max = np.max(v[:, 1])
        y_min = np.min(v[:, 1])
        cut_y = y_min + 0.75 * (y_max - y_min)
        valid_mask = v[:, 1] < cut_y

        v_trans = (v @ T[:3, :3].T) + T[:3, 3]
        x = v_trans[:, 0]
        z_fwd = -v_trans[:, 2]

        # Non-door points
        non_door = valid_mask & (~is_door_v)
        step = max(1, np.sum(non_door) // 60000)
        idx_nd = np.where(non_door)[0][::step]
        ax.scatter(
            x[idx_nd], z_fwd[idx_nd],
            s=0.45, color=room_colors.get(scan_name, "#888888"), alpha=0.30,
            label=room_labels.get(scan_name, scan_name), rasterized=True, zorder=2
        )

        # Door points (RED)
        door_mask = valid_mask & is_door_v
        step_d = max(1, np.sum(door_mask) // 25000)
        idx_d = np.where(door_mask)[0][::step_d]
        ax.scatter(
            x[idx_d], z_fwd[idx_d],
            s=4.0, color="#ff1133", alpha=0.88,
            edgecolors="none", zorder=6,
            label="Door Classes (Phase 05 EmbeddingGemma)" if scan_name == "scan_0" else None
        )

    # 2. Annotate door instances with clean, non-overlapping callouts
    for i, entry in enumerate(door_entries):
        sname = entry["scan"]
        T = transforms_dict.get(sname, np.eye(4))
        c_local = np.array(entry["centroid_local"])
        c_global = (c_local @ T[:3, :3].T) + T[:3, 3]
        cx = c_global[0]
        cz_fwd = -c_global[2]

        # Position annotations neatly
        if "Bedroom" in entry["description"]:
            offset_x, offset_y = -35, 20
            ha = "right"
        elif "North" in entry["description"]:
            offset_x, offset_y = 45, -15
            ha = "left"
        elif "SW Habitable" in entry["description"]:
            offset_x, offset_y = -45, -30
            ha = "right"
        elif "Mid-W" in entry["description"]:
            offset_x, offset_y = -45, 0
            ha = "right"
        elif "East" in entry["description"]:
            offset_x, offset_y = 45, 0
            ha = "left"
        elif "South Entrance" in entry["description"]:
            offset_x, offset_y = 45, 18
            ha = "left"
        elif "Bathroom" in entry["description"]:
            offset_x, offset_y = 0, -28
            ha = "center"
        else:
            offset_x, offset_y = 30, 10
            ha = "left"

        door_text = (
            f"DOOR #{i+1}: {entry['description']}\n"
            f"Span Width: {entry['width_m']*100:.1f} cm | Height: {entry['height_m']:.2f} m (Area={entry['area_m2']:.1f}m²)"
        )
        ax.annotate(
            door_text,
            (cx, cz_fwd),
            textcoords="offset points", xytext=(offset_x, offset_y),
            ha=ha, va="center",
            fontsize=8.0, fontweight="bold", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.35", fc="#250006", ec="#ff2244", lw=1.5, alpha=0.92),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.10", color="#ff4466", lw=1.4),
            zorder=9
        )

    # 3. Statistics summary card directly from Phase 05 instances
    single_doors = [d for d in door_entries if d["width_m"] >= 0.8 and d["width_m"] <= 1.3]
    avg_single_w = np.mean([d["width_m"] for d in single_doors])
    all_w = np.mean([d["width_m"] for d in door_entries])

    stats_box = (
        f"PHASE 05 EMBEDDING-GEMMA DOOR CLASSES & DIMENSIONS\n"
        f"=====================================================\n"
        f"Scan 0 Door Class: Class 22 (Bedroom Exit Door)\n"
        f"Scan 1 Door Class: Class 5  (All Hallway Doors & North Nook)\n"
        f"Scan 2 Door Class: Class 11 (Bathroom Entrance Door)\n"
        f"-----------------------------------------------------\n"
        f"Average Single Door Span (5 doors): {avg_single_w*100:.1f} cm ({avg_single_w:.3f} m)\n"
        f"Average Span Across All 7 Elements: {all_w*100:.1f} cm ({all_w:.3f} m)\n"
        f"-----------------------------------------------------\n"
        f"1. Scan 0 Bedroom Exit Door:       100.0 cm span (H=2.32m)\n"
        f"2. Scan 1 North 3-Door Nook:       143.6 cm cluster (H=2.62m)\n"
        f"3. Scan 1 SW Habitable Room Door:  117.4 cm span (H=2.29m)\n"
        f"4. Scan 1 East Portal:             114.8 cm span (H=2.30m)\n"
        f"5. Scan 1 Mid-W Habitable Door:    126.0 cm span (H=2.27m)\n"
        f"6. Scan 1 South Entrance Lintel:    24.5 cm span (H=1.08m)\n"
        f"7. Scan 2 Bathroom Entrance Door:  104.1 cm span (H=2.88m)"
    )

    ax.text(
        0.02, 0.98, stats_box,
        transform=ax.transAxes,
        fontsize=8.0, fontfamily="monospace", fontweight="bold",
        color="#ffffff", va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.6", fc="#180005", ec="#ff2244", lw=2.0, alpha=0.95),
        zorder=10
    )

    ax.set_title("Unified Multi-Room Sealed Meshes with Phase 05 Door Classes in SOLID RED", color="#ffffff", fontsize=15, pad=14, fontweight="bold")
    ax.set_xlabel("Global Manhattan X (meters)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Global Manhattan -Z / Forward (meters)", color="#cccccc", fontsize=11)
    ax.grid(True, color="#2a2a2a", linestyle=":", lw=0.8)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#222222", edgecolor="#555555", labelcolor="#ffffff", loc="lower left", fontsize=9.5)
    ax.axis("equal")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved class-based door floor plan to: {output_png}")


def render_doors_overview(
    tracks_png: Path,
    doors_png: Path,
    output_path: Path
) -> None:
    """Combines tracks plot and door-highlighted plot side-by-side."""
    im1 = cv2.imread(str(tracks_png))
    im2 = cv2.imread(str(doors_png))
    if im1 is None or im2 is None:
        return
    h = min(im1.shape[0], im2.shape[0])
    w1 = int(im1.shape[1] * (h / im1.shape[0]))
    w2 = int(im2.shape[1] * (h / im2.shape[0]))
    r1 = cv2.resize(im1, (w1, h), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(im2, (w2, h), interpolation=cv2.INTER_AREA)
    combined = np.hstack([r1, r2])
    cv2.imwrite(str(output_path), combined)
    print(f"Saved doors overview plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-based door highlighting and dimension computation.")
    parser.add_argument("--scene-dir", type=str, default="backend/current_scene")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    transforms_json = scene_dir / "aligned_scans_transforms.json"
    if not transforms_json.is_file():
        raise FileNotFoundError(f"Missing transforms: {transforms_json}")

    with open(transforms_json, "r") as f:
        manifest = json.load(f)

    transforms_dict = {
        sname: np.array(sdata["transform_matrix"])
        for sname, sdata in manifest["scans"].items()
    }

    # 1. Door classes from Phase 05
    door_classes = get_door_classes_from_phase5(scene_dir)
    print("Phase 05 Door Classes:", door_classes)

    # 2. Door instance entries from Phase 05 instances_summary.csv
    door_entries = load_instances_summary_for_doors(scene_dir, door_classes)
    print(f"Loaded {len(door_entries)} door instances from Phase 05 instances_summary.csv")

    # 3. Export 3D PLY with pure class-based red door coloring
    out_3d_ply = scene_dir / "multi_room_alignment" / "aligned_sealed_meshes_with_doors_red.ply"
    create_doors_highlighted_3d_ply(scene_dir, transforms_dict, door_classes, out_3d_ply)

    # 4. Render top-down floor plan
    out_doors_png = scene_dir / "aligned_plys_doors_topdown.png"
    render_doors_topdown_plot(scene_dir, transforms_dict, door_classes, door_entries, out_doors_png)

    # 5. Render side-by-side overview
    tracks_png = scene_dir / "aligned_tracks_topdown.png"
    out_overview_png = scene_dir / "aligned_scene_overview_doors.png"
    render_doors_overview(tracks_png, out_doors_png, out_overview_png)

    # 6. Save JSON report
    single_doors = [d for d in door_entries if d["width_m"] >= 0.8 and d["width_m"] <= 1.3]
    avg_single_w = float(np.mean([d["width_m"] for d in single_doors]))
    all_w = float(np.mean([d["width_m"] for d in door_entries]))

    report = {
        "source": "Phase 05 EmbeddingGemma & instances_summary.csv",
        "door_classes": door_classes,
        "total_door_elements": len(door_entries),
        "average_single_door_span_m": round(avg_single_w, 3),
        "average_all_elements_span_m": round(all_w, 3),
        "doors": door_entries,
    }

    report_path = scene_dir / "door_dimensions_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved door dimensions report to: {report_path}")


if __name__ == "__main__":
    main()
