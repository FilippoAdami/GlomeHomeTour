#!/usr/bin/env python3
"""
Interactive Real-Time 3D Class & Instance Inspector (Solution 1)

Allows visual inspection of individual semantic classes and their connected-component
instances without visual clutter from other classes.

Controls:
  [A] / [D] or [Left] / [Right] : Cycle previous / next class
  [S]                            : Toggle View Mode (SPOTLIGHT -> SOLO -> RAINBOW)
  [I] / [Up]                     : Next Instance within active class (highlighted in gold)
  [Down]                         : Previous Instance
  [0]                            : Clear instance selection (highlight all instances)
  [F]                            : Focus camera on active class / instance centroid
  [R]                            : Reset camera to overview angle
  [H]                            : Print live metadata and keyboard shortcuts
  [Q] / [ESC]                    : Quit viewer
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# On Linux Wayland (e.g. Pop!_OS COSMIC / Ubuntu / Fedora), Open3D's bundled GLFW
# fails to initialize GLEW under native Wayland EGL, causing create_window() to fail.
# Falling back to XWayland (via DISPLAY) allows GLFW + GLEW to initialize cleanly.
if "WAYLAND_DISPLAY" in os.environ and "DISPLAY" in os.environ:
    os.environ.pop("WAYLAND_DISPLAY", None)

import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

# Import project color and graph utilities
try:
    from detect_structural_classes import get_class_color, build_mesh_dual_graph
except ImportError:
    # Fallback HSV color generator
    import colorsys

    def get_class_color(mid: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
        h = ((mid * 37) % 360) / 360.0
        s = 0.85
        v = 0.90
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255)), (r, g, b)


# Default visual colors
COLOR_CLAY = np.array([0.82, 0.82, 0.82], dtype=np.float32)       # Soft clay backdrop
COLOR_INSTANCE_GOLD = np.array([1.0, 0.88, 0.05], dtype=np.float32) # Bright electric gold


class ClassInspector:
    def __init__(
        self,
        mesh_path: Path,
        structural_json_path: Path,
        instances_csv_path: Optional[Path] = None,
        per_class_dir: Optional[Path] = None,
        initial_cid: Optional[int] = None,
        initial_mode: str = "spotlight",
    ):
        self.mesh_path = mesh_path
        self.structural_json_path = structural_json_path
        self.instances_csv_path = instances_csv_path
        self.per_class_dir = per_class_dir
        self.mode = initial_mode.lower() # "spotlight", "solo", "rainbow"

        # Load metadata
        self._load_metadata()

        # Load base mesh
        print(f"[ClassInspector] Loading full room mesh: {self.mesh_path}...")
        t0 = time.perf_counter()
        self.full_mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        self.full_mesh.compute_vertex_normals()
        self.verts = np.asarray(self.full_mesh.vertices)
        self.tris = np.asarray(self.full_mesh.triangles)
        self.num_verts = len(self.verts)
        self.num_tris = len(self.tris)
        self.orig_v_colors = np.asarray(self.full_mesh.vertex_colors).copy()
        if len(self.orig_v_colors) == 0:
            self.orig_v_colors = np.ones((self.num_verts, 3), dtype=np.float32) * 0.7
        print(f"  Loaded {self.num_verts:,} vertices, {self.num_tris:,} triangles in {time.perf_counter() - t0:.2f}s")

        # Build mesh dual graph for exact component clustering and filament rejection
        print("[ClassInspector] Building dual mesh topology graph...")
        t0_g = time.perf_counter()
        self.adj_u, self.adj_v = build_mesh_dual_graph(self.tris)
        print(f"  Dual graph built in {time.perf_counter() - t0_g:.2f}s ({len(self.adj_u):,} edges)")

        # Recover triangle labels
        self._compute_triangle_labels()

        # Setup active class state
        if initial_cid and initial_cid in self.active_classes:
            self.class_idx = self.active_classes.index(initial_cid)
        else:
            self.class_idx = 0

        # Instance state within current class
        self.active_instance_idx: int = -1 # -1 means all instances
        self.class_instances: List[Dict[str, Any]] = []

        # Geometry displayed in Open3D
        self.display_mesh = o3d.geometry.TriangleMesh()
        self.display_mesh.vertices = self.full_mesh.vertices
        self.display_mesh.triangles = self.full_mesh.triangles
        self.display_mesh.vertex_normals = self.full_mesh.vertex_normals
        self.display_mesh.vertex_colors = o3d.utility.Vector3dVector(self.orig_v_colors.copy())

        # Update to initial class
        self._load_current_class_instances()
        self._apply_colors(update_geometry_buffers=False)

    def _load_metadata(self) -> None:
        with open(self.structural_json_path) as f:
            struct_data = json.load(f)
        self.classifications = struct_data.get("classifications", {})
        self.active_classes = sorted([int(k) for k in self.classifications.keys()])
        if not self.active_classes:
            raise ValueError(f"No active classes found in {self.structural_json_path}")

        # Load instances CSV if present
        self.instances_by_class: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in self.active_classes}
        if self.instances_csv_path and self.instances_csv_path.is_file():
            with open(self.instances_csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = int(row["Class ID"])
                    if cid in self.instances_by_class:
                        self.instances_by_class[cid].append({
                            "instance_id": row["Instance ID"],
                            "assigned_role": row["Assigned Role"],
                            "triangle_count": int(row["Triangle Count"]),
                            "area_m2": float(row["Area (m2)"]),
                            "centroid": np.array([float(row["Centroid X"]), float(row["Centroid Y"]), float(row["Centroid Z"])]),
                            "min_pt": np.array([float(row["Min X"]), float(row["Min Y"]), float(row["Min Z"])]),
                            "max_pt": np.array([float(row["Max X"]), float(row["Max Y"]), float(row["Max Z"])]),
                            "spans": np.array([float(row["Span X"]), float(row["Span Y"]), float(row["Span Z"])]),
                        })
            for cid in self.instances_by_class:
                self.instances_by_class[cid].sort(key=lambda x: -x["triangle_count"])

    def _compute_triangle_labels(self) -> None:
        """Computes or recovers per-triangle class ID labels."""
        cache_file = self.mesh_path.parent / "triangle_class_labels.npy"
        if cache_file.is_file():
            print(f"[ClassInspector] Loading cached triangle labels from: {cache_file.name}...")
            self.tri_labels = np.load(cache_file)
            if len(self.tri_labels) == self.num_tris:
                return

        print("[ClassInspector] Mapping vertex colors to class labels across 2.7M triangles...")
        t0 = time.perf_counter()
        palette = np.array([get_class_color(c)[1] for c in self.active_classes], dtype=np.float32) * 255.0
        v_colors_uint8 = (self.orig_v_colors * 255.0).astype(np.uint8)
        tri_c = v_colors_uint8[self.tris].mean(axis=1)
        d_palette = np.linalg.norm(tri_c[:, None, :] - palette[None, :, :], axis=2)
        self.tri_labels = np.array(self.active_classes)[d_palette.argmin(axis=1)]
        print(f"  Mapped triangle labels in {time.perf_counter() - t0:.2f}s")
        try:
            np.save(cache_file, self.tri_labels)
        except Exception:
            pass

    @property
    def current_cid(self) -> int:
        return self.active_classes[self.class_idx]

    def _load_current_class_instances(self) -> None:
        """
        Extracts valid solid instances (>= 30 triangles) for the active class.
        Completely excludes stray boundary filaments, corner seam lines, and micro-floaters.
        """
        cid = self.current_cid
        self.active_instance_idx = -1

        c_mask = (self.tri_labels == cid)
        c_indices = np.flatnonzero(c_mask)
        if len(c_indices) == 0:
            self.class_instances = []
            self.valid_class_tris = np.array([], dtype=int)
            self.valid_class_v_indices = np.array([], dtype=int)
            return

        is_c = np.zeros(self.num_tris, dtype=bool)
        is_c[c_indices] = True
        c_edges = is_c[self.adj_u] & is_c[self.adj_v]
        compact_map = {t: i for i, t in enumerate(c_indices)}
        if np.any(c_edges):
            u_loc = np.array([compact_map[t] for t in self.adj_u[c_edges]], dtype=np.int32)
            v_loc = np.array([compact_map[t] for t in self.adj_v[c_edges]], dtype=np.int32)
            adj_mat = sp.csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
            n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
        else:
            n_comp = len(c_indices)
            comp_ids = np.arange(n_comp, dtype=np.int32)

        comp_sizes = np.bincount(comp_ids)
        # Filter out all micro-filaments, seam lines, and sub-scale crumbs (< 150 triangles)
        valid_comp_indices = [c for c in np.argsort(-comp_sizes) if comp_sizes[c] >= 150]

        # Merge proximate components within 8 cm (using exact cKDTree)
        if len(valid_comp_indices) > 1:
            comp_tris_list = [c_indices[comp_ids == c] for c in valid_comp_indices]
            tri_c = self.verts[self.tris].mean(axis=1)
            trees = [cKDTree(tri_c[t]) for t in comp_tris_list]
            n_c = len(valid_comp_indices)
            adj_m = np.zeros((n_c, n_c), dtype=bool)
            np.fill_diagonal(adj_m, True)
            for i in range(n_c):
                for j in range(i + 1, n_c):
                    m = trees[i].query_ball_tree(trees[j], r=0.08)
                    if any(len(match) > 0 for match in m):
                        adj_m[i, j] = True
                        adj_m[j, i] = True
            n_g, g_ids = csgraph.connected_components(sp.csr_matrix(adj_m), directed=False)
            merged_comps = []
            for g in range(n_g):
                members = [i for i in range(n_c) if g_ids[i] == g]
                merged_comps.append(np.concatenate([comp_tris_list[i] for i in members]))
            merged_comps.sort(key=len, reverse=True)
        elif len(valid_comp_indices) == 1:
            merged_comps = [c_indices[comp_ids == valid_comp_indices[0]]]
        else:
            merged_comps = []

        c_info = self.classifications.get(str(cid), self.classifications.get(cid, {}))
        c_role = c_info.get("structural_role", "object")

        self.class_instances = []
        for t_arr in merged_comps:
            inst_v = np.unique(self.tris[t_arr])
            t_verts = self.verts[inst_v]
            spans = t_verts.max(axis=0) - t_verts.min(axis=0) if len(t_verts) > 0 else np.zeros(3)
            tot_a = float(len(t_arr) * 0.00004) # approximate area
            # Enforce physical dimension checks (reject crumbs < 150 tris, < 0.015 m2, or < 0.15 m span)
            if len(t_arr) < 150 or tot_a < 0.015 or max(spans) < 0.15:
                continue

            inst_idx = len(self.class_instances)
            self.class_instances.append({
                "instance_id": f"inst_{cid}_{inst_idx}",
                "assigned_role": c_role,
                "triangle_indices": t_arr,
                "vertex_indices": inst_v,
                "triangle_count": len(t_arr),
                "area_m2": tot_a,
                "centroid": t_verts.mean(axis=0),
                "min_pt": t_verts.min(axis=0),
                "max_pt": t_verts.max(axis=0),
                "spans": spans,
            })

        if self.class_instances:
            self.valid_class_tris = np.concatenate([inst["triangle_indices"] for inst in self.class_instances])
            self.valid_class_v_indices = np.unique(self.tris[self.valid_class_tris])
        else:
            self.valid_class_tris = np.array([], dtype=int)
            self.valid_class_v_indices = np.array([], dtype=int)

    def _apply_colors(self, update_geometry_buffers: bool = True) -> None:
        """
        Updates geometry vertex colors based on the current mode and active class/instance.
        Completely eliminates boundary filaments and seam lines from the display.
        """
        cid = self.current_cid
        _, c_rgb = get_class_color(cid)
        c_rgb_arr = np.array(c_rgb, dtype=np.float32)

        active_triangles = self.tris[self.valid_class_tris] if len(self.valid_class_tris) > 0 else np.zeros((0, 3), dtype=int)
        active_v_indices = self.valid_class_v_indices

        if self.mode == "solo":
            # SOLO MODE: display ONLY valid solid class sub-mesh (no filaments, no room backdrop)
            if len(active_triangles) == 0:
                self.display_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                self.display_mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=int))
                return

            used_v_idx, new_tri_idx = np.unique(active_triangles, return_inverse=True)
            sub_verts = self.verts[used_v_idx]
            sub_tris = new_tri_idx.reshape(active_triangles.shape)
            sub_colors = np.tile(c_rgb_arr, (len(sub_verts), 1))

            # If specific instance selected in solo mode, highlight it
            if self.active_instance_idx >= 0 and self.active_instance_idx < len(self.class_instances):
                inst = self.class_instances[self.active_instance_idx]
                target_v_set = set(inst["vertex_indices"])
                inst_v_mask = np.array([v in target_v_set for v in used_v_idx], dtype=bool)
                sub_colors[~inst_v_mask] = sub_colors[~inst_v_mask] * 0.40
                sub_colors[inst_v_mask] = COLOR_INSTANCE_GOLD

            self.display_mesh.vertices = o3d.utility.Vector3dVector(sub_verts)
            self.display_mesh.triangles = o3d.utility.Vector3iVector(sub_tris)
            self.display_mesh.vertex_colors = o3d.utility.Vector3dVector(sub_colors)
            self.display_mesh.compute_vertex_normals()

        else:
            # Full mesh backdrop modes (SPOTLIGHT or RAINBOW)
            if self.mode == "rainbow":
                # Original multi-class colors
                new_colors = self.orig_v_colors.copy()
            else:
                # SPOTLIGHT (Default): room in neutral clay, ONLY valid instances colored in class color
                # All boundary filaments, seam lines, and non-instance bleeds remain in neutral clay!
                new_colors = np.tile(COLOR_CLAY, (self.num_verts, 1))
                if len(active_v_indices) > 0:
                    new_colors[active_v_indices] = c_rgb_arr

            # Instance highlighting in Spotlight / Rainbow
            if self.active_instance_idx >= 0 and self.active_instance_idx < len(self.class_instances):
                inst = self.class_instances[self.active_instance_idx]
                inst_v_indices = inst["vertex_indices"]
                new_colors[inst_v_indices] = COLOR_INSTANCE_GOLD

            # Re-assign full geometry if we were previously in Solo mode
            if len(self.display_mesh.vertices) != self.num_verts:
                self.display_mesh.vertices = self.full_mesh.vertices
                self.display_mesh.triangles = self.full_mesh.triangles
                self.display_mesh.vertex_normals = self.full_mesh.vertex_normals

            if self.mode == "spotlight":
                # Split display-only seams so shared vertices cannot tint other classes.
                inactive = np.ones(self.num_tris, dtype=bool)
                inactive[self.valid_class_tris] = False
                used_inactive = np.zeros(self.num_verts, dtype=bool)
                used_inactive[self.tris[inactive].ravel()] = True
                boundary = active_v_indices[used_inactive[active_v_indices]]
                if len(boundary):
                    remap = np.arange(self.num_verts, dtype=np.int32)
                    remap[boundary] = self.num_verts + np.arange(len(boundary))
                    display_tris = self.tris.copy()
                    display_tris[inactive] = remap[self.tris[inactive]]
                    normals = np.asarray(self.full_mesh.vertex_normals)
                    self.display_mesh.vertices = o3d.utility.Vector3dVector(np.vstack([self.verts, self.verts[boundary]]))
                    self.display_mesh.triangles = o3d.utility.Vector3iVector(display_tris)
                    self.display_mesh.vertex_normals = o3d.utility.Vector3dVector(np.vstack([normals, normals[boundary]]))
                    new_colors = np.vstack([new_colors, np.tile(COLOR_CLAY, (len(boundary), 1))])

            self.display_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    def print_hud(self) -> None:
        """Prints live metadata and keyboard guide to the terminal."""
        cid = self.current_cid
        c_info = self.classifications.get(str(cid), self.classifications.get(cid, {}))
        role = c_info.get("structural_role", "object")
        conf = c_info.get("confidence", 1.0)
        details = c_info.get("details", "")

        c_tri_mask = (self.tri_labels == cid)
        tri_count = int(c_tri_mask.sum())
        tri_pct = (tri_count / max(1, self.num_tris)) * 100.0

        n_inst = len(self.class_instances)
        active_tris = self.tris[c_tri_mask]
        active_v_idx = np.unique(active_tris) if len(active_tris) > 0 else np.array([], dtype=int)
        c_verts = self.verts[active_v_idx] if len(active_v_idx) > 0 else np.zeros((1, 3))

        centroid = c_verts.mean(axis=0)
        min_pt = c_verts.min(axis=0)
        max_pt = c_verts.max(axis=0)
        spans = max_pt - min_pt

        mode_desc = {
            "spotlight": "[SPOTLIGHT] Active Class Colored, Room in Soft Clay",
            "solo": "[SOLO] Only Active Class Visible (Room Hidden)",
            "rainbow": "[RAINBOW] Full Scene Semantic Palette",
        }.get(self.mode, self.mode.upper())

        print("\n" + "=" * 80)
        print(f"  [CLASS {self.class_idx + 1:02d} / {len(self.active_classes):02d}]  CID #{cid:03d} - {role.upper()} (Confidence: {conf:.2f})")
        print("-" * 80)
        print(f"  • Triangles      : {tri_count:,} ({tri_pct:.2f}% of full scene)")
        print(f"  • Instances      : {n_inst} connected components (Filtered >= 150 tris, >= 0.02 m²)")
        print(f"  • 3D Center      : X={centroid[0]:+.2f}m, Y={centroid[1]:+.2f}m, Z={centroid[2]:+.2f}m")
        print(f"  • Dimensions     : {spans[0]:.2f}m (W) x {spans[1]:.2f}m (H) x {spans[2]:.2f}m (D)")
        print(f"  • Mode           : {mode_desc}")
        if details:
            print(f"  • Detection Note : {details}")

        if self.active_instance_idx >= 0 and self.active_instance_idx < len(self.class_instances):
            inst = self.class_instances[self.active_instance_idx]
            print("-" * 80)
            print(f"  >>> HIGHLIGHTED INSTANCE [{self.active_instance_idx + 1} / {n_inst}]: {inst['instance_id']}")
            print(f"      Triangles : {inst['triangle_count']:,} ({inst['area_m2']:.3f} m²)")
            print(f"      Centroid  : X={inst['centroid'][0]:+.2f}m, Y={inst['centroid'][1]:+.2f}m, Z={inst['centroid'][2]:+.2f}m")
            print(f"      Spans     : {inst['spans'][0]:.2f}m x {inst['spans'][1]:.2f}m x {inst['spans'][2]:.2f}m")
            print(f"      Role      : {inst['assigned_role']}")
        elif n_inst > 0:
            print("-" * 80)
            print(f"  • All {n_inst} instances highlighted. Press [I] or [Up] to isolate an individual instance.")

        print("-" * 80)
        print("  Controls: [A/D or ←/→]: Class | [I or ↑/↓]: Instance | [S]: Mode | [F]: Focus | [H]: Help")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # Visualizer Callbacks
    # -------------------------------------------------------------------------
    def next_class(self, vis: o3d.visualization.Visualizer) -> bool:
        self.class_idx = (self.class_idx + 1) % len(self.active_classes)
        self._load_current_class_instances()
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def prev_class(self, vis: o3d.visualization.Visualizer) -> bool:
        self.class_idx = (self.class_idx - 1) % len(self.active_classes)
        self._load_current_class_instances()
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def toggle_mode(self, vis: o3d.visualization.Visualizer) -> bool:
        modes = ["spotlight", "solo", "rainbow"]
        curr_i = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(curr_i + 1) % len(modes)]
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def next_instance(self, vis: o3d.visualization.Visualizer) -> bool:
        if not self.class_instances:
            print(f"[ClassInspector] Class #{self.current_cid} has no multi-instances to cycle.")
            return False
        self.active_instance_idx = (self.active_instance_idx + 1)
        if self.active_instance_idx >= len(self.class_instances):
            self.active_instance_idx = -1  # Loop back to all instances
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def prev_instance(self, vis: o3d.visualization.Visualizer) -> bool:
        if not self.class_instances:
            return False
        if self.active_instance_idx == -1:
            self.active_instance_idx = len(self.class_instances) - 1
        else:
            self.active_instance_idx -= 1
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def clear_instance(self, vis: o3d.visualization.Visualizer) -> bool:
        self.active_instance_idx = -1
        self._apply_colors()
        vis.update_geometry(self.display_mesh)
        vis.update_renderer()
        self.print_hud()
        return False

    def focus_camera(self, vis: o3d.visualization.Visualizer) -> bool:
        """Centers camera lookat on active class or instance."""
        ctr = vis.get_view_control()
        if self.active_instance_idx >= 0 and self.active_instance_idx < len(self.class_instances):
            target = self.class_instances[self.active_instance_idx]["centroid"]
            print(f"[ClassInspector] Focusing camera on instance centroid: {target}")
        else:
            cid = self.current_cid
            c_tri_mask = (self.tri_labels == cid)
            active_tris = self.tris[c_tri_mask]
            active_v_idx = np.unique(active_tris) if len(active_tris) > 0 else np.array([], dtype=int)
            if len(active_v_idx) > 0:
                target = self.verts[active_v_idx].mean(axis=0)
            else:
                target = np.zeros(3)
            print(f"[ClassInspector] Focusing camera on Class #{cid} centroid: {target}")

        ctr.set_lookat(target.reshape(3, 1))
        vis.update_renderer()
        return False

    def print_help(self, vis: o3d.visualization.Visualizer) -> bool:
        self.print_hud()
        return False


def run_inspector(
    mesh_path: Path,
    structural_json_path: Path,
    instances_csv_path: Optional[Path] = None,
    per_class_dir: Optional[Path] = None,
    initial_cid: Optional[int] = None,
    initial_mode: str = "spotlight",
    headless: bool = False,
) -> None:
    inspector = ClassInspector(
        mesh_path=mesh_path,
        structural_json_path=structural_json_path,
        instances_csv_path=instances_csv_path,
        per_class_dir=per_class_dir,
        initial_cid=initial_cid,
        initial_mode=initial_mode,
    )

    if headless:
        print("[ClassInspector] Running headless verification test...")
        inspector.print_hud()
        # Test class cycle
        inspector.class_idx = 1
        inspector._load_current_class_instances()
        inspector._apply_colors()
        # Test instance cycle
        inspector.active_instance_idx = 0
        inspector._apply_colors()
        # Test solo mode
        inspector.mode = "solo"
        inspector._apply_colors()
        print("[ClassInspector] Headless verification passed successfully!")
        return

    # Create visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_created = vis.create_window(
        window_name=f"3D Semantic Class & Instance Inspector — {mesh_path.parent.name}",
        width=1440,
        height=900,
        visible=True,
    )
    if not window_created:
        print("[ClassInspector ERROR] Failed to create Open3D window. Please verify DISPLAY environment.")
        return

    vis.add_geometry(inspector.display_mesh)

    # Set pleasant visual render options
    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([0.10, 0.10, 0.12]) # Deep dark slate
        opt.light_on = True
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    # Register Key Callbacks
    # GLFW Key codes:
    # Right: 262, Left: 263, Up: 265, Down: 264, ESC: 256
    vis.register_key_callback(ord("D"), inspector.next_class)
    vis.register_key_callback(ord("d"), inspector.next_class)
    vis.register_key_callback(262, inspector.next_class) # Right arrow

    vis.register_key_callback(ord("A"), inspector.prev_class)
    vis.register_key_callback(ord("a"), inspector.prev_class)
    vis.register_key_callback(263, inspector.prev_class) # Left arrow

    vis.register_key_callback(ord("S"), inspector.toggle_mode)
    vis.register_key_callback(ord("s"), inspector.toggle_mode)

    vis.register_key_callback(ord("I"), inspector.next_instance)
    vis.register_key_callback(ord("i"), inspector.next_instance)
    vis.register_key_callback(265, inspector.next_instance) # Up arrow

    vis.register_key_callback(264, inspector.prev_instance) # Down arrow

    vis.register_key_callback(ord("0"), inspector.clear_instance)

    vis.register_key_callback(ord("F"), inspector.focus_camera)
    vis.register_key_callback(ord("f"), inspector.focus_camera)

    vis.register_key_callback(ord("H"), inspector.print_help)
    vis.register_key_callback(ord("h"), inspector.print_help)

    inspector.print_hud()
    print("\n[ClassInspector] Starting 3D window. Click inside window to navigate. Press [H] for help.")
    vis.run()
    vis.destroy_window()


def _resolve_path(p: Path) -> Path:
    if p.exists():
        return p
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    backend_root = script_dir.parent
    if (repo_root / p).exists():
        return repo_root / p
    if (backend_root / p).exists():
        return backend_root / p
    stripped = Path(*p.parts[1:]) if p.parts and p.parts[0] == "backend" else p
    if (backend_root / stripped).exists():
        return backend_root / stripped
    return p


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Class & Instance Inspector")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_refined.ply"),
        help="Path to input 3D mesh PLY",
    )
    parser.add_argument(
        "--structural-json",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/structural_classes.json"),
        help="Path to structural_classes.json",
    )
    parser.add_argument(
        "--instances-csv",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/instances_summary.csv"),
        help="Path to instances_summary.csv",
    )
    parser.add_argument(
        "--per-class-dir",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/per_class_meshes"),
        help="Path to per_class_meshes directory",
    )
    parser.add_argument("--class-id", type=int, default=None, help="Initial class ID to display")
    parser.add_argument("--mode", type=str, default="spotlight", choices=["spotlight", "solo", "rainbow"])
    parser.add_argument("--headless", action="store_true", help="Run automated test without opening window")
    args = parser.parse_args()

    run_inspector(
        mesh_path=_resolve_path(args.mesh),
        structural_json_path=_resolve_path(args.structural_json),
        instances_csv_path=_resolve_path(args.instances_csv),
        per_class_dir=_resolve_path(args.per_class_dir),
        initial_cid=args.class_id,
        initial_mode=args.mode,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
