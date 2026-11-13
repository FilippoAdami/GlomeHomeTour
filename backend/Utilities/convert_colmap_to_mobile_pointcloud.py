#!/usr/bin/env python3
"""
Convert backend COLMAP point clouds into binary PLY point clouds compatible
with the mobile app's ScanPointCloudAccumulator and PointCloudRenderer.

Features:
1. Filters bounding box outliers based on reconstructed mesh extents (+0.35m margin).
2. Spatially downsamples points using a 3.5cm voxel grid matching mobile voxel size.
3. Computes 3D rigid Umeyama alignment (s, R, t) to register COLMAP points into the
   phone scan's native ARCore trajectory coordinate system.
4. Serializes standard binary little-endian PLY files (float x, y, z).
5. Injects the converted point clouds into the respective phone scan archives and
   app storage directories via ADB.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
import numpy as np
import trimesh

ADB_PATH = "/home/monday/android-sdk/platform-tools/adb"
PHONE_DOCUMENTS = "/sdcard/Documents/GlomeHomeTour"
PHONE_APP_FILES = "/storage/emulated/0/Android/data/com.glomehometour.arscan/files"

def write_binary_ply(filepath: str, points: np.ndarray):
    """Writes (N, 3) float array to standard binary little-endian PLY."""
    pts = np.ascontiguousarray(points, dtype="<f4")
    count = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pts.tobytes())
    print(f"  Wrote {count} points to {filepath} ({os.path.getsize(filepath):,} bytes)")

def parse_colmap_camera_centers(images_txt_path: str):
    """Parses COLMAP image poses and computes camera centers in world coordinates."""
    centers = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 10 and parts[-1].endswith(".jpg"):
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                # Rotation matrix from quaternion (world to camera)
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])
                t = np.array([tx, ty, tz])
                # Camera center in world space is -R^T * t
                center = -R.T @ t
                fname = os.path.basename(parts[-1])
                centers[fname] = center
    return centers

def compute_umeyama_alignment(colmap_centers: dict, phone_transforms_json: dict):
    """Computes similarity transform (scale, R, t) from COLMAP to ARCore phone frame."""
    arcore_dict = {}
    for f in phone_transforms_json["frames"]:
        fname = os.path.basename(f["file_path"])
        mat = np.array(f["transform_matrix"])
        arcore_dict[fname] = mat[:3, 3]

    common_keys = [k for k in colmap_centers if k in arcore_dict]
    if len(common_keys) < 3:
        raise ValueError(f"Too few matching frames: {len(common_keys)}")

    X = np.array([colmap_centers[k] for k in common_keys])
    Y = np.array([arcore_dict[k] for k in common_keys])

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my
    Sx = np.mean(np.sum(Xc**2, axis=1))
    H = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (1.0 / Sx) * np.sum(D * np.diag(S))
    t = my - s * R @ mx

    X_aligned = (s * (R @ X.T)).T + t
    err = np.linalg.norm(X_aligned - Y, axis=1)
    print(f"  Alignment: {len(common_keys)} frames, scale={s:.4f}, mean err={err.mean():.4f}m, median err={np.median(err):.4f}m")
    return s, R, t

def filter_and_voxelize(points: np.ndarray, mesh_path: str, voxel_size_m: float = 0.035):
    """Removes bbox outliers beyond mesh + 0.35m, then downsamples using a spatial voxel grid."""
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        m_min = mesh.vertices.min(axis=0) - 0.35
        m_max = mesh.vertices.max(axis=0) + 0.35
        in_box = np.all((points >= m_min) & (points <= m_max), axis=1)
        points = points[in_box]

    coords = np.floor(points / voxel_size_m).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx]

def inject_file_into_zip(zip_path: str, entry_name: str, file_bytes: bytes):
    """Replaces or adds an entry into a zip archive atomically."""
    temp_zip = zip_path + ".tmp"
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename != entry_name:
                    zout.writestr(item, zin.read(item.filename))
            zout.writestr(entry_name, file_bytes)
    os.replace(temp_zip, zip_path)

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    current_scene = os.path.join(repo_root, "backend", "current_scene")
    work_dir = tempfile.mkdtemp(prefix="colmap_pointclouds_")

    scans_meta = [
        {
            "session_name": "scan_20261024_164826",
            "backend_dir": "scan_0",
            "label": "bedroom",
        },
        {
            "session_name": "scan_20261024_234617",
            "backend_dir": "scan_1",
            "label": "hallway_1",
        },
        {
            "session_name": "scan_20261024_235426",
            "backend_dir": "scan_2",
            "label": "hallway_2",
        },
    ]

    print("=== Step 1: Processing COLMAP Point Clouds for Individual Rooms ===")
    room_ply_files = {}
    aligned_building_points = []

    for item in scans_meta:
        session = item["session_name"]
        bdir = os.path.join(current_scene, item["backend_dir"])
        colmap_ply = os.path.join(bdir, "sparse", "0", "points3D.ply")
        images_txt = os.path.join(bdir, "sparse", "0", "images.txt")
        mesh_path = os.path.join(bdir, "mesh.ply")

        print(f"\nProcessing {session} ({item['label']}, backend {item['backend_dir']})...")
        if not os.path.exists(colmap_ply):
            print(f"  ERROR: {colmap_ply} not found!")
            continue

        # Load raw COLMAP points
        raw_cloud = trimesh.load(colmap_ply)
        pts_colmap = np.array(raw_cloud.vertices, dtype=np.float32)
        print(f"  Loaded {len(pts_colmap):,} raw COLMAP points")

        # Clean bbox outliers and voxel downsample
        pts_voxel = filter_and_voxelize(pts_colmap, mesh_path, voxel_size_m=0.035)
        print(f"  Filtered to {len(pts_voxel):,} voxels at 3.5cm grid")
        aligned_building_points.append(pts_voxel)

        # Pull transforms.json from phone zip to compute Umeyama alignment
        cmd = [ADB_PATH, "shell", f"unzip -p {PHONE_DOCUMENTS}/{session}.zip scan_0/transforms.json"]
        out = subprocess.check_output(cmd)
        phone_tr = json.loads(out)

        colmap_centers = parse_colmap_camera_centers(images_txt)
        s, R, t = compute_umeyama_alignment(colmap_centers, phone_tr)

        # Transform to phone scan ARCore frame
        pts_phone = (s * (R @ pts_voxel.T)).T + t

        out_ply = os.path.join(work_dir, f"{session}.ply")
        write_binary_ply(out_ply, pts_phone)
        room_ply_files[session] = (out_ply, item["backend_dir"])

    print("\n=== Step 2: Processing Multi-Room Combined Building Cloud ===")
    merged_pts = np.vstack(aligned_building_points)
    # Voxel downsample combined cloud to 3.5cm
    coords_m = np.floor(merged_pts / 0.035).astype(np.int64)
    _, u_idx = np.unique(coords_m, axis=0, return_index=True)
    merged_pts = merged_pts[u_idx]
    building_ply = os.path.join(work_dir, "Via_Bassanese_313_A_building.ply")
    write_binary_ply(building_ply, merged_pts)

    print("\n=== Step 3: Injecting Point Clouds into Phone Scan Zips ===")
    for session, (ply_path, scan_id) in room_ply_files.items():
        phone_zip_remote = f"{PHONE_DOCUMENTS}/{session}.zip"
        local_zip = os.path.join(work_dir, f"{session}.zip")

        print(f"Pulling {phone_zip_remote} from phone...")
        subprocess.check_call([ADB_PATH, "pull", phone_zip_remote, local_zip])

        with open(ply_path, "rb") as f:
            ply_data = f.read()

        print(f"Injecting pointcloud.ply into {session}.zip...")
        inject_file_into_zip(local_zip, "scan_0/pointcloud.ply", ply_data)
        inject_file_into_zip(local_zip, "pointcloud.ply", ply_data)

        print(f"Pushing updated {session}.zip to phone...")
        subprocess.check_call([ADB_PATH, "push", local_zip, phone_zip_remote])

        # Also push to external files fallback directory on phone
        remote_app_scan_dir = f"{PHONE_APP_FILES}/{session}"
        remote_app_scan_sub = f"{PHONE_APP_FILES}/{session}/scan_0"
        subprocess.check_call([ADB_PATH, "shell", f"mkdir -p {remote_app_scan_sub}"])
        subprocess.check_call([ADB_PATH, "push", ply_path, f"{remote_app_scan_dir}/pointcloud.ply"])
        subprocess.check_call([ADB_PATH, "push", ply_path, f"{remote_app_scan_sub}/pointcloud.ply"])
        print(f"  Updated phone app files at {remote_app_scan_dir}")

    # Now handle Via_Bassanese_313_A_scan.zip
    via_zip_name = "Via_Bassanese_313_A_scan.zip"
    via_remote = f"{PHONE_DOCUMENTS}/{via_zip_name}"
    via_local = os.path.join(work_dir, via_zip_name)
    print(f"\nPulling {via_remote} from phone...")
    subprocess.check_call([ADB_PATH, "pull", via_remote, via_local])

    # Add each room's point cloud
    for session, (ply_path, bdir) in room_ply_files.items():
        with open(ply_path, "rb") as f:
            ply_data = f.read()
        print(f"Injecting {bdir}/pointcloud.ply into {via_zip_name}...")
        inject_file_into_zip(via_local, f"{bdir}/pointcloud.ply", ply_data)

    # Add root building multi-room point cloud
    with open(building_ply, "rb") as f:
        b_data = f.read()
    print(f"Injecting root pointcloud.ply (building) into {via_zip_name}...")
    inject_file_into_zip(via_local, "pointcloud.ply", b_data)

    print(f"Pushing updated {via_zip_name} to phone...")
    subprocess.check_call([ADB_PATH, "push", via_local, via_remote])

    # Push to fallback directory for Via_Bassanese_313_A_scan
    via_app_dir = f"{PHONE_APP_FILES}/Via_Bassanese_313_A_scan"
    subprocess.check_call([ADB_PATH, "shell", f"mkdir -p {via_app_dir}"])
    subprocess.check_call([ADB_PATH, "push", building_ply, f"{via_app_dir}/pointcloud.ply"])
    for session, (ply_path, bdir) in room_ply_files.items():
        subprocess.check_call([ADB_PATH, "shell", f"mkdir -p {via_app_dir}/{bdir}"])
        subprocess.check_call([ADB_PATH, "push", ply_path, f"{via_app_dir}/{bdir}/pointcloud.ply"])

    print("\nCleaning up temp directory...")
    shutil.rmtree(work_dir, ignore_errors=True)
    print("\n✓ All COLMAP point clouds successfully aligned, converted, and installed onto the phone!")

if __name__ == "__main__":
    main()
