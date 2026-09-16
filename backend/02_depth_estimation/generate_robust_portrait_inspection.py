import os
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen"

import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, "backend")
sys.path.insert(0, "backend/Utilities/third_party/depth_anything_3/src")
from Utilities.pipeline_paths import bootstrap
bootstrap()

from package_loader import PackageLoader
from quality_gate import QualityGate
from pose_aligner import PoseAligner
from depth_anything_3.api import DepthAnything3

def main():
    print("=== Step 1: Loading and synchronizing acquisition ===")
    zip_path = "backend/scenes/bedroom_complete.zip"
    loader = PackageLoader()
    package = loader.load(zip_path)
    
    gate = QualityGate()
    gate_result = gate.evaluate(package.keyframes)
    print(f"Accepted keyframes: {len(gate_result.accepted_keyframes)} / {len(package.keyframes)}")
    
    aligner = PoseAligner(package.trajectory)
    synced = aligner.synchronize_keyframes(gate_result.accepted_keyframes)
    print(f"Synchronized keyframes: {len(synced)}")

    print("\n=== Step 2: Running Portrait-Aware Depth-Adaptive SIFT + RANSAC Selection ===")
    # Smartphone Vertical Portrait Optical Parameters (1080 x 1920)
    # fx ~ 1450 on 1080x1920 sensor
    HFOV_DEG = 2.0 * np.degrees(np.arctan(1080.0 / (2.0 * 1450.0))) # 40.8 deg (narrow!)
    VFOV_DEG = 2.0 * np.degrees(np.arctan(1920.0 / (2.0 * 1450.0))) # 67.0 deg (tall)
    print(f"Portrait Optics: HFOV={HFOV_DEG:.1f}° (horizontal), VFOV={VFOV_DEG:.1f}° (vertical)")

    sift = cv2.SIFT_create(nfeatures=600)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract_features(kf):
        img_raw = kf.load_image_rgb()
        img_upright = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(img_upright, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (270, 480))
        return sift.detectAndCompute(small, None)

    def compute_inliers(des1, des2, kp1, kp2):
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0, 0, 2.0
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return len(good), len(good), 2.0
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(p1.reshape(-1, 1, 2), p2.reshape(-1, 1, 2), cv2.FM_RANSAC, 3.0)
        inl = int(mask.sum()) if mask is not None else 0
        
        if mask is not None and inl >= 6:
            inlier_p1 = p1[mask.ravel() == 1]
            inlier_p2 = p2[mask.ravel() == 1]
            disp = float(np.median(np.linalg.norm(inlier_p2 - inlier_p1, axis=1)))
        else:
            disp = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
            
        return inl, len(good), disp

    def compute_portrait_rotation(r1, r2):
        r_rel = np.dot(r2, r1.T)
        # Camera axes: X=right, Y=down, Z=forward
        yaw = np.degrees(np.arctan2(r_rel[0, 2], r_rel[2, 2]))      # horizontal pan
        pitch = np.degrees(np.arctan2(-r_rel[1, 2], np.sqrt(r_rel[1, 0]**2 + r_rel[1, 1]**2))) # vertical tilt
        roll = np.degrees(np.arctan2(r_rel[1, 0], r_rel[1, 1]))     # roll
        
        # Horizontal pan consumes narrow HFOV (40.8°) ~1.64x faster than pitch consumes VFOV (67°)
        norm_rot = np.sqrt((yaw / HFOV_DEG)**2 + (pitch / VFOV_DEG)**2 + (roll / HFOV_DEG)**2) * HFOV_DEG
        total_rot = np.degrees(np.arccos(np.clip((np.trace(r_rel) - 1.0)/2.0, -1.0, 1.0)))
        return float(norm_rot), float(total_rot)

    selected_indices = [0]
    c2w_last = synced[0].transform_matrix
    kp_last, des_last = extract_features(synced[0])
    last_idx = 0
    current_Z = 2.5  # Reference depth
    frame_depths = [2.5]

    while last_idx < len(synced) - 1:
        chosen = None
        # Perfected Dynamic Depth Scaling:
        # Near objects (Z ~ 1m): d_max ~ 0.65m, rot_max ~ 16° (dense & tight to prevent shearing)
        # Far scene   (Z ~ 3m): d_max ~ 1.55m, rot_max ~ 30° (wide steps across open room)
        z_ratio = np.clip(current_Z / 2.0, 0.5, 2.5)
        d_max = float(np.clip(1.20 * (z_ratio ** 0.75), 0.65, 1.65))
        d_target = float(d_max * 0.75)
        rot_max = float(np.clip(24.0 * (z_ratio ** 0.50), 16.0, 33.0))
        rot_target = float(rot_max * 0.75)
        
        for curr_idx in range(last_idx + 1, min(last_idx + 55, len(synced))):
            c2w = synced[curr_idx].transform_matrix
            d = float(np.linalg.norm(c2w[:3, 3] - c2w_last[:3, 3]))
            norm_rot, total_rot = compute_portrait_rotation(c2w_last[:3, :3], c2w[:3, :3])
            
            # Hard limit: if we exceed rot_max or d_max, lock in previous frame
            if norm_rot >= rot_max or total_rot >= 34.0 or d >= d_max:
                chosen = max(last_idx + 1, curr_idx - 1)
                break
                
            if d >= d_target or norm_rot >= rot_target:
                kp_c, des_c = extract_features(synced[curr_idx])
                inl, good, disp = compute_inliers(des_last, des_c, kp_last, kp_c)
                
                # Update rolling scene depth Z
                if disp > 2.0 and d > 0.08:
                    z_est = (360.0 * d) / disp
                    current_Z = float(np.clip(0.7 * current_Z + 0.3 * z_est, 0.7, 5.0))
                    
                if good >= 10 and inl >= 8:
                    chosen = curr_idx
                    if d >= d_max * 0.90 or norm_rot >= rot_max * 0.90:
                        break
                elif good < 10:
                    chosen = curr_idx
                    break
                    
        if chosen is None or chosen <= last_idx:
            chosen = min(last_idx + 1, len(synced) - 1)
            
        selected_indices.append(chosen)
        frame_depths.append(current_Z)
        c2w_last = synced[chosen].transform_matrix
        kp_last, des_last = extract_features(synced[chosen])
        last_idx = chosen

    print(f"Total keyframes selected: {len(selected_indices)}")
    
    # Verify sequence stats
    dists = [float(np.linalg.norm(synced[selected_indices[k]].transform_matrix[:3, 3] - synced[selected_indices[k-1]].transform_matrix[:3, 3])) for k in range(1, len(selected_indices))]
    rots = []
    for k in range(1, len(selected_indices)):
        r_rel = np.dot(synced[selected_indices[k]].transform_matrix[:3, :3], synced[selected_indices[k-1]].transform_matrix[:3, :3].T)
        rots.append(float(np.degrees(np.arccos(np.clip((np.trace(r_rel) - 1.0)/2.0, -1.0, 1.0)))))
        
    print(f"Max rotation between ANY consecutive pair: {max(rots):.1f}° (Hard Cap < 35.0°)")
    print(f"Max distance between ANY consecutive pair: {max(dists):.2f}m")
    print(f"Median rotation: {np.median(rots):.1f}°, Median distance: {np.median(dists):.2f}m")

    # Clear inspection directory
    out_dir = "backend/scenes/bedroom_portrait_inspection"
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".png"):
            try:
                os.remove(os.path.join(out_dir, f))
            except Exception:
                pass
    print(f"Cleaned output directory: {out_dir}")

    print("\n=== Step 3: Loading Depth Anything 3 Model on GPU ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device)
    model.eval()

    print(f"\n=== Step 4: Generating {len(selected_indices)} Side-by-Side Upright Portrait Frames ===")

    for k in range(len(selected_indices)):
        idx = selected_indices[k]
        kf = synced[idx]
        
        # Load raw image and rotate 90 degrees clockwise to upright portrait
        img_raw = kf.load_image_rgb()
        img_upright = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
        
        # Step metrics relative to previous frame
        if k > 0:
            step_d = dists[k-1]
            step_rot = rots[k-1]
        else:
            step_d, step_rot = 0.0, 0.0
            
        z_curr = frame_depths[k]
            
        # Run depth prediction with DA3
        with torch.no_grad():
            pred = model.inference([img_upright])
            depth_map = pred.depth[0]
            
        # Depth colormap using OpenCV inferno
        d_min, d_max = np.percentile(depth_map, 2), np.percentile(depth_map, 98)
        d_norm = np.clip((depth_map - d_min) / max(d_max - d_min, 1e-4), 0.0, 1.0)
        d_u8 = (d_norm * 255.0).astype(np.uint8)
        d_colored_bgr = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
        
        # Resize both for clean inspection display: (height=960, width=540)
        disp_rgb_bgr = cv2.resize(cv2.cvtColor(img_upright, cv2.COLOR_RGB2BGR), (540, 960))
        disp_depth_bgr = cv2.resize(d_colored_bgr, (540, 960))
        
        # Header banner
        header = np.zeros((80, 540 * 2 + 20, 3), dtype=np.uint8)
        header[:] = 30
        title_text = f"Frame {k:03d} / {len(selected_indices)-1} [{os.path.basename(kf.file_path)}]"
        info_text = f"Step dist: {step_d:.2f}m | Step rot: {step_rot:.1f}° | Scene Z: {z_curr:.1f}m | Depth: [{d_min:.2f}m - {d_max:.2f}m]"
        cv2.putText(header, title_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(header, info_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
        
        body = np.zeros((960, 540 * 2 + 20, 3), dtype=np.uint8)
        body[:, :540] = disp_rgb_bgr
        body[:, 540:560] = 50
        body[:, 560:] = disp_depth_bgr
        
        combined = np.vstack([header, body])
        out_path = os.path.join(out_dir, f"frame_{k:03d}_side_by_side.png")
        cv2.imwrite(out_path, combined)
        
        if (k + 1) % 20 == 0 or (k + 1) == len(selected_indices):
            print(f"Rendered [{k+1:03d}/{len(selected_indices)}] frames -> {out_path}")
            
        # Compositor yield for GPU safety
        time.sleep(0.18)
        if k % 25 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll {len(selected_indices)} frames successfully generated in: {out_dir}")

if __name__ == "__main__":
    main()
