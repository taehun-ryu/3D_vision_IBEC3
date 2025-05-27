#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import argparse
import torch
from corner import *
from viewer import run_calibration_gui

def compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)

        imgp_gt = imgpoints[i].reshape(-1, 2).astype(np.float32)
        imgp_proj = imgpoints_proj.reshape(-1, 2).astype(np.float32)

        error = cv2.norm(imgp_gt, imgp_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgp_gt)

    mean_error = np.sqrt(total_error / total_points)
    return mean_error
    
def visualize_detected_points(iwe, imgp):
    vis = cv2.cvtColor(iwe, cv2.COLOR_GRAY2BGR)
    for pt in imgp.reshape(-1, 2):
        pt = tuple(np.round(pt).astype(int))
        cv2.circle(vis, pt, 3, (0, 255, 0), -1)
    cv2.imshow("Detected Corners", vis)
    cv2.waitKey(1000)

def visualize_reprojection(iwe, imgp_gt, imgp_proj):
    vis = cv2.cvtColor(iwe, cv2.COLOR_GRAY2BGR)
    for pt_gt, pt_proj in zip(imgp_gt, imgp_proj):
        pt_gt = tuple(np.round(pt_gt).astype(int))
        pt_proj = tuple(np.round(pt_proj).astype(int))
        cv2.circle(vis, pt_gt, 3, (0, 255, 0), -1)
        cv2.circle(vis, pt_proj, 3, (0, 0, 255), -1)
        cv2.line(vis, pt_gt, pt_proj, (255, 255, 255), 1)
    cv2.imshow("Reprojection", vis)
    cv2.waitKey(0)
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_dir", help="Folder of IWE images")
    p.add_argument("--board_w", type=int, default=4)
    p.add_argument("--vis", action="store_true", help="Visualize calibration results")
    p.add_argument("--board_h", type=int, default=4)
    p.add_argument("--square_size", type=float, default=4.0)
    args = p.parse_args()

    print(f"[INFO] Starting calibration with board size {args.board_w}x{args.board_h}, square size = {args.square_size} mm")

    # prepare 3D object points
    objp = np.zeros((args.board_h * args.board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_w, 0:args.board_h].T.reshape(-1, 2) * args.square_size

    objpoints, imgpoints = [], []
    image_size = None

    files = sorted(glob.glob(os.path.join(args.image_dir, "*.*")))
    print(f"[INFO] Found {len(files)} files in {args.image_dir}")

    valid_count = 0
    valid_iwes = []
    for i, fname in enumerate(files):
        iwe = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if iwe is None:
            print(f"[WARNING] Failed to read image: {fname}")
            continue
        if iwe.ndim == 3:
            iwe = cv2.cvtColor(iwe, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            h, w = iwe.shape
            image_size = (w, h)
            print(f"[INFO] Image size set to {image_size}")

        print(f"[INFO] Processing {os.path.basename(fname)} ({i+1}/{len(files)})")
        imgp = detect_corners(iwe, args.board_w, args.board_h)
        if imgp is None:
            print(f"[WARNING] Corner detection failed for {fname}")
            continue
        if check_grid_validity(imgp):
            imgpoints.append(imgp.reshape(-1, 2).astype(np.float32))
            objpoints.append(objp.copy())
            valid_count += 1
            print(f"[OK] Corners validated and added (Total valid: {valid_count})")
            imgp_flat = imgp.reshape(-1, 2).astype(np.float32)
            visualize_detected_points(iwe, imgp_flat)
            valid_iwes.append(iwe)
        else:
            print(f"[WARNING] Grid structure invalid, skipping {fname}")
    cv2.destroyAllWindows()
    print(f"[SUMMARY] {valid_count} / {len(files)} images passed corner validation.")

    if len(imgpoints) == 0:
        print("[ERROR] No valid images found. Calibration aborted.")
        return

    print("[INFO] Performing camera calibration...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if ret:
        print("[SUCCESS] Calibration completed.")
        print("Camera matrix:\n", K)
        print("Distortion coefficients:\n", dist.ravel())
    else:
        print("[ERROR] Calibration failed.")
        
    reprojection_error = compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)
    print(f"[INFO] Reprojection error: {reprojection_error:.4f} pixels")
    if args.vis:
        print("[INFO] Visualizing calibration results...")
        for i, iwe in enumerate(valid_iwes):
            imgp_gt = imgpoints[i].reshape(-1, 2).astype(np.float32)
            imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            imgp_proj = imgpoints_proj.reshape(-1, 2).astype(np.float32)

            visualize_reprojection(iwe, imgp_gt, imgp_proj)
        cv2.destroyAllWindows()
        run_calibration_gui(K, dist, rvecs, tvecs, objpoints, imgpoints, image_size)

if __name__ == "__main__":
    main()
