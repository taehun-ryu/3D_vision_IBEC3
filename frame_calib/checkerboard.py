#!/usr/bin/env python3
import cv2
import glob
import numpy as np
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viewer import run_calibration_gui

def compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgpoints[i])

    mean_error = np.sqrt(total_error / total_points)
    print(f"[INFO] Mean Reprojection Error: {mean_error:.4f} pixels")
    return mean_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", help="Directory containing checkerboard images")
    parser.add_argument("--board_w", type=int, default=4)
    parser.add_argument("--board_h", type=int, default=4)
    parser.add_argument("--square_size", type=float, default=4.0)
    args = parser.parse_args()

    board_size = (args.board_w, args.board_h)
    square_size = args.square_size

    objp = np.zeros((args.board_h * args.board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_w, 0:args.board_h].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points in world space
    imgpoints = []  # 2D points in image plane
    image_size = None

    image_paths = sorted(glob.glob(os.path.join(args.frame_dir, "*.*")))
    print(f"[INFO] Found {len(image_paths)} images")
    if len(image_paths) > 50:
        print(f"[INFO] Too many images ({len(image_paths)}), randomly selecting 50 for calibration")
        np.random.seed(42)  # 재현성 위해 고정
        image_paths = sorted(np.random.choice(image_paths, size=50, replace=False).tolist())

    for i, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARNING] Cannot load image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        print(f"[INFO] Processing {fname} ({i+1}/{len(image_paths)})")

        ret, corners = cv2.findChessboardCorners(gray, board_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            corners_subpix = cv2.cornerSubPix(
                gray, corners, winSize=(11,11), zeroZone=(-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            imgpoints.append(corners_subpix)
            objpoints.append(objp.copy())

            vis = cv2.drawChessboardCorners(img, board_size, corners_subpix, ret)
            cv2.imshow("Corners", vis)
            cv2.waitKey(100)
        else:
            print(f"[WARNING] Chessboard not found in {fname}")

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("[ERROR] No valid calibration images found.")
        return

    print("[INFO] Performing calibration...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if ret:
        print("[SUCCESS] Calibration completed")
        print("Camera Matrix:\n", K)
        print("Distortion Coefficients:\n", dist.ravel())
    else:
        print("[ERROR] Calibration failed.")

    reprojection_error = compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)
    print(f"[INFO] Reprojection error: {reprojection_error:.4f} pixels")
    run_calibration_gui(K, dist, rvecs, tvecs, objpoints, imgpoints, image_size)

if __name__ == "__main__":
    main()
