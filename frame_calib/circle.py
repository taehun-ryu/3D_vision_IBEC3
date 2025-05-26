#!/usr/bin/env python3
import cv2
import glob
import numpy as np
import os
import argparse

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
    parser.add_argument("frame_dir", help="Directory containing circle grid images")
    parser.add_argument("--board_w", type=int, default=4)
    parser.add_argument("--board_h", type=int, default=11)
    parser.add_argument("--circle_size", type=float, default=1.0)
    args = parser.parse_args()

    pattern_size = (args.board_w, args.board_h)
    circle_size = args.circle_size

    # asymmetric circle grid: row is board_h, col is board_w
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp[:, 0] *= circle_size
    objp[:, 1] *= circle_size

    objpoints, imgpoints = [], []
    image_size = None

    image_paths = sorted(glob.glob(os.path.join(args.frame_dir, "*.*")))
    print(f"[INFO] Found {len(image_paths)} images")

    for i, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARNING] Failed to load image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        print(f"[INFO] Processing {fname} ({i+1}/{len(image_paths)})")
        ret, centers = cv2.findCirclesGrid(
            gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )

        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(centers)

            vis = cv2.drawChessboardCorners(img, pattern_size, centers, ret)
            cv2.imshow("Detected Circles", vis)
            cv2.waitKey(100)
        else:
            print(f"[WARNING] Circles grid not detected in {fname}")

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("[ERROR] No valid images found.")
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

if __name__ == "__main__":
    main()
