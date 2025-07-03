#!/usr/bin/env python3
import cv2
import numpy as np

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

def undistort_iwe(iwe, K, dist):
    h, w = iwe.shape
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = cv2.undistort(iwe, K, dist, None, new_K)
    return undistorted


