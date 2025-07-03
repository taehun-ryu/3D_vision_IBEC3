import numpy as np
import cv2
from src.calib.corner import get_valid_corners
from src.calib import eval as eval
from src.cmax.iwe import get_valid_iwes
from utils.event_utils import read_h5_event_components, split_events_time_interval
from utils.viewer import *
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Get config
    config_path = Path(__file__).parent / "config" / "calibration.yaml"
    cfg = load_config(config_path)
    path = cfg["path"]
    img_size = tuple(cfg["img_size"])
    board_w = cfg["board_w"]
    board_h = cfg["board_h"]
    square_size = cfg["square_size"]
    vis_iwe    = cfg.get("visualization", {}).get("iwe")
    vis_corner = cfg.get("visualization", {}).get("corner")
    vis_calib  = cfg.get("visualization", {}).get("calib")
    is_user_selecting = cfg.get("user_selecting")
    print("[INFO] Starting calibration framework with")
    print("       H5 path:", path)
    print("       Image size:", img_size)
    print("       Board size:", board_w, "x", board_h)
    print("       Square size:", square_size)
    print("       Do user selection", is_user_selecting)
    print("       Do visualization")
    print("               IWE: ", vis_iwe)
    print("               Corner: ", vis_corner)
    print("               Calibration: ", vis_calib)

    # Load event data
    xs, ys, ts, ps = read_h5_event_components(path)
    ts = ts-ts[0]
    print("[INFO] Loaded {} events".format(len(xs)))
    dt = 0.1
    coarse_bin = split_events_time_interval(xs, ys, ts, ps, dt, drop_last=True)
    print("[INFO] Splited into {} bins including {} events".format(len(coarse_bin), len(coarse_bin[0][0])))
    
    # Contrast Maximization
    iwes = get_valid_iwes(coarse_bin[:50], img_size, vis_iwe)
    if len(iwes) == 0:
        raise RuntimeError("[ERROR] No valid images found. Corner detection aborted.")
    
    # Corner Detection
    used_iwes, imgpoints = get_valid_corners(iwes, board_w, board_h, vis_corner, is_user_selecting)
    if len(imgpoints) == 0:
        raise RuntimeError("[ERROR] No valid corners found. Calibration aborted.")

    # Calibration
    ## Prepare 3D object points
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size
    objpoints = [objp.copy() for _ in range(len(imgpoints))]
    print("************************************************************")
    print("[INFO] Performing camera calibration...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    ## Evaluation
    if ret:
        print("[SUCCESS] Calibration completed.")
        print("     Camera matrix:\n", K)
        print("     Distortion coefficients:\n", dist.ravel())
    else:
        print("[ERROR] Calibration failed.")
        
    reprojection_error = eval.compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)
    print(f"[INFO] Reprojection error: {reprojection_error:.4f} pixels")
    print("************************************************************")
    if vis_calib:
        print("[INFO] Visualizing calibration results...")
        for i, iwe in enumerate(used_iwes):
            imgp_gt = imgpoints[i].reshape(-1, 2).astype(np.float32)
            imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            imgp_proj = imgpoints_proj.reshape(-1, 2).astype(np.float32)
            undistorted = eval.undistort_iwe(iwe, K, dist)
            cv2.imshow("Undistorted", undistorted)
            cv2.waitKey(100)
            eval.visualize_reprojection(iwe, imgp_gt, imgp_proj)
                
        cv2.destroyAllWindows()
        run_calibration_gui(K, dist, rvecs, tvecs, objpoints, imgpoints, img_size)