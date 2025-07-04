import numpy as np
import math
import cv2
from src.cmax import objectives as obj
from src.cmax import warps as wf
from src.cmax import events_cmax as cmax
from scipy.ndimage import gaussian_filter
from abc import ABC, abstractmethod

class IweMetric(ABC):
    def __init__(self, name):
        self.name = name
        self.blur_sigma = 1.0

    def preprocess(self, iwe):
        # Apply Gaussian blur with sigma = 1.0
        return gaussian_filter(iwe, self.blur_sigma)

    @abstractmethod
    def evaluate(self, iwe):
        pass

class GradientEnergyMetric(IweMetric):
    def __init__(self):
        super().__init__('gradient_energy')

    def evaluate(self, iwe):
        iwe_blur = self.preprocess(iwe)
        gx = cv2.Sobel(iwe_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(iwe_blur, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx*gx + gy*gy).sum()

def compute_final_iwe(params, xs, ys, ts, ps, warpfunc, img_size):
    xw, yw, _, _ = warpfunc.warp(xs, ys, ts, ps, ts[-1], params, compute_grad=False)

    H, W = img_size
    iwe = np.zeros((H, W), dtype=np.float32)

    # Compute integer and fractional parts of the coordinates
    x0 = np.floor(xw).astype(int)
    x1 = x0 + 1
    y0 = np.floor(yw).astype(int)
    y1 = y0 + 1

    wx1 = xw - x0  # Weight for x1
    wx0 = 1 - wx1  # Weight for x0
    wy1 = yw - y0  # Weight for y1
    wy0 = 1 - wy1  # Weight for y0

    # Mask to ensure coordinates are within bounds
    mask = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    # Apply bilinear interpolation
    np.add.at(iwe, (y0[mask], x0[mask]), ps[mask] * wx0[mask] * wy0[mask])
    np.add.at(iwe, (y0[mask], x1[mask]), ps[mask] * wx1[mask] * wy0[mask])
    np.add.at(iwe, (y1[mask], x0[mask]), ps[mask] * wx0[mask] * wy1[mask])
    np.add.at(iwe, (y1[mask], x1[mask]), ps[mask] * wx1[mask] * wy1[mask])

    return iwe

def is_near_angle(angle_deg, target_angles, tolerance):
    return any(
        abs(angle_deg - ref) <= tolerance or abs(angle_deg - ref + 360) <= tolerance or abs(angle_deg - ref - 360) <= tolerance
        for ref in target_angles
    )

def get_valid_iwes(coarse_bin, img_size, vis, tor_ge, tor_theta):
    # Optimization parameters
    objective = obj.r1_objective()
    warp = wf.linvel_warp()
    metric = GradientEnergyMetric()
    argmax = [0, 0]
    # Etc
    target_angles = [0, 90, 180, 270]
    iwes = []
    valid_count = 0
    skipped_count = 0
    total_num = len(coarse_bin)
    print("************************************************************")
    print("[INFO] Start optimization process")
    for xs, ys, ts, ps in coarse_bin:
        argmax = cmax.optimize(xs, ys, ts, ps, warp, objective, numeric_grads=True, init=argmax)
        theta_deg = np.rad2deg(math.atan2(argmax[1], argmax[0]))
        theta_360 = (theta_deg + 360) % 360
        print(f"    [PROGRESS] Bin {valid_count+skipped_count}/{total_num}")
        print(f"        Optimized: vx = {argmax[0]:.2f}, vy = {argmax[1]:.2f}, theta(deg) = {theta_360:.2f}")

        # Filtering pure motion
        if is_near_angle(theta_360, target_angles, tor_theta):  #TODO we need to consider relative pose of checkerboard on image plane.
            print("         [SKIPED] One-direction edge")
            skipped_count+=1
            continue

        # Filtering unreliable optimizaiton result
        iwe_eval = objective.evaluate_function(argmax, xs, ys, ts, ps, warp, img_size=img_size)
        unwarp_eval = objective.evaluate_function([0, 0], xs, ys, ts, ps, warp, img_size=img_size)
        if abs(iwe_eval) <= abs(unwarp_eval):
            print("         [SKIPED] Unreliable optimization result")
            skipped_count +=1
            continue

        # Construct Image of Warped Event
        iwe = compute_final_iwe(argmax, xs, ys, ts, ps, warp, img_size)
        iwe = cv2.normalize(iwe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Filtering based on Gradient Energy
        if metric.evaluate(iwe) < tor_ge:    #THINK Too heuristic?
            print("         [SKIPED] Low Gradient Energy")
            skipped_count +=1
            continue

        if vis:
            unwarp_img = compute_final_iwe([0, 0], xs, ys, ts, ps, warp, img_size)
            unwarp_img = cv2.normalize(unwarp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("IWE", iwe)
            cv2.imshow("Unwarping", unwarp_img)
            cv2.waitKey(250)
        print("         [ACCEPTED] IWE is valid!!!!!!!!!!")
        valid_count += 1
        iwes.append(iwe)
    if vis:
        cv2.destroyAllWindows()
    # Logging
    acceptance_rate = valid_count/total_num * 10
    print("[INFO] Optimization Process is done!")
    print(f"        acceptance ratio: {acceptance_rate}%")
    print(f"        #accpeted: {valid_count}, #rejected: {skipped_count}")
    print("************************************************************")

    return iwes