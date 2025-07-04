#!/usr/bin/env python3
import cv2
import numpy as np
import torch

def draw_corner_candidates(iwe, corners, radius=3, color=(0,255,0)):
    vis = cv2.cvtColor(iwe, cv2.COLOR_GRAY2BGR)
    for x,y in corners.reshape(-1,2):
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis

def initialize_grid(board_w, board_h, corner_candidates):
    mean = np.mean(corner_candidates, axis=0)
    std = np.std(corner_candidates, axis=0)

    xs = np.linspace(mean[0] - std[0], mean[0] + std[0], board_w)
    ys = np.linspace(mean[1] - std[1], mean[1] + std[1], board_h)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)  # shape: (h, w, 2)
    return torch.tensor(grid, dtype=torch.float32)

def compute_loss(G, corner_candidates):
    loss = 0.0
    h, w, _ = G.shape
    eps = 1e-6

    # Spacing Ratio Loss
    spacing_ratio_loss = 0.0
    # row direction
    for i in range(h):
        for j in range(1, w - 1):
            a = torch.norm(G[i, j+1] - G[i, j])
            b = torch.norm(G[i, j] - G[i, j-1])
            spacing_ratio_loss += ((a / (b + eps) - 1.0) ** 2)

    # column direction
    for j in range(w):
        for i in range(1, h - 1):
            a = torch.norm(G[i+1, j] - G[i, j])
            b = torch.norm(G[i, j] - G[i-1, j])
            spacing_ratio_loss += ((a / (b + eps) - 1.0) ** 2)

    loss += spacing_ratio_loss

    # Orthogonality Loss
    for i in range(h - 1):
        for j in range(w - 1):
            a = G[i+1, j] - G[i, j]
            b = G[i, j+1] - G[i, j]
            cos_angle = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + eps)
            loss += cos_angle**2

    # Candidate Proximity Loss
    cand = torch.tensor(corner_candidates, dtype=torch.float32)
    G_flat = G.reshape(-1, 2)
    dists = torch.cdist(G_flat, cand)  # shape (16, N)
    min_dists, _ = torch.min(dists, dim=1)
    loss += torch.sum(min_dists**2) * 0.5  # weighting factor

    return loss

def optimize_corners(corner_candidates, board_w, board_h):
    G = torch.nn.Parameter(initialize_grid(board_w, board_h, corner_candidates))
    optimizer = torch.optim.Adam([G], lr=0.1)

    for iter in range(200):
        optimizer.zero_grad()
        loss = compute_loss(G, corner_candidates)
        loss.backward()
        optimizer.step()

    return G.detach().numpy()

def snap_to_candidates(G_numpy, corner_candidates):
    from scipy.spatial import cKDTree

    h, w, _ = G_numpy.shape
    tree = cKDTree(corner_candidates)
    snapped = np.empty_like(G_numpy)
    used = set()

    for i in range(h):
        for j in range(w):
            g = G_numpy[i, j]
            dists, indices = tree.query(g, k=len(corner_candidates))
            for idx in indices:
                idx = int(idx)
                if idx not in used:
                    used.add(idx)
                    snapped[i, j] = corner_candidates[idx]
                    break

    return snapped

def detect_corners(iwe, board_w, board_h, quality=0.1, min_dist=10, vis=False):
    gray = cv2.normalize(iwe, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    max_corners =  2 * board_w * board_h
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                  qualityLevel=quality,
                                  minDistance=min_dist)

    if pts is None or len(pts) < board_w * board_h:
        return None
    corner_candidates = pts.reshape(-1,2).astype(np.float32)
    ip = initialize_grid(board_w, board_h, corner_candidates)
    G_numpy = optimize_corners(corner_candidates, board_w=board_w, board_h=board_h)
    snapped = snap_to_candidates(G_numpy, corner_candidates)
    if vis:
        vis = draw_corner_candidates(iwe, corner_candidates, color=(0,0,255))
        cv2.imshow("Corner candidate", vis)
        cv2.waitKey(100)
        vis = draw_corner_candidates(iwe, ip, color=(255,0,0))
        cv2.imshow("Before Opt; contineous points", vis)
        cv2.waitKey(100)
        vis = draw_corner_candidates(iwe, G_numpy, color=(255,0,0))
        cv2.imshow("After Opt; contineous points", vis)
        cv2.waitKey(100)
        vis = draw_corner_candidates(iwe, snapped)
        cv2.imshow("snapped", vis)
        cv2.waitKey(100)

    return snapped

def is_corner_valid(grid: np.ndarray, tor_spacing, tor_orth) -> bool:
    """
    Check whether a grid of 2D points has regular spacing and near-orthogonal structure.
    """
    H, W, _ = grid.shape
    if H < 2 or W < 2:
        return False  # not enough points to check structure

    # Row-wise spacing
    row_dists = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2)  # (H, W-1)
    row_std = np.std(row_dists)
    row_mean = np.mean(row_dists)
    row_ok = (row_std / (row_mean + 1e-6)) < tor_spacing
    if not row_ok:
        print("     [WARNING] Row-wise spacing is unreliable")

    # Column-wise spacing
    col_dists = np.linalg.norm(grid[1:, :] - grid[:-1, :], axis=2)  # (H-1, W)
    col_std = np.std(col_dists)
    col_mean = np.mean(col_dists)
    col_ok = (col_std / (col_mean + 1e-6)) < tor_spacing
    if not col_ok:
        print("     [WARNING] Column-wise spacing is unreliable")

    # Orthogonality check (dot product of row and column vectors at internal corners)
    orth_errors = []
    for i in range(H - 1):
        for j in range(W - 1):
            dx = grid[i, j + 1] - grid[i, j]
            dy = grid[i + 1, j] - grid[i, j]
            norm_dx = np.linalg.norm(dx)
            norm_dy = np.linalg.norm(dy)
            if norm_dx < 1e-6 or norm_dy < 1e-6:
                continue  # skip degenerate
            cos_angle = np.dot(dx, dy) / (norm_dx * norm_dy)
            orth_errors.append(abs(cos_angle))  # 0 when orthogonal

    orth_ok = np.mean(orth_errors) < tor_orth if orth_errors else False
    if not col_ok:
        print("     [WARNING] Orthogonality is unreliable")

    return row_ok and col_ok and orth_ok


def get_valid_corners(iwes, board_w, board_h, vis, is_user_selecting, tor_spacing, tor_orth):
    imgpoints = []
    used_iwes = []
    valid_count = 0
    num = len(iwes)
    print("[INFO] Start corner detection process")
    for i, iwe in enumerate(iwes):
        imgp = detect_corners(iwe, board_w, board_h, vis=vis)
        if imgp is None:
            continue
        if not is_corner_valid(imgp, tor_spacing, tor_orth):
            print(f"    [SKIPPED] Grid structure invalid ({i}/{num})")
            continue
        if is_user_selecting:
            corner_checker = draw_corner_candidates(iwe, imgp)
            print("Press 'y' to accpet this corner, and any keys to reject")
            cv2.imshow("Corner checker", corner_checker)
            key = cv2.waitKey(0)
            if key != ord('y'):
                print(f"    [SKIPPED] User rejection ({i}/{num})")
                continue

        imgpoints.append(imgp.reshape(-1, 2).astype(np.float32))
        used_iwes.append(iwe)
        valid_count += 1
        print(f"    [OK] Corners validated and added ({i}/{num})")
    if vis:
        cv2.destroyAllWindows()
    print("[INFO] Corner detection process is done!")
    print(f"        #accepted: {valid_count}")
    print("************************************************************")

    return used_iwes, imgpoints