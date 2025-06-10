import argparse
import numpy as np
from event_utils import *
from objectives  import *
from warps       import *
from events_cmax import *
from metric      import *
from scipy.ndimage import gaussian_filter
from datetime import datetime

def split_events_time_interval(xs, ys, ts, ps, dt, t_start=None, t_end=None, drop_last=False):
    if t_start is None:
        t_start = ts[0]
    if t_end is None:
        t_end = ts[-1]

    edges = np.arange(t_start, t_end, dt, dtype=ts.dtype)
    edges = np.concatenate([edges, np.array([t_end], dtype=ts.dtype)])

    boundaries = np.searchsorted(ts, edges, side='left')

    bins = []
    n_bins = len(boundaries) - 1

    for i in range(n_bins):
        t0, t1 = edges[i], edges[i+1]
        start_idx, end_idx = boundaries[i], boundaries[i+1]

        is_last = (i == n_bins - 1)
        width = t1 - t0

        if is_last and drop_last and (width < dt):
            break

        xs_bin = xs[start_idx:end_idx]
        ys_bin = ys[start_idx:end_idx]
        ts_bin = ts[start_idx:end_idx]
        ps_bin = ps[start_idx:end_idx]

        bins.append((xs_bin, ys_bin, ts_bin, ps_bin))

    return bins

def split_events_fixed_count(xs, ys, ts, ps, packet_size, drop_last=False):
    """
    Split events into bins of exactly packet_size events each.
    If drop_last=True, the final bin with fewer than packet_size events is discarded.
    """
    N = len(ts)
    bins = []
    # number of full bins
    n_full = N // packet_size

    for b in range(n_full):
        start = b * packet_size
        end   = start + packet_size
        bins.append((xs[start:end], ys[start:end], ts[start:end], ps[start:end]))

    # handle the leftover
    rem = N % packet_size
    if rem and not drop_last:
        start = n_full * packet_size
        bins.append((xs[start:], ys[start:], ts[start:], ps[start:]))

    return bins

def compute_final_iwe(params, xs, ys, ts, ps, warpfunc, img_size):
    """
    Warp events with final flow params and accumulate them
    into a fixed-resolution IWE (bilinear interpolation).
    """
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

if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        img_size The size of the event camera sensor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    parser.add_argument("--img_size", nargs='+', type=int, default=(260,346))
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts-ts[0]
    img_size=tuple(args.img_size)
    print("Loaded {} events".format(len(xs)))
    
    dt = 0.1
    #coarse_bin = split_events_time_interval(xs, ys, ts, ps, dt, drop_last=True)
    packet_size = 20000
    coarse_bin = split_events_fixed_count(xs, ys, ts, ps, packet_size, drop_last=True)
    print("Split into {} bins including {} events".format(len(coarse_bin), len(coarse_bin[0][0])))

    objectives = [r1_objective(), zhu_timestamp_objective(), variance_objective(), sos_objective(), soe_objective(),
                  moa_objective(),isoa_objective(), sosa_objective(), rms_objective()]
    warp = linvel_warp()
    metrics = [VarianceMetric(), SumOfSquaresMetric(), GradientEnergyMetric(), EntropyMetric()]
    blur = None

    best_loss = np.inf
    iwes = []
    min_flow_mag = 50.0
    
    metric = metrics[2]
    objective = objectives[0]
    argmax = None
    num = 0
    valid_count = 0
    skipped_count = 0

    for xs, ys, ts, ps in coarse_bin:
        argmax = optimize(xs, ys, ts, ps, warp, objective, numeric_grads=True, init=argmax)
        mag = np.linalg.norm(argmax)

        if abs(argmax[0]) < 1 and abs(argmax[1]) < 1 and mag < min_flow_mag:
            skipped_count += 1
            print(f"Skipped: [argmax] = {argmax}, mag = {mag}")
            continue

        loss = objective.evaluate_function(argmax, xs, ys, ts, ps, warp, img_size=img_size)

        if objective.has_derivative:
            argmax = optimize(xs, ys, ts, ps, warp, objective, numeric_grads=False)
            loss_an = objective.evaluate_function(argmax, xs, ys, ts, ps, warp, img_size=img_size)

        iwe = compute_final_iwe(argmax, xs, ys, ts, ps, warp, img_size)

        if metric.evaluate(iwe) < 85000:
            skipped_count += 1
            print(f"Skipped: loss = {loss}, metric = {metric.evaluate(iwe)}")
            continue

        valid_count += 1
        num += 1
        print(f"Valid: loss = {loss:.2f}, metric = {metric.evaluate(iwe):.2f}, num = {num}")
        iwes.append((iwe, f"image_{num:04d}.png"))

    # 저장
    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    for iwe, name in iwes:
        save_image(iwe, save_dir=f"./iwe/{now_str}", save_name=name)