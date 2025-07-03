import argparse
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

def read_h5_events(hdf_path):
    """
    Read events from HDF5 file into 4xN numpy array (N=number of events)
    """
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = np.stack((f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1)), axis=1)
    else:
        events = np.stack((f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1)), axis=1)
    return events

def read_h5_event_components(hdf_path):
    """
    Read events from HDF5 file. Return x,y,t,p components.
    """
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        return (f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1))
    else:
        return (f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1))

def save_image(image, lognorm=False, cmap='gray', save_dir="/tmp/img.jpg", save_name="img.jpg"):
    """
    Save an image
    """
    os.makedirs(save_dir, exist_ok=True)
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    save_path = os.path.join(save_dir, save_name)
    cv.imwrite(save_path, image)

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