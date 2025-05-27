from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys


class CalibrationViewer(QtWidgets.QTabWidget):
    def __init__(self, K, dist, rvecs, tvecs, objpoints, imgpoints, image_size):
        super().__init__()
        self.setWindowTitle("Calibration Report")

        self.addTab(self.create_pose_tab(rvecs, tvecs), "Estimated Poses")
        self.addTab(self.create_reproj_tab(K, dist, rvecs, tvecs, objpoints, imgpoints, image_size), "Reprojection Errors")

    def create_pose_tab(self, rvecs, tvecs):
        fig = Figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        tv_arr = np.array([t.reshape(3) for t in tvecs])
        mins = tv_arr.min(axis=0)
        maxs = tv_arr.max(axis=0)
        diffs = maxs - mins
        tv_n = (tv_arr - mins) / diffs

        axis_len = 0.1
        prev = None
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            o = tv_n[i]
            R, _ = cv2.Rodrigues(rvec)
            x_v = R[:,0] * axis_len
            y_v = R[:,1] * axis_len
            z_v = R[:,2] * axis_len

            ax.plot([o[0], o[0]+x_v[0]], [o[1], o[1]+x_v[1]], [o[2], o[2]+x_v[2]], 'r')
            ax.plot([o[0], o[0]+y_v[0]], [o[1], o[1]+y_v[1]], [o[2], o[2]+y_v[2]], 'g')
            ax.plot([o[0], o[0]+z_v[0]], [o[1], o[1]+z_v[1]], [o[2], o[2]+z_v[2]], 'b')

            if prev is not None:
                ax.plot([prev[0], o[0]], [prev[1], o[1]], [prev[2], o[2]], 'k', linewidth=0.8)
            prev = o

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1))

        canvas = FigureCanvas(fig)
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        l.addWidget(canvas)
        w.setLayout(l)

        return w


    def create_reproj_tab(self, K, dist, rvecs, tvecs, objpoints, imgpoints, image_size):
        fig = Figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_xlim(0, image_size[0])
        ax1.set_ylim(image_size[1], 0)
        ax1.set_aspect('equal')
        ax2.set_xlabel("error x (pix)")
        ax2.set_ylabel("error y (pix)")
        ax2.set_aspect('equal')
        ax2.grid(True)

        cmap = plt.get_cmap("jet")
        all_dx, all_dy, all_idx = [], [], []

        n_imgs = len(imgpoints)
        for i in range(n_imgs):
            color = cmap(i / max(n_imgs - 1, 1))
            gt = imgpoints[i].reshape(-1, 2)
            prj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            pr = prj.reshape(-1, 2)

            for g, p in zip(gt, pr):
                ax1.plot([g[0], p[0]], [g[1], p[1]], color=color, alpha=0.5, linewidth=0.6)
            ax1.scatter(gt[:, 0], gt[:, 1], c=[color], s=50, marker='o',alpha=0.5)
            ax1.scatter(pr[:, 0], pr[:, 1], c=[color], s=50, marker='x',alpha=0.5)

            dx = pr[:, 0] - gt[:, 0]
            dy = pr[:, 1] - gt[:, 1]
            all_dx.extend(dx); all_dy.extend(dy)
            all_idx.extend([i] * len(dx))

        sc = ax2.scatter(all_dx, all_dy, c=all_idx, cmap='jet', s=50, alpha=0.5)
        fig.colorbar(sc, ax=(ax1, ax2), label="image index")

        canvas = FigureCanvas(fig)
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(); l.addWidget(canvas); w.setLayout(l)

        return w

def run_calibration_gui(K, dist, rvecs, tvecs, objpoints, imgpoints, image_size):
    app = QtWidgets.QApplication(sys.argv)
    win = CalibrationViewer(K, dist, rvecs, tvecs, objpoints, imgpoints, image_size)
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())
