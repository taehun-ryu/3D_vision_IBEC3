import numpy as np
import cv2
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter

class IweMetric(ABC):
    """
    Base class for IWE quality metrics with fixed blur_sigma = 1.0.
    """
    def __init__(self, name):
        self.name = name
        self.blur_sigma = 1.0

    def preprocess(self, iwe):
        # Apply Gaussian blur with sigma = 1.0
        return gaussian_filter(iwe, self.blur_sigma)

    @abstractmethod
    def evaluate(self, iwe):
        pass

class VarianceMetric(IweMetric):
    def __init__(self):
        super().__init__('variance')

    def evaluate(self, iwe):
        iwe_blur = self.preprocess(iwe)
        return iwe_blur.var()

class SumOfSquaresMetric(IweMetric):
    def __init__(self):
        super().__init__('sum_of_squares')

    def evaluate(self, iwe):
        iwe_blur = self.preprocess(iwe)
        return np.mean(iwe_blur**2)

class GradientEnergyMetric(IweMetric):
    def __init__(self):
        super().__init__('gradient_energy')

    def evaluate(self, iwe):
        iwe_blur = self.preprocess(iwe)
        gx = cv2.Sobel(iwe_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(iwe_blur, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx*gx + gy*gy).sum()

class EntropyMetric(IweMetric):
    def __init__(self, n_bins=256):
        super().__init__('entropy')
        self.n_bins = n_bins

    def evaluate(self, iwe):
        iwe_blur = self.preprocess(iwe)
        hist, _ = np.histogram(iwe_blur, bins=self.n_bins, density=True)
        hist = hist[hist > 0]
        return -(hist * np.log(hist)).sum()

# Example usage:
if __name__ == "__main__":
    # Assume iwe is a 2D numpy array of your motion‐compensated image
    iwe = np.random.randn(260, 346).astype(np.float32)

    metrics = [
        VarianceMetric(),
        SumOfSquaresMetric(),
        GradientEnergyMetric(),
        EntropyMetric(n_bins=128)
    ]

    for metric in metrics:
        score = metric.evaluate(iwe)
        print(f"{metric.name} (blur_sigma={metric.blur_sigma}): {score:.4f}")
