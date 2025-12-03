# Estadísticas de primer orden

import numpy as np
from scipy.stats import skew, kurtosis, entropy

def compute_first_order(image):
    
    hist, _ = np.histogram(image, bins=256, range=(0,256), density=True)

    return {
        "mean": np.mean(image),
        "variance": np.var(image),
        "skewness": skew(image.ravel()),
        "kurtosis": kurtosis(image.ravel()),
        "entropy": entropy(hist + 1e-8)
    }
