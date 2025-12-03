# Local Binary Patterns (LBP) 

import numpy as np
from skimage.feature import local_binary_pattern

def compute_lbp(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(),
                           bins=P+2,
                           range=(0, P+2),
                           density=True)
    return lbp, hist