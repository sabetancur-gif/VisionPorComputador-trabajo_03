# Filtros de Gabor

import cv2
import numpy as np

def compute_gabor_features(image, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi/4, np.pi/2]):
    
    feats = {}

    idx = 0
    for freq in frequencies:
        for theta in thetas:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5,
                theta=theta,
                lambd=1/freq,
                gamma=0.5,
                psi=0
            )
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)

            feats[f"gabor_mean_{idx}"] = filtered.mean()
            feats[f"gabor_std_{idx}"] = filtered.std()

            idx += 1

    return feats