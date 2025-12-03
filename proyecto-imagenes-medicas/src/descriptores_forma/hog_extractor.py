"""
hog_extractor.py

Extrae HOG (Histogram of Oriented Gradients) de una imagen.
Function: extract_hog(img, pixels_per_cell=(16,16), orientations=9)
"""

import numpy as np
import cv2
from skimage.feature import hog as sk_hog
from typing import Tuple

def extract_hog(img: np.ndarray, pixels_per_cell=(16, 16), orientations: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae HOG (Histogram of Oriented Gradients) de una imagen.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (gris o color).
    pixels_per_cell : tuple
        Tamaño del pixel cell para HOG.
    orientations : int
        Número de orientaciones.

    Returns
    -------
    hog_features, hog_image : tuple
        Vector de características HOG y la imagen visualizada de HOG.
    """
    if img is None:
        raise ValueError("`img` es None")

    # convertir a gris si viene en color
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    hog_features, hog_image = sk_hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return hog_features, hog_image
