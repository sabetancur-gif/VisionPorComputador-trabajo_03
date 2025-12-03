"""
hu_moments.py

Calcula los 7 momentos de Hu de una imagen.
Function: extract_hu_moments(img)
"""

import numpy as np
import cv2

def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """Calcula los momentos de Hu de una imagen.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (gris o color).

    Returns
    -------
    hu : np.ndarray
        Array de 7 valores con los momentos de Hu.
    """
    if img is None:
        raise ValueError("`img` es None")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return hu
