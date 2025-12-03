"""
fourier_descriptor.py

Extrae descriptores de forma mediante Fourier a partir de una imagen.
Function: extract_fourier(img, n_coeff=20)
"""

import numpy as np
import cv2

def _descriptor_fourier_from_contour(cnt: np.ndarray, n_coeff: int = 20) -> np.ndarray:
    """Calcula la DFT (magnitud) sobre un contorno dado."""
    if cnt is None or len(cnt) == 0:
        return np.zeros(n_coeff)
    pts = cnt[:, 0, :]
    complex_pts = pts[:, 0].astype(float) + 1j * pts[:, 1].astype(float)
    fft_vals = np.fft.fft(complex_pts)
    fft_vals = np.abs(fft_vals)
    if len(fft_vals) >= n_coeff:
        result = fft_vals[:n_coeff]
    else:
        result = np.pad(fft_vals, (0, n_coeff - len(fft_vals)))
    return result

def extract_fourier(img: np.ndarray, n_coeff: int = 20) -> np.ndarray:
    """Extrae descriptores de forma mediante Fourier a partir de una imagen.

    Steps:
    - Convierte a gris si es necesario.
    - Binariza con OTSU.
    - Extrae el contorno más grande.
    - Calcula la DFT sobre los puntos del contorno y devuelve las magnitudes
      de los primeros `n_coeff` coeficientes.
    """
    if img is None:
        raise ValueError("`img` es None")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return np.zeros(n_coeff)

    c = max(cnts, key=cv2.contourArea)
    return _descriptor_fourier_from_contour(c, n_coeff=n_coeff)
