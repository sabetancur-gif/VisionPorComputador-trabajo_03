"""
contour_features.py

Extrae descriptores geométricos del contorno más grande de la imagen.
Function: extract_contour_features(img)
"""

import numpy as np
import cv2

def extract_contour_features(img: np.ndarray) -> dict:
    """Extrae descriptores geométricos del contorno más grande de la imagen.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (gris o color).

    Returns
    -------
    dict
        Diccionario con claves: area, perimeter, circularity, eccentricity.
    """
    if img is None:
        raise ValueError("`img` es None")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Binarización (OTSU)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Encontrar contornos (compatibilidad con OpenCV 4.x)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return {"area": 0, "perimeter": 0, "circularity": 0, "eccentricity": 0}

    c = max(cnts, key=cv2.contourArea)

    # Área y perímetro
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    # Circularidad
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0

    # Excentricidad mediante ajuste de elipse
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        (_, axes, _) = ellipse
        major, minor = max(axes), min(axes)
        eccentricity = np.sqrt(1 - (minor / major)**2) if major != 0 else 0
    else:
        eccentricity = 0

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "eccentricity": eccentricity
    }
