"""
before_after.py.

Módulo con funciones auxiliares para el análisis sencillo de imágenes en escala de
grises: entropía de Shannon y estimación de desviación estándar local mediante
filtro de caja. Las funciones aceptan arreglos numpy (uint8 o float) y devuelven
arreglos numpy. Está pensado para imágenes en escala de grises con valores en
el rango [0, 255].

Funciones
---------
shannon_entropy(img)
    Calcula la entropía de Shannon de una imagen (uint8 o convertible) en escala
    de grises.

local_std_img(img, ksize=31)
    Estima la desviación estándar local usando un blur de caja (cv2.blur).

Ejemplo
-------
>>> import cv2, numpy as np
>>> from imagen_metrics import shannon_entropy, local_std_img
>>> img = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)
>>> print(shannon_entropy(img))
>>> s = local_std_img(img, ksize=15)

"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

__all__ = ["shannon_entropy", "local_std_img"]


def _as_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convertir entrada a imagen grayscale uint8.

    - Si la imagen ya es uint8 se devuelve una copia (o la misma referencia si es
      inmutable).
    - Si la imagen es float se asume que sus valores están en [0, 255] y se
      realiza un `np.rint` con recorte.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (H, W) o (H, W, 1).

    Returns
    -------
    np.ndarray
        Imagen en uint8 con forma (H, W).
    """
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.reshape(img.shape[:2])

    if img.dtype == np.uint8:
        return img

    # Para imágenes float, asumimos rango [0,255]
    if np.issubdtype(img.dtype, np.floating):
        f = np.rint(img).astype(np.int32)
        f = np.clip(f, 0, 255).astype(np.uint8)
        return f

    # Para otros dtypes enteros, recortar y convertir
    if np.issubdtype(img.dtype, np.integer):
        a = img.astype(np.int64)
        a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    # Fallback: intentar convertir a float y luego a uint8
    f = np.array(img, dtype=np.float32)
    f = np.rint(f)
    f = np.clip(f, 0, 255).astype(np.uint8)
    return f


def shannon_entropy(img: np.ndarray) -> float:
    """Calcular la entropía de Shannon de una imagen en escala de grises.

    La función espera una imagen de 8 bits (valores 0-255). Si se le pasa una
    imagen en otro tipo, intentará convertirla razonablemente.

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises (H, W) con dtype uint8 o convertible.

    Returns
    -------
    float
        Entropía de Shannon en bits.

    Notes
    -----
    - Se utiliza un histograma de 256 bins (0..255). Para evitar log(0) se
      descartan probabilidades nulas.
    - Para imágenes vacías (todos ceros) la entropía será 0.

    Examples
    --------
    >>> import numpy as np
    >>> shannon_entropy(np.zeros((10,10), dtype=np.uint8))
    0.0

    """
    if not isinstance(img, np.ndarray):
        raise TypeError("`img` debe ser un numpy.ndarray")

    u8 = _as_uint8_gray(img)

    # histograma - usa bincount por ser muy rápido
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0

    prob = hist / total
    prob = prob[prob > 0.0]

    # entropía en bits
    ent = -(prob * np.log2(prob)).sum()
    return float(ent)


def local_std_img(img: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Estimar la desviación estándar local usando un blur de caja.

    La desviación estándar se estima mediante las identidades:

        var = E[X^2] - (E[X])^2
        std = sqrt(max(var, 0))

    donde las esperanzas locales se aproximan con `cv2.blur` (media de la
    ventana). La función acepta imágenes uint8 o float y devuelve un arreglo
    float32 con la desviación estándar local por píxel.

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises (H, W). Puede ser uint8 o float con rango
        aproximadamente en [0, 255].
    ksize : int, optional
        Tamaño de la ventana cuadrada (debe ser impar), por defecto 31.

    Returns
    -------
    np.ndarray
        Imagen (H, W) de tipo float32 con la desviación estándar local.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((100,100), dtype=np.uint8)
    >>> s = local_std_img(img, ksize=15)
    >>> s.dtype
    dtype('float32')

    """
    if not isinstance(img, np.ndarray):
        raise TypeError("`img` debe ser un numpy.ndarray")

    if ksize <= 0:
        raise ValueError("`ksize` debe ser un entero positivo")

    # forzar ksize impar
    if ksize % 2 == 0:
        ksize += 1

    # Convertir a float32 para evitar overflow en cuadrados
    f = img.astype(np.float32)
    f2 = f * f

    k = (ksize, ksize)
    mean = cv2.blur(f, k)
    mean2 = cv2.blur(f2, k)

    var = mean2 - mean * mean
    # corregir pequeños valores negativos por errores numéricos
    var = np.maximum(var, 0.0)

    return np.sqrt(var).astype(np.float32)


if __name__ == "__main__":
    # breve demostración si se ejecuta como script
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"No se pudo leer la imagen: {path}")
            sys.exit(1)

        print(f"Entropía: {shannon_entropy(im):.6f} bits")
        s = local_std_img(im, ksize=31)
        print(f"Desviación estándar local: min={s.min():.3f}, max={s.max():.3f}")
    else:
        print("Ejecuta: python imagen_metrics.py <ruta_imagen>")
