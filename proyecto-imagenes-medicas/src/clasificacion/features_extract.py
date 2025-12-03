
"""Módulo para extracción de características de imágenes."""

from typing import List, Tuple, Union, Optional
import numpy as np

def flatten_feature(x: Union[dict, list, tuple, np.ndarray, float, int]) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Convierte `x` en un np.ndarray 1-D de tipo float32.
    Si `x` es dict devuelve también la lista de keys (ordenadas) para poder
    construir nombres de características consistentes.
    """
    if x is None:
        raise ValueError("Se recibió None como característica")

    # Diccionario -> vector con keys ordenadas (determinista)
    if isinstance(x, dict):
        keys = sorted(x.keys())
        vals = [float(x[k]) for k in keys]
        return np.asarray(vals, dtype=np.float32).ravel(), keys

    # ndarray -> ravel
    if isinstance(x, np.ndarray):
        return x.ravel().astype(np.float32), None

    # lista/tupla -> convertir a ndarray y ravel
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float32)
        # Si es array de objetos intentamos forzar a float
        if arr.dtype == object:
            arr = np.array([float(xx) for xx in arr], dtype=np.float32)
        return arr.ravel(), None

    # scalar -> array de 1 elemento
    try:
        f = float(x)
    except Exception as e:
        raise TypeError(f"No se pudo convertir la característica a float: {e}") from e
    return np.asarray([f], dtype=np.float32), None


