"""
Módulo: 'preprocess_image'

Contiene una sola función `preprocess_image_save` para cargar, convertir a
escala de grises, aplicar CLAHE opcionalmente, redimensionar y guardar una
imagen procesada.

La lógica es exactamente la misma que la función original proporcionada.
"""
import cv2
from pathlib import Path


def preprocess_image_save(in_path: Path,
                           out_path: Path,
                           size=(224, 224),
                           apply_clahe: bool = True) -> bool:
    """Preprocesa una imagen y la guarda.

    Pasos:
    - Crea el directorio padre de out_path si no existe.
    - Carga la imagen (BGR) con cv2.imread.
    - Convierte a escala de grises.
    - (Opcional) Aplica CLAHE.
    - Redimensiona a `size` usando INTER_AREA.
    - Guarda como PNG (uint8) en out_path.

    Devuelve True si la imagen se procesó correctamente, False si no se pudo
    cargar.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(in_path))
    if img_bgr is None:
        return False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_path), resized)
    return True
