"""
Módulo 'extract_selected'

Funciones para extraer (desde ficheros ZIP/TAR) o copiar (desde un directorio)
solo las subcarpetas de interés (por defecto: 'train', 'val', 'test').

Todas las funciones mantienen la lógica original solicitada y devuelven `None`.

Ejemplo rápido:
    from pathlib import Path
    from extract_selected import extract_selected_from_zip

    extract_selected_from_zip(Path("dataset.zip"), Path("/tmp/out"))

Constante exportada:
    DESIRED -- tupla con los nombres de subcarpetas que queremos extraer/mover.
"""
from pathlib import Path
import zipfile
import tarfile
import shutil
from typing import Tuple

# Carpetas que buscamos dentro del zip/tar o dentro del directorio de origen
DESIRED: Tuple[str, ...] = ("train", "val", "test")


def extract_selected_from_zip(zip_path: Path, dest: Path) -> None:
    """Extrae desde un archivo .zip solo las rutas que contienen cualquiera de
    los componentes listados en DESIRED (por ejemplo 'train', 'val', 'test').

    - zip_path: ruta al archivo .zip.
    - dest: carpeta destino donde se recrearán únicamente las subrutas que
      incluyan alguno de los componentes en DESIRED.

    La lógica sigue exactamente el comportamiento original: se inspeccionan
    las entradas del zip, se busca el índice del primer componente que
    pertenezca a DESIRED y se extrae desde ese punto en adelante.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            fn = info.filename
            parts = Path(fn).parts
            # buscar índice donde aparezca 'train'/'val'/'test'
            idx = next((i for i, p in enumerate(parts) if p in DESIRED), None)
            if idx is None:
                continue
            target_sub = Path(*parts[idx:])  # p.ej. train/XXX/...
            target_path = dest / target_sub
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target_path, "wb") as out:
                    shutil.copyfileobj(src, out)
    print("ZIP: extraídas las carpetas seleccionadas en", dest)


def extract_selected_from_tar(tar_path: Path, dest: Path) -> None:
    """Extrae desde un archivo .tar (o .tar.gz/.tgz, etc.) solo los miembros
    cuyas rutas contengan alguno de los componentes en DESIRED.

    - tar_path: ruta al archivo tar (se abre con modo 'r:*' para auto-detectar
      compresión).
    - dest: carpeta destino donde se recrearán únicamente las subrutas que
      incluyan alguno de los componentes en DESIRED.

    Se preserva la lógica original: se recorre getmembers(), se busca el
    índice del componente deseado y se extrae desde ese punto.
    """
    with tarfile.open(tar_path, 'r:*') as t:
        for member in t.getmembers():
            parts = Path(member.name).parts
            idx = next((i for i, p in enumerate(parts) if p in DESIRED), None)
            if idx is None:
                continue
            # construir ruta destino
            target_sub = Path(*parts[idx:])
            target_path = dest / target_sub
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                f = t.extractfile(member)
                if f is not None:
                    with open(target_path, "wb") as out:
                        shutil.copyfileobj(f, out)
    print("TAR: extraídas las carpetas seleccionadas en", dest)


def move_selected_from_dir(src_dir: Path, dest: Path) -> None:
    """Copia las subcarpetas indicadas en DESIRED desde `src_dir` hacia `dest`.

    - Busca en `src_dir` y en sus hijos de primer nivel (se consideran
      candidatos: src_dir y cada subdirectorio directo de src_dir).
    - Para cada candidato, si existe una carpeta con nombre en DESIRED, la
      copia completa al destino (se elimina la carpeta destino previa si
      existe).

    Mantiene la implementación original: no modifica la lógica de búsqueda
    ni de copia (se usa shutil.copytree y shutil.rmtree cuando procede).
    """
    # buscar subcarpetas train/val/test a profundidad 1 o dentro de una carpeta top-level
    # comprobamos src_dir itself, y sus hijos
    candidates = [src_dir] + [p for p in src_dir.iterdir() if p.is_dir()]
    found = False
    for base in candidates:
        for name in DESIRED:
            src = base / name
            if src.exists() and src.is_dir():
                found = True
                target = dest / name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(src, target)
                print(f"Movida {src} -> {target}")
    if not found:
        print("No se encontraron carpetas train/val/test en", src_dir)
