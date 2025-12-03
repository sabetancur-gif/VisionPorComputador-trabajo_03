<<<<<<< HEAD
# Visión por Computador: QuantumViz

## 🧠 Trabajo03 — Clasificación de Imágenes Médicas (Descriptores Clásicos vs Deep Learning)

---

### 📌 Resumen

Este repositorio documenta, implementa y evalúa un pipeline completo para clasificación de radiografías de tórax (NORMAL vs PNEUMONIA) usando:

1. **Descriptores handcrafted** de forma y textura + clasificadores tradicionales (SVM, Random Forest, k-NN, Logistic Regression), y
2. **Redes Neuronales Convolucionales** (CNNs) entrenadas sobre imágenes.

El objetivo es explorar creativamente distintos descriptores, construir un flujo reproducible desde data raw hasta modelos finales, comparar desempeño y extraer conclusiones técnicas.

---

### 📁 Estructura del repositorio

```
proyecto-clasificacion-imagenes-medicas/
├── data/
│   ├── processed/
│   │   ├── test/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   └── val/
│   │       ├── NORMAL/
│   │       └── PNEUMONIA/
│   ├── raw/
│   │   └── chest_xray/
│   │       ├── test/
│   │       │   ├── NORMAL/
│   │       │   └── PNEUMONIA/
│   │       ├── train/
│   │       │   ├── NORMAL/
│   │       │   └── PNEUMONIA/
│   │       └── val/
│   │           ├── NORMAL/
│   │           └── PNEUMONIA/
│   └── person1946_bacteria_4875.jpeg
│
├── resultados/
│   ├── descriptores_forma/
│   │   ├── figures/
│   │   └── tables/
│   ├── descriptores_textura/
│   │   ├── figures/
│   │   └── tables/
│   ├── exploracion/
│   │   ├── figures/
│   │   └── tables/
│   └── models/
│       ├── best_chest_cnn.pth
│       ├── figures/
│       └── tables/
│
├── src/
│   ├── clasificacion/
│   │   └── features_extract.py
│   ├── descriptores_forma/
│   │   ├── contour_features.py
│   │   ├── fourier_descriptor.py
│   │   ├── hog_extractory.py
│   │   └── hu_moments.py
│   ├── descriptores_textura/
│   │   ├── texture_firstorder.py
│   │   ├── texture_gabor.py
│   │   ├── texture_glcm.py
│   │   └── texture_lbp.py
│   ├── parte_1/
│   │   ├── before_after.py
│   │   └── preprocess_image.py
│   ├── extract_selected.py
│   └── __init__.py
│
├── main.ipynb (único notebook reproducible)
├── README.md   # <- este archivo
└── requirements.txt
```

---

### 🎯 Objetivos específicos

* Implementar pipelines para extracción de descriptores de **forma** y **textura**.
* Entrenar y comparar clasificadores tradicionales usando las features handcrafted.
* Implementar y entrenar al menos una CNN para clasificación directa sobre imágenes.
* Evaluar y reportar métricas: **Accuracy, Precision, Recall, F1, AUC-ROC**, matrices de confusión y análisis de error.
* Entregar notebooks reproducibles, resultados y discusión técnica.

---

### 📦 Datos

* El dataset base proviene de un conjunto público de radiografías de tórax (casos NORMAL vs PNEUMONIA). Colocar los archivos raw en `data/raw/chest_xray/` respetando la partición `train/ val/ test/`.
* `data/processed/` contiene las imágenes normalizadas y preprocesadas usadas por los scripts.

**Nota:** mantener `person1946_bacteria_4875.jpeg` (u otros ejemplos) en `data/` para pruebas rápidas.

---

### 🛠 Instalación y entorno

Se recomienda usar un entorno virtual (conda o venv). A continuación instrucciones con `venv` (Windows / Linux / macOS).

```bash
python -m venv venv
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` incluye (ejemplos): `numpy, pandas, scikit-learn, matplotlib, opencv-python, scikit-image, torch, torchvision, tqdm`.

---

### 🚀 Ejecución — Notebooks (recomendado)

Abrir `main.ipynb` o los notebooks de cada parte si los hay. De existir notebooks de cada parte:

* `notebooks/01_exploration.ipynb` — Exploración de datos y decisiones de preprocesamiento.
* `notebooks/02_features_extraction.ipynb` — Implementación y visualización de descriptores de forma y textura.
* `notebooks/03_classification_handcrafted.ipynb` — Entrenamiento y evaluación de clasificadores con features.
* `notebooks/04_cnn_training.ipynb` — Arquitectura y entrenamiento de la CNN.

**Nota:** Se decidió realizar todo el proceso en un unico notebook `main.ipynb` por cuestiones de estructura.

Todos los notebooks (en este caso solo uno) registran figuras y métricas en `resultados/` por defecto.

---

### 🧭 Uso desde Python — API mínima

Ejemplos de import desde `src`:

```python
from src.clasificacion.features_extract import build_feature_matrix
from src.descriptores_forma.hog_extractory import extract_hog
from src.descriptores_textura.texture_lbp import extract_lbp
from src.parte_1.preprocess_image import preprocess_image
```
---

### 🔬 Preprocesamiento y parte 1 — Recomendaciones

1. **Normalización de tamaño:** redimensionar a un tamaño fijo manteniendo relación de aspecto (p. ej. 224×224 o 256×256) según la red usada.
2. **Ecualización:** aplicar CLAHE local para mejorar contraste en radiografías.
3. **Denoising opcional:** filtros median o bilateral si hay ruido.
4. **Segmentación opcional:** para descriptores de forma, segmentar región pulmonar (umbral adaptativo, o U-Net entrenado si disponible).
5. **Augmentations (solo para CNN):** rotaciones pequeñas, flips horizontales, cambios leves de brillo/contraste, pero evitar transformaciones que alteren la anatomía.

Guardad versiones originales y procesadas en `data/processed/`.

---

### 🧩 Descriptores implementados

Se recomienda como mínimo implementar 3 descriptores de forma y 3 de textura. Los módulos en `src/descriptores_*` implementan estas funciones.

**Forma (ejemplos):**

* HOG (visualización del descriptor y parámetros: cell_size, bins)
* Fourier Shape Descriptors (primeros N coeficientes del contorno)
* Momentos de Hu
* Contour features (área, perímetro, circularidad, excentricidad)

**Textura (ejemplos):**

* LBP (histograma de patrones, experimentar radios y vecinos)
* GLCM (contraste, correlación, energía, homogeneidad — direcciones y distancias)
* Filtros de Gabor (estadísticas de respuestas)
* Estadísticas de primer orden (media, varianza, skewness, kurtosis, entropía)

Cada extractor debe documentarse (entrada, salida, shape) y tener tests básicos (por ejemplo: vector con longitud esperada, no NaNs).

---

### 📈 Clasificación y evaluación

#### Pipeline de features (handcrafted):

1. Extraer features para todo el dataset → `X` (N × D) y etiquetas `y`.
2. Normalizar features (StandardScaler o MinMax).
3. Reducción dimensional (opcional): PCA, SelectKBest.
4. Entrenar clasificadores: SVM (lineal/RBF), RandomForest, k-NN, LogisticRegression.
5. Validación: cross-validation estratificada y partición `train/val/test`.
6. Métricas: Accuracy, Precision, Recall, F1, AUC, matriz de confusión.

Generar reportes y comparar combinaciones de descriptores y clasificadores en `resultados/`.

#### Pipeline CNN (imagen → etiqueta):

* Arquitectura base (ejemplo): ResNet18/Custom CNN.
* Entrenamiento con augmentations moderadas, criterio: CrossEntropyLoss.
* Callbacks: early stopping por validación, guardar `best_chest_cnn.pth`.
* Evaluar en split test final y comparar con aproximaciones handcrafted.

---

### 🧪 Validación sintética y tests

* Crear casos sintéticos (rotaciones, escalados, ruido) para comprobar estabilidad de descriptores.
* Tests unitarios sugeridos en `tests/` (no incluidos por defecto, pero el lector puede incluirlos si así lo desea):

  * `test_features_shape.py` (comprobar longitudes de vectores)
  * `test_preprocess.py` (salida esperada para una imagen de ejemplo)
  * `test_model_io.py` (guardar/cargar checkpoint)

---

### ⚠️ Casos límite y recomendaciones técnicas

* **Desequilibrio:** usar técnicas como oversampling, class weights o focal loss para CNN.
* **Pocas correspondencias (shape descriptors):** mejorar segmentación o usar descriptores globales.
* **Radiografías con artefactos:** aplicar preprocesamiento robusto (CLAHE + denoise).
* **Evaluación cuidadosa:** reportar desviación estándar en cross-validation.

---

### 🧾 Buenas prácticas de reproducibilidad

* Registrar seeds (`numpy`, `torch`, `random`).
* Versionar `requirements.txt` y anotar la versión de Python.
* Guardar hiperparámetros en `resultados/models/<exp_name>_hparams.json`.
* Documentar contribuciones por equipo en el reporte.

---

### 🛠 Issues conocidos / Próximos pasos

* Consolidar tests automatizados.
* Añadir un notebook que genere un benchmark comparativo final (tabla resumen y figuras).
* Experimentar con técnicas de explainability (Grad-CAM) para la CNN.

---
=======
# VisionPorComputador-trabajo_03_v2
>>>>>>> fb8046cb2902f7e3f267f3acfde6ed6204fe4380
