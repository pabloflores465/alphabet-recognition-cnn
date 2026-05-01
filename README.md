# 🔤 A-Z Alphabet Recognition — CNN + Webcam Demo

Reconocimiento de letras del alfabeto A-Z con red neuronal convolucional (CNN) y demo en tiempo real con webcam. Optimizado para Apple Silicon (M1/M2/M3) usando **MLX**.

---

## 📁 Estructura del proyecto

```
Project_AI/
├── README.md                       ← Este archivo
├── webcam_demo.py                  ← Demo principal con webcam 🎥
├── train_alphabet_cnn.py           ← Entrenar modelo base (99.2% acc)
├── train_robust_cnn.py             ← Entrenar modelo robusto (91.7% acc)
├── train_robust_v3.py              ← Entrenar v3 (manuscrito + impreso v2)
├── test_alphabet_cnn.py            ← Evaluación + visualizaciones
├── generate_printed_data.py        ← Generar datos impresos v1
├── generate_printed_data_v2.py     ← Generar datos impresos v2 (webcam-compatible)
├── models/                         ← Modelos entrenados (.safetensors)
├── output/                         ← Imágenes de visualización (.png)
├── printed_dataset/                ← Datos sintéticos impresos v1
├── printed_dataset_v2/             ← Datos sintéticos impresos v2
└── venv/                           ← Entorno virtual Python
```

---

## 🚀 Instalación

```bash
# 1. Clonar / entrar al proyecto
cd /Users/pabloflores/Documents/Project_AI

# 2. Crear entorno virtual (si no existe)
python3 -m venv venv

# 3. Activar entorno
source venv/bin/activate

# 4. Instalar dependencias
pip install mlx mlx-metal kagglehub numpy pillow opencv-python
```

> **Nota:** MLX solo funciona en Apple Silicon (M1/M2/M3/M4). Requiere macOS >= 13.3.

---

## ⬇️ Descargar dataset

El dataset se descarga automáticamente al entrenar. Si quieres descargarlo manualmente:

```bash
source venv/bin/activate
python3 -c "
import kagglehub
path = kagglehub.dataset_download('sachinpatel21/az-handwritten-alphabets-in-csv-format')
print('Dataset en:', path)
"
```

- **Dataset:** A-Z Handwritten Alphabets in .csv format
- **Tamaño:** ~667 MB (CSV), ~372,450 imágenes 28×28
- **Clases:** 26 (A-Z), etiquetas 0-25

---

## 🏋️ Entrenamiento

### Modelo base (manuscritas) — ⭐ 99.2% accuracy

```bash
source venv/bin/activate
python3 train_alphabet_cnn.py
```

- ~426K parámetros
- 5 épocas (~11 min en M1)
- Guarda: `models/alphabet_cnn_mlx.safetensors`

### Modelo robusto (manuscritas + impresas) — 91.7% accuracy

```bash
# Primero generar datos impresos
python3 generate_printed_data.py

# Luego entrenar
python3 train_robust_cnn.py
```

- ~492K parámetros
- 5 épocas (~17 min en M1)
- Guarda: `models/alphabet_robust.safetensors`

### Modelo robusto v3 (con preprocesamiento webcam-compatible)

```bash
# Generar datos impresos v2 (mejor preprocesamiento)
python3 generate_printed_data_v2.py

# Entrenar
python3 train_robust_v3.py
```

---

## 🧪 Evaluación y pruebas

```bash
source venv/bin/activate
python3 test_alphabet_cnn.py
```

Salida:
- Precisión global en test set
- Matriz de confusión por letra
- Ejemplos individuales con Top-3 predicciones
- Visualizaciones guardadas en `output/`
- Demo interactiva por terminal

---

## 🎥 Webcam Demo

### Opción 1: GUI (recomendada) 🖥️

```bash
source venv/bin/activate
python3 gui_app.py
```

Interfaz gráfica moderna con:
- Vista previa de webcam con ROI ajustable
- Predicción principal en grande + barras de confianza Top-3
- Historial de predicciones recientes
- Botones: Freeze, ROI Bigger/Smaller, Reset
- Atajos de teclado: flechas, +/-, F, R, Q/ESC

### Opción 2: OpenCV Window

```bash
source venv/bin/activate
python3 webcam_demo.py
```

Ventana OpenCV con los mismos controles.

### Controles

| Tecla | Acción |
|-------|--------|
| `S` | Alternar modo **ROI** / **Auto-detección** |
| `F` | Congelar / descongelar frame |
| `← ↑ ↓ →` | Mover la región de interés (ROI) |
| `+` / `-` | Aumentar / disminuir tamaño del ROI |
| `1` / `2` / `3` | Ajustar umbral de confianza (40% / 60% / 80%) |
| `R` | Resetear posición del ROI al centro |
| `Q` / `ESC` | Salir |

### Modos de funcionamiento

**Modo ROI (por defecto):**
- Recuadro verde en el centro de la pantalla
- Coloca la letra dentro del recuadro
- Muestra predicción principal + barras de confianza Top-3
- Vista previa del preprocesamiento (binarización) en esquina superior derecha

**Modo Auto-detección:**
- Detecta automáticamente contornos de caracteres en toda la imagen
- Útil para reconocer múltiples letras a la vez
- Umbral de confianza ajustable (teclas 1-3)

### Pipeline de preprocesamiento

```
Webcam (BGR)
  → Gris (GRAY)
  → CLAHE (mejora contraste)
  → OTSU Threshold (binarización)
  → Contornos (centrar carácter)
  → Redimensionar 28×28
  → Normalizar [0, 1]
  → CNN (inferencia)
```

---

## 📊 Resultados

### Modelo base — alphabet_cnn_mlx

| Métrica | Valor |
|---------|-------|
| Test Accuracy (manuscritas) | **99.21%** |
| Validation Accuracy | 99.23% |
| Train Accuracy | 99.37% |
| Parámetros | 426,010 |
| Tiempo entrenamiento | ~11 min (M1) |

### Reconocimiento de letras impresas (Arial, Times, Verdana...)

El mismo modelo reconoce letras impresas con alta precisión gracias al pipeline de preprocesamiento:

| Fuente | Accuracy |
|--------|----------|
| Verdana | 100% |
| Comic Sans MS | 100% |
| Arial | 96% |
| Georgia | 92% |
| Times New Roman | 88% |
| Courier New | 77% |
| Impact | 77% |

> **Nota:** El preprocesamiento (CLAHE + OTSU + centrado por contornos) convierte caracteres impresos a un formato similar a los manuscritos, permitiendo que el modelo los reconozca sin reentrenamiento.

### Letras con menor accuracy (base)

| Letra | Accuracy | Confusiones principales |
|-------|----------|------------------------|
| D | 94.6% | O (21), A (3) |
| I | 97.8% | N (1) |
| K | 97.9% | L (5) |

### Modelo robusto — alphabet_robust

| Métrica | Valor |
|---------|-------|
| Test Accuracy | **91.67%** |
| Validation Accuracy | 91.91% |
| Parámetros | 491,802 |
| Datos | 372K manuscritas + 52K impresas |

---

## 🧠 Arquitectura CNN

```
Input (28×28×1, NHWC)
  → Conv2D(1→32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → 14×14×32
  → Conv2D(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → 7×7×64
  → Conv2D(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → 3×3×128
  → Conv2D(128→256, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → 1×1×256
  → Flatten → Dense(256→128) + ReLU + Dropout(0.4)
  → Dense(128→26) + Softmax
```

---

## 🔧 Solución de problemas

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `No trained model found!`
Ejecuta el entrenamiento primero:
```bash
python3 train_alphabet_cnn.py
```

### Webcam no abre
- Verifica permisos de cámara en Preferencias del Sistema → Privacidad → Cámara
- Prueba con Photo Booth / FaceTime primero

### MLX no instalado (solo Apple Silicon)
```bash
pip install mlx mlx-metal
```

### Error de timeout en entrenamiento
El entrenamiento puede tomar >20 min. Ejecuta sin timeout:
```bash
python3 train_alphabet_cnn.py
```

---

## 📝 Notas

- **MLX** es el framework de Apple para machine learning en Apple Silicon. Usa la GPU/Neural Engine nativamente.
- El formato de imagen en MLX es **NHWC** (batch, height, width, channels), no NCHW como PyTorch.
- Los pesos de Conv2d en MLX tienen formato `(out_channels, kH, kW, in_channels)`.
- Para reentrenar con más épocas, edita la variable `EPOCHS` en el script correspondiente.
- El dataset original es un subset de NIST Special Database 19.

---

## 📚 Referencias

- [Kaggle Dataset: A-Z Handwritten Alphabets](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
