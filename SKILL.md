# SKILL.md - Catálogo de Habilidades del Agente

## Habilidad: `neural-network-lifecycle`

### Capacidades del Agente

#### 1. Gestión de Datos y Canalizaciones (`src/dataset.py`)
- Carga de datos, división en conjuntos (train/val/test) y aplicación de transformaciones/normalización.

#### 2. Modelado y Arquitectura (`src/model.py`)
- Definición de bloques de redes neuronales (convolucionales, densas o recurrentes) con soporte para parametrización.

#### 3. Bucle de Entrenamiento y Optimización (`src/train.py`)
- Ejecución de ciclos de entrenamiento (*training loops*), cálculo de funciones de pérdida, optimizadores (Adam, SGD) y guardado del mejor *checkpoint*.

#### 4. Evaluación de Rendimiento (`src/evaluate.py`)
- Cálculo de métricas principales (Loss, Accuracy, Precision, Recall, F1-Score) y exportación de reportes de desempeño.

#### 5. Registro de Bitácora (`MEMORIA.md`)
- Actualización sistemática del historial de experimentos para garantizar la trazabilidad del desarrollo.