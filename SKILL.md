# SKILL.md - Catálogo de Habilidades del Agente

## Habilidad: `code-analysis-neural-pipeline`

### Capacidades del Agente

#### 1. Gestión y Tokenización de Código (`src/dataset.py`)
- Carga de fragmentos de código fuente en Python, tokenización de secuencias, padding/truncado y preparación de DataLoaders.

#### 2. Modelado de Secuencias de Código (`src/model.py`)
- Arquitectura en PyTorch con capas de Embedding, red recurrente bidireccional (BiLSTM) / Transformer y clasificador denso para 4 clases.

#### 3. Bucle de Entrenamiento Parametrizado (`src/train.py`)
- Control de épocas, tamaño de lote (*batch size*), tasa de aprendizaje (*learning rate*), función de pérdida (*CrossEntropyLoss*) y optimizador (*AdamW*).

#### 4. Evaluación de Métricas (`src/evaluate.py`)
- Cálculo e informe detallado de Accuracy, Precision, Recall y F1-Score general y por categoría de error.

#### 5. Registro de Bitácora Persistente (`MEMORIA.md`)
- Actualización sistemática del historial de experimentos para alimentar el *Feedback Loop*.