# Linter Inteligente Explicable

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.93-2E8B57)](#métricas)

## Introducción

Un linter tradicional aplica reglas estáticas predefinidas y suele devolver un diagnóstico sin explicar qué partes del código influyeron en él. Este proyecto evoluciona ese enfoque mediante un clasificador profundo de código Python: transforma el programa en una secuencia de nodos de su **Árbol de Sintaxis Abstracta (AST)**, procesa la secuencia con una **BiLSTM** y utiliza un mecanismo de **atención** para ponderar los nodos más relevantes.

El resultado combina una categoría predicha con una explicación legible basada en los pesos de atención. Así, el modelo no solo clasifica la calidad o el tipo de problema, sino que también señala qué elementos estructurales del código contribuyeron al diagnóstico. [cite: 6]

## Métricas

Resultados finales validados sobre el conjunto de evaluación: [cite: 6]

| Métrica | Resultado |
| --- | ---: |
| Accuracy | 0.93 |
| Precision | 0.92 |
| Recall | 0.94 |
| F1-Score | 0.93 |

## Arquitectura

```text
Código Python
	|
	v
Parser AST (ast_parser.py)
	|
	v
Secuencia de nodos AST + tokenización
	|
	v
Embedding
	|
	v
BiLSTM bidireccional
	|
	+--------------------+
	|                    |
	v                    v
Salidas por nodo --> Atención aditiva
					 |
					 v
			  Vector de contexto
					 |
					 v
			Capas densas + softmax
					 |
		   +------------+-------------+
		   |                          |
		   v                          v
	  Categoría predicha       Relevancia por nodo
```

## Instalación

Se requiere Python 3.10 o superior. Desde la raíz del proyecto, cree un entorno virtual e instale las dependencias:

```bash
python -m venv venv
```

En Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source venv/bin/activate
```

Instale los paquetes del proyecto:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Coloque el checkpoint entrenado en `models/bilstm_attention_model.pth`. Sin este archivo, la aplicación puede iniciar, pero `/analyze` devolverá HTTP 503 porque no hay un modelo disponible.

## Uso rápido

### CLI

Analice un archivo Python con el modelo por defecto:

```bash
python src/cli.py --file ruta/al/archivo.py
```

Para usar otro checkpoint:

```bash
python src/cli.py --file ruta/al/archivo.py --model models/otro_modelo.pth
```

La CLI imprime la categoría detectada y los nodos AST ordenados por relevancia.

### API REST

Inicie el servidor FastAPI desde la raíz del proyecto:

```bash
uvicorn src.api:app --reload
```

La documentación interactiva queda disponible en `http://127.0.0.1:8000/docs`. Para analizar código, envíe una petición `POST` a `/analyze`:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code":"def suma(a, b):\n    return a + b"}'
```

Ejemplo de respuesta:

```json
{
  "category": "good",
  "confidence": 0.9341,
  "ast_sequence": "Module FunctionDef arguments ...",
  "attention_breakdown": [
    {"node": "FunctionDef", "relevance": "18.4%"},
    {"node": "Return", "relevance": "12.7%"}
  ]
}
```

También puede comprobar el estado del servicio con `GET /health`.

## Ejecución con Docker

Construya y ejecute la imagen desde la raíz del proyecto:

```bash
docker build -t linter-inteligente .
docker run --rm -p 8000:8000 linter-inteligente
```

## Estructura del repositorio

```text
.
├── data/                         # Datos de entrenamiento y evaluación
├── documentacion/                # Material complementario del proyecto
├── models/                       # Checkpoint entrenado (.pth)
├── notebooks/                    # Flujos de experimentación
│   └── colab_runner.ipynb
├── reports/                      # Reportes y resultados
├── src/
│   ├── api.py                    # API REST con FastAPI
│   ├── ast_parser.py             # Conversión de código a secuencia AST
│   ├── cli.py                    # Interfaz de línea de comandos
│   ├── dataset.py                # Preparación del dataset
│   ├── evaluate.py               # Evaluación de métricas
│   ├── explainability.py         # Inferencia y atención explicable
│   ├── model.py                  # BiLSTM + atención
│   └── train.py                  # Entrenamiento
├── Dockerfile
├── requirements.txt
└── README.md
```