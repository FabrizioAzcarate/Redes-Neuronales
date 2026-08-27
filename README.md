# Clasificador de Errores y Calidad de Código en Python

## Descripción
Proyecto de Deep Learning enfocado en analizar fragmentos de código fuente y clasificar la presencia de errores sintácticos, malas prácticas o riesgos de seguridad mediante un modelo de lenguaje modular operado por un Agente autónomo en Google Colab.

## Arquitectura y Parámetros Base
- **Backbone:** ResNet18 preentrenada en ImageNet (capa final adaptada a 2 clases).
- **Optimizador:** Adam / SGD.
- **Función de Pérdida:** CrossEntropyLoss.
- **Entorno de Ejecución:** Google Colab (GPU T4).

## Estructura del Proyecto
- `AGENT.md`: Protocolo y reglas del agente autónomo.
- `SKILL.md`: Catálogo de habilidades del agente.
- `MEMORIA.md`: Registro histórico de experimentos y feedback loop.
- `COMMANDS.md`: Lista de comandos autorizados para ejecutar en Colab / Terminal.
- `src/`: Código fuente modularizado (`dataset.py`, `model.py`, `train.py`, `evaluate.py`).