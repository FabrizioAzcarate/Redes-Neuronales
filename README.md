# Deep Learning Base Framework - Agente Autónomo

## Descripción del Proyecto
Este repositorio constituye una infraestructura base modular para la creación, entrenamiento, evaluación e iteración de modelos de redes neuronales utilizando PyTorch. Está diseñado específicamente para ser operado por un **Agente de IA Autónomo**, el cual puede ejecutar experimentos, ajustar hiperparámetros y registrar métricas de manera independiente.

## Estructura del Repositorio
- `AGENT.md`: Definición del rol, reglas de interacción y directrices del agente autónomo.
- `SKILL.md`: Descripción detallada de las capacidades y habilidades técnicas del agente.
- `MEMORIA.md`: Bitácora y registro de métricas de los experimentos realizados.
- `COMMANDS.md`: Lista de comandos autorizados para la ejecución en terminal.
- `src/`: Modulos de Python para la arquitectura, carga de datos, entrenamiento y evaluación.

## Guía de Configuración
1. Crear el entorno virtual:
   ```bash
   python -m venv venv --without-pip
   .\venv\Scripts\Activate.ps1
   python -m ensurepip
   python -m pip install --upgrade pip