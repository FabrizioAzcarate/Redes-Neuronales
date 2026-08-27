# MEMORIA.md - Registro de Experimentos Persistente

## Estado Actual del Proyecto
- **Objetivo:** Clasificación de fragmentos de código Python (`Clean`, `SyntaxError`, `SecurityRisk`, `BadPractice`).
- **Métrica Objetivo:** F1-Score >= 0.88 en el conjunto de evaluación.
- **Entorno:** Google Colab / GPU T4.

## Historial de Experimentos

| ID Exp | Fecha | Modelo / Arq | LR | Batch Size | Épocas | Train Loss | Val Loss | F1-Score | Decisiones / Feedback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 2026-08-27 | CodeClassifier (BiLSTM Base) | - | - | - | - | - | - | Transición a clasificación de código configurada. |

## Conclusiones y Próximos Pasos
- Ejecutar el pipeline en Google Colab para obtener las métricas base del primer experimento.