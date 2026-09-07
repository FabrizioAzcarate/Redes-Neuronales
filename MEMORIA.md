# MEMORIA.md - Registro de Experimentos Persistente

## Estado Actual del Proyecto
- **Objetivo:** Clasificación de fragmentos de código Python (`Clean`, `SyntaxError`, `SecurityRisk`, `BadPractice`).
- **Métrica Objetivo:** F1-Score >= 0.88 en el conjunto de evaluación.
- **Estado:** **APROBADO** — Meta superada (F1-Score alcanzado: **0.89**).
- **Entorno:** Google Colab / GPU T4 (PyTorch).

## Historial de Experimentos

| ID Exp | Fecha | Modelo / Arq | LR | Batch Size | Épocas | Train Loss | Val Loss | F1-Score | Decisiones / Feedback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 2026-08-27 | CodeClassifier (BiLSTM Base) | - | - | - | - | - | - | Transición a clasificación de código configurada. |
| EXP-01 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0010 | 32 | 10 | 0.42 | 0.38 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Se sugiere reducir Learning Rate o aumentar Épocas. |
| EXP-02 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0005 | 32 | 15 | 0.40 | 0.36 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Requiere mayor número de épocas y menor tamaño de batch. |
| EXP-03 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0001 | 16 | 30 | 0.21 | 0.18 | **0.89** | **[ÉXITO]** F1-Score >= 0.88 alcanzado. Modelo validado y guardado en `models/bilstm_model.pth`. |

## Métricas Finales del Modelo Validado (EXP-03)
- **Accuracy:** 0.89
- **Precision:** 0.88
- **Recall:** 0.90
- **F1-Score:** 0.89

## Conclusiones y Próximos Pasos
- La arquitectura BiLSTM demostró alta capacidad para vectorizar y clasificar sintaxis y patrones de código Python en las 4 categorías definidas.
- El ajuste de hiperparámetros (reducción del *learning rate* a `0.0001` y aumento a `30` épocas con batch size de `16`) permitió converger al F1-Score de **0.89**, superando el umbral académico exigido.
- Se mantiene el modelo final exportado en `models/bilstm_model.pth` y la infografía estructurada guardada para la entrega del proyecto.