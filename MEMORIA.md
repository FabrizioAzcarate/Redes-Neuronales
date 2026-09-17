# MEMORIA.md - Registro de Experimentos Persistente

## Estado Actual del Proyecto
- **Objetivo:** Clasificación de fragmentos de código Python (`Clean`, `SyntaxError`, `SecurityRisk`, `BadPractice`).
- **Métrica Objetivo:** F1-Score >= 0.88 en el conjunto de evaluación.
- **Estado:** **FASE 2 COMPLETADA** — Mecanismo de atención y explicabilidad validados (**F1-Score: 0.93**).
- **Entorno:** Google Colab / GPU T4 (PyTorch).

## Historial de Experimentos

| ID Exp | Fecha | Modelo / Arq | LR | Batch Size | Épocas | Train Loss | Val Loss | F1-Score | Decisiones / Feedback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 2026-08-27 | CodeClassifier (BiLSTM Base) | - | - | - | - | - | - | Transición a clasificación de código configurada. |
| EXP-01 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0010 | 32 | 10 | 0.42 | 0.38 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Se sugiere reducir LR o aumentar Épocas. |
| EXP-02 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0005 | 32 | 15 | 0.40 | 0.36 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Requiere mayor número de épocas y menor batch size. |
| EXP-03 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0001 | 16 | 30 | 0.21 | 0.18 | 0.89 | [ÉXITO] F1-Score >= 0.88 alcanzado. Modelo base validado y guardado en `models/bilstm_model.pth`. |
| EXP-04 | 2026-09-17 | AST + BiLSTM | 0.0005 | 16 | 20 | 0.16 | 0.14 | 0.91 | [ÉXITO] Transformación a secuencias AST completada (`src/ast_parser.py`). Modelo guardado en `models/bilstm_ast_model.pth`. |
| EXP-05 | 2026-09-17 | AST + BiLSTM + Attention | 0.0005 | 16 | 20 | 0.12 | 0.10 | **0.93** | **[ÉXITO]** Incorporación de capa de atención y módulo `src/explainability.py`. Modelo guardado en `models/bilstm_attention_model.pth`. |

## Métricas Finales del Modelo Validado (EXP-05 - Fase 2 Atención)
- **Accuracy:** 0.93
- **Precision:** 0.92
- **Recall:** 0.94
- **F1-Score:** 0.93

## Conclusiones y Próximos Pasos
- El mecanismo de atención incrementó el F1-Score a **0.93** y aporta mapas de relevancia sintáctica por nodo AST.
- **Próximos Pasos (Fase 3):** Prototipado y despliegue local mediante interfaz CLI / API REST ligera.