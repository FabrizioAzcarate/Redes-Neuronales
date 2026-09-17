# MEMORIA.md - Registro de Experimentos Persistente

## Estado Actual del Proyecto
- **Objetivo:** Clasificación de fragmentos de código Python (`Clean`, `SyntaxError`, `SecurityRisk`, `BadPractice`).
- **Métrica Objetivo:** F1-Score >= 0.88 en el conjunto de evaluación.
- **Estado:** **FASE 1 COMPLETADA** — Meta superada con arquitectura AST (F1-Score alcanzado: **0.91**).
- **Entorno:** Google Colab / GPU T4 (PyTorch).

## Historial de Experimentos

| ID Exp | Fecha | Modelo / Arq | LR | Batch Size | Épocas | Train Loss | Val Loss | F1-Score | Decisiones / Feedback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 2026-08-27 | CodeClassifier (BiLSTM Base) | - | - | - | - | - | - | Transición a clasificación de código configurada. |
| EXP-01 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0010 | 32 | 10 | 0.42 | 0.38 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Se sugiere reducir Learning Rate o aumentar Épocas. |
| EXP-02 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0005 | 32 | 15 | 0.40 | 0.36 | 0.85 | [FEEDBACK LOOP] F1-Score < 0.88. Requiere mayor número de épocas y menor tamaño de batch. |
| EXP-03 | 2026-09-07 | CodeClassifier (BiLSTM Base) | 0.0001 | 16 | 30 | 0.21 | 0.18 | 0.89 | **[ÉXITO]** F1-Score >= 0.88 alcanzado. Modelo base validado y guardado en `models/bilstm_model.pth`. |
| EXP-04 | 2026-09-17 | AST + BiLSTM | 0.0005 | 16 | 20 | 0.16 | 0.14 | **0.91** | **[ÉXITO]** Transformación a secuencias AST completada (`src/ast_parser.py`). F1-Score incrementado a 0.91. Modelo guardado en `models/bilstm_ast_model.pth`. |

## Métricas Finales del Modelo Validado (EXP-04 - Fase 1 AST)
- **Accuracy:** 0.91
- **Precision:** 0.90
- **Recall:** 0.92
- **F1-Score:** 0.91

## Conclusiones y Próximos Pasos
- La integración de secuencias de nodos del Árbol de Sintaxis Abstracta (AST) mediante `src/ast_parser.py` mejoró la extracción de características gramaticales, elevando el F1-Score de **0.89** a **0.91**.
- Se valida la persistencia del modelo de la Fase 1 en `models/bilstm_ast_model.pth`.
- **Próximos Pasos (Fase 2):** Implementación de mecanismos de atención (Self-Attention / Additive Attention) para aportar explicabilidad y generación del módulo `src/explainability.py`.