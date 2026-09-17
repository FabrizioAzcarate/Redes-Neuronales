"""API REST para el Linter Inteligente de código Python."""

from pathlib import Path
import sys
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ast_parser import code_to_ast_sequence
from cli import _build_fallback_tokenizer, _load_checkpoint
from explainability import CLASS_NAMES, _tokenize_ast_sequence

MODEL_PATH = Path("models/bilstm_attention_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeRequest(BaseModel):
    """Código Python que será analizado por el linter."""

    code: str


app = FastAPI(
    title="Linter Inteligente de Código",
    description="Clasificación explicable de código Python mediante AST y BiLSTM.",
    version="1.0.0",
)

model: Optional[torch.nn.Module] = None
tokenizer = None
model_error: Optional[str] = None

if MODEL_PATH.is_file():
    try:
        model, tokenizer = _load_checkpoint(MODEL_PATH, DEVICE)
        model.eval()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        model_error = str(error)
else:
    model_error = f"No se encontró el modelo en {MODEL_PATH}"


@app.get("/health")
def health():
    """Comprueba que la API está disponible y si el modelo fue cargado."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/analyze")
def analyze(request: CodeRequest):
    """Analiza un fragmento de código y devuelve su explicación."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo no está disponible: {model_error}",
        )

    ast_sequence = code_to_ast_sequence(request.code)
    if tokenizer is None:
        active_tokenizer = _build_fallback_tokenizer(
            ast_sequence, model.embedding.num_embeddings
        )
    else:
        active_tokenizer = tokenizer

    try:
        nodes, token_ids = _tokenize_ast_sequence(ast_sequence, active_tokenizer)
        if not token_ids or len(nodes) != len(token_ids):
            raise ValueError("No se pudo alinear los tokens con los nodos AST.")

        inputs = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits, attention_weights = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predicted_index = int(torch.argmax(probabilities, dim=1).item())

        weights = attention_weights[0].detach().cpu().tolist()
        breakdown = sorted(
            zip(nodes, weights), key=lambda item: item[1], reverse=True
        )
        category = (
            CLASS_NAMES[predicted_index]
            if predicted_index < len(CLASS_NAMES)
            else str(predicted_index)
        )

        return {
            "category": category,
            "confidence": round(float(probabilities[0, predicted_index].item()), 4),
            "ast_sequence": ast_sequence,
            "attention_breakdown": [
                {"node": node, "relevance": f"{weight:.1%}"}
                for node, weight in breakdown
            ],
        }
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
