"""API REST para el Linter Inteligente de código Python."""

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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


class RunRequest(CodeRequest):
    """Código y valores opcionales para las llamadas a input()."""

    inputs: list[str] = Field(default_factory=list)


app = FastAPI(
    title="Linter Inteligente de Código",
    description="Clasificación explicable de código Python mediante AST y BiLSTM.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/")
def serve_frontend():
    """Sirve el frontend desde la raíz del proyecto o desde src."""
    project_root = Path(__file__).resolve().parent.parent
    html_path = project_root / "index.html"
    if not html_path.is_file():
        html_path = Path(__file__).resolve().parent / "index.html"
    if not html_path.is_file():
        html_path = Path(__file__).resolve().parent / "app.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend no encontrado")
    return FileResponse(html_path)


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
    try:
        ast.parse(request.code)
    except SyntaxError as error:
        line = error.lineno or 1
        column = error.offset or 1
        description = error.msg or "error sintáctico no especificado"
        message = f"Error de sintaxis en la línea {line}, columna {column}: {description}."
        return {
            "category": "SyntaxError",
            "confidence": 1.0,
            "code": request.code,
            "ast_sequence": "SyntaxError InvalidSyntax",
            "issues": [
                {
                    "severity": "error",
                    "code": "SyntaxError",
                    "message": message,
                    "node": None,
                    "line": line,
                }
            ],
            "attention_breakdown": [],
            "logs": ["Validación AST fallida", message],
        }

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo no está disponible: {model_error}",
        )

    confidence_threshold = 0.60
    logs = ["Inicio del análisis de código"]
    ast_sequence = code_to_ast_sequence(request.code)
    logs.append("Parsing AST completado")
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
        logs.append("Inferencia BiLSTM exitosa")

        weights = attention_weights[0].detach().cpu().tolist()
        breakdown = sorted(
            zip(nodes, weights), key=lambda item: item[1], reverse=True
        )
        category = (
            CLASS_NAMES[predicted_index]
            if predicted_index < len(CLASS_NAMES)
            else str(predicted_index)
        )
        confidence = float(probabilities[0, predicted_index].item())
        is_error_category = category.lower() not in {"clean", "good", "ok"}
        logs.append("Cálculo de pesos de atención aditiva completado")
        if confidence < confidence_threshold:
            clean_message = (
                "No se detectaron anomalías críticas en el flujo; "
                "el código supera la validación sintáctica."
            )
            return {
                "category": "Clean",
                "confidence": round(confidence, 4),
                "code": request.code,
                "ast_sequence": ast_sequence,
                "issues": [
                    {
                        "severity": "info",
                        "code": "Clean",
                        "message": clean_message,
                        "node": None,
                        "line": None,
                    }
                ],
                "attention_breakdown": [],
                "logs": logs + ["Confianza inferior al umbral; análisis considerado limpio"],
            }

        if not is_error_category:
            clean_message = "No se detectaron anomalías críticas en el flujo."
            return {
                "category": category,
                "confidence": round(confidence, 4),
                "code": request.code,
                "ast_sequence": ast_sequence,
                "issues": [
                    {
                        "severity": "info",
                        "code": category,
                        "message": clean_message,
                        "node": None,
                        "line": None,
                    }
                ],
                "attention_breakdown": [],
                "logs": logs + ["El modelo clasificó el código como limpio"],
            }

        input_lines = [
            line_number
            for line_number, line in enumerate(request.code.splitlines(), start=1)
            if "input(" in line
        ]
        issues = [
            {
                "severity": "info" if category.lower() in {"good", "ok"} else "warning",
                "code": category,
                "message": f"Clasificación detectada: {category}.",
                "node": breakdown[0][0] if breakdown else None,
                "line": input_lines[0] if input_lines else None,
            }
        ]

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "code": request.code,
            "ast_sequence": ast_sequence,
            "issues": issues,
            "attention_breakdown": [
                {"node": node, "relevance": f"{weight:.1%}"}
                for node, weight in breakdown
            ],
            "logs": logs + ["Análisis finalizado"],
        }
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/run")
def run_code(request: RunRequest):
    """Ejecuta código en un proceso aislado con un límite de tres segundos."""
    script_path: Optional[Path] = None
    user_inputs_string = "\n".join(request.inputs)
    if request.inputs:
        user_inputs_string += "\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py", delete=False
        ) as script:
            script.write(request.code)
            script_path = Path(script.name)

        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(script_path)],
            input=user_inputs_string,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            shell=False,
            cwd=script_path.parent,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "return_code": completed.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "stdout": stdout,
            "stderr": stderr
            + "\nEjecución cancelada: superó el límite de 3 segundos.",
            "return_code": None,
            "timed_out": True,
        }
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500, detail=f"No se pudo ejecutar el código: {error}"
        ) from error
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
