"""CLI para el análisis explicable de código Python."""

import argparse
from pathlib import Path
import sys

import torch

# Permite ejecutar el script desde la raíz del proyecto con ``python src/cli.py``.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ast_parser import code_to_ast_sequence
from explainability import CLASS_NAMES, explain_code
from model import CodeClassifier

DEFAULT_MODEL_PATH = Path("models/bilstm_attention_model.pth")


def _load_checkpoint(model_path, device):
    """Carga el modelo y devuelve también el tokenizer guardado, si existe."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        tokenizer = checkpoint.get("tokenizer", checkpoint.get("vocabulary"))
        model_kwargs = checkpoint.get("model_kwargs", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        tokenizer = None
        model_kwargs = {}
    else:
        raise ValueError("El checkpoint no contiene un state_dict válido.")

    model = CodeClassifier(
        num_classes=model_kwargs.get("num_classes", len(CLASS_NAMES)),
        vocab_size=model_kwargs.get("vocab_size", 5000),
        embed_dim=model_kwargs.get("embed_dim", 128),
        hidden_dim=model_kwargs.get("hidden_dim", 128),
    )
    model.load_state_dict(state_dict)
    return model.to(device), tokenizer


def _build_fallback_tokenizer(ast_sequence, vocab_size):
    """Construye un vocabulario reproducible cuando el checkpoint no lo incluye."""
    vocabulary = {"<UNK>": 0}
    for token in ast_sequence.split():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary) % vocab_size
    return vocabulary


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Linter inteligente explicable para código Python"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Ruta al archivo Python que se desea analizar",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta al checkpoint del modelo (por defecto: models/bilstm_attention_model.pth)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    file_path = Path(args.file)
    model_path = Path(args.model)

    if not file_path.is_file():
        print(f"[ERROR] No se encontró el archivo: {file_path}", file=sys.stderr)
        return 2

    try:
        code = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"[ERROR] No se pudo leer el archivo '{file_path}': {error}", file=sys.stderr)
        return 2

    if not model_path.is_file():
        print(
            f"[ERROR] No se encontró el modelo: {model_path}\n"
            "Genera o copia el checkpoint entrenado en esa ruta, "
            "o usa --model para indicar otra ubicación.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, tokenizer = _load_checkpoint(model_path, device)
        ast_sequence = code_to_ast_sequence(code)
        if tokenizer is None:
            tokenizer = _build_fallback_tokenizer(
                ast_sequence, model.embedding.num_embeddings
            )

        predicted_class, attention_weights = explain_code(
            code, model, tokenizer, device
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"[ERROR] No se pudo analizar el archivo: {error}", file=sys.stderr)
        return 1

    nodes = ast_sequence.split()
    weights = attention_weights[0].detach().cpu().tolist()
    ranked_nodes = sorted(zip(nodes, weights), key=lambda item: item[1], reverse=True)

    print("\n" + "=" * 64)
    print("           INFORME DEL LINTER INTELIGENTE")
    print("=" * 64)
    print(f"Archivo:    {file_path}")
    print(f"Diagnóstico: {predicted_class}")
    print(f"Dispositivo: {device}")
    print("\nNodos AST más críticos:")
    print(f"{'#':>3}  {'Nodo':<24} {'Relevancia':>12}")
    print("-" * 44)
    for index, (node, weight) in enumerate(ranked_nodes, start=1):
        print(f"{index:>3}  {node:<24} {weight:>11.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
