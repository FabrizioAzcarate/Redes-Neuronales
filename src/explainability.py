"""Herramientas de explicabilidad para el clasificador de código basado en AST."""

import ast
from pathlib import Path
import sys

import torch

# Permite ejecutar el archivo como ``python src/explainability.py`` o importarlo
# como ``src.explainability`` desde la raíz del proyecto.
try:
    from ast_parser import code_to_ast_sequence
    from model import get_code_model
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ast_parser import code_to_ast_sequence
    from model import get_code_model


CLASS_NAMES = ("Clean", "SyntaxError", "SecurityRisk", "BadPractice")


def _tokenize_ast_sequence(ast_sequence, tokenizer):
    """Convierte una secuencia AST en ids usando un tokenizer compatible."""
    tokens = ast_sequence.split()

    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(ast_sequence, add_special_tokens=False)
    elif callable(tokenizer):
        token_ids = tokenizer(tokens)
    elif isinstance(tokenizer, dict):
        unknown_id = tokenizer.get("<UNK>", tokenizer.get("[UNK]", 0))
        token_ids = [tokenizer.get(token, unknown_id) for token in tokens]
    else:
        raise TypeError(
            "tokenizer debe tener encode(), ser callable o ser un diccionario"
        )

    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().flatten().tolist()

    return tokens, [int(token_id) for token_id in token_ids]


def _node_labels(code_string, ast_sequence):
    """Obtiene las etiquetas AST para conservar una alineación verificable."""
    if ast_sequence in {"SyntaxError InvalidSyntax", "UnknownError"}:
        return ast_sequence.split()

    try:
        return [type(node).__name__ for node in ast.walk(ast.parse(code_string))]
    except (SyntaxError, TypeError):
        return ast_sequence.split()


def explain_code(code_string, model, tokenizer, device):
    """Predice una categoría e imprime la relevancia de cada nodo AST.

    Args:
        code_string: Fragmento de código Python a analizar.
        model: Modelo cuyo ``forward`` devuelve logits y pesos de atención.
        tokenizer: Objeto con ``encode``, callable o diccionario token -> id.
        device: Dispositivo de PyTorch, por ejemplo ``"cpu"`` o ``"cuda"``.

    Returns:
        Tupla ``(predicted_class, attention_weights)``.
    """
    ast_sequence = code_to_ast_sequence(code_string)
    nodes = _node_labels(code_string, ast_sequence)
    tokens, token_ids = _tokenize_ast_sequence(ast_sequence, tokenizer)

    if len(token_ids) != len(nodes):
        raise ValueError(
            "El tokenizer debe producir un id por nodo AST para poder explicar "
            "los pesos de atención."
        )

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits, attention_weights = model(input_tensor)
        predicted_index = int(torch.argmax(logits, dim=1).item())

    weights = attention_weights[0].detach().cpu().tolist()
    predicted_class = (
        CLASS_NAMES[predicted_index]
        if predicted_index < len(CLASS_NAMES)
        else str(predicted_index)
    )

    print(f"Secuencia AST: {ast_sequence}")
    print(f"Clase predicha: {predicted_class} ({predicted_index})")
    print("Relevancia por nodo AST:")
    for node, token, weight in zip(nodes, tokens, weights):
        print(f"  {node:<20} ({token:<20}) {weight:.2%}")

    return predicted_class, attention_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code = "result = eval(user_input)"
    ast_sequence = code_to_ast_sequence(code)
    vocabulary = {token: index + 1 for index, token in enumerate(ast_sequence.split())}
    vocabulary["<UNK>"] = 0

    model = get_code_model(num_classes=len(CLASS_NAMES))
    model_vocab_size = model.embedding.num_embeddings
    tokenizer = {
        token: token_id % model_vocab_size
        for token, token_id in vocabulary.items()
    }

    explain_code(code, model, tokenizer, device)
