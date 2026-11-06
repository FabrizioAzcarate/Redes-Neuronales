"""Herramientas de explicabilidad robustas para el clasificador de código basado en AST."""

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


CLASS_NAMES = ("Clean", "SyntaxError", "SecurityRisk")


def class_name_from_index(predicted_index):
    """Mapea salidas del modelo a categorías públicas conservadoras.

    El índice 3 corresponde a ``BadPractice`` en checkpoints antiguos. Se
    conserva como salida del modelo para poder cargar esos checkpoints, pero
    se clasifica públicamente como código limpio.
    """
    if predicted_index < len(CLASS_NAMES):
        return CLASS_NAMES[predicted_index]
    return "Clean"


def has_critical_security_risk(code_string):
    """Detecta únicamente llamadas AST consideradas riesgos críticos."""
    try:
        tree = ast.parse(code_string)
    except (SyntaxError, TypeError):
        return False

    critical_names = {"eval", "exec"}
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in critical_names
        for node in ast.walk(tree)
    )


def _tokenize_ast_sequence(ast_sequence, tokenizer):
    """Convierte una secuencia AST en ids usando un tokenizer compatible de forma segura."""
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
        token_ids = token_ids.get("input_ids", [])
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().flatten().tolist()

    return tokens, [int(token_id) for token_id in token_ids]


def _node_labels(code_string, ast_sequence):
    """Obtiene las etiquetas AST de manera segura y ordenada."""
    if ast_sequence in {"SyntaxError InvalidSyntax", "UnknownError"}:
        return ast_sequence.split()

    try:
        # Recorrido lineal estricto utilizando ast.NodeVisitor o iter_child_nodes
        # para mantener coherencia con la secuencia de tokens.
        nodes = []
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            nodes.append(type(node).__name__)
        return nodes
    except (SyntaxError, TypeError):
        return ast_sequence.split()


def explain_code(code_string, model, tokenizer, device):
    """Predice una categoría y calcula la relevancia explicativa de cada nodo AST.

    Args:
        code_string: Fragmento de código Python a analizar.
        model: Modelo de PyTorch entrenado.
        tokenizer: Tokenizador compatible.
        device: Dispositivo ("cpu" o "cuda").

    Returns:
        Tupla ``(predicted_class, attention_weights)``.
    """
    ast_sequence = code_to_ast_sequence(code_string)
    nodes = _node_labels(code_string, ast_sequence)
    tokens, token_ids = _tokenize_ast_sequence(ast_sequence, tokenizer)

    # Blindaje contra discrepancias de longitud entre tokens y nodos
    if len(token_ids) != len(nodes):
        min_len = min(len(token_ids), len(nodes))
        token_ids = token_ids[:min_len]
        nodes = nodes[:min_len]
        tokens = tokens[:min_len]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits, attention_weights = model(input_tensor)
        predicted_index = int(torch.argmax(logits, dim=1).item())

    weights = attention_weights[0].detach().cpu().tolist()
    predicted_class = class_name_from_index(predicted_index)

    print(f"Secuencia AST: {ast_sequence}")
    print(f"Clase predicha: {predicted_class} ({predicted_index})")
    print("Relevancia por nodo AST (Atención Aditiva):")
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