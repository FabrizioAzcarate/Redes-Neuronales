"""Validaciones AST deterministas para errores funcionales evidentes."""

import ast
from typing import Optional


_NUMBER_TYPES = {"bool", "int", "float", "complex"}
_SEQUENCE_TYPES = {"str", "bytes", "list", "tuple"}


def _literal_type(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "NoneType"
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, float):
            return "float"
        if isinstance(node.value, complex):
            return "complex"
        if isinstance(node.value, str):
            return "str"
        if isinstance(node.value, bytes):
            return "bytes"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    return None


def _is_numeric(type_name: Optional[str]) -> bool:
    return type_name in _NUMBER_TYPES


def _invalid_binary_operation(
    operator: ast.operator,
    left_type: Optional[str],
    right_type: Optional[str],
) -> bool:
    if left_type is None or right_type is None:
        return False

    if isinstance(operator, ast.Add):
        return (
            left_type != right_type
            and not (_is_numeric(left_type) and _is_numeric(right_type))
        )
    if isinstance(operator, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        return not (_is_numeric(left_type) and _is_numeric(right_type))
    if isinstance(operator, ast.Mult):
        return not (
            (_is_numeric(left_type) and _is_numeric(right_type))
            or (
                left_type in _SEQUENCE_TYPES
                and right_type == "int"
            )
            or (
                right_type in _SEQUENCE_TYPES
                and left_type == "int"
            )
        )
    return False


class _FunctionalErrorVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.known_types: dict[str, str] = {}
        self.error: Optional[dict[str, object]] = None

    def _infer_type(self, node: ast.AST) -> Optional[str]:
        literal_type = _literal_type(node)
        if literal_type is not None:
            return literal_type
        if isinstance(node, ast.Name):
            return self.known_types.get(node.id)
        if isinstance(node, ast.UnaryOp):
            return self._infer_type(node.operand)
        if isinstance(node, ast.BinOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if _invalid_binary_operation(node.op, left_type, right_type):
                self._set_error(node, left_type, right_type)
                return None
            if left_type == right_type:
                return left_type
            if _is_numeric(left_type) and _is_numeric(right_type):
                return "complex" if "complex" in {left_type, right_type} else "float"
        return None

    def _set_error(
        self,
        node: ast.BinOp,
        left_type: Optional[str],
        right_type: Optional[str],
    ) -> None:
        if self.error is not None:
            return
        operator = type(node.op).__name__
        self.error = {
            "error": "TypeError",
            "message": (
                f"Operación inválida: no se puede aplicar {operator} entre "
                f"'{left_type}' y '{right_type}'."
            ),
            "line": node.lineno,
            "column": node.col_offset + 1,
        }

    def visit_Assign(self, node: ast.Assign) -> None:
        value_type = self._infer_type(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and value_type is not None:
                self.known_types[target.id] = value_type
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value_type = self._infer_type(node.value) if node.value is not None else None
        if isinstance(node.target, ast.Name) and value_type is not None:
            self.known_types[node.target.id] = value_type
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._infer_type(node)
        self.generic_visit(node)


def validate_functional_errors(code_str: str) -> Optional[dict[str, object]]:
    """Devuelve el primer error funcional que pueda probarse estáticamente."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as error:
        return {
            "error": "SyntaxError",
            "message": error.msg or "Error de sintaxis.",
            "line": error.lineno or 1,
            "column": error.offset or 1,
        }

    visitor = _FunctionalErrorVisitor()
    visitor.visit(tree)
    return visitor.error
