"""Validaciones AST deterministas para errores funcionales evidentes."""

import ast
from typing import Optional


_NUMBER_TYPES = {"bool", "int", "float", "complex"}
_INTEGER_TYPES = {"bool", "int"}
_NON_COMPLEX_NUMBER_TYPES = {"bool", "int", "float"}
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


def _annotation_type(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Name) and node.id in {
        "bool",
        "int",
        "float",
        "complex",
        "str",
        "bytes",
        "list",
        "tuple",
    }:
        return node.id
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id in {"list", "tuple"}:
            return node.value.id
    return None


def _is_numeric(type_name: Optional[str]) -> bool:
    return type_name in _NUMBER_TYPES


def _is_integer(type_name: Optional[str]) -> bool:
    return type_name in _INTEGER_TYPES


def _invalid_binary_operation(
    operator: ast.operator,
    left_type: Optional[str],
    right_type: Optional[str],
) -> bool:
    if left_type is None or right_type is None:
        return False

    if isinstance(operator, ast.Add):
        return (
            not (
                (_is_numeric(left_type) and _is_numeric(right_type))
                or (
                    left_type == right_type
                    and left_type in _SEQUENCE_TYPES
                )
            )
        )
    if isinstance(operator, (ast.Sub, ast.Div, ast.Pow)):
        return not (_is_numeric(left_type) and _is_numeric(right_type))
    if isinstance(operator, (ast.FloorDiv, ast.Mod)):
        return not (
            left_type in _NON_COMPLEX_NUMBER_TYPES
            and right_type in _NON_COMPLEX_NUMBER_TYPES
        )
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
    if isinstance(operator, (ast.LShift, ast.RShift)):
        return not (_is_integer(left_type) and _is_integer(right_type))
    if isinstance(operator, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return not (_is_integer(left_type) and _is_integer(right_type))
    return False


def _binary_result_type(
    operator: ast.operator,
    left_type: Optional[str],
    right_type: Optional[str],
) -> Optional[str]:
    if left_type is None or right_type is None:
        return None
    if isinstance(operator, ast.Div):
        return "complex" if "complex" in {left_type, right_type} else "float"
    if isinstance(operator, ast.Mult):
        if left_type in _SEQUENCE_TYPES:
            return left_type
        if right_type in _SEQUENCE_TYPES:
            return right_type
    if isinstance(operator, (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)):
        return "int"
    if _is_numeric(left_type) and _is_numeric(right_type):
        if "complex" in {left_type, right_type}:
            return "complex"
        if "float" in {left_type, right_type}:
            return "float"
        return "int"
    if left_type == right_type:
        return left_type
    return None


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
            operand_type = self._infer_type(node.operand)
            if operand_type is None:
                return None
            if isinstance(node.op, (ast.UAdd, ast.USub)) and not _is_numeric(
                operand_type
            ):
                self._set_error(node, operand_type, None)
                return None
            if isinstance(node.op, ast.Invert) and not _is_integer(operand_type):
                self._set_error(node, operand_type, None)
                return None
            if isinstance(node.op, ast.Not):
                return "bool"
            return operand_type
        if isinstance(node, ast.BinOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if _invalid_binary_operation(node.op, left_type, right_type):
                self._set_error(node, left_type, right_type)
                return None
            if (
                isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float, complex))
                and node.right.value == 0
            ):
                self._set_error(
                    node,
                    left_type,
                    right_type,
                    error_code="ZeroDivisionError",
                    message="División por cero detectada estáticamente.",
                )
                return None
            return _binary_result_type(node.op, left_type, right_type)
        return None

    def _set_error(
        self,
        node: ast.BinOp | ast.UnaryOp | ast.AugAssign,
        left_type: Optional[str],
        right_type: Optional[str],
        *,
        error_code: str = "TypeError",
        message: Optional[str] = None,
    ) -> None:
        if self.error is not None:
            return
        operator = type(node.op).__name__
        self.error = {
            "error": error_code,
            "message": message
            or (
                f"Operación inválida: no se puede aplicar {operator} entre "
                f"'{left_type}' y '{right_type}'."
            ),
            "line": node.lineno,
            "column": node.col_offset + 1,
            "node": type(node).__name__,
        }

    def visit_Assign(self, node: ast.Assign) -> None:
        value_type = self._infer_type(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value_type is None:
                    self.known_types.pop(target.id, None)
                else:
                    self.known_types[target.id] = value_type
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value_type = self._infer_type(node.value) if node.value is not None else None
        if isinstance(node.target, ast.Name):
            inferred_type = value_type or _annotation_type(node.annotation)
            if inferred_type is None:
                self.known_types.pop(node.target.id, None)
            else:
                self.known_types[node.target.id] = inferred_type
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        left_type = self._infer_type(node.target)
        right_type = self._infer_type(node.value)
        if _invalid_binary_operation(node.op, left_type, right_type):
            self._set_error(node, left_type, right_type)
        elif isinstance(node.target, ast.Name):
            result_type = _binary_result_type(node.op, left_type, right_type)
            if result_type is None:
                self.known_types.pop(node.target.id, None)
            else:
                self.known_types[node.target.id] = result_type
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._infer_type(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_suite_isolated(node.body)
        self._visit_suite_isolated(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_suite_isolated(node.body)
        self._visit_suite_isolated(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_suite_isolated(node.body)
        self._visit_suite_isolated(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_suite_isolated(node.body)
        self._visit_suite_isolated(node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_suite_isolated(node.body)
        for handler in node.handlers:
            self._visit_suite_isolated([handler])
        self._visit_suite_isolated(node.orelse)
        self._visit_suite_isolated(node.finalbody)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self._visit_suite_isolated([case])

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for expression in node.decorator_list + node.args.defaults + [
            default for default in node.args.kw_defaults if default is not None
        ]:
            self.visit(expression)

        previous_types = self.known_types
        self.known_types = {}
        try:
            arguments = (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                argument_type = _annotation_type(argument.annotation)
                if argument_type is not None:
                    self.known_types[argument.arg] = argument_type
            for statement in node.body:
                self.visit(statement)
        finally:
            self.known_types = previous_types

    def _visit_suite_isolated(self, statements: list[ast.AST]) -> None:
        previous_types = self.known_types
        self.known_types = previous_types.copy()
        try:
            for statement in statements:
                self.visit(statement)
        finally:
            self.known_types = previous_types


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
