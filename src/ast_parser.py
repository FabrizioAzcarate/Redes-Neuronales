import ast

def code_to_ast_sequence(code_str: str) -> str:
    """
    Convierte un string de código Python en una secuencia de nodos AST.
    Si el código tiene error de sintaxis, captura la excepción y retorna el fallo sintáctico.
    """
    try:
        parsed_ast = ast.parse(code_str)
        nodes = [type(node).__name__ for node in ast.walk(parsed_ast)]
        return " ".join(nodes)
    except SyntaxError:
        # Si el código falla en sintaxis (categoría SyntaxError), devolvemos el evento
        return "SyntaxError InvalidSyntax"
    except Exception:
        return "UnknownError"

# Ejemplo rápido de prueba
if __name__ == "__main__":
    test_code = "if x > 0:\n    print('Hola')"
    print("Secuencia AST generada:", code_to_ast_sequence(test_code))