import ast

def code_to_ast_sequence(code_str: str) -> str:
    """
    Convierte un string de código Python en una secuencia enriquecida de nodos AST,
    capturando operadores (Add, Sub, etc.) y tipos de literales (str, int) 
    para dotar de semántica real a la red neuronal.
    """
    try:
        parsed_ast = ast.parse(code_str)
        sequence = []
        for node in ast.walk(parsed_ast):
            node_type = type(node).__name__
            
            # Enriquecemos nodos críticos para que la red detecte operaciones y tipos
            if isinstance(node, ast.BinOp):
                op_name = type(node.op).__name__
                sequence.append(f"{node_type}_{op_name}")
            elif isinstance(node, ast.Constant):
                val_type = type(node.value).__name__
                sequence.append(f"{node_type}_{val_type}")
            else:
                sequence.append(node_type)
                
        return " ".join(sequence)
    except SyntaxError:
        return "SyntaxError InvalidSyntax"
    except Exception:
        return "UnknownError"

# Ejemplo rápido de prueba
if __name__ == "__main__":
    test_code = "precio + '15'"
    print("Secuencia AST enriquecida:", code_to_ast_sequence(test_code))