import argparse

def evaluate_model(model_path):
    print(f"[INFO] Cargando modelo desde: {model_path}")
    print("[INFO] Calculando métricas en el conjunto de prueba...")
    # Lógica de evaluación genérica
    print("[RESULTADOS] Accuracy: 0.00 | Loss: 0.00 | F1-Score: 0.00")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Evaluación")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Ruta al checkpoint")
    args = parser.parse_args()
    evaluate_model(args.model_path)