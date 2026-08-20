import argparse
import torch
from model import GenericNeuralNetwork

def run_training(model_type, epochs, batch_size, lr):
    print(f"[INFO] Configurando modelo: {model_type}")
    print(f"[INFO] Hiperparámetros -> Épocas: {epochs} | Batch: {batch_size} | LR: {lr}")
    
    # Inicialización del modelo base
    model = GenericNeuralNetwork()
    print(f"[INFO] Red instanciada con éxito:\n{model}")
    
    print("[INFO] Simulación de entrenamiento finalizada. Guardando modelo en models/best_model.pth...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Genérico de Entrenamiento")
    parser.add_argument("--model_type", type=str, default="base", help="Tipo de arquitectura")
    parser.add_argument("--epochs", type=int, default=10, help="Cantidad de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje")
    
    args = parser.parse_args()
    run_training(args.model_type, args.epochs, args.batch_size, args.lr)