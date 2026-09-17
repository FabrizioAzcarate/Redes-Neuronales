import argparse
import pandas as pd
from torch.utils.data import Dataset

from ast_parser import code_to_ast_sequence

class CodeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def preprocess_dataset(input_csv="data/raw_dataset.csv", output_csv="data/ast_dataset.csv"):
    df = pd.read_csv(input_csv)

    print("[INFO] Convirtiendo fragmentos de código a secuencias AST...")
    df["ast_sequence"] = df["code"].apply(code_to_ast_sequence)

    df.to_csv(output_csv, index=False)
    print(f"[ÉXITO] Dataset procesado con AST guardado en: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento del dataset de código")
    parser.add_argument("--input-csv", default="data/raw_dataset.csv", help="CSV de entrada")
    parser.add_argument("--output-csv", default="data/ast_dataset.csv", help="CSV procesado de salida")
    args = parser.parse_args()
    preprocess_dataset(args.input_csv, args.output_csv)