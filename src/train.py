"""Entrenamiento del clasificador de código con BiLSTM y atención."""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ast_parser import code_to_ast_sequence
from dataset import CodeDataset
from explainability import CLASS_NAMES
from model import CodeClassifier

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bilstm_attention_model.pth"
DEFAULT_DATA_PATHS = (
    PROJECT_ROOT / "data" / "ast_dataset.csv",
    PROJECT_ROOT / "data" / "raw_dataset.csv",
)

def _find_dataset():
    for path in DEFAULT_DATA_PATHS:
        if path.is_file():
            return path
    paths = ", ".join(str(path) for path in DEFAULT_DATA_PATHS)
    raise FileNotFoundError(
        f"No se encontró un dataset. Cree uno en una de estas rutas: {paths}"
    )

def _load_samples(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    code_column = "code" if "code" in dataframe.columns else None
    sequence_column = "ast_sequence" if "ast_sequence" in dataframe.columns else None
    label_column = next(
        (column for column in ("label", "category", "target", "class") if column in dataframe.columns),
        None,
    )

    if not (code_column or sequence_column):
        raise ValueError("El dataset debe contener una columna 'code' o 'ast_sequence'.")
    if label_column is None:
        raise ValueError("El dataset debe contener una columna de etiquetas: 'label', 'category', 'target' o 'class'.")

    label_lookup = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
    sequences = []
    labels = []
    for _, row in dataframe.iterrows():
        sequence = row[sequence_column] if sequence_column else code_to_ast_sequence(row[code_column])
        if pd.isna(sequence) or not str(sequence).strip():
            continue
        raw_label = row[label_column]
        if pd.isna(raw_label):
            continue
        if isinstance(raw_label, str):
            normalized_label = raw_label.strip().lower()
            if normalized_label not in label_lookup:
                raise ValueError(
                    f"Etiqueta desconocida '{raw_label}'. Use: {', '.join(CLASS_NAMES)}."
                )
            label = label_lookup[normalized_label]
        else:
            label = int(raw_label)
            if not 0 <= label < len(CLASS_NAMES):
                raise ValueError(f"La etiqueta numérica debe estar entre 0 y {len(CLASS_NAMES) - 1}.")
        sequences.append(str(sequence).split())
        labels.append(label)

    if not sequences:
        raise ValueError("El dataset no contiene muestras válidas para entrenar.")
    return sequences, labels

def _build_tokenizer(sequences):
    tokenizer = {"<UNK>": 0}
    for sequence in sequences:
        for token in sequence:
            if token not in tokenizer:
                tokenizer[token] = len(tokenizer)
    return tokenizer

def _make_dataset(sequences, labels, tokenizer):
    encoded = [
        torch.tensor([tokenizer.get(token, 0) for token in sequence], dtype=torch.long)
        for sequence in sequences
    ]
    inputs = pad_sequence(encoded, batch_first=True, padding_value=0)
    return CodeDataset(list(zip(inputs, torch.tensor(labels, dtype=torch.long))))

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def train(epochs, batch_size, learning_rate, model_path=DEFAULT_MODEL_PATH):
    dataset_path = _find_dataset()
    sequences, labels = _load_samples(dataset_path)
    tokenizer = _build_tokenizer(sequences)
    training_dataset = _make_dataset(sequences, labels, tokenizer)
    loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeClassifier(
        vocab_size=len(tokenizer),
        num_classes=len(CLASS_NAMES),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"[INFO] Entrenamiento iniciado en {device} | muestras: {len(training_dataset)}")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"[ÉPOCA {epoch}/{epochs}] loss: {loss:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "model_kwargs": {
                "vocab_size": len(tokenizer),
                "num_classes": len(CLASS_NAMES),
            },
        },
        model_path,
    )
    print(f"[ÉXITO] Pesos guardados en: {model_path}")
    return model

def _parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del linter inteligente explicable")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("--epochs y --batch_size deben ser positivos; --lr debe ser mayor que cero.")
    train(args.epochs, args.batch_size, args.lr)
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_waste_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_waste_model().to(device)
    print(f"[INFO] Entrenamiento iniciado en {device} | LR: {args.lr} | Batch: {args.batch_size}")