import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Atención aditiva para ponderar las salidas de una BiLSTM de forma explicable."""

    def __init__(self, input_dim: int, attention_dim: int = None):
        super().__init__()
        attention_dim = attention_dim or input_dim
        # Capas de proyección para el mecanismo de atención aditiva (Bahdanau-style)
        self.projection = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_outputs: torch.Tensor):
        """
        Calcula un contexto ponderado y los pesos de atención normalizados.

        Args:
            sequence_outputs: Tensor con forma (batch, seq_len, input_dim).

        Returns:
            Una tupla con el contexto (batch, input_dim) y los pesos
            normalizados (batch, seq_len).
        """
        # Proyección y activación no lineal
        energy = torch.tanh(self.projection(sequence_outputs))
        # Puntuaciones escalares para cada elemento de la secuencia
        scores = self.score(energy).squeeze(-1)  # (batch, seq_len)
        # Normalización mediante softmax para obtener pesos de atención sumando 1.0
        attention_weights = torch.softmax(scores, dim=1)

        # Combinación lineal ponderada de las salidas de la secuencia
        context = torch.sum(sequence_outputs * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights


class CodeClassifier(nn.Module):
    """Red neuronal robusta basada en BiLSTM y Atención para clasificación y explicabilidad de código Python."""

    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if hidden_dim > 1 else 0.0,
        )
        # Como la LSTM es bidireccional, la dimensión de salida se duplica (hidden_dim * 2)
        self.attention = AttentionModule(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Paso hacia adelante (forward pass) de la red.
        Args:
            x: Tensor de enteros con los IDs de tokens (batch, seq_len).
        """
        embedded = self.embedding(x)
        sequence_outputs, _ = self.lstm(embedded)
        context, attention_weights = self.attention(sequence_outputs)
        logits = self.fc(context)
        return logits, attention_weights


def get_code_model(num_classes: int = 4) -> CodeClassifier:
    return CodeClassifier(num_classes=num_classes)


if __name__ == "__main__":
    model = get_code_model()
    print("[INFO] Arquitectura CodeClassifier (BiLSTM + Atención Aditiva) optimizada correctamente:")
    print(model)