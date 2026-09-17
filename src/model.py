import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Atención aditiva para ponderar las salidas de una BiLSTM."""

    def __init__(self, input_dim, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or input_dim
        self.projection = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_outputs):
        """
        Calcula un contexto ponderado y los pesos de atención.

        Args:
            sequence_outputs: Tensor con forma (batch, seq_len, input_dim).

        Returns:
            Una tupla con el contexto (batch, input_dim) y los pesos
            normalizados (batch, seq_len).
        """
        # Cada posición de la secuencia recibe una puntuación aprendida.
        scores = self.score(torch.tanh(self.projection(sequence_outputs))).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=1)

        # Combina las representaciones usando los pesos como coeficientes.
        context = torch.sum(sequence_outputs * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights


class CodeClassifier(nn.Module):
    """Red neuronal para clasificación de errores y calidad en código Python."""
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=128, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = AttentionModule(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        sequence_outputs, _ = self.lstm(embedded)
        context, attention_weights = self.attention(sequence_outputs)
        logits = self.fc(context)
        return logits, attention_weights

def get_code_model(num_classes=4):
    return CodeClassifier(num_classes=num_classes)

if __name__ == "__main__":
    model = get_code_model()
    print("[INFO] Arquitectura CodeClassifier (BiLSTM) instanciada correctamente:")
    print(model)