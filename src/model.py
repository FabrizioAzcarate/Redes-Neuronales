import torch
import torch.nn as nn

class CodeClassifier(nn.Module):
    """Red neuronal para clasificación de errores y calidad en código Python."""
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=128, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)

def get_code_model(num_classes=4):
    return CodeClassifier(num_classes=num_classes)

if __name__ == "__main__":
    model = get_code_model()
    print("[INFO] Arquitectura CodeClassifier (BiLSTM) instanciada correctamente:")
    print(model)