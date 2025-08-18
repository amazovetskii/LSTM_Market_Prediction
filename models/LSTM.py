import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2, pred_days=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Fully connected head → predicts n closes
        self.fc = nn.Linear(hidden_size, pred_days)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)                # out: [batch, seq_len, hidden]
        last_hidden = out[:, -1, :]          # take last time step: [batch, hidden]
        preds = self.fc(last_hidden)         # [batch, pred_days]
        return preds