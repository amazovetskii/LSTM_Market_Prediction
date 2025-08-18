import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, output_seq_len=5, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_seq_len = output_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)  # predicting single feature

    def forward(self, src, tgt):
        """
        src: (batch_size, src_seq_len, input_dim)
        tgt: (batch_size, tgt_seq_len, 1) - usually previous known values, can be zeros for training
        """
        src = self.input_proj(src)
        src = self.pos_encoder(src)

        tgt = self.input_proj(tgt)  # project to d_model
        tgt = self.pos_encoder(tgt)

        out = self.transformer(src, tgt)  # (batch_size, tgt_seq_len, d_model)
        out = self.output_proj(out)       # (batch_size, tgt_seq_len, 1)
        return out