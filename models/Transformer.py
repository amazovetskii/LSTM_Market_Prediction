import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        input_size=5,          # OHLCV
        d_model=16,            # divisible by nhead
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        output_steps=5,
        max_len=512            # max sequence length for positional embeddings
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_steps = output_steps
        self.max_len = max_len

        # Embeddings
        self.encoder_embedding = nn.Linear(input_size, d_model)
        self.decoder_embedding = nn.Linear(input_size, d_model)

        # Learnable positional embeddings (batch_first)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.pos_decoder = nn.Embedding(max_len, d_model)

        # Encoder / Decoder with batch_first=True
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Output projection back to OHLCV
        self.output_layer = nn.Linear(d_model, input_size)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device):
        # Causal mask: allow attention to current and previous, block future
        # Shape: [sz, sz]
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src, tgt=None, teacher_forcing=True):
        """
        src: [batch, src_len, input_size]
        tgt: [batch, tgt_len (=output_steps), input_size] (optional during training)
        returns: [batch, output_steps, input_size]
        """
        bsz, src_len, _ = src.size()
        device = src.device

        # ----- Encoder -----
        src_pos_ids = torch.arange(src_len, device=device)  # [src_len]
        src_emb = self.encoder_embedding(src) + self.pos_encoder(src_pos_ids)[None, :, :]  # [B, S, D]
        memory = self.encoder(src_emb)  # [B, S, D]

        # ----- Decoder (autoregressive) -----
        # Start token = last observed step (can swap to learned BOS if you prefer)
        if tgt is None:
            tgt_input = src[:, -1:, :]  # [B, 1, F]
        else:
            tgt_input = tgt[:, :1, :]   # teacher-forced first step

        outputs = []

        for t in range(self.output_steps):
            tlen = tgt_input.size(1)
            tgt_pos_ids = torch.arange(tlen, device=device)  # [tlen]
            tgt_emb = self.decoder_embedding(tgt_input) + self.pos_decoder(tgt_pos_ids)[None, :, :]  # [B, tlen, D]

            tgt_mask = self._generate_square_subsequent_mask(tlen, device)  # [tlen, tlen]

            dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # [B, tlen, D]
            step_hidden = dec_out[:, -1, :]                 # [B, D]
            step_pred = self.output_layer(step_hidden)      # [B, F]
            outputs.append(step_pred.unsqueeze(1))          # [B, 1, F]

            # Next input (teacher forcing vs. autoregressive)
            if teacher_forcing and (tgt is not None) and (t + 1 < tgt.size(1)):
                next_in = tgt[:, t+1:t+2, :]                # [B, 1, F]
            else:
                next_in = step_pred.unsqueeze(1)            # [B, 1, F]

            tgt_input = torch.cat([tgt_input, next_in], dim=1)  # grow sequence

        return torch.cat(outputs, dim=1)  # [B, output_steps, F]