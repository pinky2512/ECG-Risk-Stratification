"""
CNN-Transformer Hybrid for ECG Risk Stratification
Architecture:
  1D CNN feature extractor  → local morphology (QRS, P-wave, T-wave)
  Positional Encoding       → inject time position info
  Transformer Encoder       → global temporal + inter-lead dependencies
  CLS token pooling         → classification head → softmax(3)

Input : (batch, 12, 1000)
Output: (batch, 3)
"""

import math
import torch
import torch.nn as nn


# ── Positional Encoding ────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── CNN Feature Extractor ──────────────────────────────────
class CNNExtractor(nn.Module):
    """
    3-layer 1D CNN to extract local ECG morphology features.
    Input : (batch, 12, 1000)
    Output: (batch, d_model, T') where T' ~ 125
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(12,  32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                              # 1000 → 250

            # Block 2
            nn.Conv1d(32,  64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),   # 250 → 125

            # Block 3
            nn.Conv1d(64, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True)  # 125 → 125
        )

    def forward(self, x):
        return self.cnn(x)   # (batch, d_model, 125)


# ── CNN-Transformer ────────────────────────────────────────
class CNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid for 12-lead ECG classification.

    Args:
        num_classes : number of output classes (3)
        d_model     : transformer hidden dim (128)
        nhead       : number of attention heads (4)
        num_layers  : transformer encoder layers (3)
        dim_ff      : feedforward dim inside transformer (256)
        dropout     : dropout rate
    """
    def __init__(
        self,
        num_classes: int = 3,
        d_model:     int = 128,
        nhead:       int = 4,
        num_layers:  int = 3,
        dim_ff:      int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # CNN feature extractor
        self.cnn = CNNExtractor(d_model=d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq_len = 125 + 1 CLS = 126)
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classification head
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # x: (batch, 12, 1000)
        B = x.size(0)

        # CNN feature extraction
        feat = self.cnn(x)              # (B, d_model, 125)
        feat = feat.permute(0, 2, 1)    # (B, 125, d_model)

        # Prepend CLS token
        cls  = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        feat = torch.cat([cls, feat], dim=1)       # (B, 126, d_model)

        # Positional encoding
        feat = self.pos_enc(feat)                  # (B, 126, d_model)

        # Transformer encoder
        feat = self.transformer(feat)              # (B, 126, d_model)

        # CLS token output → classifier
        cls_out = self.norm(feat[:, 0])            # (B, d_model)
        cls_out = self.dropout(cls_out)
        return self.fc(cls_out)                    # (B, num_classes)


if __name__ == "__main__":
    model = CNNTransformer(num_classes=3)
    x     = torch.randn(32, 12, 1000)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")    # expect (32, 3)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")