"""
1D ResNet-34 for ECG Classification
Input : (batch, 12, 1000)
Output: (batch, 3)  — Low / Medium / High risk
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Basic residual block with two 1D conv layers."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2        = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """
    1D ResNet-34 adapted for 12-lead ECG.
    Architecture:
      Stem → Layer1(64) → Layer2(128) → Layer3(256) → Layer4(512)
      → GlobalAvgPool → Dropout → FC(3)
    """
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers (ResNet-34: 3,4,6,3 blocks)
        self.layer1 = self._make_layer(64,  64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        # Head
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        layers = [ResBlock1D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)      # (B, 64, 250)
        x = self.layer1(x)    # (B, 64, 250)
        x = self.layer2(x)    # (B, 128, 125)
        x = self.layer3(x)    # (B, 256, 63)
        x = self.layer4(x)    # (B, 512, 32)
        x = self.gap(x)       # (B, 512, 1)
        x = x.squeeze(-1)     # (B, 512)
        x = self.dropout(x)
        return self.fc(x)     # (B, 3)


if __name__ == "__main__":
    model = ResNet1D(num_classes=3)
    x     = torch.randn(32, 12, 1000)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")   # expect (32, 3)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")  # expect ~1.5M