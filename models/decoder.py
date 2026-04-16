"""
models/decoder.py  —  [B, latent_dim] -> [B, 3, 224, 224]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch))
    def forward(self, x):
        return F.gelu(x + self.b(x))

class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.GELU())
    def forward(self, x):
        return self.net(x)

class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_ch=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(latent_dim, 512*7*7), nn.GELU())
        self.net  = nn.Sequential(
            ResBlock(512),
            Up(512,256), ResBlock(256),   # 7 ->14
            Up(256,128), ResBlock(128),   # 14->28
            Up(128, 64), ResBlock(64),    # 28->56
            Up( 64, 32), ResBlock(32),    # 56->112
            Up( 32, 16), ResBlock(16),    # 112->224
            nn.Conv2d(16, out_ch, 3, padding=1),
            nn.Sigmoid())
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.proj(z).view(z.size(0), 512, 7, 7)
        return self.net(x)
