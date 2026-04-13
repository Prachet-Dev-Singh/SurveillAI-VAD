"""
models/decoder.py

Reconstruction decoder: latent_dim → [B, 3, 224, 224]

Design notes:
- Input is a 1D latent vector from the MemoryBank [B, latent_dim]
- We project it to a small spatial seed [B, 512, 7, 7] then upsample
- Each block doubles spatial resolution: 7→14→28→56→112→224
- BatchNorm + GELU throughout for stable training
- Final Tanh-free sigmoid clamps output to [0, 1] matching normalised input frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Lightweight residual block used inside the decoder."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class UpsampleBlock(nn.Module):
    """2× bilinear upsample followed by a conv to refine features."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class ReconstructionDecoder(nn.Module):
    """
    Decodes a [B, latent_dim] vector to a [B, 3, 224, 224] image.

    Args:
        latent_dim: must match the bottleneck dimension in MemoryBankAutoencoder
        out_channels: 3 for RGB (matches MambaVision input format)
    """
    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Project latent vector to spatial seed ─────────────────
        # 512 * 7 * 7 = 25088
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.GELU(),
        )

        # ── Residual refinement at seed resolution ────────────────
        self.seed_refine = ResidualBlock(512)

        # ── Progressive upsampling ─────────────────────────────────
        # 7  → 14  (512 → 256)
        # 14 → 28  (256 → 128)
        # 28 → 56  (128 → 64)
        # 56 → 112 (64  → 32)
        # 112→ 224 (32  → 16)
        self.upsample = nn.Sequential(
            UpsampleBlock(512, 256),
            ResidualBlock(256),
            UpsampleBlock(256, 128),
            ResidualBlock(128),
            UpsampleBlock(128, 64),
            ResidualBlock(64),
            UpsampleBlock(64, 32),
            ResidualBlock(32),
            UpsampleBlock(32, 16),
            ResidualBlock(16),
        )

        # ── Final output head ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()   # output in [0, 1] — matches frames normalised to [0,1]
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
        Returns:
            recon: [B, 3, 224, 224]
        """
        # Project and reshape to spatial seed
        x = self.project(z)                 # [B, 512*7*7]
        x = x.view(z.size(0), 512, 7, 7)   # [B, 512, 7, 7]
        x = self.seed_refine(x)             # [B, 512, 7, 7]

        # Upsample to 224×224
        x = self.upsample(x)               # [B, 16, 224, 224]
        return self.head(x)                # [B, 3, 224, 224]
