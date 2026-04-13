import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import numpy as np
import sys
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
import transformers
import pytorch_msssim

sys.path.append(os.getcwd())

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
class ConfigObject(dict):
    def __getattr__(self, name):
        if name in self: return self[name]
        if name == 'data_dir': return self.get('dataset_dir', self.get('data_path', 'data/processed'))
        if name == 'batch_size': return 4
        if name == 'epochs': return 50
        if name == 'learning_rate': return 1e-4
        if name == 'model_type': return 'mamba'
        raise AttributeError(f"'ConfigObject' has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value

def load_config(config_path):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return ConfigObject(data)


# ─────────────────────────────────────────────────────────────
# 2. STRUCTURAL LOSS
# ─────────────────────────────────────────────────────────────
class StructuralLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        mse_loss = self.mse(recon, target)
        ssim_val = pytorch_msssim.ssim(recon, target, data_range=1.0, size_average=True)
        return (self.alpha * mse_loss) + (self.beta * (1.0 - ssim_val))


# ─────────────────────────────────────────────────────────────
# 3. ENCODER — FIX #1: Extract backbone features, NOT logits
#    MambaVision-T head is Linear(640 → 1000). We replace it
#    with Identity() so the encoder outputs 640-d feature vectors
#    instead of meaningless ImageNet class scores.
# ─────────────────────────────────────────────────────────────
transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

MAMBA_FEATURE_DIM = 640  # MambaVision-T backbone output dimension

class MambaFeatureExtractor(nn.Module):
    """
    Loads MambaVision-T and removes the classification head so we get
    real 640-d spatial feature vectors instead of 1000-d logits.
    """
    def __init__(self, freeze=True):
        super().__init__()
        # Patch linspace for CPU compat during init
        _orig = torch.linspace
        def _patched(*a, **kw):
            kw['device'] = 'cpu'
            return _orig(*a, **kw)
        torch.linspace = _patched
        self.backbone = AutoModelForImageClassification.from_pretrained(
            "nvidia/MambaVision-T-1K",
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            _fast_init=False
        )
        torch.linspace = _orig

        # ── KEY FIX: replace classifier head with Identity ──────────
        # This makes forward() return 640-d features, not 1000-d logits
        replaced = False
        for head_name in ('head', 'classifier', 'fc'):
            if hasattr(self.backbone, head_name):
                head = getattr(self.backbone, head_name)
                # Detect actual feature dim from the Linear layer
                if hasattr(head, 'in_features'):
                    global MAMBA_FEATURE_DIM
                    MAMBA_FEATURE_DIM = head.in_features
                setattr(self.backbone, head_name, nn.Identity())
                replaced = True
                break
        if not replaced:
            print("WARNING: Could not find classification head to replace. "
                  "Features may still be logits.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        # Extract raw tensor from HuggingFace output wrapper
        if hasattr(out, 'logits'):
            feat = out.logits          # after head→Identity this is 640-d features
        elif isinstance(out, dict):
            feat = list(out.values())[0]
        elif isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            feat = out

        # Handle unexpected spatial output shapes (safety net)
        if feat.dim() == 4:            # [B, C, H, W]
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.dim() == 3:          # [B, T, C] from SSM layers
            feat = feat.mean(dim=1)

        return feat  # [B, MAMBA_FEATURE_DIM]


class TrainableMambaExtractor(MambaFeatureExtractor):
    """Same as above but with backbone weights unfrozen for fine-tuning."""
    def __init__(self):
        super().__init__(freeze=False)


# ─────────────────────────────────────────────────────────────
# 4. MEMORY BANK — FIX #4: More slots + L2 normalisation
#    With only 5 soft-attention slots, anomalous patterns can
#    still be reconstructed as a mixture. More normalised slots
#    make it harder to represent out-of-distribution inputs.
# ─────────────────────────────────────────────────────────────
class MemoryBank(nn.Module):
    def __init__(self, num_slots=512, dim=256):
        super().__init__()
        # L2-normalised random init is much more stable than raw randn
        mem = torch.randn(num_slots, dim)
        mem = F.normalize(mem, dim=1)
        self.memory = nn.Parameter(mem)

    def forward(self, query):
        # Normalise both query and memory before cosine similarity
        q_norm = F.normalize(query, dim=1)               # [B, D]
        m_norm = F.normalize(self.memory, dim=1)          # [N, D]
        sim = torch.mm(q_norm, m_norm.t())                # [B, N]
        weights = F.softmax(sim * 10, dim=1)              # [B, N]
        return weights @ self.memory                       # [B, D]


# ─────────────────────────────────────────────────────────────
# 5. FULL AUTOENCODER
# ─────────────────────────────────────────────────────────────
from models.decoder import ReconstructionDecoder

class MemoryBankAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, trainable_encoder=False, num_slots=512):
        super().__init__()
        self.encoder = TrainableMambaExtractor() if trainable_encoder \
                       else MambaFeatureExtractor(freeze=True)
        # GRU input size must match encoder output, NOT hardcoded to 1000
        self.gru = nn.GRU(
            input_size=MAMBA_FEATURE_DIM,  # 640, not 1000
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.memory_bank = MemoryBank(num_slots=num_slots, dim=latent_dim)
        self.decoder = ReconstructionDecoder(latent_dim=latent_dim)

    def forward(self, x):
        # x: [B, T, C, H, W] — never accept raw [B, C, H, W] silently
        # FIX #3: removed the single-frame duplication fallback.
        # If you get [B, C, H, W] it means the dataset is broken — fix the dataset.
        if x.dim() == 4:
            raise ValueError(
                f"Expected 5D input [B, T, C, H, W] but got shape {x.shape}. "
                "Check your SlidingWindowDataset — it should always return clips."
            )
        B, T, C, H, W = x.shape

        # Encode each frame — only no_grad if encoder is frozen
        encode_ctx = torch.no_grad() if not self.encoder.training else torch.enable_grad()
        spatial_feats = []
        with encode_ctx:
            for t in range(T):
                spatial_feats.append(self.encoder(x[:, t]))  # [B, 640]
        sequence = torch.stack(spatial_feats, dim=1)          # [B, T, 640]

        _, hidden = self.gru(sequence)   # hidden: [num_layers, B, 512]
        z = self.bottleneck(hidden[-1])  # take last layer: [B, latent_dim]

        mem_out = self.memory_bank(z)    # [B, latent_dim]
        return self.decoder(mem_out)     # [B, 3, H, W]


# ─────────────────────────────────────────────────────────────
# 6. BATCH HELPER
# ─────────────────────────────────────────────────────────────
def process_batch(batch_data, device):
    if isinstance(batch_data, (tuple, list)):
        clips = batch_data[0]
        target = batch_data[1] if len(batch_data) > 1 else clips[:, -1]
    else:
        clips = batch_data
        target = clips[:, -1]
    if target.dim() == 5:
        target = target[:, -1]
    # Channel safety: match recon channels (always 3)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)
    return clips.to(device), target.to(device)


# ─────────────────────────────────────────────────────────────
# 7. TRAINING — FIX #5: gradient clipping + LR scheduler
# ─────────────────────────────────────────────────────────────
def train_model(config, model, train_loader, val_loader, device):
    lr = float(config.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing: smoothly decays LR to near-zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=lr * 0.01
    )
    criterion = StructuralLoss(alpha=0.3, beta=0.7)
    best_val_loss = float('inf')

    print(f"\n🚀 Training | {config.epochs} epochs | "
          f"LR={lr} | Cosine scheduler | Grad clip=1.0")

    for epoch in range(config.epochs):
        # ── Train ───────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch_data in train_loader:
            clips, target = process_batch(batch_data, device)
            optimizer.zero_grad()
            recon = model(clips)

            # Ensure channel match after decode
            if recon.shape[1] != target.shape[1]:
                if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)
                elif recon.shape[1] == 1: recon = recon.repeat(1, 3, 1, 1)

            loss = criterion(recon, target)
            loss.backward()

            # FIX #5a: Gradient clipping prevents explosion when encoder is unfrozen
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # FIX #5b: Step LR scheduler after each epoch
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                clips, target = process_batch(batch_data, device)
                recon = model(clips)
                if recon.shape[1] != target.shape[1]:
                    if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)
                    elif recon.shape[1] == 1: recon = recon.repeat(1, 3, 1, 1)
                val_losses.append(criterion(recon, target).item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        cur_lr    = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}/{config.epochs} | "
              f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {cur_lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{config.model_type}_best.pth")
            print(f"   ⭐ Best model saved (Val: {best_val_loss:.6f})")


# ─────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          required=True)
    parser.add_argument('--device',          default='cuda')
    parser.add_argument('--trainable',       action='store_true',
                        help='Unfreeze Mamba encoder for fine-tuning')
    parser.add_argument('--latent_dim',      type=int, default=256)
    parser.add_argument('--num_slots',       type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    config.device = args.device

    from data.dataset import SlidingWindowDataset
    train_dir = os.path.join(config.data_dir, 'train')
    val_dir   = os.path.join(config.data_dir, 'test')

    # FIX #3: stride=4 during training is fine, but window_size must always
    # return proper clips — no more single-frame fallback
    train_dataset = SlidingWindowDataset(frame_dir=train_dir, window_size=8, stride=4)
    val_dataset   = SlidingWindowDataset(frame_dir=val_dir,   window_size=8, stride=4)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = MemoryBankAutoencoder(
        latent_dim=args.latent_dim,
        trainable_encoder=args.trainable,
        num_slots=args.num_slots
    ).to(args.device)

    train_model(config, model, train_loader, val_loader, args.device)


if __name__ == "__main__":
    main()
