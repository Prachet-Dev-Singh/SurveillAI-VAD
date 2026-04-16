import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, yaml, sys, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
import transformers
import pytorch_msssim
from tqdm.autonotebook import tqdm

# Allow importing from project root
sys.path.append(os.getcwd())

# ── CONFIG ────────────────────────────────────────────────────────────
class ConfigObject(dict):
    _DEFAULTS = {
        'data_dir': 'data/processed', 'batch_size': 16,
        'epochs': 100, 'learning_rate': 2e-4, 'model_type': 'mamba',
    }
    def __getattr__(self, name):
        if name in self:
            return self[name]
        if name == 'learning_rate' and 'lr' in self:
            return self['lr']
        if name in self._DEFAULTS:
            return self._DEFAULTS[name]
        raise AttributeError(f"Config missing: '{name}'")
    def __setattr__(self, n, v):
        self[n] = v

def load_config(p):
    with open(p) as f:
        return ConfigObject(yaml.safe_load(f))

# ── LOSS: 0.3*MSE + 0.7*SSIM (simple, proven) ────────────────────────
class StructuralLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        mse_l  = self.mse(recon, target)
        ssim_l = 1.0 - pytorch_msssim.ssim(
            recon, target, data_range=1.0, size_average=True)
        return self.alpha * mse_l + self.beta * ssim_l

# ── ENCODER ───────────────────────────────────────────────────────────
transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
ENCODER_DIM = 640

class MambaFeatureExtractor(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        _orig = torch.linspace
        def _p(*a, **kw):
            kw["device"] = "cpu"
            return _orig(*a, **kw)
        torch.linspace = _p
        self.backbone = AutoModelForImageClassification.from_pretrained(
            "nvidia/MambaVision-T-1K", trust_remote_code=True,
            low_cpu_mem_usage=False, _fast_init=False)
        torch.linspace = _orig
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat, _ = self.backbone.model.forward_features(x)
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.dim() == 3:
            feat = feat.mean(1)
        return feat   # [B, 640]

# ── MEMORY BANK: Soft attention ALWAYS (train + eval match) ───────────
class MemoryBank(nn.Module):
    def __init__(self, num_slots=512, dim=256, temperature=15.0):
        super().__init__()
        self.memory = nn.Parameter(
            F.normalize(torch.randn(num_slots, dim), dim=1))
        self.temperature = temperature

    def forward(self, q):
        qn = F.normalize(q, dim=1)
        mn = F.normalize(self.memory, dim=1)
        sim = torch.mm(qn, mn.t()) * self.temperature
        w = F.softmax(sim, dim=1)
        return w @ self.memory

    def diversity_loss(self, weight=0.0005):
        mn = F.normalize(self.memory, dim=1)
        sim = torch.mm(mn, mn.t())
        mask = 1.0 - torch.eye(sim.shape[0], device=sim.device)
        return weight * (sim * mask).pow(2).mean()

# ── AUTOENCODER ───────────────────────────────────────────────────────
from models.decoder import ReconstructionDecoder

class MemoryBankAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, num_slots=512, freeze_encoder=True):
        super().__init__()
        self.encoder = MambaFeatureExtractor(freeze=freeze_encoder)
        self.gru = nn.GRU(
            input_size=ENCODER_DIM, hidden_size=512,
            num_layers=2, batch_first=True, dropout=0.1)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU())
        self.memory_bank = MemoryBank(
            num_slots=num_slots, dim=latent_dim, temperature=15.0)
        self.decoder = ReconstructionDecoder(latent_dim=latent_dim)

    def forward(self, context):
        B, T, C, H, W = context.shape
        enc_ctx = (torch.no_grad()
                   if not self.encoder.backbone.training
                   else torch.enable_grad())
        feats = []
        with enc_ctx:
            for t in range(T):
                feats.append(self.encoder(context[:, t]))
        _, h = self.gru(torch.stack(feats, dim=1))
        z = self.bottleneck(h[-1])
        return self.decoder(self.memory_bank(z))

# ── BATCH HELPER ──────────────────────────────────────────────────────
def process_batch(batch, device):
    clips = batch[0] if isinstance(batch, (tuple, list)) else batch
    context = clips[:, :-1]
    target  = clips[:, -1]
    if target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)
    return context.to(device), target.to(device)

# ── TRAINING ──────────────────────────────────────────────────────────
def train_model(config, model, train_loader, val_loader, device):
    lr  = float(config.learning_rate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
              opt, T_max=config.epochs, eta_min=lr * 0.01)
    crit     = StructuralLoss(alpha=0.3, beta=0.7)
    best_val = float("inf")
    best_path   = f"checkpoints/{config.model_type}_best.pth"
    master_path = f"checkpoints/{config.model_type}_master.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config.epochs):
        # ── Train ──────────────────────────────────────────────
        model.train()
        t_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", leave=False):
            context, target = process_batch(batch, device)
            opt.zero_grad()
            recon = model(context)
            if recon.shape[1] != target.shape[1]:
                target = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
            loss = crit(recon, target) + model.memory_bank.diversity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_losses.append(loss.item())
        sch.step()

        # ── Validate ───────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]", leave=False):
                context, target = process_batch(batch, device)
                recon = model(context)
                if recon.shape[1] != target.shape[1]:
                    target = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
                v_losses.append(crit(recon, target).item())

        avg_t = np.mean(t_losses)
        avg_v = np.mean(v_losses)
        is_best = avg_v < best_val and epoch >= 5   # skip first 5 noisy epochs

        # ── Single clean line per epoch ────────────────────────
        tag = "  ⭐best" if is_best else ""
        print(f"Epoch {epoch+1:03d}/{config.epochs} | "
              f"Train {avg_t:.6f} | Val {avg_v:.6f} | "
              f"LR {sch.get_last_lr()[0]:.2e}{tag}")

        # ── Save ───────────────────────────────────────────────
        torch.save({"epoch": epoch, "model": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "scheduler": sch.state_dict(),
                     "best_val": best_val}, master_path)
        if is_best:
            best_val = avg_v
            torch.save(model.state_dict(), best_path)

    print(f"\n✅ Done. Best val: {best_val:.6f}")

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/mamba.yaml")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--num_slots",  type=int, default=512)
    p.add_argument("--freeze",     action="store_true", default=True)
    p.add_argument("--resume",     type=str, default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    cfg.device = a.device

    from data.dataset import SlidingWindowDataset
    tr = SlidingWindowDataset(
        os.path.join(cfg.data_dir, "train"), window_size=8, stride=4)
    va = SlidingWindowDataset(
        os.path.join(cfg.data_dir, "test"),  window_size=8, stride=4)
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    model = MemoryBankAutoencoder(
        latent_dim=a.latent_dim, num_slots=a.num_slots,
        freeze_encoder=a.freeze).to(a.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n📐 Model: {total:,} total params, {trainable:,} trainable")

    train_model(cfg, model, tl, vl, a.device)

if __name__ == "__main__":
    main()
