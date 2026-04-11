"""Iteration 071: Efficient Channel Attention (ECA) on TinyDeep.

ECA (Wang et al. 2020) uses a 1D convolution across channels instead of
dimensionality reduction (like SE blocks). For C=46 input channels to the
transformer, kernel k=5 captures local inter-channel relationships.

Architecture:
  MultiScaleConv -> ECA block -> downsample -> transformer -> upsample -> output

Hypothesis: ECA learns inter-channel dependencies more expressively than
learnable scalar weights (iter062), improving spatial selectivity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class ECABlock(nn.Module):
    """Efficient Channel Attention: 1D conv across channel descriptors."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        # AdaptiveAvgPool1d squeezes temporal dim to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 1D conv across channels (no dimensionality reduction)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = self.pool(x)          # (B, C, 1)
        w = w.transpose(1, 2)     # (B, 1, C) — treat channels as sequence
        w = self.conv(w)          # (B, 1, C)
        w = w.transpose(1, 2)     # (B, C, 1)
        w = torch.sigmoid(w)      # channel weights in [0, 1]
        return x * w


class MultiScaleConv(nn.Module):
    def __init__(self, C_in: int, H: int, kernels: tuple = (3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeepECA(nn.Module):
    """TinyDeep with ECA after MultiScaleConv (H=64, 2 transformer blocks)."""

    def __init__(self, C_in: int, C_out: int, T: int = 256,
                 H: int = 64, n_blocks: int = 2, dropout: float = 0.1,
                 eca_kernel: int = 5):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.eca = ECABlock(channels=H, kernel_size=eca_kernel)
        self.down = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = self.temporal(x)       # (B, H, T)
        h = self.eca(h)            # ECA re-weights channels
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


class CorrMSELoss(nn.Module):
    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model: nn.Module, loader: DataLoader,
                         device: torch.device) -> float:
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build TinyDeep+ECA, train with CF-init skip, mixup, channel dropout."""

    # CF init for skip connection
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = TinyDeepECA(
        C_in=C_scalp, C_out=C_inear, T=256,
        H=64, n_blocks=2, dropout=0.1, eca_kernel=5,
    ).to(device)

    # Init skip from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeepECA params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1.0, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"  ECA Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"TinyDeepECA best val_r: {best_r:.4f}")
    return model
