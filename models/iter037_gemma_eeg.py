"""Iteration 037: Gemma-4 inspired EEG architecture.

Motivation: Training diagnostics show the 2268-param FIR model barely learns
(1% loss decrease over 150 epochs). The CF init captures everything; SGD adds
nothing. We need MORE PARAMETERS with proper regularization.

Architecture inspired by Gemma 4's alternating local/global attention:
1. Spatial embedding: project 27 scalp channels to H hidden dims
2. Alternating blocks:
   a. LOCAL: depthwise temporal conv (per-channel, captures temporal patterns)
   b. GLOBAL: channel mixing MLP (cross-channel spatial features)
   c. Residual connections + LayerNorm
3. Output projection: H dims → 12 in-ear channels

Target: ~20K-50K parameters (10-20x more than FIR).
CF-initialized output projection for stable starting point.

Expected: significant improvement from actually LEARNING temporal-spatial features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class TemporalConvBlock(nn.Module):
    """Local temporal processing: depthwise 1D conv per channel."""

    def __init__(self, dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2,
                              groups=dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.act(x)
        x = self.drop(x)
        return residual + x


class ChannelMixBlock(nn.Module):
    """Global spatial processing: MLP across channels."""

    def __init__(self, dim, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expand, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return residual + x


class GemmaEEG(nn.Module):
    """Alternating local/global blocks for EEG prediction."""

    def __init__(self, C_in, C_out, hidden_dim=64, n_blocks=3,
                 kernel_size=7, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(C_in, hidden_dim)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(TemporalConvBlock(hidden_dim, kernel_size, dropout))
            blocks.append(ChannelMixBlock(hidden_dim, expand=2, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, C_out)

    def forward(self, x):
        # x: (B, C_in, T)
        x = x.transpose(1, 2)  # (B, T, C_in)
        x = self.input_proj(x)  # (B, T, H)
        x = self.blocks(x)  # (B, T, H)
        x = self.out_norm(x)
        x = self.output_proj(x)  # (B, T, C_out)
        return x.transpose(1, 2)  # (B, C_out, T)


class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * (1.0 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for scalp, inear in loader:
            scalp, inear = scalp.to(device), inear.to(device)
            pred = model(scalp)
            pred_m = pred - pred.mean(dim=-1, keepdim=True)
            target_m = inear - inear.mean(dim=-1, keepdim=True)
            cov = (pred_m * target_m).sum(dim=-1)
            r = cov / ((pred_m**2).sum(dim=-1).sqrt() * (target_m**2).sum(dim=-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    # CF for output projection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = GemmaEEG(
        C_in=C_scalp, C_out=C_inear,
        hidden_dim=64, n_blocks=3,
        kernel_size=7, dropout=0.15,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GemmaEEG params: {n_params}")

    # Initialize output projection close to CF solution
    # CF: W is (C_out, C_in). Our output_proj is (H, C_out)
    # We can't directly copy CF weights since dims differ,
    # but we initialize input_proj and output_proj to approximate CF:
    # output ≈ output_proj(input_proj(x)) ≈ W_out @ W_in @ x
    # where W_out @ W_in ≈ CF.W
    # Use SVD: CF.W = U @ S @ V^T, set W_in = V^T[:H, :], W_out = U[:, :H] @ S[:H]
    with torch.no_grad():
        W_cf = cf.W.float()  # (C_out, C_in)
        U, S, Vh = torch.linalg.svd(W_cf, full_matrices=False)
        H = model.input_proj.weight.shape[0]  # hidden_dim
        K = min(H, C_scalp, C_inear)
        # input_proj: (H, C_in) — first K rows from Vh
        model.input_proj.weight.zero_()
        model.input_proj.weight[:K] = Vh[:K]
        model.input_proj.bias.zero_()
        # output_proj: (C_out, H) — first K cols from U@diag(S)
        model.output_proj.weight.zero_()
        model.output_proj.weight[:, :K] = U[:, :K] * S[:K].unsqueeze(0)
        model.output_proj.bias.zero_()

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 201):
        # Training with channel dropout
        model.train()
        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device) > 0.15).float()
            scalp = scalp * mask / 0.85
            optimizer.zero_grad()
            pred = model(scalp)
            loss = loss_fn(pred, inear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: val_r={val_r:.4f} (best={best_val_r:.4f})")

    print(f"Final best val_r: {best_val_r:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model
