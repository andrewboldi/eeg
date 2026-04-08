"""Iteration 023: Channel-attention transformer (spatial self-attention).

Hypothesis: While iter022 uses temporal self-attention, the key challenge
in this task is SPATIAL — which scalp channels best predict which in-ear
channels, and how does this vary across subjects. Channel-attention
treats each EEG channel as a token and learns input-dependent spatial
relationships via self-attention.

Architecture:
1. Temporal embedding: Conv1d per-channel to create features
2. Self-attention over CHANNELS (each channel = token)
3. Cross-attention: in-ear queries attend to scalp keys/values
4. Output: temporal convolution to refine

This is more like a spatial transformer that adapts its mixing weights
based on the input, rather than using fixed spatial filters.

Design: H=32 per timestep, 2 layers, 4 heads. ~25K params.
Best run on GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class ChannelAttentionBlock(nn.Module):
    """Self-attention over EEG channels."""

    def __init__(self, d_model, nhead=4, dim_ff=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, n_channels, d_model)
        h = self.norm1(x)
        h = self.attn(h, h, h)[0]
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ChannelTransformer(nn.Module):
    """Transformer that operates over EEG channels.

    Each channel is a token with T-dimensional features.
    Self-attention learns spatial relationships.
    """

    def __init__(self, C_in=27, C_out=12, T=40, d_model=40,
                 nhead=4, num_layers=2, dim_ff=80, dropout=0.1):
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.d_model = d_model

        # Input normalization
        self.inorm = nn.InstanceNorm1d(C_in, affine=True)

        # Temporal embedding per channel: T → d_model
        # For T=40 and d_model=40, this is effectively identity-like
        self.temporal_embed = nn.Linear(T, d_model)

        # Learnable channel position embeddings
        self.channel_embed = nn.Parameter(torch.randn(1, C_in, d_model) * 0.02)

        # Self-attention over channels
        self.layers = nn.ModuleList([
            ChannelAttentionBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output: project from C_in channels to C_out channels
        # Each output channel is a learned query that attends to input channels
        self.out_queries = nn.Parameter(torch.randn(1, C_out, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

        # Map d_model back to T
        self.temporal_proj = nn.Linear(d_model, T)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T)
        Returns:
            (B, C_out, T)
        """
        B, C, T = x.shape

        # Normalize
        x = self.inorm(x)  # (B, C_in, T)

        # Temporal embedding: each channel becomes a d_model vector
        tokens = self.temporal_embed(x)  # (B, C_in, d_model)

        # Add channel position embeddings
        tokens = tokens + self.channel_embed

        # Self-attention over channels
        for layer in self.layers:
            tokens = layer(tokens)  # (B, C_in, d_model)

        # Cross-attention: output queries attend to channel tokens
        queries = self.out_queries.expand(B, -1, -1)  # (B, C_out, d_model)
        out = self.cross_attn(
            self.out_norm(queries), tokens, tokens
        )[0]  # (B, C_out, d_model)

        # Add residual from queries
        out = out + queries

        # Map back to temporal dimension
        return self.temporal_proj(out)  # (B, C_out, T)


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


def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    grad_clip=1.0, channel_drop_prob=0.15):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for scalp, inear in loader:
        scalp, inear = scalp.to(device), inear.to(device)
        if channel_drop_prob > 0:
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device)
                    > channel_drop_prob).float()
            scalp = scalp * mask / (1 - channel_drop_prob)
        optimizer.zero_grad()
        pred = model(scalp)
        loss = loss_fn(pred, inear)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return {"train_loss": total_loss / max(n_batches, 1)}


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build and train channel-attention transformer."""

    T = train_ds.scalp.shape[2]  # Should be 40

    model = ChannelTransformer(
        C_in=C_scalp,
        C_out=C_inear,
        T=T,
        d_model=40,  # Match T for simplicity
        nhead=4,
        num_layers=2,
        dim_ff=80,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Channel Transformer params: {n_params:,}")

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 201):
        train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: val_r={val_r:.4f} (best={best_val_r:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    return model
