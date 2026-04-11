"""Iteration 052: Cross-Attention Decoder.

Architecture with explicit cross-attention from output queries to input channels:
1. Per-channel encoder: Conv1d on each of 46 input channels -> (B, 46, D)
2. 12 learnable output query embeddings, each D=64
3. Cross-attention: queries=output(12), keys/values=input(46) -> (B, 12, 64)
4. Temporal decoder: Linear(64, 256) per output channel -> (B, 12, 256)
5. CF skip connection

This explicitly learns which input channels are relevant for each output channel.

Confidence: 60% — cross-attention is principled for channel selection but may overfit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class PerChannelEncoder(nn.Module):
    """Encode each input channel's temporal signal into a feature vector."""

    def __init__(self, T, D, n_layers=2):
        super().__init__()
        # Shared temporal encoder applied per channel
        layers = []
        in_dim = T
        for i in range(n_layers):
            out_dim = D * 2 if i < n_layers - 1 else D
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        # x: (B, C_in, T) -> treat T as feature dim per channel
        # Apply shared encoder to each channel
        h = self.encoder(x)  # (B, C_in, D)
        return self.norm(h)


class CrossAttentionDecoder(nn.Module):
    """Cross-attention from output queries to encoded input channels."""

    def __init__(self, C_in, C_out, T=256, D=64, n_heads=4, n_cross_layers=2, dropout=0.1):
        super().__init__()
        self.C_out = C_out
        self.T = T
        self.D = D

        # Per-channel temporal encoder
        self.channel_encoder = PerChannelEncoder(T, D)

        # Learnable output query embeddings (one per output channel)
        self.output_queries = nn.Parameter(torch.randn(1, C_out, D) * 0.02)

        # Multi-layer cross-attention
        self.cross_layers = nn.ModuleList()
        for _ in range(n_cross_layers):
            self.cross_layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=D, num_heads=n_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(D),
                'norm2': nn.LayerNorm(D),
                'ffn': nn.Sequential(
                    nn.Linear(D, D * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D * 4, D),
                    nn.Dropout(dropout),
                ),
            }))

        # Temporal decoder: map D features back to T time points per output channel
        self.temporal_decoder = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, T),
        )

        # CF skip connection
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        # x: (B, C_in, T)
        B = x.shape[0]
        skip = self.skip(x)  # (B, C_out, T)

        # Encode each input channel
        encoded = self.channel_encoder(x)  # (B, C_in, D)

        # Expand output queries for batch
        queries = self.output_queries.expand(B, -1, -1)  # (B, C_out, D)

        # Cross-attention layers
        for layer in self.cross_layers:
            # Cross-attention: queries attend to encoded input channels
            residual = queries
            queries = layer['norm1'](queries)
            attn_out, _ = layer['cross_attn'](
                query=queries,
                key=encoded,
                value=encoded,
            )
            queries = residual + attn_out

            # FFN
            residual = queries
            queries = residual + layer['ffn'](layer['norm2'](queries))

        # Decode temporal signal for each output channel
        out = self.temporal_decoder(queries)  # (B, C_out, T)

        return out + skip


class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm, tm = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm, tm = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build CrossAttentionDecoder
    model = CrossAttentionDecoder(
        C_in=C_scalp, C_out=C_inear, T=256, D=64,
        n_heads=4, n_cross_layers=2, dropout=0.1,
    ).to(device)

    # Init skip with CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CrossAttentionDecoder params: {n_params:,}")

    # Training setup
    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam*x + (1-lam)*x[idx]; y = lam*y + (1-lam)*y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad(); loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")
    return model
