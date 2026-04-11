"""Iteration 070: Stochastic Depth (Layer Dropout) for TinyDeep.

Randomly skip transformer blocks during training with probability p=0.2.
This forces each layer to produce useful features independently, acting as
strong regularization. At test time, outputs are scaled by (1-p).

Architecture: TinyDeep (H=64, 2 blocks) with stochastic depth applied to
each transformer block.

Hypothesis: Stochastic depth will reduce overfitting on the small EEG dataset,
improving cross-subject generalization beyond the 0.378 plateau.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class MultiScaleConv(nn.Module):
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class StochasticDepthBlock(nn.Module):
    """Wraps a TransformerEncoderLayer with stochastic depth.

    During training, the block is skipped with probability `drop_prob`,
    returning the input unchanged (identity). During eval, the output
    is scaled by (1 - drop_prob) to compensate.
    """

    def __init__(self, layer: nn.Module, drop_prob: float = 0.2):
        super().__init__()
        self.layer = layer
        self.drop_prob = drop_prob

    def forward(self, x, **kwargs):
        if self.training:
            if torch.rand(1).item() < self.drop_prob:
                return x  # Skip this block entirely
            return x + (self.layer(x, **kwargs) - x)
        else:
            # At test time, scale residual by survival probability
            return x + (1 - self.drop_prob) * (self.layer(x, **kwargs) - x)


class StochasticDepthTransformerEncoder(nn.Module):
    """Transformer encoder where each layer has stochastic depth.

    Uses linearly increasing drop probability from 0 to max_drop_prob
    (earlier layers are more important, so dropped less often).
    """

    def __init__(self, encoder_layer_fn, num_layers, max_drop_prob=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Linear schedule: layer 0 gets lower drop prob, last layer gets max
            drop_prob = max_drop_prob * (i + 1) / num_layers
            layer = encoder_layer_fn()
            self.layers.append(StochasticDepthBlock(layer, drop_prob))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TinyDeepSD(nn.Module):
    """TinyDeep with Stochastic Depth on transformer blocks."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1,
                 sd_max_prob=0.2):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())

        # Stochastic depth transformer
        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=H, nhead=4, dim_feedforward=H * 4,
                dropout=dropout, batch_first=True,
                norm_first=True,
            )

        self.transformer = StochasticDepthTransformerEncoder(
            make_layer, num_layers=n_blocks, max_drop_prob=sd_max_prob
        )

        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm, tm = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm, tm = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip-connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build TinyDeep with stochastic depth
    model = TinyDeepSD(
        C_in=C_scalp, C_out=C_inear, T=256, H=64,
        n_blocks=2, dropout=0.1, sd_max_prob=0.2,
    ).to(device)

    # Init skip connection from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeepSD params: {n_params:,}")

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
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss_fn(model(x), y).backward()
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
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")

    return model
