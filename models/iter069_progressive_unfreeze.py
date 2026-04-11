"""Iteration 069: Progressive Layer Unfreezing (ULMFiT-inspired).

Training schedule:
  Phase 1 (epochs 1-30):  Only output projection + skip connection trainable
  Phase 2 (epochs 31-60): Unfreeze Transformer blocks
  Phase 3 (epochs 61+):   Unfreeze everything with discriminative LR

LR: output=3e-4, transformer=1e-4, conv=3e-5

Architecture: TinyDeep (H=64, 2 blocks).
Hypothesis: Gradual unfreezing prevents early layers from being corrupted by
noisy gradients before the output head has stabilized, similar to ULMFiT.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
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


class TinyDeep(nn.Module):
    """Tiny deep model (H=64, 2 transformer blocks)."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
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
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def _freeze(module):
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze(module):
    """Unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = True


def _set_phase(model, phase):
    """Configure which layers are trainable based on training phase.

    Phase 1: Only output projection + skip connection
    Phase 2: + Transformer blocks (up, out_norm, transformer)
    Phase 3: + Conv layers (temporal, down) with discriminative LR
    """
    # Start by freezing everything
    _freeze(model)

    # Phase 1+: output projection and skip are always trainable
    _unfreeze(model.out_proj)
    _unfreeze(model.skip)
    _unfreeze(model.out_norm)
    _unfreeze(model.up)

    if phase >= 2:
        # Phase 2+: unfreeze transformer
        _unfreeze(model.transformer)

    if phase >= 3:
        # Phase 3: unfreeze conv layers
        _unfreeze(model.temporal)
        _unfreeze(model.down)


def _build_optimizer(model, phase):
    """Build optimizer with discriminative learning rates per phase."""
    # Output group: out_proj, skip, out_norm, up
    output_params = (
        list(model.out_proj.parameters()) +
        list(model.skip.parameters()) +
        list(model.out_norm.parameters()) +
        list(model.up.parameters())
    )

    param_groups = [{"params": output_params, "lr": 3e-4}]

    if phase >= 2:
        transformer_params = list(model.transformer.parameters())
        param_groups.append({"params": transformer_params, "lr": 1e-4})

    if phase >= 3:
        conv_params = (
            list(model.temporal.parameters()) +
            list(model.down.parameters())
        )
        param_groups.append({"params": conv_params, "lr": 3e-5})

    return torch.optim.AdamW(param_groups, weight_decay=1e-2)


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # CF init for skip connection
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    max_epochs = 150

    # Phase boundaries
    phase1_end = 30
    phase2_end = 60

    current_phase = 0  # Will be set on first iteration
    opt = None

    for ep in range(1, max_epochs + 1):
        # Determine phase
        if ep <= phase1_end:
            phase = 1
        elif ep <= phase2_end:
            phase = 2
        else:
            phase = 3

        # Reconfigure when phase changes
        if phase != current_phase:
            current_phase = phase
            _set_phase(model, phase)
            opt = _build_optimizer(model, phase)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Phase {phase} (epoch {ep}): {trainable:,}/{total:,} params trainable")

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

        if ep % 10 == 0 or phase != current_phase:
            print(f"  Epoch {ep} (phase {current_phase}): val_r={vr:.4f} (best={best_r:.4f})")

        # Early stopping: patience 30 but never stop before phase 2 starts
        if no_imp >= 30 and ep > phase1_end:
            print(f"  Early stopping at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    # Ensure all params unfrozen for inference
    _unfreeze(model)
    print(f"Best val_r: {best_r:.4f}")
    return model
