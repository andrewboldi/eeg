"""Iteration 047: Spatial Positional Encoding Transformer (REVE-inspired).

Key innovation from REVE: 3D Fourier positional encoding for electrode positions.
Each input channel becomes a token with temporal features + spatial PE.
Cross-attention decoder maps to in-ear channels using their spatial PE.

Architecture:
  1. Per-channel temporal encoding: Conv1d(1, D, kernel_size=T) -> single D-dim embedding
  2. 3D Fourier PE from (x,y,z) electrode coordinates, projected to D
  3. Transformer encoder: 4 layers, 4 heads, dim=64
  4. Cross-attention decoder: 12 output queries with in-ear spatial PE
  5. Linear projection back to T time steps
  6. Skip connection from CF weights (Conv1d C_in->C_out, kernel=1)

Confidence: 60% -- spatial PE is a good inductive bias for EEG, but the
single-token-per-channel compression is aggressive and may lose temporal info.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Approximate 3D electrode coordinates (unit sphere, RAS convention)
# ---------------------------------------------------------------------------

# 27 scalp channels (standard 10-20 + extensions)
SCALP_COORDS = {
    "Fp1": (0.31, 0.95, 0.0),
    "Fp2": (-0.31, 0.95, 0.0),
    "F3": (0.55, 0.67, 0.5),
    "F4": (-0.55, 0.67, 0.5),
    "C3": (0.71, 0.0, 0.71),
    "C4": (-0.71, 0.0, 0.71),
    "P3": (0.55, -0.67, 0.5),
    "P4": (-0.55, -0.67, 0.5),
    "O1": (0.31, -0.95, 0.0),
    "O2": (-0.31, -0.95, 0.0),
    "F7": (0.81, 0.59, 0.0),
    "F8": (-0.81, 0.59, 0.0),
    "T7": (1.0, 0.0, 0.0),
    "T8": (-1.0, 0.0, 0.0),
    "P7": (0.81, -0.59, 0.0),
    "P8": (-0.81, -0.59, 0.0),
    "Fz": (0.0, 0.72, 0.69),
    "Cz": (0.0, 0.0, 1.0),
    "Pz": (0.0, -0.72, 0.69),
    "Oz": (0.0, -1.0, 0.0),
    "FC1": (0.35, 0.35, 0.87),
    "FC2": (-0.35, 0.35, 0.87),
    "CP1": (0.35, -0.35, 0.87),
    "CP2": (-0.35, -0.35, 0.87),
    "FC5": (0.81, 0.31, 0.5),
    "FC6": (-0.81, 0.31, 0.5),
    "CP5": (0.81, -0.31, 0.5),
}

# We may also have CP6 as the 28th scalp channel in some configs
EXTRA_SCALP = {
    "CP6": (-0.81, -0.31, 0.5),
}

# 19 around-ear channels (cEEGrid, approximate positions)
# Left ear: cEL1-cEL9, roughly at left temporal-mastoid
_LEFT_EAR_BASE = np.array([0.95, -0.20, -0.20])
AROUND_EAR_COORDS = {}
for i in range(1, 10):
    offset = np.array([0.0, 0.03 * (i - 5), 0.02 * (i - 5)])
    pos = _LEFT_EAR_BASE + offset
    AROUND_EAR_COORDS[f"cEL{i}"] = tuple(pos.tolist())

# Right ear: cER1-cER10
_RIGHT_EAR_BASE = np.array([-0.95, -0.20, -0.20])
for i in range(1, 11):
    offset = np.array([0.0, 0.03 * (i - 5.5), 0.02 * (i - 5.5)])
    pos = _RIGHT_EAR_BASE + offset
    AROUND_EAR_COORDS[f"cER{i}"] = tuple(pos.tolist())

# 12 in-ear channels (6 left, 6 right)
_LEFT_INEAR_BASE = np.array([0.97, -0.15, -0.25])
_RIGHT_INEAR_BASE = np.array([-0.97, -0.15, -0.25])
INEAR_COORDS = {}
for i, suffix in enumerate(["A", "B", "C", "S", "T", "L"]):
    offset = np.array([0.0, 0.01 * (i - 2.5), 0.01 * (i - 2.5)])
    INEAR_COORDS[f"EL{suffix}"] = tuple((_LEFT_INEAR_BASE + offset).tolist())
    INEAR_COORDS[f"ER{suffix}"] = tuple((_RIGHT_INEAR_BASE + offset).tolist())


def _get_input_coords(C_in: int) -> torch.Tensor:
    """Build (C_in, 3) coordinate tensor for input channels.

    Handles C_in=27 (scalp only) or C_in=46 (scalp + around-ear).
    Falls back to learnable-compatible random coords if channel count is unexpected.
    """
    coords = []

    # Always start with 27 scalp channels (ordered as in SCALP_COORDS)
    scalp_names = list(SCALP_COORDS.keys())
    for name in scalp_names:
        coords.append(SCALP_COORDS[name])

    if C_in > 27:
        # Add CP6 if we have 28 scalp channels
        if C_in >= 28 and "CP6" in EXTRA_SCALP:
            coords.append(EXTRA_SCALP["CP6"])

        # Add around-ear channels
        ear_names = list(AROUND_EAR_COORDS.keys())
        for name in ear_names:
            coords.append(AROUND_EAR_COORDS[name])

    # Pad or truncate to match C_in
    while len(coords) < C_in:
        # Use small random offsets around centroid for unknown channels
        coords.append((np.random.randn() * 0.1, np.random.randn() * 0.1, np.random.randn() * 0.1))
    coords = coords[:C_in]

    return torch.tensor(coords, dtype=torch.float32)


def _get_output_coords(C_out: int) -> torch.Tensor:
    """Build (C_out, 3) coordinate tensor for output (in-ear) channels."""
    coords = []
    inear_names = list(INEAR_COORDS.keys())
    for name in inear_names[:C_out]:
        coords.append(INEAR_COORDS[name])

    while len(coords) < C_out:
        coords.append((np.random.randn() * 0.1, np.random.randn() * 0.1, np.random.randn() * 0.1))
    coords = coords[:C_out]

    return torch.tensor(coords, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3D Fourier Positional Encoding
# ---------------------------------------------------------------------------

class FourierPE3D(nn.Module):
    """3D Fourier positional encoding for electrode coordinates.

    For each (x, y, z), computes sin/cos at n_freq frequency bands per axis.
    Output dim = 2 * 3 * n_freq (sin + cos, 3 axes, n_freq bands).
    Then projects to model dimension.
    """

    def __init__(self, d_model: int, n_freq: int = 8):
        super().__init__()
        self.n_freq = n_freq
        raw_dim = 2 * 3 * n_freq  # sin+cos for x,y,z at each frequency
        self.proj = nn.Linear(raw_dim, d_model)

        # Frequency bands: logarithmically spaced from 1 to 2^(n_freq-1)
        freqs = 2.0 ** torch.arange(n_freq).float()  # [1, 2, 4, 8, ...]
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) electrode positions

        Returns:
            (N, d_model) positional embeddings
        """
        # coords: (N, 3), freqs: (n_freq,)
        # Expand: (N, 3, 1) * (1, 1, n_freq) -> (N, 3, n_freq)
        scaled = coords.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0) * math.pi

        # sin and cos: each (N, 3, n_freq)
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)

        # Concatenate: (N, 3, 2*n_freq) -> (N, 6*n_freq)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (N, 3, 2*n_freq)
        enc = enc.reshape(coords.shape[0], -1)  # (N, 6*n_freq)

        return self.proj(enc)


# ---------------------------------------------------------------------------
# Cross-Attention Decoder
# ---------------------------------------------------------------------------

class CrossAttentionDecoder(nn.Module):
    """Decoder that uses learnable output queries with spatial PE."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(d_model),
            }))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, C_out, D) output queries
            memory: (B, C_in, D) encoder output

        Returns:
            (B, C_out, D)
        """
        x = queries
        for layer in self.layers:
            # Cross-attention: queries attend to encoder memory
            attn_out, _ = layer["cross_attn"](x, memory, memory)
            x = layer["norm1"](x + attn_out)

            # FFN
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class SpatialPETransformer(nn.Module):
    """REVE-inspired transformer with 3D Fourier positional encoding.

    Each input channel is a token. Temporal features are extracted per-channel
    then combined with spatial positional encoding. A transformer encoder
    processes cross-channel interactions, and a cross-attention decoder
    maps to output channels using their spatial PE.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        T: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_freq: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T
        self.d_model = d_model

        # Per-channel temporal encoding: maps T time steps to d_model features
        # Use a small 1D conv stack instead of single huge kernel for efficiency
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=min(7, T), padding=min(7, T) // 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (1, d_model, 1) -> pool to single embedding
        )

        # 3D Fourier PE for input and output electrodes
        self.input_pe = FourierPE3D(d_model, n_freq=n_freq)
        self.output_pe = FourierPE3D(d_model, n_freq=n_freq)

        # Register coordinate buffers
        input_coords = _get_input_coords(C_in)
        output_coords = _get_output_coords(C_out)
        self.register_buffer("input_coords", input_coords)
        self.register_buffer("output_coords", output_coords)

        # Learnable output query embeddings (in addition to spatial PE)
        self.output_queries = nn.Parameter(torch.randn(C_out, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Cross-attention decoder
        self.decoder = CrossAttentionDecoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_dec_layers, dropout=dropout
        )

        # Output projection: d_model -> T time steps per output channel
        self.output_proj = nn.Linear(d_model, T)

        # Skip connection: simple linear spatial map (initialized from CF)
        self.skip = nn.Conv1d(C_in, C_out, kernel_size=1, bias=False)

        # Mixing coefficient for skip connection
        self.skip_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) input EEG

        Returns:
            (B, C_out, T) predicted EEG
        """
        B, C_in, T = x.shape

        # --- Skip connection ---
        skip_out = self.skip(x)  # (B, C_out, T)

        # --- Temporal encoding per channel ---
        # Reshape to process each channel independently: (B*C_in, 1, T)
        x_flat = x.reshape(B * C_in, 1, T)
        temporal_features = self.temporal_enc(x_flat)  # (B*C_in, d_model, 1)
        temporal_features = temporal_features.squeeze(-1)  # (B*C_in, d_model)
        temporal_features = temporal_features.reshape(B, C_in, self.d_model)  # (B, C_in, D)

        # --- Add 3D Fourier spatial PE ---
        spatial_pe = self.input_pe(self.input_coords)  # (C_in, D)
        tokens = temporal_features + spatial_pe.unsqueeze(0)  # (B, C_in, D)

        # --- Transformer encoder ---
        encoded = self.encoder(tokens)  # (B, C_in, D)

        # --- Cross-attention decoder ---
        # Build output queries: learnable embedding + spatial PE
        output_spatial_pe = self.output_pe(self.output_coords)  # (C_out, D)
        queries = self.output_queries + output_spatial_pe  # (C_out, D)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, C_out, D)

        decoded = self.decoder(queries, encoded)  # (B, C_out, D)

        # --- Project to time domain ---
        transformer_out = self.output_proj(decoded)  # (B, C_out, T)

        # --- Combine with skip ---
        alpha = torch.sigmoid(self.skip_alpha)
        out = alpha * skip_out + (1 - alpha) * transformer_out

        return out


# ---------------------------------------------------------------------------
# Loss and validation (same as other iterations)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# build_and_train
# ---------------------------------------------------------------------------

def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build and train REVE-inspired spatial PE transformer."""

    # Infer T from data
    T = train_ds.scalp.shape[2]
    print(f"Spatial PE Transformer: C_in={C_scalp}, C_out={C_inear}, T={T}")

    # Step 1: Fit closed-form baseline for skip initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build model
    model = SpatialPETransformer(
        C_in=C_scalp,
        C_out=C_inear,
        T=T,
        d_model=64,
        n_heads=4,
        n_enc_layers=4,
        n_dec_layers=2,
        n_freq=8,
        dropout=0.1,
    ).to(device)

    # Initialize skip connection from CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))  # (C_out, C_in, 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Spatial PE Transformer params: {n_params:,}")

    # Step 3: Train
    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_val_r = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, 151):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)

            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(scalp.shape[0], device=device)
            scalp = lam * scalp + (1 - lam) * scalp[idx]
            inear = lam * inear + (1 - lam) * inear[idx]

            # Channel dropout (15%)
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device) > 0.15).float()
            scalp = scalp * mask / 0.85

            optimizer.zero_grad()
            pred = model(scalp)
            loss = loss_fn(pred, inear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_r = validate_correlation(model, val_loader, device)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f} val_r={val_r:.4f} (best={best_val_r:.4f})")

        # Early stopping
        if no_improve >= 30:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Final best val_r: {best_val_r:.4f}")

    return model
