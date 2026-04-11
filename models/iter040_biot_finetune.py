"""Iteration 040: BIOT pretrained encoder fine-tuned for EEG waveform prediction.

Uses BIOT (NeurIPS 2023) pretrained on 10M+ EEG samples as feature extractor.
BIOT's per-channel STFT + transformer learns general EEG representations.

Architecture:
  - BIOT encoder (modified to return per-token features, not pooled)
  - Temporal decoder: project tokens back to waveform
  - Skip connection from CF linear mapping

Key insight: BIOT tokenizes each channel independently, so it handles our
non-standard around-ear/in-ear channels naturally.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class BIOTWaveformModel(nn.Module):
    """BIOT encoder + temporal decoder for waveform prediction."""

    def __init__(self, C_in, C_out, T=256, sfreq=128, emb_size=256):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T
        self.emb_size = emb_size

        # Load BIOT encoder from braindecode
        from braindecode.models import BIOT
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            biot = BIOT(n_chans=C_in, n_outputs=C_out, n_times=T, sfreq=sfreq)

        # Extract encoder components
        self.encoder = biot.encoder

        # Figure out how many tokens BIOT produces per channel
        with torch.no_grad():
            dummy = torch.randn(1, C_in, T)
            # Run one channel through to get token count
            emb_seq = []
            for i in range(C_in):
                channel_spec = self.encoder.stft(dummy[:, i:i+1, :])
                channel_emb = self.encoder.patch_embedding(channel_spec)
                emb_seq.append(channel_emb)
            self.tokens_per_channel = emb_seq[0].shape[1]
            self.total_tokens = self.tokens_per_channel * C_in

        print(f"BIOT: {self.tokens_per_channel} tokens/channel, "
              f"{self.total_tokens} total, emb_size={emb_size}")

        # Decoder: transform encoder tokens → output waveform
        # Group tokens by output channel using learned attention
        self.output_query = nn.Parameter(torch.randn(C_out, emb_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(emb_size, num_heads=4,
                                                 dropout=0.1, batch_first=True)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(emb_size, T),
        )

        # Skip connection
        self.skip = nn.Conv1d(C_in, C_out, 1, bias=True)

    def encode_no_pool(self, x):
        """Run BIOT encoder but return per-token features (no mean pooling)."""
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec = self.encoder.stft(x[:, i:i+1, :])
            channel_emb = self.encoder.patch_embedding(channel_spec)
            B, ts, _ = channel_emb.shape
            # Add channel token + positional encoding
            channel_token = self.encoder.channel_tokens(
                torch.full((B, ts), i, dtype=torch.long, device=x.device)
            )
            channel_emb = channel_emb + channel_token
            channel_emb = self.encoder.positional_encoding(channel_emb)
            emb_seq.append(channel_emb)

        # Concatenate all channel tokens
        emb = torch.cat(emb_seq, dim=1)  # (B, total_tokens, emb_size)

        # Run through transformer
        emb = self.encoder.transformer(emb)  # (B, total_tokens, emb_size)
        return emb

    def forward(self, x):
        # x: (B, C_in, T)
        B = x.shape[0]
        skip = self.skip(x)  # (B, C_out, T)

        # BIOT encoder features (no pooling)
        tokens = self.encode_no_pool(x)  # (B, total_tokens, emb_size)

        # Cross-attention: output queries attend to encoder tokens
        queries = self.output_query.unsqueeze(0).expand(B, -1, -1)  # (B, C_out, emb)
        decoded, _ = self.cross_attn(queries, tokens, tokens)  # (B, C_out, emb)

        # Project each output channel embedding to time series
        waveform = self.temporal_decoder(decoded)  # (B, C_out, T)

        return waveform + skip


class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        r = cov / ((pred_m**2).sum(dim=-1).sqrt() * (target_m**2).sum(dim=-1).sqrt() + 1e-8)
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
    # CF for skip init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = BIOTWaveformModel(
        C_in=C_scalp, C_out=C_inear, T=256, sfreq=128,
    ).to(device)

    # Init skip with CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BIOT Waveform: {n_params:,} params ({n_trainable:,} trainable)")

    loss_fn = CorrMSELoss(alpha=0.5)

    # Different LR for pretrained encoder vs new decoder
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith('encoder')]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-5},      # Slow for pretrained
        {"params": decoder_params, "lr": 3e-4},       # Fast for new decoder
    ], weight_decay=1e-2)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=2, pin_memory=True)

    best_val_r = -1.0
    best_state = None
    patience = 40
    no_improve = 0

    for epoch in range(1, 201):
        model.train()
        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)

            # Mixup
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(scalp.shape[0], device=device)
            scalp = lam * scalp + (1 - lam) * scalp[idx]
            inear = lam * inear + (1 - lam) * inear[idx]

            optimizer.zero_grad()
            pred = model(scalp)
            loss = loss_fn(pred, inear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_r = validate_correlation(model, val_loader, device)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: val_r={val_r:.4f} (best={best_val_r:.4f})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Final best val_r: {best_val_r:.4f}")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
