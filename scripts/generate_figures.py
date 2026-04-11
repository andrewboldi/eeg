#!/usr/bin/env python3
"""Generate publication-quality figures for the EEG scalp-to-in-ear prediction paper.

Outputs PNG (300 DPI) and PDF to results/figures/.

Figures:
  1. Leaderboard progression (narrowband, broadband 27ch, broadband 46ch)
  2. Scaling law (log params vs test_r on Subject 13)
  3. Per-subject LOSO bar chart (colored by difficulty)
  4. Architecture sweep (iter039-054 horizontal bar chart)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH = RESULTS / "benchmark"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette
C_NARROW = "#5B7FA5"      # steel blue
C_BROAD27 = "#E07A3A"     # burnt orange
C_BROAD46 = "#4CAF50"     # green
C_ACCENT = "#D32F2F"      # red for breakthrough markers
C_EASY = "#66BB6A"
C_MEDIUM = "#FFA726"
C_HARD = "#EF5350"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_num(model_name: str) -> int | None:
    """Extract iteration number from model name like 'iter038_broadband_fir'."""
    for part in model_name.split("_"):
        if part.startswith("iter"):
            try:
                return int(part[4:])
            except ValueError:
                pass
    # Handle closed_form_baseline -> iter 7
    if "closed_form" in model_name and "broadband" not in model_name and "46ch" not in model_name:
        return 7
    return None


# ======================================================================
# Figure 1: Leaderboard Progression
# ======================================================================
def fig_leaderboard_progression():
    # --- Narrowband (leaderboard.jsonl) ---
    narrow_raw = _load_jsonl(BENCH / "leaderboard.jsonl")
    # Deduplicate: keep best result per iteration
    narrow_best: dict[int, float] = {}
    for row in narrow_raw:
        it = _iter_num(row["model"])
        if it is not None:
            narrow_best[it] = max(narrow_best.get(it, 0), row["mean_r"])
    narrow_iters = sorted(narrow_best.keys())
    narrow_rs = [narrow_best[i] for i in narrow_iters]

    # --- Broadband 27ch ---
    broad27_raw = _load_jsonl(BENCH / "leaderboard_broadband.jsonl")
    broad27_best: dict[int, float] = {}
    for row in broad27_raw:
        it = _iter_num(row["model"])
        if it is not None:
            broad27_best[it] = max(broad27_best.get(it, 0), row["mean_r"])
        elif "closed_form" in row["model"]:
            broad27_best[7] = max(broad27_best.get(7, 0), row["mean_r"])
    broad27_iters = sorted(broad27_best.keys())
    broad27_rs = [broad27_best[i] for i in broad27_iters]

    # --- Broadband 46ch ---
    broad46_raw = _load_jsonl(BENCH / "leaderboard_broadband_46ch.jsonl")
    broad46_best: dict[int, float] = {}
    for row in broad46_raw:
        it = _iter_num(row["model"])
        if it is not None:
            broad46_best[it] = max(broad46_best.get(it, 0), row["mean_r"])
        elif "closed_form" in row["model"]:
            broad46_best[7] = max(broad46_best.get(7, 0), row["mean_r"])
    broad46_iters = sorted(broad46_best.keys())
    broad46_rs = [broad46_best[i] for i in broad46_iters]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(narrow_iters, narrow_rs, "o-", color=C_NARROW, markersize=5,
            linewidth=1.5, label="Narrowband 1--9 Hz (27 ch)", zorder=3)
    ax.plot(broad27_iters, broad27_rs, "s-", color=C_BROAD27, markersize=6,
            linewidth=1.5, label="Broadband 1--45 Hz (27 ch)", zorder=3)
    ax.plot(broad46_iters, broad46_rs, "D-", color=C_BROAD46, markersize=6,
            linewidth=1.5, label="Broadband 1--45 Hz (46 ch)", zorder=3)

    # --- Breakthrough annotations ---
    breakthroughs = [
        (9, narrow_best.get(9, 0.373),
         "FIR spatio-temporal\nfilter (r=0.373)"),
        (38, broad27_best.get(38, 0.465),
         "Broadband FIR\n(r=0.465)"),
        (39, broad46_best.get(39, 0.638),
         "46-ch deep broadband\n(r=0.638)"),
    ]
    for bx, by, label in breakthroughs:
        ax.annotate(
            label, xy=(bx, by),
            xytext=(0, 22), textcoords="offset points",
            fontsize=8, fontweight="bold", color=C_ACCENT,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=1.2),
        )
        ax.plot(bx, by, "*", color=C_ACCENT, markersize=14, zorder=5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Pearson $r$")
    ax.set_title("Leaderboard Progression: Scalp-to-In-Ear EEG Prediction")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_ylim(0.30, 0.70)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"leaderboard_progression.{fmt}")
    plt.close(fig)
    print(f"  [1/4] leaderboard_progression saved")


# ======================================================================
# Figure 2: Scaling Law
# ======================================================================
def fig_scaling_law():
    with open(RESULTS / "scaling_law.json") as f:
        data = json.load(f)

    names = [d["name"] for d in data]
    params = np.array([d["n_params"] for d in data])
    test_r = np.array([d["test_r"] for d in data])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter
    ax.semilogx(params, test_r, "o", color=C_BROAD46, markersize=9, zorder=4)

    # Labels
    offsets = {
        "cf_baseline": (-8, -14),
        "tiny": (10, -10),
        "small": (10, 5),
        "medium": (-10, -14),
        "large": (10, 5),
        "xl": (-10, -14),
    }
    for name, p, r in zip(names, params, test_r):
        dx, dy = offsets.get(name, (8, -8))
        ax.annotate(
            f"{name}\n({p:,.0f})", xy=(p, r),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )

    # Horizontal line at CF baseline
    ax.axhline(test_r[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(params[-1] * 1.1, test_r[0] - 0.002, "CF baseline", fontsize=8,
            color="gray", va="top")

    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Test Pearson $r$ (Subject 13)")
    ax.set_title("Scaling Law: Model Size vs. Prediction Quality")
    ax.set_ylim(0.71, 0.77)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.grid(axis="y", alpha=0.3)

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"scaling_law.{fmt}")
    plt.close(fig)
    print(f"  [2/4] scaling_law saved")


# ======================================================================
# Figure 3: Per-Subject LOSO Bar Chart
# ======================================================================
def fig_per_subject():
    # Data from subject_analysis.md / leaderboard_loso_full.jsonl
    loso_data = _load_jsonl(BENCH / "leaderboard_loso_full.jsonl")
    per_subject = loso_data[0]["per_subject_r"]

    subjects = sorted(per_subject.keys(), key=lambda s: int(s))
    rs = [per_subject[s] for s in subjects]
    labels = [f"S{s}" for s in subjects]

    # Colour by difficulty
    colors = []
    for r in rs:
        if r >= 0.7:
            colors.append(C_EASY)
        elif r >= 0.55:
            colors.append(C_MEDIUM)
        else:
            colors.append(C_HARD)

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(subjects))
    bars = ax.bar(x, rs, color=colors, edgecolor="white", linewidth=0.5, width=0.7)

    # Value labels on bars
    for xi, r in zip(x, rs):
        ax.text(xi, r + 0.01, f"{r:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Subject")
    ax.set_ylabel("LOSO Pearson $r$")
    ax.set_title("Per-Subject LOSO Performance (CF Baseline, 46-ch Broadband)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=C_EASY, label="Easy ($r \\geq 0.70$)"),
        Patch(facecolor=C_MEDIUM, label="Medium ($0.55 \\leq r < 0.70$)"),
        Patch(facecolor=C_HARD, label="Hard ($r < 0.55$)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.9)

    # Horizontal mean line
    mean_r = np.mean(rs)
    ax.axhline(mean_r, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(subjects) - 0.5, mean_r + 0.015, f"mean = {mean_r:.3f}",
            fontsize=8, ha="right", color="black", alpha=0.7)

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"per_subject_loso.{fmt}")
    plt.close(fig)
    print(f"  [3/4] per_subject_loso saved")


# ======================================================================
# Figure 4: Architecture Sweep (iter039-054)
# ======================================================================
def fig_architecture_sweep():
    broad46_raw = _load_jsonl(BENCH / "leaderboard_broadband_46ch.jsonl")

    # Collect all entries with iteration numbers in [39, 54] range plus CF baseline
    entries: dict[str, float] = {}
    for row in broad46_raw:
        it = _iter_num(row["model"])
        name = row["model"]
        if it is not None and 39 <= it <= 54:
            # Keep best per model name
            entries[name] = max(entries.get(name, 0), row["mean_r"])
        elif "closed_form" in name:
            entries[name] = max(entries.get(name, 0), row["mean_r"])

    # Sort by mean_r ascending (best at top for horizontal bar chart)
    sorted_entries = sorted(entries.items(), key=lambda x: x[1])
    names = [e[0] for e in sorted_entries]
    rs = [e[1] for e in sorted_entries]

    # Prettify names
    pretty = []
    for n in names:
        p = n.replace("_46ch", "").replace("_", " ")
        # Capitalize first letter of each word
        p = p.title()
        pretty.append(p)

    fig, ax = plt.subplots(figsize=(7, 6))

    y = np.arange(len(names))
    colors = []
    best_r = max(rs)
    for r in rs:
        if r == best_r:
            colors.append(C_BROAD46)
        elif "closed_form" in names[rs.index(r)]:
            colors.append("#9E9E9E")  # grey for baseline
        else:
            colors.append(C_NARROW)

    bars = ax.barh(y, rs, color=colors, edgecolor="white", linewidth=0.5, height=0.65)

    # Value labels
    for yi, r in zip(y, rs):
        ax.text(r + 0.002, yi, f"{r:.4f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=8.5)
    ax.set_xlabel("Mean Pearson $r$")
    ax.set_title("Architecture Sweep: Broadband 46-ch Models")

    # Vertical line at CF baseline
    cf_r = entries.get("closed_form_46ch", None)
    if cf_r is not None:
        ax.axvline(cf_r, color="#9E9E9E", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlim(0.54, max(rs) + 0.03)
    ax.grid(axis="x", alpha=0.3)

    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"architecture_sweep.{fmt}")
    plt.close(fig)
    print(f"  [4/4] architecture_sweep saved")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("Generating publication figures...")
    fig_leaderboard_progression()
    fig_scaling_law()
    fig_per_subject()
    fig_architecture_sweep()
    print(f"\nAll figures saved to {FIGURES}/")
    print("  Formats: PNG (300 DPI) and PDF")
