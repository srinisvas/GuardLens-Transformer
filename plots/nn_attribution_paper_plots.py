"""
Paper-quality plotting utilities for NN token attribution / multi-turn guardrail attribution results.

Expected JSON inputs:
  - causal_eval_results.json
  - attribution_utility.json
  - deconfounded_results.json
  - paraphrase_eval_full.json (optional)

Usage:
  python nn_attribution_paper_plots.py \
    --causal causal_eval_results.json \
    --utility attribution_utility.json \
    --deconfounded deconfounded_results.json \
    --paraphrase paraphrase_eval_full.json \
    --outdir paper_figures

Outputs PDF and PNG versions of each figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Display names used in paper figures. Keep method names implementation-neutral.
# -----------------------------------------------------------------------------
METHOD_LABELS: Dict[str, str] = {
    "guardlens": "Proposed",
    "surface_risk": "Surface-risk",
    "grad_x_input": "Grad×Input",
    "integrated_gradients": "IG",
    "attention": "Attention-based",
    "random": "Random",
}

# Muted, print-friendly palette. All markers are distinct for grayscale reading.
METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "guardlens": {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "surface_risk": {"color": "#8b2f2f", "marker": "s", "linestyle": "--"},
    "grad_x_input": {"color": "#5f7f3f", "marker": "^", "linestyle": "-."},
    "integrated_gradients": {"color": "#6b5b95", "marker": "D", "linestyle": ":"},
    "attention": {"color": "#8c6d31", "marker": "v", "linestyle": "-"},
    "random": {"color": "#666666", "marker": "x", "linestyle": ":"},
}

K_ORDER = ["5%", "10%", "15%", "20%"]
K_VALUES = np.array([5, 10, 15, 20])


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_paper_style() -> None:
    """Set a compact ACL/EMNLP-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def finish_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def get_full_test(causal: Mapping) -> Mapping:
    if "full_test" not in causal:
        raise KeyError("Expected causal_eval_results.json to contain key 'full_test'.")
    return causal["full_test"]


def metric_curve(method_block: Mapping, metric: str) -> List[float]:
    if metric == "dd":
        d = method_block["deviation_drops"]
        return [float(d[k]) for k in K_ORDER]
    if metric == "flip":
        d = method_block["flip_rates"]
        return [float(d[f"flip@{k}"]) for k in K_ORDER]
    if metric == "necessity":
        d = method_block["necessity"]
        return [float(d[k]) for k in K_ORDER]
    raise ValueError(f"Unknown metric: {metric}")


# -----------------------------------------------------------------------------
# Figure 1: DD@k and Flip@k curves
# -----------------------------------------------------------------------------

def plot_dd_flip_curves(
    causal_json: str | Path,
    outdir: str | Path,
    methods: Iterable[str] = ("guardlens", "surface_risk", "grad_x_input", "integrated_gradients"),
    name: str = "fig_dd_flip_curves",
) -> None:
    """Plot DD@k and Flip@k curves in a compact two-panel figure."""
    set_paper_style()
    causal = load_json(causal_json)
    full = get_full_test(causal)
    methods = [m for m in methods if m in full]

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), sharex=True)

    for metric, ax, title, ylabel in [
        ("dd", axes[0], "Deviation drop", "DD@k"),
        ("flip", axes[1], "Prediction flip", "Flip@k"),
    ]:
        for method in methods:
            style = METHOD_STYLES.get(method, {})
            ax.plot(
                K_VALUES,
                metric_curve(full[method], metric),
                label=METHOD_LABELS.get(method, method),
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
            )
        ax.set_title(title)
        ax.set_xlabel("Removed tokens (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(-0.04, 0.66)
        finish_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(labels), 5),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    save_figure(fig, Path(outdir), name)


# -----------------------------------------------------------------------------
# Figure 2: Pivot-window localization curve
# -----------------------------------------------------------------------------

def plot_pivot_window(
    utility_json: str | Path,
    outdir: str | Path,
    name: str = "fig_pivot_window",
) -> None:
    """Plot windowed pivot accuracy from attribution_utility.json."""
    set_paper_style()
    data = load_json(utility_json)
    win = data["pivot_window"]["window_accuracy"]

    labels = ["Exact", "±1", "±2", "±3", "±5"]
    keys = ["within_0", "within_1", "within_2", "within_3", "within_5"]
    values = [float(win[k]["accuracy"]) for k in keys]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    ax.plot(
        x,
        values,
        color=METHOD_STYLES["guardlens"]["color"],
        marker="o",
        linewidth=1.9,
    )
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.025, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Pivot window")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.76)
    finish_axis(ax)
    save_figure(fig, Path(outdir), name)


# -----------------------------------------------------------------------------
# Figure 3: Utility tradeoff scatter
# -----------------------------------------------------------------------------

def plot_utility_tradeoff(
    utility_json: str | Path,
    outdir: str | Path,
    methods: Iterable[str] = ("guardlens", "surface_risk", "grad_x_input", "integrated_gradients", "attention", "random"),
    name: str = "fig_utility_tradeoff",
) -> None:
    """Plot DD@15 versus boundary FPR with iso-utility guides."""
    set_paper_style()
    data = load_json(utility_json)
    util = data["utility_boundary"]
    methods = [m for m in methods if m in util]

    fig, ax = plt.subplots(figsize=(3.45, 2.45))

    # Iso-utility guide lines, utility = DD - FPR.
    x = np.linspace(0.0, 0.42, 200)
    for c in (0.20, 0.50):
        y = x + c
        mask = y <= 0.68
        ax.plot(x[mask], y[mask], color="0.55", linestyle="--", linewidth=0.8, zorder=1)
        if np.any(mask):
            idx = np.where(mask)[0][-1]
            ax.text(x[idx] - 0.075, y[idx] + 0.012, f"U={c:.2f}", fontsize=6.8, color="0.35")

    for method in methods:
        p = util[method]
        style = METHOD_STYLES.get(method, {})
        fpr = float(p["fpr"])
        dd = float(p["dd"])
        ax.scatter(
            fpr,
            dd,
            s=52,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            edgecolor="black" if method != "random" else "none",
            linewidth=0.45,
            zorder=3,
        )

        label = METHOD_LABELS.get(method, method)
        # Manual offsets chosen for the current metric range.
        offsets = {
            "guardlens": (0.010, 0.018),
            "surface_risk": (-0.175, 0.018),
            "grad_x_input": (0.010, 0.012),
            "integrated_gradients": (0.010, -0.030),
            "attention": (0.010, -0.030),
            "random": (0.010, 0.012),
        }
        dx, dy = offsets.get(method, (0.010, 0.012))
        ax.text(fpr + dx, dd + dy, label, fontsize=6.9)

    ax.set_xlabel("Boundary FPR")
    ax.set_ylabel("DD@15")
    ax.set_xlim(-0.02, 0.43)
    ax.set_ylim(-0.05, 0.68)
    finish_axis(ax, grid_axis="both")
    save_figure(fig, Path(outdir), name)


# -----------------------------------------------------------------------------
# Optional: Deconfounded DD@15 grouped bar chart
# -----------------------------------------------------------------------------

def plot_deconfounded_dd(
    deconfounded_json: str | Path,
    outdir: str | Path,
    name: str = "fig_deconfounded_dd",
) -> None:
    """Plot DD@15 across deconfounded variants for Proposed and Surface-risk."""
    set_paper_style()
    data = load_json(deconfounded_json)

    conds = [
        ("original", "Original"),
        ("sr_neutralized", "SR-neutralized"),
        ("noise_equalized", "Noise-eq."),
        ("combined", "Combined"),
    ]
    methods = ["guardlens", "surface_risk"]

    vals = {
        m: [float(data[c][m]["deviation_drops"]["15%"])
            for c, _ in conds]
        for m in methods
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = np.arange(len(conds))
    width = 0.34

    for i, method in enumerate(methods):
        offset = (i - 0.5) * width
        style = METHOD_STYLES[method]
        ax.bar(
            x + offset,
            vals[method],
            width=width,
            label=METHOD_LABELS[method],
            color=style["color"],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in conds], rotation=15, ha="right")
    ax.set_ylabel("DD@15")
    ax.set_ylim(0.35, 0.58)
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax)
    save_figure(fig, Path(outdir), name)


# -----------------------------------------------------------------------------
# Optional: Paraphrase robustness summary
# -----------------------------------------------------------------------------

def plot_paraphrase_stability(
    paraphrase_json: str | Path,
    outdir: str | Path,
    name: str = "fig_paraphrase_stability",
) -> None:
    """Plot turn/token rank stability for full and subset paraphrase evaluations."""
    set_paper_style()
    data = load_json(paraphrase_json)

    rows: List[Tuple[str, float, float]] = [
        (
            "Full",
            float(data["aggregate"]["turn_rho_mean"]),
            float(data["aggregate"]["token_rho_mean"]),
        ),
        (
            "Implicit",
            float(data["implicit_subset"]["turn_rho"]),
            float(data["implicit_subset"]["token_rho"]),
        ),
    ]
    # Some runs call explicit lexical-pivot subset explicit_subset.
    if "explicit_subset" in data:
        rows.append(
            (
                "Explicit",
                float(data["explicit_subset"]["turn_rho"]),
                float(data["explicit_subset"]["token_rho"]),
            )
        )

    labels = [r[0] for r in rows]
    turn = [r[1] for r in rows]
    token = [r[2] for r in rows]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.bar(x - width / 2, turn, width=width, label="Turn $\\rho$", color="#1f4e79")
    ax.bar(x + width / 2, token, width=width, label="Token $\\rho$", color="#8b2f2f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_ylim(0.90, 1.00)
    ax.legend(frameon=False, loc="lower right")
    finish_axis(ax)
    save_figure(fig, Path(outdir), name)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-quality figures for attribution evaluation.")
    parser.add_argument("--causal", type=str, default="causal_eval_results.json")
    parser.add_argument("--utility", type=str, default="attribution_utility.json")
    parser.add_argument("--deconfounded", type=str, default="deconfounded_results.json")
    parser.add_argument("--paraphrase", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="paper_figures")
    parser.add_argument("--skip-optional", action="store_true", help="Only generate the three main figures.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = Path(args.outdir)

    plot_dd_flip_curves(args.causal, outdir)
    plot_pivot_window(args.utility, outdir)
    plot_utility_tradeoff(args.utility, outdir)

    if not args.skip_optional:
        if Path(args.deconfounded).exists():
            plot_deconfounded_dd(args.deconfounded, outdir)
        if args.paraphrase and Path(args.paraphrase).exists():
            plot_paraphrase_stability(args.paraphrase, outdir)

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
