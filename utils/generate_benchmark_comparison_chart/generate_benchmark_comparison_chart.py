#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Generate a benchmark comparison image for two model results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_metrics(path):
    with open(path) as f:
        return json.load(f)["aggregate_metrics"]


def draw_bar(ax, x, y, value, bar_width=1.0, bar_height=0.18):
    """Draw a bar with solid white fill and dark gray remainder."""
    filled_w = bar_width * value
    remainder_w = bar_width - filled_w
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), filled_w, bar_height,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none",
    ))
    if remainder_w > 0.005:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + filled_w, y), remainder_w, bar_height,
            boxstyle="square,pad=0", facecolor="#333333", edgecolor="none",
        ))


def generate(result_a_path, result_b_path, label_a, label_b, output_path):
    a = load_metrics(result_a_path)
    b = load_metrics(result_b_path)

    levels = [
        ("Funder", "funder"),
        ("Award ID", "award_id"),
        ("Funding Scheme", "funding_scheme"),
        ("Award Title", "award_title"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("black")

    font = {"fontfamily": "monospace", "color": "white"}
    font_gray = {"fontfamily": "monospace", "color": "#999999"}

    lbl_x = 0.2
    a_p, a_r, a_f1 = 2.8, 4.2, 5.6
    b_p, b_r, b_f1 = 7.6, 9.0, 10.4
    bar_w = 1.0

    ax.text((a_p + a_f1) / 2, 7.5, label_a, fontsize=13, fontweight="bold",
            ha="center", va="center", **font)
    ax.text((b_p + b_f1) / 2, 7.5, label_b, fontsize=13, fontweight="bold",
            ha="center", va="center", **font)

    for x in [a_p, b_p]:
        ax.text(x + bar_w / 2, 7.0, "P", fontsize=11, ha="center", va="center", **font_gray)
    for x in [a_r, b_r]:
        ax.text(x + bar_w / 2, 7.0, "R", fontsize=11, ha="center", va="center", **font_gray)
    for x in [a_f1, b_f1]:
        ax.text(x + bar_w / 2, 7.0, "F1", fontsize=11, ha="center", va="center", **font_gray)

    ax.plot([0.1, 11.8], [6.65, 6.65], color="white", linewidth=1.5)

    div_x = (a_f1 + bar_w + b_p) / 2
    ax.plot([div_x, div_x], [0.5, 7.8], color="#555555", linewidth=1)

    y_start = 6.0
    row_h = 1.45

    for i, (label, key) in enumerate(levels):
        y = y_start - i * row_h
        ma = a[key]
        mb = b[key]

        ax.text(lbl_x, y, label, fontsize=12, fontweight="bold", va="center", **font)

        for x, val in [(a_p, ma["precision"]), (a_r, ma["recall"]), (a_f1, ma["f1"])]:
            ax.text(x + bar_w / 2, y + 0.05, f"{val:.3f}", fontsize=11,
                    ha="center", va="center", **font)
            draw_bar(ax, x, y - 0.42, val, bar_w)

        for x, val in [(b_p, mb["precision"]), (b_r, mb["recall"]), (b_f1, mb["f1"])]:
            ax.text(x + bar_w / 2, y + 0.05, f"{val:.3f}", fontsize=11,
                    ha="center", va="center", **font)
            draw_bar(ax, x, y - 0.42, val, bar_w)

        ax.plot([0.1, 11.8], [y - 0.65, y - 0.65], color="#555555", linewidth=0.5)

    bottom_y = y_start - len(levels) * row_h + row_h - 0.65
    ax.plot([0.1, 11.8], [bottom_y, bottom_y], color="white", linewidth=1.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a benchmark comparison chart for two model results."
    )
    parser.add_argument("result_a", type=Path, help="Path to first model's benchmark JSON")
    parser.add_argument("result_b", type=Path, help="Path to second model's benchmark JSON")
    parser.add_argument("--label-a", required=True, help="Display label for the first model")
    parser.add_argument("--label-b", required=True, help="Display label for the second model")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmark_comparison.png"),
                        help="Output image path (default: benchmark_comparison.png)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.result_a, args.result_b, args.label_a, args.label_b, args.output)
