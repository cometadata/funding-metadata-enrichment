import matplotlib.pyplot as plt
import numpy as np

out_dir = "/Users/adambuttrick/Documents/GitHub/cometadata/funding-metadata-enrichment/evals/benchmark_results"

cross_configs = [
    "Q3-8B\nbase",
    "Q3-8B\nbase Think",
    "Q3-8B\nglm-distil",
    "Q3-8B\nglm-distil Think",
    "Q3-8B\nep2p1 twostage",
    "Q3-8B\nep2p1 two. Think",
    "Q3.5-9B\nbase",
    "Q3.5-9B\nbase Think",
    "Q3.5-9B\nglm-distil",
    "Q3.5-9B\nglm-distil Think",
]

cross_precision = {
    "Funder":         [0.767, 0.844, 0.827, 0.855, 0.847, 0.847, 0.822, 0.863, 0.823, 0.859],
    "Award ID":       [0.722, 0.776, 0.740, 0.784, 0.756, 0.771, 0.789, 0.817, 0.782, 0.799],
    "Funding Scheme": [0.507, 0.531, 0.560, 0.598, 0.546, 0.525, 0.833, 0.917, 0.658, 0.701],
    "Award Title":    [0.545, 0.387, 0.214, 0.278, 0.500, 0.526, 0.446, 0.643, 0.300, 0.681],
}

cross_recall = {
    "Funder":         [0.944, 0.903, 0.895, 0.909, 0.930, 0.914, 0.927, 0.863, 0.931, 0.914],
    "Award ID":       [0.775, 0.799, 0.766, 0.790, 0.763, 0.768, 0.834, 0.789, 0.820, 0.807],
    "Funding Scheme": [0.098, 0.215, 0.196, 0.282, 0.318, 0.296, 0.084, 0.062, 0.140, 0.229],
    "Award Title":    [0.070, 0.140, 0.070, 0.116, 0.047, 0.116, 0.430, 0.209, 0.384, 0.372],
}

line_styles = {
    "Funder":         {"color": "#2ca02c", "marker": "s", "ms": 10, "lw": 3},
    "Award ID":       {"color": "#d4881c", "marker": "s", "ms": 10, "lw": 3},
    "Funding Scheme": {"color": "#1f77b4", "marker": "D", "ms": 9, "lw": 2.5},
    "Award Title":    {"color": "#9467bd", "marker": "D", "ms": 9, "lw": 2.5},
}


def make_cross_chart(data, metric_label, filename, y_range):
    fig, ax = plt.subplots(figsize=(20, 9))

    # Background shading for model families
    ax.axvspan(-0.5, 5.5, alpha=0.06, color="#2196F3", zorder=0)
    ax.axvspan(5.5, 9.5, alpha=0.06, color="#FF9800", zorder=0)
    ax.text(2.5, 0.97, "Qwen3-8B", ha="center", fontsize=13, fontweight="bold",
            color="#1565C0", alpha=0.5, transform=ax.get_xaxis_transform())
    ax.text(7.5, 0.97, "Qwen3.5-9B", ha="center", fontsize=13, fontweight="bold",
            color="#E65100", alpha=0.5, transform=ax.get_xaxis_transform())

    # Think variant shading
    for i, label in enumerate(cross_configs):
        if "Think" in label:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="#FFC107", zorder=0)

    x = np.arange(len(cross_configs))

    for field_name, values in data.items():
        style = line_styles[field_name]
        ax.plot(x, values, color=style["color"], marker=style["marker"],
                markersize=style["ms"], linewidth=style["lw"],
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{field_name} {metric_label}", zorder=3)

        for i, val in enumerate(values):
            ax.annotate(f"{val:.3f}", (i, val),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                        color=style["color"])

        best_idx = np.argmax(values)
        ax.plot(x[best_idx], values[best_idx], "o",
                markersize=18, markeredgecolor=style["color"],
                markerfacecolor="none", markeredgewidth=2.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(cross_configs, fontsize=9, fontweight="bold")
    ax.set_xlabel("Training Config", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
    ax.set_title(f"Cross-Family {metric_label} Comparison: Qwen3-8B vs Qwen3.5-9B",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(y_range)
    ax.legend(fontsize=11, loc="best", framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.3, linestyle="-")
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


make_cross_chart(cross_precision, "Precision", "cross_family_precision.png", y_range=(0.0, 1.0))
make_cross_chart(cross_recall, "Recall", "cross_family_recall.png", y_range=(0.0, 1.0))
