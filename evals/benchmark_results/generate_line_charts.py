import matplotlib.pyplot as plt
import numpy as np

out_dir = "/Users/adambuttrick/Documents/GitHub/cometadata/funding-metadata-enrichment/evals/benchmark_results"

# --- Qwen3-8B family data (test split) ---
qwen3_8b_configs = [
    "base",
    "base\nThink",
    "glm-air-4.5\n-distil",
    "glm-air-4.5\n-distil Think",
    "ep2\nsynth",
    "ep2\nsynth Think",
    "ep2p1\ntwostage",
    "ep2p1\ntwostage Think",
]

qwen3_8b = {
    "Funder F1":         [0.847, 0.872, 0.860, 0.881, 0.882, 0.866, 0.886, 0.879],
    "Award ID F1":       [0.747, 0.787, 0.753, 0.787, 0.755, 0.737, 0.759, 0.769],
    "Funding Scheme F1": [0.164, 0.306, 0.290, 0.383, 0.347, 0.335, 0.402, 0.379],
    "Award Title F1":    [0.124, 0.205, 0.105, 0.164, 0.043, 0.121, 0.085, 0.190],
}

# --- Qwen3.5-9B family data (test split) ---
qwen35_9b_configs = [
    "base",
    "base\nThink",
    "glm-air-4.5\n-distil",
    "glm-air-4.5\n-distil Think",
]

qwen35_9b = {
    "Funder F1":         [0.871, 0.863, 0.874, 0.885],
    "Award ID F1":       [0.811, 0.803, 0.800, 0.803],
    "Funding Scheme F1": [0.152, 0.115, 0.230, 0.345],
    "Award Title F1":    [0.438, 0.316, 0.337, 0.481],
}

# Style config
line_styles = {
    "Funder F1":         {"color": "#2ca02c", "marker": "s", "ms": 10, "lw": 3},
    "Award ID F1":       {"color": "#d4881c", "marker": "s", "ms": 10, "lw": 3},
    "Funding Scheme F1": {"color": "#1f77b4", "marker": "D", "ms": 9, "lw": 2.5},
    "Award Title F1":    {"color": "#9467bd", "marker": "D", "ms": 9, "lw": 2.5},
}


def make_line_chart(configs, data, title, filename, y_range=None, highlight_best=True):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Light background shading - alternate groups
    # Shade "Think" variants
    for i in range(len(configs)):
        label = configs[i]
        if "Think" in label:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color="#FFC107", zorder=0)

    x = np.arange(len(configs))

    for metric_name, values in data.items():
        style = line_styles[metric_name]
        line, = ax.plot(x, values, color=style["color"], marker=style["marker"],
                        markersize=style["ms"], linewidth=style["lw"],
                        markeredgecolor="white", markeredgewidth=1.5,
                        label=metric_name, zorder=3)

        # Annotate each point
        for i, val in enumerate(values):
            offset_y = 0.012 if val == max(values) else 0.008
            ax.annotate(f"{val:.3f}", (i, val),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=style["color"])

        # Circle the best value
        if highlight_best:
            best_idx = np.argmax(values)
            ax.plot(x[best_idx], values[best_idx], "o",
                    markersize=18, markeredgecolor=style["color"],
                    markerfacecolor="none", markeredgewidth=2.5, zorder=4)

    # Add base model reference lines (dashed)
    for metric_name, values in data.items():
        base_val = values[0]
        style = line_styles[metric_name]
        ax.axhline(y=base_val, color=style["color"], linestyle="--",
                   alpha=0.4, linewidth=1.5, zorder=1)
        ax.text(len(configs) - 0.3, base_val, f"  base ({base_val:.3f})",
                color=style["color"], alpha=0.6, fontsize=8, va="center",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Config", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    if y_range:
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


# --- Chart 1: Qwen3-8B Funder + Award ID ---
make_line_chart(
    qwen3_8b_configs,
    {"Funder F1": qwen3_8b["Funder F1"], "Award ID F1": qwen3_8b["Award ID F1"]},
    "Qwen3-8B: Funder & Award ID F1 Across Training Configs",
    "qwen3_8b_funder_award_id.png",
    y_range=(0.70, 0.92),
)

# --- Chart 2: Qwen3-8B Funding Scheme + Award Title ---
make_line_chart(
    qwen3_8b_configs,
    {"Funding Scheme F1": qwen3_8b["Funding Scheme F1"], "Award Title F1": qwen3_8b["Award Title F1"]},
    "Qwen3-8B: Funding Scheme & Award Title F1 Across Training Configs",
    "qwen3_8b_scheme_title.png",
    y_range=(0.0, 0.50),
)

# --- Chart 3: Qwen3.5-9B Funder + Award ID ---
make_line_chart(
    qwen35_9b_configs,
    {"Funder F1": qwen35_9b["Funder F1"], "Award ID F1": qwen35_9b["Award ID F1"]},
    "Qwen3.5-9B: Funder & Award ID F1 Across Training Configs",
    "qwen35_9b_funder_award_id.png",
    y_range=(0.76, 0.92),
)

# --- Chart 4: Qwen3.5-9B Funding Scheme + Award Title ---
make_line_chart(
    qwen35_9b_configs,
    {"Funding Scheme F1": qwen35_9b["Funding Scheme F1"], "Award Title F1": qwen35_9b["Award Title F1"]},
    "Qwen3.5-9B: Funding Scheme & Award Title F1 Across Training Configs",
    "qwen35_9b_scheme_title.png",
    y_range=(0.0, 0.55),
)

# --- Chart 5: Cross-family best comparison (all 4 metrics) ---
cross_configs = [
    "Q3-8B\nbase",
    "Q3-8B\nbase Think",
    "Q3-8B\nglm-distil",
    "Q3-8B\nglm-distil Think",
    "Q3-8B\nep2p1 twostage",
    "Q3-8B\nep2p1 two. Think",
    "Q3-8B\n397B-distil-3ep",
    "Q3.5-9B\nbase",
    "Q3.5-9B\nbase Think",
    "Q3.5-9B\nglm-distil",
    "Q3.5-9B\nglm-distil Think",
]

cross_data = {
    "Funder F1":         [0.847, 0.872, 0.860, 0.881, 0.886, 0.879, 0.810, 0.871, 0.863, 0.874, 0.885],
    "Award ID F1":       [0.747, 0.787, 0.753, 0.787, 0.759, 0.769, 0.709, 0.811, 0.803, 0.800, 0.803],
    "Funding Scheme F1": [0.164, 0.306, 0.290, 0.383, 0.402, 0.379, 0.157, 0.152, 0.115, 0.230, 0.345],
    "Award Title F1":    [0.124, 0.205, 0.105, 0.164, 0.085, 0.190, 0.065, 0.438, 0.316, 0.337, 0.481],
}

fig, ax = plt.subplots(figsize=(20, 9))

# Background shading for model families
ax.axvspan(-0.5, 6.5, alpha=0.06, color="#2196F3", zorder=0)
ax.axvspan(6.5, 10.5, alpha=0.06, color="#FF9800", zorder=0)
ax.text(3.0, 0.97, "Qwen3-8B", ha="center", fontsize=13, fontweight="bold",
        color="#1565C0", alpha=0.5, transform=ax.get_xaxis_transform())
ax.text(8.5, 0.97, "Qwen3.5-9B", ha="center", fontsize=13, fontweight="bold",
        color="#E65100", alpha=0.5, transform=ax.get_xaxis_transform())

# Think variant shading
for i, label in enumerate(cross_configs):
    if "Think" in label:
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="#FFC107", zorder=0)

x = np.arange(len(cross_configs))

for metric_name, values in cross_data.items():
    style = line_styles[metric_name]
    ax.plot(x, values, color=style["color"], marker=style["marker"],
            markersize=style["ms"], linewidth=style["lw"],
            markeredgecolor="white", markeredgewidth=1.5,
            label=metric_name, zorder=3)

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
ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
ax.set_title("Cross-Family F1 Comparison: Qwen3-8B vs Qwen3.5-9B", fontsize=15, fontweight="bold", pad=15)
ax.set_ylim(0.0, 0.95)
ax.legend(fontsize=11, loc="upper left", framealpha=0.9, edgecolor="#ccc")
ax.grid(axis="y", alpha=0.3, linestyle="-")
ax.grid(axis="x", alpha=0.15, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{out_dir}/cross_family_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved cross_family_comparison.png")
