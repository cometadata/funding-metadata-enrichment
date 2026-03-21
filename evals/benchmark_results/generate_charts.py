import matplotlib.pyplot as plt
import numpy as np

# Model data (test split) - ordered by overall avg F1
models = {
    # Base models
    "Qwen3-8B base": {
        "funder":  {"p": 0.767, "r": 0.944, "f1": 0.847},
        "award_id": {"p": 0.722, "r": 0.775, "f1": 0.747},
        "funding_scheme": {"p": 0.507, "r": 0.098, "f1": 0.164},
        "award_title": {"p": 0.545, "r": 0.070, "f1": 0.124},
    },
    "Qwen3-8B base Think": {
        "funder":  {"p": 0.844, "r": 0.903, "f1": 0.872},
        "award_id": {"p": 0.776, "r": 0.799, "f1": 0.787},
        "funding_scheme": {"p": 0.531, "r": 0.215, "f1": 0.306},
        "award_title": {"p": 0.387, "r": 0.140, "f1": 0.205},
    },
    "Qwen3.5-9B base": {
        "funder":  {"p": 0.822, "r": 0.927, "f1": 0.871},
        "award_id": {"p": 0.789, "r": 0.834, "f1": 0.811},
        "funding_scheme": {"p": 0.833, "r": 0.084, "f1": 0.152},
        "award_title": {"p": 0.446, "r": 0.430, "f1": 0.438},
    },
    "Qwen3.5-9B base Think": {
        "funder":  {"p": 0.863, "r": 0.863, "f1": 0.863},
        "award_id": {"p": 0.817, "r": 0.789, "f1": 0.803},
        "funding_scheme": {"p": 0.917, "r": 0.062, "f1": 0.115},
        "award_title": {"p": 0.643, "r": 0.209, "f1": 0.316},
    },
    # Fine-tuned models
    "Qwen3-8B glm-air-4.5-distil": {
        "funder":  {"p": 0.827, "r": 0.895, "f1": 0.860},
        "award_id": {"p": 0.740, "r": 0.766, "f1": 0.753},
        "funding_scheme": {"p": 0.560, "r": 0.196, "f1": 0.290},
        "award_title": {"p": 0.214, "r": 0.070, "f1": 0.105},
    },
    "Qwen3-8B glm-air-4.5-distil Think": {
        "funder":  {"p": 0.855, "r": 0.909, "f1": 0.881},
        "award_id": {"p": 0.784, "r": 0.790, "f1": 0.787},
        "funding_scheme": {"p": 0.598, "r": 0.282, "f1": 0.383},
        "award_title": {"p": 0.278, "r": 0.116, "f1": 0.164},
    },
    "Qwen3-8B ep2 synthetic": {
        "funder":  {"p": 0.840, "r": 0.930, "f1": 0.882},
        "award_id": {"p": 0.750, "r": 0.760, "f1": 0.755},
        "funding_scheme": {"p": 0.574, "r": 0.249, "f1": 0.347},
        "award_title": {"p": 0.286, "r": 0.023, "f1": 0.043},
    },
    "Qwen3-8B ep2 synthetic Think": {
        "funder":  {"p": 0.829, "r": 0.906, "f1": 0.866},
        "award_id": {"p": 0.730, "r": 0.743, "f1": 0.737},
        "funding_scheme": {"p": 0.527, "r": 0.246, "f1": 0.335},
        "award_title": {"p": 0.462, "r": 0.070, "f1": 0.121},
    },
    "Qwen3-8B ep2p1 twostage": {
        "funder":  {"p": 0.847, "r": 0.930, "f1": 0.886},
        "award_id": {"p": 0.756, "r": 0.763, "f1": 0.759},
        "funding_scheme": {"p": 0.546, "r": 0.318, "f1": 0.402},
        "award_title": {"p": 0.500, "r": 0.047, "f1": 0.085},
    },
    "Qwen3-8B ep2p1 twostage Think": {
        "funder":  {"p": 0.847, "r": 0.914, "f1": 0.879},
        "award_id": {"p": 0.771, "r": 0.768, "f1": 0.769},
        "funding_scheme": {"p": 0.525, "r": 0.296, "f1": 0.379},
        "award_title": {"p": 0.526, "r": 0.116, "f1": 0.190},
    },
    "Qwen3.5-9B glm-air-4.5-distil": {
        "funder":  {"p": 0.823, "r": 0.931, "f1": 0.874},
        "award_id": {"p": 0.782, "r": 0.820, "f1": 0.800},
        "funding_scheme": {"p": 0.658, "r": 0.140, "f1": 0.230},
        "award_title": {"p": 0.300, "r": 0.384, "f1": 0.337},
    },
    "Qwen3.5-9B glm-air-4.5-distil Think": {
        "funder":  {"p": 0.859, "r": 0.914, "f1": 0.885},
        "award_id": {"p": 0.799, "r": 0.807, "f1": 0.803},
        "funding_scheme": {"p": 0.701, "r": 0.229, "f1": 0.345},
        "award_title": {"p": 0.681, "r": 0.372, "f1": 0.481},
    },
}

fields = ["funder", "award_id", "funding_scheme", "award_title"]
field_labels = ["Funder", "Award ID", "Funding Scheme", "Award Title"]
model_names = list(models.keys())
n_models = len(model_names)

# Color palette
colors = plt.cm.tab20(np.linspace(0, 1, n_models))

out_dir = "/Users/adambuttrick/Documents/GitHub/cometadata/funding-metadata-enrichment/evals/benchmark_results"

# --- Chart 1: Overall F1 comparison (grouped bar) ---
fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(fields))
bar_width = 0.065
offsets = np.arange(n_models) - (n_models - 1) / 2

for i, model in enumerate(model_names):
    f1_vals = [models[model][f]["f1"] for f in fields]
    bars = ax.bar(x + offsets[i] * bar_width, f1_vals, bar_width * 0.9,
                  label=model, color=colors[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, f1_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(field_labels, fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Benchmark F1 Comparison — Test Split", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{out_dir}/benchmark_f1_comparison.png", dpi=200)
plt.close()
print("Saved benchmark_f1_comparison.png")

# --- Charts 2-5: Precision & Recall per field ---
for field, field_label in zip(fields, field_labels):
    fig, ax = plt.subplots(figsize=(14, 9))

    # Sort models by F1 for this field (descending)
    sorted_models = sorted(model_names, key=lambda m: models[m][field]["f1"], reverse=True)

    y = np.arange(len(sorted_models))
    bar_height = 0.35

    p_vals = [models[m][field]["p"] for m in sorted_models]
    r_vals = [models[m][field]["r"] for m in sorted_models]
    f1_vals = [models[m][field]["f1"] for m in sorted_models]

    bars_p = ax.barh(y + bar_height / 2, p_vals, bar_height, label="Precision",
                     color="#4C72B0", edgecolor="white", linewidth=0.5)
    bars_r = ax.barh(y - bar_height / 2, r_vals, bar_height, label="Recall",
                     color="#DD8452", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars_p, p_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=8)
    for bar, val in zip(bars_r, r_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=8)

    # Add F1 annotation on right
    for i, (m, f1) in enumerate(zip(sorted_models, f1_vals)):
        ax.text(1.02, i, f"F1={f1:.3f}", ha="left", va="center",
                fontsize=8, fontweight="bold", transform=ax.get_yaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=8)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(f"{field_label} — Precision & Recall (Test Split)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()

    fname = f"benchmark_{field}_pr.png"
    plt.savefig(f"{out_dir}/{fname}", dpi=200)
    plt.close()
    print(f"Saved {fname}")
