"""Render the measured RIM-ONE whole-image versus disc-crop comparison."""
from pathlib import Path
import json

import matplotlib.pyplot as plt


root = Path("outputs/rim_whole_vs_crop")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
names = ("whole", "crop")
labels = ["Whole image", "Predicted disc crop"]
auc = [metrics[name]["auc"] for name in names]
accuracy = [metrics[name]["accuracy"] for name in names]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=160)
for ax, values, title, ylabel in (
    (axes[0], auc, "Held-out ROC AUC", "AUC"),
    (axes[1], accuracy, "Held-out accuracy", "Accuracy"),
):
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("RIM-ONE r3 matched split: whole image vs disc crop")
fig.tight_layout()
output = root / "comparison.png"
fig.savefig(output, bbox_inches="tight")
plt.close(fig)
print(output)
