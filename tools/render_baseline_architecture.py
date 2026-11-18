"""Draw the two implemented baseline pipelines without implying a final model."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


fig, ax = plt.subplots(figsize=(15, 7.2))
fig.patch.set_facecolor("#f8fbfc")
ax.set_facecolor("#f8fbfc")
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, detail, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.12", linewidth=1.3, edgecolor=color, facecolor="white")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", color="#173038", fontsize=11, weight="bold")
    ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center", color="#53656d", fontsize=9)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color="#7a9199"))


ax.text(0.25, 6.65, "Implemented glaucoma baselines", fontsize=19, weight="bold", color="#173038")
ax.text(0.25, 6.25, "Measured on ACRIMA and RIM-ONE r3; final five-dataset comparison is pending", fontsize=10, color="#53656d")

ax.text(0.25, 5.72, "CLASSIFICATION + EXPLANATION", fontsize=11, weight="bold", color="#147d86")
classification = [
    (0.3, "ACRIMA fundus", "RGB image"),
    (3.25, "Preprocess", "resize 256; crop 224"),
    (6.2, "ResNet-18", "ImageNet weights; frozen"),
    (9.15, "Logistic head", "512 features; balanced"),
    (12.1, "Prediction + XAI", "probability; Grad-CAM"),
]
for x, title, detail in classification:
    box(x, 4.45, 2.55, 0.95, title, detail, "#147d86")
for x in (2.85, 5.8, 8.75, 11.7):
    arrow(x, 4.93, x + 0.4, 4.93)
ax.text(12.1, 4.15, "Grad-CAM uses ResNet layer 4", fontsize=8.5, color="#53656d")

ax.text(0.25, 3.5, "OPTIC DISC + CUP SEGMENTATION", fontsize=11, weight="bold", color="#a46425")
segmentation = [
    (0.3, "RIM-ONE stereo", "left view only"),
    (3.25, "Preprocess", "RGB + masks at 256²"),
    (6.2, "Compact U-Net", "encoder 16–32–64–128"),
    (9.15, "Two-channel head", "disc and cup logits"),
    (12.1, "Evaluation", "0.5 mask; Dice + IoU"),
]
for x, title, detail in segmentation:
    box(x, 2.25, 2.55, 0.95, title, detail, "#a46425")
for x in (2.85, 5.8, 8.75, 11.7):
    arrow(x, 2.73, x + 0.4, 2.73)
ax.text(6.2, 1.87, "U-Net skip connections pass encoder features to decoder stages", fontsize=8.5, color="#53656d")

ax.text(0.3, 0.95, "Protocol: fixed train / validation / test splits • test data used only for final evaluation • ACRIMA has no segmentation masks", fontsize=10, color="#384e56")
destination = Path("outputs/architecture_baselines.png")
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(destination)
