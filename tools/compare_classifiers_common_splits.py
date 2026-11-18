"""Compare two classifiers only on identical held-out images in four datasets."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def rows(path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


original = Path("outputs/acrima_resnet18_frozen")
pooled = Path("outputs/pooled_four_dataset_frozen")
acrima_members = {r["image_id"]: r["member"] for r in rows(original / "split.csv")}
sources = {
    "ACRIMA": (original / "test_predictions.csv", pooled / "acrima_test_predictions.csv", lambda r: acrima_members[r["image_id"]]),
    "ORIGA": (original / "origa_external_predictions.csv", pooled / "origa_test_predictions.csv", lambda r: r["image"]),
    "REFUGE": (original / "refuge_val_external_predictions.csv", pooled / "refuge_test_predictions.csv", lambda r: r["image"]),
    "RIM-ONE r3": (original / "rim_external_predictions.csv", pooled / "rim_one_r3_test_predictions.csv", lambda r: r["image"]),
}


def calculate(records):
    y = np.array([int(r["true_label"]) for r in records])
    p = np.array([float(r["probability_glaucoma"]) for r in records])
    prediction = np.array([int(r["predicted_label"]) for r in records])
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"n_test": len(records), "accuracy": float(accuracy_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": float(roc_auc_score(y, p)), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


comparison = {}
for dataset, (original_path, pooled_path, key) in sources.items():
    baseline = {key(r): r for r in rows(original_path)}
    pooled_rows = rows(pooled_path)
    assert len(pooled_rows) == len({r["member"] for r in pooled_rows})
    original_rows = [baseline[r["member"]] for r in pooled_rows]
    assert all(int(a["true_label"]) == int(b["true_label"]) for a, b in zip(original_rows, pooled_rows)), dataset
    comparison[dataset] = {"ACRIMA-only frozen": calculate(original_rows), "Pooled four-dataset frozen": calculate(pooled_rows)}

destination = Path("outputs/classifier_common_split_comparison.json")
destination.write_text(json.dumps({"protocol": "Both models evaluated on identical held-out images; ORIGA B, REFUGE official validation, ACRIMA fixed test, RIM-ONE non-suspect fixed test subset", "results": comparison}, indent=2), encoding="utf-8")

datasets = list(comparison)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
x = np.arange(len(datasets))
for ax, metric, title in zip(axes, ("auc", "balanced_accuracy"), ("ROC AUC", "Balanced accuracy")):
    for offset, model, color in ((-0.18, "ACRIMA-only frozen", "#b66a36"), (0.18, "Pooled four-dataset frozen", "#14858a")):
        values = [comparison[d][model][metric] for d in datasets]
        ax.bar(x + offset, values, width=0.35, label=model, color=color)
    ax.set_xticks(x, datasets, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
axes[1].legend(loc="upper left", fontsize=8)
fig.suptitle("Same held-out images per dataset; frozen-feature baselines", fontsize=13)
figure = Path("outputs/classifier_common_split_comparison.png")
fig.savefig(figure, dpi=170)
plt.close(fig)
print(destination)
print(figure)
for dataset, result in comparison.items():
    print(dataset, "n=", result["ACRIMA-only frozen"]["n_test"], "AUC", round(result["ACRIMA-only frozen"]["auc"], 3), "->", round(result["Pooled four-dataset frozen"]["auc"], 3))
