"""Four-dataset pooled frozen ResNet-18 baseline with declared held-out splits."""
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Normalize, ToTensor


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
root = Path("outputs/pooled_four_dataset_frozen")
root.mkdir(parents=True, exist_ok=True)
archive_paths = {
    "ACRIMA": Path("data/raw/ACRIMA_Database.zip"),
    "RIM-ONE r3": Path("data/raw/RIM-ONE-r3.zip"),
    "ORIGA": Path("data/raw/ORIGA_candidate_figshare.zip"),
    "REFUGE": Path("data/raw/ORIGA_candidate_figshare.zip"),
}


def add_record(records, dataset, member, label, split, left_stereo=False):
    records.append({"dataset": dataset, "member": member, "label": int(label), "split": split, "left_stereo": left_stereo})


records = []
for row in csv.DictReader(Path("outputs/acrima_resnet18_frozen/split.csv").open(encoding="utf-8")):
    add_record(records, "ACRIMA", row["member"], row["label"] == "glaucoma", row["split"])

origa = list(csv.DictReader(Path("data/audit/origa_manifest.csv").open(encoding="utf-8")))
origa_a = [r for r in origa if r["provided_set"] == "A"]
origa_b = [r for r in origa if r["provided_set"] == "B"]
assert len(origa_a) == len(origa_b) == 325
a_train, a_val = train_test_split(np.arange(len(origa_a)), test_size=0.2, stratify=[r["label"] for r in origa_a], random_state=SEED)
for i, row in enumerate(origa_a):
    add_record(records, "ORIGA", row["image"], row["label"], "val" if i in set(a_val) else "train")
for row in origa_b:
    add_record(records, "ORIGA", row["image"], row["label"], "test")

refuge = list(csv.DictReader(Path("data/audit/refuge_manifest.csv").open(encoding="utf-8")))
refuge_train = [r for r in refuge if r["split"] == "train"]
refuge_val = [r for r in refuge if r["split"] == "val"]
assert len(refuge_train) == len(refuge_val) == 400
r_train, r_val = train_test_split(np.arange(len(refuge_train)), test_size=0.2, stratify=[r["label"] for r in refuge_train], random_state=SEED)
for i, row in enumerate(refuge_train):
    add_record(records, "REFUGE", row["image"], row["label"], "val" if i in set(r_val) else "train")
for row in refuge_val:
    add_record(records, "REFUGE", row["image"], row["label"], "test")

rim_labels = {r["image"]: r["label_original"] for r in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))}
for row in csv.DictReader(Path("outputs/rim_unet_baseline/split.csv").open(encoding="utf-8")):
    label = rim_labels[row["image"]]
    if label in ("normal", "glaucoma"):
        add_record(records, "RIM-ONE r3", row["image"], label == "glaucoma", row["split"], True)

assert len(records) == 705 + 650 + 800 + 124
assert len({(r["dataset"], r["member"]) for r in records}) == len(records)
with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model.eval()
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
features = []
digests = defaultdict(list)
with zipfile.ZipFile(archive_paths["ACRIMA"]) as acrima_zip, zipfile.ZipFile(archive_paths["RIM-ONE r3"]) as rim_zip, zipfile.ZipFile(archive_paths["ORIGA"]) as combined_zip, torch.no_grad():
    archives = {"ACRIMA": acrima_zip, "RIM-ONE r3": rim_zip, "ORIGA": combined_zip, "REFUGE": combined_zip}
    for start in range(0, len(records), 16):
        batch = []
        for row in records[start:start + 16]:
            raw = archives[row["dataset"]].read(row["member"])
            digests[hashlib.sha256(raw).hexdigest()].append((row["dataset"], row["member"], row["split"]))
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
                if row["left_stereo"]:
                    image = image.crop((0, 0, image.width // 2, image.height))
                image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
                batch.append(normalize(to_tensor(image)))
        features.append(model(torch.stack(batch)).numpy())
        if (start // 16) % 10 == 0:
            print(f"features {min(start + 16, len(records))}/{len(records)}", flush=True)
X = np.concatenate(features)
duplicates = [group for group in digests.values() if len(group) > 1]
cross_split = [group for group in duplicates if len({item[2] for item in group}) > 1]
assert not cross_split, f"Cross-split exact duplicates: {cross_split[:3]}"
(root / "duplicate_audit.json").write_text(json.dumps({"exact_duplicate_groups": duplicates, "cross_split_groups": cross_split}, indent=2), encoding="utf-8")

dataset = np.array([r["dataset"] for r in records])
labels = np.array([r["label"] for r in records])
splits = np.array([r["split"] for r in records])
train = np.where(splits == "train")[0]
val = np.where(splits == "val")[0]
trainval = np.where(splits != "test")[0]


def balancing_weights(indices):
    counts = Counter((dataset[i], int(labels[i])) for i in indices)
    weights = np.array([1.0 / counts[(dataset[i], int(labels[i]))] for i in indices])
    return weights * (len(weights) / weights.sum())


trials = []
for c in (0.01, 0.1, 1.0, 10.0):
    classifier = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
    classifier.fit(X[train], labels[train], sample_weight=balancing_weights(train))
    probability = classifier.predict_proba(X[val])[:, 1]
    auc_by_dataset = {name: float(roc_auc_score(labels[val][dataset[val] == name], probability[dataset[val] == name])) for name in sorted(set(dataset[val]))}
    macro_auc = float(np.mean(list(auc_by_dataset.values())))
    trials.append({"C": c, "macro_val_auc": macro_auc, "per_dataset_val_auc": auc_by_dataset})
best = max(trials, key=lambda r: r["macro_val_auc"])
classifier = LogisticRegression(C=best["C"], max_iter=1000, random_state=SEED)
classifier.fit(X[trainval], labels[trainval], sample_weight=balancing_weights(trainval))
joblib.dump(classifier, root / "classifier.joblib")

full_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
full_model.fc = torch.nn.Linear(512, 1)
with torch.no_grad():
    full_model.fc.weight.copy_(torch.from_numpy(classifier.coef_).float())
    full_model.fc.bias.copy_(torch.from_numpy(classifier.intercept_).float())
torch.save(full_model.state_dict(), root / "resnet18_classifier.pt")

metrics = {}
for name in sorted(set(dataset)):
    indices = np.where((dataset == name) & (splits == "test"))[0]
    y = labels[indices]
    probabilities = classifier.predict_proba(X[indices])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics[name] = {"n_test": len(indices), "accuracy": float(accuracy_score(y, predicted)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": float(roc_auc_score(y, probabilities)), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    with (root / (name.lower().replace(" ", "_").replace("-", "_") + "_test_predictions.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "member", "true_label", "probability_glaucoma", "predicted_label"])
        writer.writerows((name, records[i]["member"], int(labels[i]), float(p), int(yhat)) for i, p, yhat in zip(indices, probabilities, predicted))

summary = {
    "status": "four_dataset_partial_baseline",
    "architecture": "frozen ImageNet ResNet-18 features + balanced logistic head",
    "preprocessing": "full RGB image letterboxed to 224x224; RIM-ONE uses left stereo view",
    "protocol": "ACRIMA fixed split; ORIGA A train/val B test; REFUGE train train/val and official validation held out; RIM-ONE fixed split excluding suspects; seed 42",
    "selection": {"metric": "macro validation AUC across four datasets", "trials": trials, "selected_C": best["C"]},
    "split_counts": {name: dict(Counter(splits[dataset == name])) for name in sorted(set(dataset))},
    "test_metrics": metrics,
    "limitations": "DRISHTI-GS unavailable; ORIGA/REFUGE come from a verified third-party repackaging; ORIGA patient identifiers unavailable; frozen backbone is not final fine-tuning",
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
