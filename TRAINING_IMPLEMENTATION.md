# Training Implementation and Next Plan

**Project:** Glaucoma MLDL  
**Research question:** Does a disc-crop classifier improve glaucoma classification compared with a whole-image classifier when both use the same classifier architecture and training budget?  
**Current phase:** Partial baselines complete; final fine-tuned five-source models are remaining.

## 1. Training work completed

The project has implemented three types of measured experiments:

1. Frozen ResNet-18 feature extraction with logistic-regression classification.
2. Compact two-output U-Net optic-disc/cup segmentation.
3. A matched whole-image versus disc-crop RIM-ONE classification comparison.

All measured runs use seed `42` and preserve fixed split and prediction files.

---

## 2. ACRIMA-only classification baseline

**Implementation file**

- `tools/train_acrima_baseline.py`

**Method**

- Extract 512-dimensional ImageNet ResNet-18 features.
- Freeze the ResNet-18 backbone.
- Train a balanced logistic-regression classification head.
- Try `C = 0.01, 0.1, 1.0, 10.0`.
- Select the regularization value using validation ROC AUC.
- Refit on train plus validation.
- Evaluate on the untouched test split.

**Split**

- Train: 458
- Validation: 106
- Test: 141
- Total: 705

**Test result**

- Accuracy: `90.78%`
- Precision: `90.24%`
- Recall: `93.67%`
- F1: `91.93%`
- ROC AUC: `97.84%`

**Saved artifacts**

- `outputs/acrima_resnet18_frozen/split.csv`
- `outputs/acrima_resnet18_frozen/metrics.json`
- `outputs/acrima_resnet18_frozen/test_predictions.csv`
- `outputs/acrima_resnet18_frozen/classifier.joblib`
- `outputs/acrima_resnet18_frozen/resnet18_classifier.pt`

**Important limitation**

ACRIMA has no segmentation masks. It must not be used to calculate Dice or IoU.

---

## 3. RIM-ONE single-dataset segmentation baseline

**Implementation file**

- `tools/train_rim_segmentation.py`

**Architecture**

- Compact custom two-channel U-Net.
- Input size: `256 x 256`.
- Output channels:
  - channel 0: optic disc
  - channel 1: optic cup
- Loss: binary cross-entropy plus soft mean Dice loss.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Epochs: `12`.
- Batch size: `4`.
- Horizontal flipping used for augmentation.
- Model checkpoint selected using mean validation disc/cup Dice.

**Split**

- Train: 103
- Validation: 24
- Test: 32
- Total non-split images: 159

**Test result**

- Disc Dice: `0.7989`
- Disc IoU: `0.6904`
- Cup Dice: `0.5620`
- Cup IoU: `0.4283`

**Saved artifacts**

- `outputs/rim_unet_baseline/best.pt`
- `outputs/rim_unet_baseline/history.json`
- `outputs/rim_unet_baseline/metrics.json`
- `outputs/rim_unet_baseline/split.csv`
- `outputs/rim_unet_baseline/test_predictions.csv`
- `outputs/rim_unet_baseline/examples/`

**Important limitation**

This is a small randomly initialized U-Net, not yet the final pretrained-encoder segmenter.

---

## 4. Whole-image versus disc-crop comparison

**Implementation file**

- `tools/train_rim_whole_vs_crop.py`

**Matched protocol**

- Dataset: RIM-ONE r3.
- Suspects excluded.
- Train: 80 non-suspect images.
- Validation: 19 non-suspect images.
- Test: 25 non-suspect images.
- Seed: 42.
- Frozen ResNet-18 feature extractor.
- Logistic regularization candidates: `0.01, 0.1, 1.0, 10.0`.

**Whole-image variant**

- Left stereo view used.
- Letterboxed to `224 x 224`.

**Crop variant**

- Training and validation crops use ground-truth disc masks.
- Test crops use the disc mask predicted by the saved RIM-ONE U-Net.
- Test ground-truth masks are never used to construct test crops.

**Test result**

| Variant | Accuracy | ROC AUC |
|---|---:|---:|
| Whole image | `0.8000` | `0.9044` |
| Predicted disc crop | `0.7600` | `0.8750` |

**Saved artifacts**

- `outputs/rim_whole_vs_crop/metrics.json`
- `outputs/rim_whole_vs_crop/split.csv`
- `outputs/rim_whole_vs_crop/whole_test_predictions.csv`
- `outputs/rim_whole_vs_crop/crop_test_predictions.csv`
- `outputs/rim_whole_vs_crop/whole_classifier.joblib`
- `outputs/rim_whole_vs_crop/crop_classifier.joblib`
- `outputs/rim_whole_vs_crop/comparison.png`

**Interpretation**

The crop did not outperform the whole image in this RIM-ONE split. This is a negative measured result for this protocol, not a general claim that disc crops never help.

---

## 5. Four-dataset pooled frozen classifier

**Implementation file**

- `tools/train_pooled_frozen_baseline.py`

**Datasets**

- ACRIMA
- RIM-ONE r3
- ORIGA
- REFUGE

**Method**

- Frozen ImageNet ResNet-18 feature extraction.
- Full image letterboxed to `224 x 224`.
- RIM-ONE left stereo view used.
- Logistic head weighted by dataset and class.
- Macro validation ROC AUC across datasets used for model selection.
- Test data remained untouched during selection.

**Split structure**

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| ACRIMA | 458 | 106 | 141 |
| ORIGA | 260 | 65 | 325 |
| REFUGE | 320 | 80 | 400 |
| RIM-ONE r3 | 80 | 19 | 25 |

**Test result**

| Dataset | Accuracy | ROC AUC |
|---|---:|---:|
| ACRIMA | `0.8582` | `0.9251` |
| ORIGA | `0.7046` | `0.7000` |
| REFUGE | `0.7800` | `0.7792` |
| RIM-ONE r3 | `0.8000` | `0.8676` |

Independent official REFUGE test: accuracy `0.7500`, ROC AUC `0.7910`.

**Saved artifacts**

- `outputs/pooled_four_dataset_frozen/`
- `outputs/pooled_four_dataset_frozen/split.csv`
- `outputs/pooled_four_dataset_frozen/metrics.json`
- `outputs/pooled_four_dataset_frozen/classifier.joblib`
- `outputs/pooled_four_dataset_frozen/resnet18_classifier.pt`
- Per-dataset test prediction CSV files
- Pooled Grad-CAM panels

**Important limitation**

This is a frozen-feature four-dataset baseline. The backbone is not fine-tuned and DRISHTI-GS1 is not included.

---

## 6. Pooled RIM-ONE/ORIGA/REFUGE segmentation

**Implementation file**

- `tools/train_pooled_segmentation.py`

**Method**

- Compact randomly initialized U-Net.
- Pooled training across RIM-ONE, ORIGA, and REFUGE.
- Binary cross-entropy plus soft Dice loss.
- 256-pixel input.
- Eight epochs.
- Validation mean disc/cup Dice used for checkpoint selection.

**Split count**

| Split | RIM-ONE | ORIGA | REFUGE | Total |
|---|---:|---:|---:|---:|
| Train | 103 | 260 | 320 | 683 |
| Validation | 24 | 65 | 80 | 169 |
| Test | 32 | 325 | 800 | 1,157 |

**Test result**

| Dataset | Disc Dice | Disc IoU | Cup Dice | Cup IoU |
|---|---:|---:|---:|---:|
| RIM-ONE r3 | 0.6920 | 0.5947 | 0.5527 | 0.4314 |
| ORIGA | 0.7510 | 0.6240 | 0.8053 | 0.6898 |
| REFUGE | 0.6930 | 0.5459 | 0.6701 | 0.5258 |

**Saved artifacts**

- `outputs/pooled_segmentation/best.pt`
- `outputs/pooled_segmentation/history.json`
- `outputs/pooled_segmentation/metrics.json`
- `outputs/pooled_segmentation/split.csv`
- `outputs/pooled_segmentation/test_predictions.csv`
- `outputs/pooled_segmentation/examples/`

**Important limitation**

This is a compact random-parameter baseline. It is not the final pretrained-encoder segmenter.

---

## 7. DRISHTI-GS1 Kaggle-mirror baselines

**Source**

- Kaggle uploader: `lokeshsaipureddi`
- Dataset: `lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation`
- Archive: `357,915,185` bytes
- Provenance and inventory: `data/provisional/drishti_gs1/`

**Inventory**

- 101 images
- 50 supplied training images
- 51 supplied test images
- 70 glaucoma / 31 normal folder labels
- 101 OD soft masks
- 101 cup soft masks

### Classification

**Implementation file**

- `tools/train_drishti_classifier.py`

**Method**

- Frozen ImageNet ResNet-18 features.
- Logistic classification head.
- Four regularization candidates.
- 40 training / 10 validation / 51 test split.
- Validation AUC used for selection.

**Test result**

- Accuracy: `78.43%`
- Precision: `78.72%`
- Recall: `97.37%`
- F1: `87.06%`
- ROC AUC: `68.02%`

**Artifacts**

- `outputs/drishti_gs1_mirror_classifier/metrics.json`
- `outputs/drishti_gs1_mirror_classifier/split.csv`
- `outputs/drishti_gs1_mirror_classifier/test_predictions.csv`
- `outputs/drishti_gs1_mirror_classifier/classifier.joblib`

### Segmentation

**Implementation file**

- `tools/train_drishti_segmentation.py`

**Method**

- Compact random-parameter U-Net.
- 256-pixel input.
- Soft OD/cup masks.
- Eight epochs.
- 40 training / 10 validation / 51 test split.
- Validation mean Dice used for checkpoint selection.

**Test result**

- Disc Dice: `0.8841`
- Disc IoU: `0.7997`
- Cup Dice: `0.7171`
- Cup IoU: `0.5796`

**Artifacts**

- `outputs/drishti_gs1_mirror/best.pt`
- `outputs/drishti_gs1_mirror/history.json`
- `outputs/drishti_gs1_mirror/metrics.json`
- `outputs/drishti_gs1_mirror/split.csv`
- `outputs/drishti_gs1_mirror/test_predictions.csv`
- `outputs/drishti_gs1_mirror/examples/`

**Source warning**

These values describe the Kaggle mirror and must not be presented as official-source-verified DRISHTI-GS1 results.

---

## 8. Explainability and export

### Grad-CAM

- ACRIMA Grad-CAM examples saved.
- Pooled four-dataset predicted-class Grad-CAM saved.
- DRISHTI mirror segmentation images saved.

### Website export

The export script writes:

- `outputs/website_results_partial.json`

The export validator confirms:

- completed experiment count;
- unique experiment IDs;
- dataset/task validity;
- prediction counts reconcile with metrics;
- referenced image/Grad-CAM files exist.

---

## 9. What remains for final training

### A. Freeze the final protocol

- Select whether DRISHTI-GS1 is used from the Kaggle mirror or replaced by another source.
- Freeze final train/validation/test assignments for all five sources.
- Define which RIM-ONE views and stereo side are used.
- Define ACRIMA preprocessing.
- Define ORIGA, REFUGE, and DRISHTI image preprocessing.
- Define patient-group or image-group overlap checks.
- Save one canonical manifest and configuration file.

### B. Train the final fine-tuned classifier

- Use pretrained ResNet-18 or another agreed backbone.
- Fine-tune only after frozen-feature baselines are complete.
- Use dataset-balanced batches.
- Use validation-only model/epoch/threshold selection.
- Add repeated seeds or cross-validation if feasible.
- Evaluate every source on a fixed test split.
- Save:
  - final checkpoint
  - configuration JSON
  - training history
  - test predictions
  - accuracy, precision, recall, F1, ROC AUC
  - confidence intervals or repeated-seed summary

### C. Train the final segmentation model

- Replace the compact random U-Net.
- Prefer a pretrained encoder U-Net or comparable maintained architecture.
- Decide whether soft masks or thresholded hard masks are used.
- Train on RIM-ONE, ORIGA, REFUGE, and DRISHTI data with paired masks.
- Preserve fixed test and validation splits.
- Save:
  - checkpoint
  - history
  - per-image Dice and IoU
  - best/worst overlays
  - final dataset metrics

### D. Final crop experiment

- Use the final trained segmentation model to predict disc masks.
- Never use test ground-truth masks to construct test crops.
- Create whole-image and crop variants from identical test images.
- Use the same backbone, preprocessing, training budget, and hyperparameters.
- Save paired per-image predictions.
- Add paired statistical comparison and confidence intervals.

### E. Final explainability

- Generate Grad-CAM for the final classifier.
- Explain the correct predicted class.
- Use TP, TN, FP, and FN panels.
- Check whether attention remains in the optic-disc region.
- Clearly distinguish attention evidence from causal evidence.

### F. Final report and website

- Replace the partial report with final results only.
- Add exact split and source provenance tables.
- Add all final classification and segmentation tables.
- Include whole-vs-crop comparison.
- Include Grad-CAM and segmentation visual galleries.
- Freeze `outputs/website_results_partial.json` or replace it with a final schema.
- Validate every displayed number against a saved prediction file.

---

## 9.1 Colab GPU launch

The project has a dedicated validated GPU launcher:

- `COLAB_TRAINING.ipynb`
- `tools/train_pooled_finetuned_classifier.py`

This workspace cannot provide a live Colab session URL because no Colab URL, GitHub remote, or Drive link is configured. Open the notebook using one of these methods:

### Upload method

1. Open https://colab.research.google.com/
2. Choose **Upload**.
3. Upload `COLAB_TRAINING.ipynb`.
4. Upload the project folder or place it in `/content/mldl_project`.
5. Select **Runtime > Change runtime type > T4 GPU**.
6. Run all cells.

### GitHub method

1. Push `COLAB_TRAINING.ipynb` to a public or accessible GitHub repository.
2. Open:

```text
https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/COLAB_TRAINING.ipynb
```

Replace `<owner>`, `<repo>`, and `<branch>` with the real repository values.

### Google Drive method

1. Upload `COLAB_TRAINING.ipynb` and the `mldl_project` folder to Google Drive.
2. Open the notebook through Google Colab.
3. Mount Drive and set `PROJECT_ROOT` to the Drive path.

---

## 10. Recommended execution order

1. Freeze the five-source split and source policy.
2. Run duplicate and split-integrity checks.
3. Fine-tune the final pooled classifier.
4. Train the final pooled segmenter.
5. Run the matched crop-vs-whole experiment.
6. Generate final Grad-CAM.
7. Export frozen metrics and predictions.
8. Update report.
9. Complete website.
10. Package reproducibility files and final handoff.

## 11. Team copy/paste prompt

```text
Continue this MLDL project from TRAINING_IMPLEMENTATION.md,
TEAM_HANDOFF.md, PROJECT.ipynb, and tools/HANDOFF.md.

Completed training includes:
- ACRIMA frozen ResNet-18 + logistic baseline
- RIM-ONE compact disc/cup U-Net
- RIM-ONE whole-image vs predicted-disc-crop comparison
- Four-dataset pooled frozen ResNet-18 baseline
- Pooled RIM-ONE/ORIGA/REFUGE compact U-Net
- DRISHTI-GS1 Kaggle-mirror frozen classifier
- DRISHTI-GS1 Kaggle-mirror compact U-Net
- Grad-CAM and partial measured exports

Do not restart completed baselines. Do not report ACRIMA Dice/IoU.
Do not use test masks to construct test crops. Do not present the Kaggle
DRISHTI-GS1 result as official-source-verified.

Next, freeze the final five-source protocol and implement:
1. fine-tuned classification;
2. pretrained-encoder pooled segmentation;
3. final predicted-crop versus whole-image comparison;
4. final Grad-CAM;
5. frozen website/report export.

Before changing code, state:
- exact split counts;
- data provenance;
- model architecture;
- hyperparameters;
- checkpoint selection rule;
- crop construction rule;
- expected output paths.

After training, reconcile every reported metric with saved per-image
predictions and run export validation.
```
