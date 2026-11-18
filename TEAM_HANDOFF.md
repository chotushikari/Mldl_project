# Team Handoff - Glaucoma MLDL Project

**Date:** 2026-11-18  
**Canonical notebook:** `PROJECT.ipynb`  
**Current status:** Partial four/five-source baselines implemented. Not yet the final report, website, or fine-tuned model submission.

## 1. Project objective

Answer the central research question:

> Does an optic-disc-crop classification strategy improve glaucoma classification compared with whole-image classification using the same architecture and training budget?

The project also requires:

- optic-disc/cup segmentation;
- Grad-CAM explanation;
- fixed, leakage-controlled train/validation/test splits;
- per-image predictions and reproducible metrics;
- a final report and measured-results website.

## 2. What is already implemented

### Dataset acquisition and audits

- [x] ACRIMA archive downloaded and audited.
- [x] RIM-ONE r3 archive downloaded and audited.
- [x] ORIGA combined third-party archive downloaded and audited.
- [x] REFUGE original RAR and repackaged archive verified.
- [x] REFUGE official test ground-truth spreadsheet recovered.
- [x] DRISHTI-GS1 Kaggle mirror downloaded for working experiments.
- [x] Dataset manifests, provenance files, and sample sheets generated.
- [x] Exact-image SHA-256 duplicate audit added.

### Classification

- [x] ACRIMA whole-image frozen ResNet-18 + logistic baseline.
- [x] ACRIMA checkpoint and per-image test predictions saved.
- [x] ACRIMA external RIM-ONE, ORIGA, and REFUGE evaluation completed.
- [x] Four-dataset pooled frozen-feature baseline implemented.
- [x] ACRIMA, ORIGA, REFUGE, and RIM-ONE held-out predictions saved.
- [x] DRISHTI-GS1 Kaggle-mirror frozen ResNet-18 + logistic baseline implemented.
- [x] Same-image fair comparison between ACRIMA-only and pooled classifiers added.

### Segmentation

- [x] Compact RIM-ONE U-Net baseline trained.
- [x] RIM-ONE per-image disc/cup Dice and IoU saved.
- [x] RIM-ONE best/worst prediction panels saved.
- [x] Pooled RIM-ONE/ORIGA/REFUGE compact U-Net trained.
- [x] 1,157 pooled held-out segmentation predictions saved.
- [x] Pooled six-panel best/worst visualization gallery saved.
- [x] DRISHTI-GS1 Kaggle-mirror compact U-Net trained.
- [x] DRISHTI-GS1 51 held-out per-image segmentation predictions saved.

### Whole-image versus crop

- [x] Matched RIM-ONE 80/19/25 binary split created.
- [x] Whole-image classifier run.
- [x] Disc-crop classifier run.
- [x] Training/validation crops used ground-truth masks.
- [x] Test crops used the saved U-Net prediction, never test ground truth.
- [x] Comparison chart and metrics saved.

### Explainability and export

- [x] ACRIMA Grad-CAM examples generated.
- [x] Pooled four-dataset predicted-class Grad-CAM panels generated.
- [x] DRISHTI segmentation best/worst panels generated.
- [x] Partial static results website created.
- [x] Partial project report created.
- [x] A 17-combination measured export is generated and validated.

## 3. Measured results completed

### ACRIMA frozen baseline

- Accuracy: `90.78%`
- Precision: `90.24%`
- Recall: `93.67%`
- F1: `91.93%`
- ROC AUC: `97.84%`

### RIM-ONE U-Net

- Disc Dice: `0.7989`
- Disc IoU: `0.6904`
- Cup Dice: `0.5620`
- Cup IoU: `0.4283`

### Whole image versus predicted disc crop on RIM-ONE

| Variant | n | Accuracy | ROC AUC |
|---|---:|---:|---:|
| Whole image | 25 | 0.8000 | 0.9044 |
| Predicted disc crop | 25 | 0.7600 | 0.8750 |

### Four-dataset pooled frozen classifier

| Dataset | Test n | Accuracy | ROC AUC |
|---|---:|---:|---:|
| ACRIMA | 141 | 0.8582 | 0.9251 |
| ORIGA | 325 | 0.7046 | 0.7000 |
| REFUGE validation | 400 | 0.7800 | 0.7792 |
| RIM-ONE non-suspect | 25 | 0.8000 | 0.8676 |

Independent official REFUGE test: accuracy `0.7500`, ROC AUC `0.7910`.

### Pooled three-dataset segmentation

| Dataset | Test n | Disc Dice | Disc IoU | Cup Dice | Cup IoU |
|---|---:|---:|---:|---:|---:|
| RIM-ONE r3 | 32 | 0.6920 | 0.5947 | 0.5527 | 0.4314 |
| ORIGA | 325 | 0.7510 | 0.6240 | 0.8053 | 0.6898 |
| REFUGE | 800 | 0.6930 | 0.5459 | 0.6701 | 0.5258 |

### DRISHTI-GS1 Kaggle-mirror baseline

Source: `lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation`

- Images: 101
- Training: 40
- Validation: 10
- Test: 51
- Classification accuracy: `78.43%`
- Classification F1: `87.06%`
- Classification ROC AUC: `68.02%`
- Segmentation disc Dice/IoU: `0.8841/0.7997`
- Segmentation cup Dice/IoU: `0.7171/0.5796`

This is a third-party mirror-specific checkpoint, not an official-source-verified DRISHTI-GS1 result.

## 4. Important files

### Canonical workflow

- `PROJECT.ipynb` - notebook entry point
- `COLAB_TRAINING.ipynb` - T4 GPU fine-tuning launcher
- `TRAINING_IMPLEMENTATION.md` - detailed training implementation and final plan
- `tools/HANDOFF.md` - current project handoff
- `README.md` - measured project summary
- `report/glaucoma_project_report.md` - current partial report

### Data

- `data/raw/ACRIMA_Database.zip`
- `data/raw/RIM-ONE-r3.zip`
- `data/raw/ORIGA_candidate_figshare.zip`
- `data/raw/REFUGE_figshare.rar`
- `data/audit/` - manifests and sample sheets
- `data/provisional/drishti_gs1/` - Kaggle mirror, provenance, and inventory

### Main experiment scripts

- `tools/train_acrima_baseline.py`
- `tools/train_rim_segmentation.py`
- `tools/train_pooled_frozen_baseline.py`
- `tools/train_pooled_segmentation.py`
- `tools/train_rim_whole_vs_crop.py`
- `tools/train_drishti_classifier.py`
- `tools/train_drishti_segmentation.py`
- `tools/evaluate_drishti_mirror.py`
- `tools/render_gradcam.py`
- `tools/render_pooled_gradcam.py`
- `tools/export_results.py`
- `tools/validate_website_export.py`

### Main saved outputs

- `outputs/acrima_resnet18_frozen/`
- `outputs/rim_unet_baseline/`
- `outputs/rim_whole_vs_crop/`
- `outputs/pooled_four_dataset_frozen/`
- `outputs/pooled_segmentation/`
- `outputs/drishti_gs1_mirror/`
- `outputs/drishti_gs1_mirror_classifier/`
- `outputs/website_results_partial.json`

### Website and report

- `website/index.html`
- `website/styles.css`
- `website/app.js`
- `report/glaucoma_project_report.md`

## 5. Remaining work checklist

### Data protocol

- [ ] Freeze one final five-source manifest and split schema.
- [ ] Reconcile the Kaggle DRISHTI mirror with the selected final source policy.
- [ ] Complete perceptual duplicate checking in addition to exact SHA-256 checks.
- [ ] Check patient-level identity overlap where identifiers exist.
- [ ] Verify all mask encodings and pixel meanings in a final protocol.
- [ ] Remove or clearly isolate every partial/third-party source that is not used in final claims.

### Final classification

- [ ] Fine-tune an end-to-end ResNet-18, not only frozen features.
- [ ] Train across the final five-source protocol.
- [ ] Use validation-only model and threshold selection.
- [ ] Evaluate fixed held-out test sets per dataset.
- [ ] Add dataset balancing and ablation tests.
- [ ] Add calibration, confusion matrices, confidence intervals, and threshold sensitivity.
- [ ] Save final checkpoint, configuration, and per-image predictions.

### Final segmentation

- [ ] Replace the compact random U-Net with a stronger pretrained-encoder architecture.
- [ ] Train a final pooled disc/cup model.
- [ ] Test on fixed held-out splits per dataset.
- [ ] Reconcile binary threshold and soft-mask threshold definitions.
- [ ] Save final checkpoints, history, per-image metrics, and examples.
- [ ] Compare singleton-dataset and pooled models.

### Final crop comparison

- [ ] Repeat with the final segmentation cropper.
- [ ] Ensure test crops use only predicted disc masks.
- [ ] Freeze identical image sets and preprocessing.
- [ ] Compare whole image and crop using the same classifier and training budget.
- [ ] Add paired statistical and confidence-interval analysis.
- [ ] Add correct/false-positive/false-negative crop panels.

### XAI

- [ ] Generate Grad-CAM for the final classifier.
- [ ] Verify normal decisions target the normal logit direction.
- [ ] Generate TP/TN/FP/FN examples across datasets.
- [ ] Add Grad-CAM stability/ablation analysis if required.
- [ ] Avoid clinical-causality claims.

### Report

- [ ] Replace partial report wording with final methodology.
- [ ] Add final result tables.
- [ ] Explain source provenance and split construction.
- [ ] Explain leakage controls and limitations.
- [ ] Add qualitative segmentation and Grad-CAM figures.
- [ ] Add repeated-run or confidence intervals if produced.
- [ ] Add final reproducibility instructions.

### Website

- [ ] Replace the partial export with the final frozen export.
- [ ] Add dataset pages.
- [ ] Add crop-vs-whole comparison.
- [ ] Add segmentation gallery.
- [ ] Add Grad-CAM gallery.
- [ ] Add per-image prediction explorer.
- [ ] Test responsive layout and browser loading.
- [ ] Deploy the selected static website.

## 6. How to run the current work

Run from the project root:

```powershell
python tools/audit_acrima.py
python tools/audit_rim_one_r3.py
python tools/audit_origa_refuge.py
python tools/audit_refuge_original.py
python tools/audit_drishti_mirror.py
python tools/validate_website_export.py
```

For the implemented baselines:

```powershell
python tools/train_acrima_baseline.py
python tools/train_rim_segmentation.py
python tools/train_pooled_frozen_baseline.py
python tools/train_pooled_segmentation.py
python tools/train_rim_whole_vs_crop.py
python tools/train_drishti_classifier.py
python tools/train_drishti_segmentation.py
python tools/evaluate_drishti_mirror.py
python tools/export_results.py
python tools/validate_website_export.py
```

Execute the canonical notebook:

```powershell
jupyter nbconvert --to notebook --execute PROJECT.ipynb --inplace --ExecutePreprocessor.timeout=120
```

Serve the partial website locally:

```powershell
python -m http.server 8765 --directory .
```

Open `http://127.0.0.1:8765/website/`.

## 7. Recommended division of work

### Teammate A - Final classification

- Extend `train_pooled_frozen_baseline.py`.
- Add fine-tuning.
- Add cross-dataset and crop baselines.
- Save final checkpoints and predictions.

### Teammate B - Final segmentation

- Extend `train_pooled_segmentation.py`.
- Add a pretrained encoder U-Net.
- Complete final per-dataset metrics.
- Render final examples.

### Teammate C - Report and website

- Consume only `outputs/website_results_partial.json`.
- Update `report/glaucoma_project_report.md`.
- Update `website/`.
- Do not add placeholder metrics.

### Teammate D - Verification

- Run duplicate, manifest, split, and export validators.
- Reconcile every summary metric with per-image predictions.
- Confirm no test leakage.

## 8. Safety and honesty rules

- Never claim a Kaggle DRISHTI result is officially sourced.
- Never report Dice or IoU for ACRIMA.
- Never use test masks to create test classification crops.
- Never publish held-out metrics without saved per-image predictions.
- Never claim statistical significance from a single split.
- Keep partial/third-party sources clearly labeled.
- Freeze the final split before final training.

## 9. Copy/paste continuation prompt

Use this prompt with the next teammate or AI coding agent:

```text
Continue the glaucoma MLDL project from TEAM_HANDOFF.md and PROJECT.ipynb.

Start by reading:
1. TEAM_HANDOFF.md
2. tools/HANDOFF.md
3. PROJECT.ipynb
4. README.md
5. report/glaucoma_project_report.md

The project currently has:
- ACRIMA whole-image classification baselines and external transfer tests
- RIM-ONE compact disc/cup U-Net and per-image segmentation results
- A matched RIM-ONE whole-image versus predicted-disc-crop experiment
- A four-dataset pooled frozen-feature classifier
- A pooled RIM-ONE/ORIGA/REFUGE segmentation baseline
- DRISHTI-GS1 Kaggle-mirror classification and segmentation baselines
- Grad-CAM, split audits, a 17-combination measured export, and a partial website/report

Do not restart prior work or overwrite saved results. Do not treat the Kaggle
DRISHTI-GS1 mirror as an official-source-verified dataset. Do not report
segmentation on ACRIMA or create test crops from test ground-truth masks.

Next priority:
1. Freeze one final five-source manifest and split configuration.
2. Resolve source-policy and duplicate-audit questions in the manifest.
3. Implement a fine-tuned whole-image ResNet-18 classifier.
4. Implement a pretrained-encoder pooled disc/cup segmenter.
5. Rebuild the crop experiment using saved predicted masks for test images.
6. Generate final Grad-CAM examples and save per-image predictions.
7. Export only reproducible final metrics to the website and report.
8. Update TEAM_HANDOFF.md and tools/HANDOFF.md after each meaningful stage.

Before training, state:
- exact split counts;
- held-out data isolation;
- source names and provenance;
- model architecture;
- hyperparameter selection protocol;
- crop mask source;
- expected output files.

After each stage, run the smallest test that verifies:
- manifest integrity;
- no train/test image duplication;
- metric values match per-image predictions;
- checkpoints load;
- export validation passes;
- the project notebook executes.
```
