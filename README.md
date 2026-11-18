# Glaucoma classification, disc/cup segmentation, and XAI

Dataset setup: [DATASETS.md](DATASETS.md). Raw images are intentionally excluded from GitHub.

For a team-ready summary of completed work, remaining tasks, key files, commands, and a continuation prompt, read `TEAM_HANDOFF.md`.

For a training-specific explanation of every implemented model, split, saved artifact, and final training plan, read `TRAINING_IMPLEMENTATION.md`.

For GPU fine-tuning in Colab, use `COLAB_TRAINING.ipynb`. This notebook requires a T4 GPU runtime and a project upload or Drive mount.

## Current checkpoint: four-dataset baselines; DRISHTI-GS pending

The project will train and test models before creating the interactive results site. All published metrics must come from saved per-image predictions on fixed test splits. The site will consume those predictions and metrics.

### First measured result (baseline, not final model)

On ACRIMA, a frozen ImageNet ResNet-18 feature extractor with balanced logistic regression achieved **90.78% accuracy, 90.24% precision, 93.67% recall, 91.93% F1, and 97.84% ROC AUC** on a fixed 141-image test split (TN 54, FP 8, FN 5, TP 74). A separate 106-image validation set selected the regularization parameter; the classifier was then refit on 564 train+validation images. See `outputs/acrima_resnet18_frozen/metrics.json`, `split.csv`, and `test_predictions.csv`. This is a single-split, same-dataset baseline; it does not establish cross-dataset generalization or meet the final training requirement.

External testing of that *unchanged* classifier on the left view of 124 RIM-ONE V3 normal/glaucoma images gave **31.45% accuracy and 61.42% ROC AUC** (TN 0, FP 85, FN 0, TP 39). All 35 suspects were excluded. This exposes severe domain shift and shows why the ACRIMA score is not a general glaucoma result. See `rim_external_metrics.json` and `rim_external_predictions.csv`.

A separate compact U-Net was trained on RIM-ONE V3 masks with a fixed 103/24/32 split. Best validation checkpoint selection used mean disc/cup Dice. On the untouched 32-image test set: **disc Dice 0.7989, IoU 0.6904; cup Dice 0.5620, IoU 0.4283**. See `outputs/rim_unet_baseline/metrics.json`, `history.json`, `split.csv`, and `best.pt`. This is a small, single-dataset baseline, not the final pooled segmentation model.

The unchanged ACRIMA classifier was also tested externally on all 650 ORIGA images and 400 labeled REFUGE validation images. It predicted almost every image as glaucoma (ORIGA accuracy 0.260, AUC 0.372; REFUGE validation accuracy 0.103, AUC 0.583). Those are measured transfer failures, not tuned results. The original-source REFUGE test ground-truth spreadsheet was recovered separately and was not used to train the models.

A second **four-dataset pooled frozen-feature baseline** uses full-image letterboxing, balanced weighting by dataset and class, and fixed held-out assignments. On those held-out images its AUC was 0.925 (ACRIMA, 141), 0.700 (ORIGA B, 325), 0.779 (REFUGE validation, 400), and 0.868 (RIM-ONE non-suspect test, 25). On the independent original-source REFUGE test set (400 images), AUC was **0.791** and accuracy **0.750**. See `outputs/pooled_four_dataset_frozen/metrics.json`, `split.csv`, and `refuge_official_test_metrics.json`. This baseline does not include DRISHTI-GS or fine-tune the ResNet backbone.

The fair same-image classifier comparison is `outputs/classifier_common_split_comparison.json` and `.png`; it filters each model to identical held-out images within each dataset. The pooled model improved AUC on ORIGA, REFUGE, and RIM-ONE and lost AUC on ACRIMA. No statistical significance is claimed from these single splits.

The generated comparison figure is `outputs/acrima_resnet18_frozen/comparison.png`. Eight held-out Grad-CAM panels are under `outputs/acrima_resnet18_frozen/gradcam/`; 16 original/ground-truth/prediction segmentation panels (including selected best and worst cases) are under `outputs/rim_unet_baseline/examples/`. The 32 held-out segmentation images now have per-image Dice/IoU in `outputs/rim_unet_baseline/test_predictions.csv`; their means match the aggregate metrics. A self-contained classifier checkpoint is `outputs/acrima_resnet18_frozen/resnet18_classifier.pt` and was checked against five saved test probabilities. The exact-image split audit found no duplicate images across splits; see `outputs/split_duplicate_audit.json`. The measured-only website export is `outputs/website_results_partial.json`; its partial status is intentional.

The architecture diagram for the original implemented baselines is `outputs/architecture_baselines.png`. Sixteen predicted-class Grad-CAM panels for the pooled classifier are under `outputs/pooled_four_dataset_frozen/gradcam/`. The partial website export currently contains 12 measured experiment combinations in `outputs/website_results_partial.json`.

A matched RIM-ONE whole-image versus disc-crop study is complete for the fixed 80/19/25 binary split. Whole-image test accuracy/AUC were **0.8000/0.9044**; the predicted-disc-crop test accuracy/AUC were **0.7600/0.8750**. Training/validation crops used ground-truth disc masks, while test crops used saved U-Net predictions so test masks were never used for crop construction. See `outputs/rim_whole_vs_crop/`.

A partial pooled compact U-Net segmentation experiment completed across RIM-ONE, ORIGA, and REFUGE. Test disc/cup Dice was **0.6920/0.5527** on RIM-ONE, **0.7510/0.8053** on ORIGA, and **0.6930/0.6701** on REFUGE. The outputs include 1,157 held-out per-image scores and six best/worst panels under `outputs/pooled_segmentation/`.

### Verified data

| Dataset | Status | Source | Notes |
|---|---|---|---|
| ACRIMA | Downloaded, integrity and labels audited | Authors' Figshare article 7613135, `Database.zip` | 705 images: 396 glaucoma, 309 normal; no segmentation masks. See `data/audit/acrima_manifest.csv`. |
| RIM-ONE V3 | Downloaded and audited | University of La Laguna `RIM-ONE-r3.zip` | 85 normal, 39 glaucoma, 35 suspect; all 159 have paired disc/cup average masks. Binary classification excludes suspects unless a separate protocol is declared. |
| DRISHTI-GS1 | Awaiting researcher access | [IIIT Hyderabad MIP datasets](https://cvit.iiit.ac.in/mip/datasets.html) | The official page requests an email for access and lists expert binary labels for GS1. Confirm the assigned variant; do not redistribute restricted images. |
| REFUGE | Original-source RAR and repackaged copy audited; both published MD5 checks passed | [Original-source Figshare article 26049574](https://figshare.com/articles/dataset/REFUGE/26049574) and combined archive | 400 train, 400 validation, 400 test; each has 360 normal/40 glaucoma. The copy omitted test labels, but the original `REFUGE-Test-GT` spreadsheet supplies them. All names and nine image/mask samples were cross-checked. |
| ORIGA-light | Repackaged copy audited; published MD5 matched | [Figshare article 24549217](https://figshare.com/articles/dataset/Retinal_Fundus_Glaucoma_Image_dataset/24549217) combined archive | 650 labeled images (482 normal, 168 glaucoma), 650 masks, and 325/325 provided A/B sets. This is a third-party repackaging, not an original release. |

### Reproduce this checkpoint

Open and run `PROJECT.ipynb` from top to bottom. Its experiment switches default to off, so it checks the saved results without retraining. Set a switch to `True` only to reproduce that stage. The notebook is executed and verified locally; a live Colab runtime was not accessible from this chat. Individual scripts are in `tools/`. No five-dataset or Colab GPU training result has been claimed.

### Experiment order

1. Audit each dataset, labels, masks, patient IDs, and available official splits.
2. Freeze manifests and train/validation/test assignments; check exact and perceptual duplicates.
3. Train a whole-image ResNet-18 classifier as the baseline.
4. Train a disc/cup U-Net on images with verified masks.
5. Train the matched disc-crop ResNet-18 using *predicted* disc locations, never test ground truth.
6. Evaluate both classifiers per dataset and segmentation only where masks exist. Save per-image predictions, checkpoints, configurations, metrics, and Grad-CAM images.
7. Generate the 2–3 page report and the later static results explorer from the same saved outputs.

### Important audit finding

An older notebook in Downloads uses a Kaggle copy of RIM-ONE and states it has no segmentation masks. That does not establish the contents of the original RIM-ONE V3 release. It also assumes particular REFUGE test labels and calls its test set the same hospital as training; these must be checked against the actual archive and official challenge protocol. Do not run that notebook as the final experiment.
