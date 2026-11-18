# Glaucoma Classification and Optic Disc/Cup Segmentation

**Project status: measured partial checkpoint**

## 1. Research aim

The central question is whether restricting a glaucoma classifier to the optic-disc region improves held-out performance compared with a whole fundus image. The project also measures optic-disc/cup segmentation and provides Grad-CAM examples.

## 2. Datasets and verification

Four dataset sources are currently usable for measured baselines. ACRIMA contains 705 classification images, but no segmentation masks. RIM-ONE r3 contains 159 stereo images and paired disc/cup masks; 35 suspect cases are excluded from binary classification. A verified combined Figshare archive provides 650 ORIGA images and 650 masks plus a 400/400/400 REFUGE repackaging. The separate official REFUGE RAR supplied 400 official test labels and was used for independent REFUGE test evaluation. DRISHTI-GS1 remains blocked by authorized registration access.

No five-dataset pooled result is claimed.

## 2.1 DRISHTI-GS1 Kaggle-mirror checkpoint

The requested Kaggle mirror by `lokeshsaipureddi` contains 101 images: 50 official-style training images and 51 official-style test images. The mirror folders contain 70 glaucoma and 31 normal images, plus paired OD/cup soft masks.

The compact frozen-classification baseline used 40 training and 10 validation images from the supplied training set, with the 51 test images held out. It reached 78.43% accuracy, 87.06% F1, and 68.02% ROC AUC.

A compact randomly initialized U-Net used the same 40/10/51 split. It reached test disc/cup Dice 0.8841/0.7171 and IoU 0.7997/0.5796.

These are working mirror-specific baselines, not official-source-verified results. The archive and masks should be described as the `lokeshsaipureddi` Kaggle mirror in any presentation or publication.

## 3. Measured baselines

The ACRIMA-only frozen ResNet-18 feature model with a logistic head reached 90.78% accuracy, 91.93% F1, and 97.84% ROC AUC on a fixed 141-image test split. The unchanged model transferred poorly to RIM-ONE, ORIGA, and REFUGE, showing substantial domain shift.

A compact RIM-ONE U-Net reached held-out disc Dice/IoU of 0.7989/0.6904 and cup Dice/IoU of 0.5620/0.4283 on 32 images. These values describe a small, single-dataset segmentation baseline.

A separate four-dataset frozen-feature baseline improved external-held-out AUC on ORIGA, REFUGE, and RIM-ONE compared with the ACRIMA-only model, while losing AUC on ACRIMA. Independent official REFUGE test AUC was 0.791 at 0.750 accuracy.

## 4. Whole image versus disc crop

On a matched fixed RIM-ONE split of 80 train, 19 validation, and 25 non-suspect test images, whole-image classification reached 0.8000 accuracy and 0.9044 ROC AUC. The predicted-disc-crop variant reached 0.7600 accuracy and 0.8750 ROC AUC.

Ground-truth disc masks were used only to construct training and validation crops. Test crops used the saved U-Net prediction, preventing test-mask leakage. This experiment therefore provides one measured domain/specification result: the crop did not outperform the whole image in this protocol. It does not establish that disc crops are inferior in general.

## 4.1 Pooled optic-disc/cup segmentation

A compact randomly initialized U-Net was trained across RIM-ONE, ORIGA, and REFUGE. The fixed test set contained 32 RIM-ONE, 325 ORIGA, and 800 REFUGE images. Held-out disc/cup Dice was 0.6920/0.5527 on RIM-ONE, 0.7510/0.8053 on ORIGA, and 0.6930/0.6701 on REFUGE. Per-image Dice and IoU were saved for all 1,157 test images, together with six best/worst visual panels.

This result is a pooled three-dataset baseline, not the final pretrained-encoder segmenter. The REFUGE test pool combines the supplied validation split and official test split. ORIGA and REFUGE come from the verified combined third-party archive, and DRISHTI-GS1 is unavailable.

## 5. Explainability and error analysis

Grad-CAM panels were generated for the ACRIMA baseline and pooled four-dataset classifier. The panels target the model's predicted class, and normal predictions use the normal-logit direction. Grad-CAM is evidence for model attention patterns, not clinical causality. Held-out RIM-ONE segmentation panels include best and worst cases, with per-image Dice and IoU saved.

## 6. Limitations

- No patient-level ORIGA identifiers were available.
- The ORIGA/REFUGE release was a third-party combined archive.
- The REFUGE copy was not byte-verified across every image against the original RAR.
- DRISHTI-GS1 is missing.
- The U-Net is compact and randomly initialized.
- Frozen-feature models are baselines, not final fine-tuned models.
- Single-split metrics do not establish statistical significance.

## 7. Reproducibility

The canonical record is `PROJECT.ipynb`. Saved per-image predictions, fixed splits, checkpoints, metrics, manifests, Grad-CAM panels, and segmentation examples are under `outputs/` and `data/audit/`. The measured-only website export is `outputs/website_results_partial.json`.
