# HANDOFF.md - Current Project State

**Last Updated**: 2026-11-18  
**Canonical workflow**: `PROJECT.ipynb`  
**Team-sharing package**: `TEAM_HANDOFF.md`
**Colab training launcher**: `COLAB_TRAINING.ipynb`
**Current Level**: Level 1 - Dataset Acquisition & Inventory  
**Status**: Five-source working experiments now include the requested Kaggle DRISHTI-GS1 mirror. Results remain partial and mirror-specific.

## Completed and verified
- ACRIMA archive and labels audited: 705 images, 396 glaucoma, 309 normal; no segmentation masks.
- RIM-ONE V3 archive audited: 85 normal, 39 glaucoma, 35 suspect; all 159 images have paired disc/cup average masks.
- Existing baseline outputs remain preserved under `outputs/` and are not presented as final five-dataset results.
- `PROJECT.ipynb` is the single execution entry point for acquisition, provenance, inventory, visual checks, and gates.
 - A Kaggle DRISHTI-GS1 mirror was downloaded and used for working classification and segmentation baselines, clearly labeled by source.
- DRISHTI-GS1 frozen classifier: 40 train / 10 val / 51 test; accuracy 78.43%, F1 87.06%, AUC 68.02%.
- DRISHTI-GS1 compact U-Net: test disc Dice/IoU 0.8841/0.7997; cup Dice/IoU 0.7171/0.5796.
- DRISHTI segmentation and classification per-image outputs are under `outputs/drishti_gs1_mirror/` and `outputs/drishti_gs1_mirror_classifier/`.
- The measured export contains 17 completed combinations, including both DRISHTI mirror experiments.
- A complete 14-cell `COLAB_TRAINING.ipynb` master pipeline is ready. It performs audits, baselines, fine-tuning, segmentation, crop, Grad-CAM, export, validation, and optional Drive persistence.

## DRISHTI-GS1 provisional mirror
- Uploader: `lokeshsaipureddi`
- Archive: 357,915,185 bytes
- SHA-256: `8b681cd260f76b825a8be0ac1b45bf13da4aa55cc157441d5a0e18067c818767`
- Mirror inventory: 101 images, 50 train and 51 test
- Folder labels: 70 glaucoma and 31 normal
- Soft masks: 101 OD and 101 cup
- Kaggle advertises CC0, but the official IIIT terms restrict use to academic/research and prohibit redistribution except within a registered team.
- Working mirror experiments are not presented as official-source-verified.

## Blockers and safeguards
- The Kaggle DRISHTI-GS1 copy is a third-party mirror, not official confirmation of source or test labels.
- DRISHTI-GS1 requires official registration/access and must not be redistributed outside the registered team.
- Official-source verification is not required for the requested mirror run.
- Replacing this mirror with an official archive will require rerunning affected splits and metrics.

## Next exact actions
1. Freeze the five-source mirror protocol and rerun cross-dataset duplicate audits.
2. Train the final fine-tuned classifier comparison across all five sources.
3. Train the final pretrained-encoder pooled segmenter across all five sources.
4. Recompute crop-vs-whole and Grad-CAM evidence.
5. Freeze the report and website only after every metric reconciles with saved predictions.
