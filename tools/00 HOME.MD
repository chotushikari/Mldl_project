# 00 HOME.md — Glaucoma MLDL Project

## Project One-Liner
Build, train, evaluate and explain deep learning models for glaucoma classification + optic disc/cup segmentation + Grad-CAM on five public retinal fundus datasets, then present results in a clean, honest, interactive way.

## Official Scope (Professor Requirements)
1. Classification (Normal vs Glaucoma)
2. Segmentation of Optic Disc and Optic Cup (where ground truth exists)
3. Explainable AI (Grad-CAM)
4. Evaluation with proper metrics
5. Comparison across datasets
6. Working code + trained weights + 2–3 page report + demonstration

## Datasets
- DRISHTI-GS
- RIM-ONE V3
- REFUGE
- ORIGA
- ACRIMA

**Critical notes**:
- ACRIMA has classification labels only (no disc/cup masks) → no Dice/IoU allowed
- Some datasets have access restrictions / registration
- Official splits must be respected where they exist
- Never redistribute restricted images publicly

## Current Strategic Approach
- Research question: Does focusing the classifier on the optic disc region improve performance vs whole-image classification?
- One segmentation model (U-Net + ResNet-18 encoder)
- Two classifiers (same architecture): whole-image vs disc-centred crop
- Grad-CAM on the classifiers
- First complete solid science → then build interactive results explorer website from verified exports only

## Folder Structure (Target)