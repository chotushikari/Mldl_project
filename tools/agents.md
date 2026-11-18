# AGENTS.md — Standing Instructions for AI Assistants

## Core Mission
Help complete a university MLDL project on glaucoma classification, segmentation and explainable AI to a high standard, using only free resources, while avoiding common student pitfalls and hallucinations.

## Mandatory Rules
1. Always read `00 HOME.md` and `HANDOFF.md` at the start of every session.
2. Never invent dataset statistics, metrics, paper results, or architecture performance.
3. Never report Dice/IoU on ACRIMA (no ground-truth masks).
4. Never use test-set ground-truth masks to create classification crops (no leakage).
5. Prefer small, verifiable steps. Complete one Level fully before moving to the next.
6. All reported numbers must come from actual saved predictions/checkpoints.
7. Design every training/evaluation step with the future website export in mind.
8. Keep code clean, modular, and reproducible.
9. Use free resources only (local RTX 3050 preferred, Colab as backup).
10. At the end of every meaningful unit of work, update `HANDOFF.md`.

## Preferred Technical Stack
- PyTorch + TorchVision
- Segmentation Models PyTorch (for U-Net with pretrained encoder)
- pytorch-grad-cam (or equivalent maintained Grad-CAM library)
- Albumentations for preprocessing
- Clean Python package structure under `src/`
- Results stored in structured JSON + figures

## Current Research Framing
Primary question we are answering:
> Does a disc-centred crop improve glaucoma classification performance compared to using the full fundus image, when both use the same classifier architecture and training budget?

## Communication Style
- Be precise and senior-engineer level.
- Call out risks and failure modes early.
- Prefer visual checkpoints the student can verify.
- Never give generic textbook answers when a stronger project-specific strategy exists.