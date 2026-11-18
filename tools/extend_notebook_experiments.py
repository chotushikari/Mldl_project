"""Keep PROJECT.ipynb aligned with the experiments actually run on disk."""
from pathlib import Path

import nbformat


path = Path("PROJECT.ipynb")
nb = nbformat.read(path, as_version=4)
marker = "## Reproducible measured experiments"
nb.cells = [cell for cell in nb.cells if marker not in cell.source and not cell.metadata.get("experiment_section")]


def add(kind, source):
    cell = nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
    cell.metadata["experiment_section"] = True
    return cell


section = [
    add("markdown", """## Reproducible measured experiments

These are **baseline experiments** on the two verified datasets. Training is opt-in because rerunning it takes time and changes saved artifacts. Set the relevant switch to `True` to reproduce a stage. The saved test split is never used for model or threshold selection. ACRIMA has no optic-disc/cup masks, and RIM-ONE suspect cases are excluded from binary classification.

| Stage | Dataset | Script | Saved output |
|---|---|---|---|
| Frozen ResNet-18 features + logistic classifier | ACRIMA | `tools/train_acrima_baseline.py` | `outputs/acrima_resnet18_frozen/` |
| Self-contained classifier checkpoint | ACRIMA | `tools/package_acrima_classifier.py` | `outputs/acrima_resnet18_frozen/resnet18_classifier.pt` |
| Untuned external classification evaluation | RIM-ONE r3 | `tools/evaluate_acrima_on_rim.py` | `outputs/acrima_resnet18_frozen/rim_external_*` |
| Compact two-channel U-Net | RIM-ONE r3 | `tools/train_rim_segmentation.py` | `outputs/rim_unet_baseline/` |
| Held-out per-image segmentation evaluation | RIM-ONE r3 | `tools/evaluate_rim_segmentation.py` | `outputs/rim_unet_baseline/test_predictions.csv` |

The external RIM-ONE result is a domain-shift check, not a RIM-trained model comparison. The U-Net baseline is not yet the planned ResNet-18 encoder model."""),
    add("code", """# Reproduce individual stages only when their required dataset archives are present.
RUN_ACRIMA_CLASSIFIER = False
RUN_RIM_EXTERNAL_EVAL = False
RUN_RIM_SEGMENTER = False

def run_project_script(name):
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools' / name)], cwd=PROJECT_ROOT, check=True)

if RUN_ACRIMA_CLASSIFIER:
    assert DATASETS['ACRIMA']['archive'].exists(), 'ACRIMA archive is required'
    run_project_script('train_acrima_baseline.py')
    run_project_script('package_acrima_classifier.py')
if RUN_RIM_EXTERNAL_EVAL:
    assert DATASETS['RIM-ONE V3']['archive'].exists(), 'RIM-ONE archive is required'
    assert (OUTPUTS / 'acrima_resnet18_frozen' / 'classifier.joblib').exists(), 'Train the ACRIMA baseline first'
    run_project_script('evaluate_acrima_on_rim.py')
if RUN_RIM_SEGMENTER:
    assert DATASETS['RIM-ONE V3']['archive'].exists(), 'RIM-ONE archive is required'
    run_project_script('train_rim_segmentation.py')
    run_project_script('evaluate_rim_segmentation.py')
print('Experiment switches:', RUN_ACRIMA_CLASSIFIER, RUN_RIM_EXTERNAL_EVAL, RUN_RIM_SEGMENTER)"""),
    add("markdown", """### Verify held-out results

The checks below compare saved predictions with the reported confusion counts and check that segmentation split membership is unique. They fail if the artifacts are incomplete or inconsistent."""),
    add("code", """classifier_dir = OUTPUTS / 'acrima_resnet18_frozen'
segmenter_dir = OUTPUTS / 'rim_unet_baseline'
required = [classifier_dir / 'metrics.json', classifier_dir / 'test_predictions.csv', classifier_dir / 'rim_external_metrics.json', classifier_dir / 'resnet18_classifier.pt', segmenter_dir / 'metrics.json', segmenter_dir / 'split.csv', segmenter_dir / 'best.pt', segmenter_dir / 'test_predictions.csv']
missing = [str(p) for p in required if not p.exists()]
if missing:
    print('Measured baselines are missing; run the switches above:', missing)
else:
    cm = json.loads((classifier_dir / 'metrics.json').read_text(encoding='utf-8'))
    with (classifier_dir / 'test_predictions.csv').open(newline='', encoding='utf-8') as f:
        predictions = list(csv.DictReader(f))
    counts = {k: 0 for k in ('TN', 'FP', 'FN', 'TP')}
    for row in predictions:
        y, p = int(row['true_label']), int(row['predicted_label'])
        counts[('TN', 'FP', 'FN', 'TP')[2*y+p]] += 1
    assert len(predictions) == cm['n_test']
    assert all(counts[k] == cm[k] for k in counts), (counts, cm)
    with (segmenter_dir / 'split.csv').open(newline='', encoding='utf-8') as f:
        splits = list(csv.DictReader(f))
    sm = json.loads((segmenter_dir / 'metrics.json').read_text(encoding='utf-8'))
    assert len({r['image'] for r in splits}) == len(splits)
    assert {s: sum(r['split'] == s for r in splits) for s in ('train', 'val', 'test')} == {s: sm['n_' + s] for s in ('train', 'val', 'test')}
    with (segmenter_dir / 'test_predictions.csv').open(newline='', encoding='utf-8') as f:
        segment_predictions = list(csv.DictReader(f))
    assert len(segment_predictions) == sm['n_test']
    assert {r['image'] for r in segment_predictions} == {r['image'] for r in splits if r['split'] == 'test'}
    for region in ('disc', 'cup'):
        for metric in ('dice', 'iou'):
            key = region + '_' + metric
            mean = sum(float(r[key]) for r in segment_predictions) / len(segment_predictions)
            assert abs(mean - sm['test_' + key]) < 1e-6, key
    print('ACRIMA held-out confusion:', counts)
    print('ACRIMA test accuracy/F1/AUC:', *(round(cm['test_' + k], 4) for k in ('accuracy', 'f1', 'auc')))
    external = json.loads((classifier_dir / 'rim_external_metrics.json').read_text(encoding='utf-8'))
    print('RIM-ONE external accuracy/AUC:', round(external['accuracy'], 4), round(external['auc'], 4))
    print('RIM-ONE disc Dice/IoU:', round(sm['test_disc_dice'], 4), round(sm['test_disc_iou'], 4))
    print('RIM-ONE cup Dice/IoU:', round(sm['test_cup_dice'], 4), round(sm['test_cup_iou'], 4))"""),
    add("code", """# Exact-image leakage audit of the frozen train/validation/test assignments.
if DATASETS['ACRIMA']['archive'].exists() and DATASETS['RIM-ONE V3']['archive'].exists():
    run_project_script('audit_split_duplicates.py')
else:
    print('Exact duplicate audit waits for both verified archives.')"""),
    add("markdown", """### Visual predictions and Grad-CAM

Render examples from the held-out split. A Grad-CAM heatmap shows sensitivity of the classifier's glaucoma logit to image regions; it does not prove the model used clinically valid features."""),
    add("code", """GENERATE_VISUALS = False
if GENERATE_VISUALS:
    for name in ('render_rim_predictions.py', 'render_gradcam.py', 'render_results.py', 'render_baseline_architecture.py'):
        run_project_script(name)

from IPython.display import display
from PIL import Image
architecture = OUTPUTS / 'architecture_baselines.png'
if architecture.exists():
    display(Image.open(architecture))
for folder in (segmenter_dir / 'examples', classifier_dir / 'gradcam'):
    examples = sorted(folder.glob('*.png')) if folder.exists() else []
    print(folder, ':', len(examples), 'saved panels')
    if examples:
        display(Image.open(examples[0]))"""),
    add("markdown", """### Export measured data for the future results website

The export contains only completed experiments and per-image predictions. The website must offer only combinations represented in this file; it must not generate placeholder metrics."""),
    add("code", """if all(p.exists() for p in required + pooled_required):
    run_project_script('export_results.py')
    run_project_script('validate_website_export.py')
    website_export = OUTPUTS / 'website_results_partial.json'
    payload = json.loads(website_export.read_text(encoding='utf-8'))
    print('Verified export:', website_export, '| experiments:', [e['id'] for e in payload['experiments']])
else:
    print('Website export waits for measured experiment artifacts.')"""),
]

section[-2:-2] = [
    add("markdown", """### Four-dataset pooled classification baseline

The original ACRIMA-only classifier predicted glaucoma for almost every ORIGA and REFUGE image. This measured domain shift motivated a second, separate experiment. It uses the full retinal view letterboxed to 224×224, a frozen ImageNet ResNet-18, and a logistic head weighted equally by dataset and class. Hyperparameter selection uses macro validation AUC. ACRIMA's fixed test, ORIGA's provided B set, REFUGE's official validation set, and non-suspect RIM-ONE test images remain held out. The separate original-source REFUGE test labels provide another independent evaluation. **This is a four-dataset baseline, not the final five-dataset model.**"""),
    add("code", """RUN_FOUR_DATASET_BASELINE = False
if RUN_FOUR_DATASET_BASELINE:
    assert all(DATASETS[n]['archive'] and DATASETS[n]['archive'].exists() for n in ('ACRIMA', 'RIM-ONE V3', 'ORIGA', 'REFUGE'))
    for name in ('evaluate_acrima_on_repackaged.py', 'train_pooled_frozen_baseline.py', 'evaluate_pooled_refuge_official_test.py', 'compare_classifiers_common_splits.py', 'render_pooled_gradcam.py'):
        run_project_script(name)
print('Four-dataset baseline reproduction:', 'ran' if RUN_FOUR_DATASET_BASELINE else 'using saved artifacts')"""),
    add("code", """pooled_dir = OUTPUTS / 'pooled_four_dataset_frozen'
pooled_required = [pooled_dir / 'metrics.json', pooled_dir / 'resnet18_classifier.pt', pooled_dir / 'refuge_official_test_metrics.json', OUTPUTS / 'classifier_common_split_comparison.json', OUTPUTS / 'classifier_common_split_comparison.png']
if all(p.exists() for p in pooled_required):
    pooled_metrics = json.loads((pooled_dir / 'metrics.json').read_text(encoding='utf-8'))
    comparison = json.loads((OUTPUTS / 'classifier_common_split_comparison.json').read_text(encoding='utf-8'))
    for dataset, result in comparison['results'].items():
        old, new = result['ACRIMA-only frozen'], result['Pooled four-dataset frozen']
        assert old['n_test'] == new['n_test']
        print(dataset, 'n=', old['n_test'], 'AUC:', round(old['auc'], 3), '→', round(new['auc'], 3))
    official = json.loads((pooled_dir / 'refuge_official_test_metrics.json').read_text(encoding='utf-8'))
    print('REFUGE independent official test:', official['n_test'], 'images, AUC', round(official['auc'], 4))
    display(Image.open(OUTPUTS / 'classifier_common_split_comparison.png'))
    pooled_cams = sorted((pooled_dir / 'gradcam').glob('*/*.png'))
    print('Pooled predicted-class Grad-CAM panels:', len(pooled_cams))
    if pooled_cams:
        display(Image.open(pooled_cams[0]))
else:
    print('Pooled experiment artifacts missing:', [str(p) for p in pooled_required if not p.exists()])"""),
]

gate_index = next(i for i, cell in enumerate(nb.cells) if "RUN_FINAL_TRAINING = False" in cell.source)
nb.cells[gate_index:gate_index] = section
for cell in nb.cells:
    if "handoff.write_text(handoff_text" in cell.source:
        cell.source = "handoff = PROJECT_ROOT / 'tools' / 'HANDOFF.md'\nprint(handoff.read_text(encoding='utf-8') if handoff.exists() else 'Handoff file missing')"
        break
nbformat.write(nb, path)
print(f"Updated {path} with {len(section)} experiment cells")
