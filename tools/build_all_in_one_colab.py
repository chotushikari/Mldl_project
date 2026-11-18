"""Build a Colab notebook that embeds every runtime source file in code cells.

The generated notebook needs no repository clone, pre-existing folders, or uploaded
dataset archives. It creates its own folders, downloads the data, and writes the
embedded Python sources before running each measured stage.
"""
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "GLAUCOMA_ALL_IN_ONE_COLAB.ipynb"
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"name": TARGET.name, "provenance": []}


def md(source):
    nb.cells.append(nbformat.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbformat.v4.new_code_cell(source))


def embed(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    if not source.strip():
        source = "# Package marker for notebook-generated models.\n"
    if relative_path == "tools/audit_refuge_original.py":
        source = source.replace("import subprocess\n", "import subprocess\nimport shutil\nRAR_TAR = shutil.which('bsdtar') or 'tar'\n")
        source = source.replace('["tar",', "[RAR_TAR,")
        assert "RAR_TAR" in source
    cell = "%%writefile " + relative_path.replace("\\", "/") + "\n" + source
    code(cell)


def stage(title, description, sources, run_code):
    md(f"## {title}\n\n{description}\n\nRun the source cells below, then the stage runner. Every file they create comes from code **inside this notebook**.")
    for source in sources:
        embed(source)
    code(run_code)


md("""# Glaucoma MLDL — one-notebook Colab experiment

**Run cells from top to bottom.** This notebook downloads ACRIMA, RIM-ONE r3, a combined ORIGA/REFUGE archive, the original-source REFUGE archive, and the DRISHTI-GS1 Kaggle mirror. No dataset ZIP or project folder must be uploaded. It writes all training code from cells inside this notebook.

The notebook trains classification and optic-disc/cup segmentation baselines, a pooled fine-tuned classifier, and a matched whole-image versus predicted-disc-crop comparison. It evaluates held-out images, creates Grad-CAM and mask overlays, builds architecture and metric figures, and exports measured results for a later static website.

**Source note:** DRISHTI-GS1 comes from a third-party Kaggle mirror because that is the source chosen for this run. Label its results as *mirror-based* in the report. Do not redistribute its images. The original IIIT Hyderabad terms allow redistribution only within a registered team. REFUGE official test labels come from its original-source RAR; the combined archive's test index alone does not contain them.

**Time and storage:** The four public archives total roughly 10.5 GB, plus the DRISHTI mirror. Download and training can take hours on a free Colab session. Choose a GPU runtime before the training cells. Results depend on actual training; no metric in this notebook is pre-filled.""")

code(r'''# 0 — Environment, folders, and reproducible configuration
import csv, io, json, os, random, shutil, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path('/content/glaucoma_mldl') if Path('/content').exists() else Path.cwd() / 'glaucoma_mldl_run'
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
for folder in ('tools', 'models', 'data/raw', 'data/audit', 'data/provisional/drishti_gs1', 'outputs', 'results', 'website', 'report'):
    Path(folder).mkdir(parents=True, exist_ok=True)

required = {'torch':'torch', 'torchvision':'torchvision', 'numpy':'numpy', 'PIL':'Pillow', 'sklearn':'scikit-learn', 'matplotlib':'matplotlib', 'openpyxl':'openpyxl', 'requests':'requests', 'joblib':'joblib'}
import importlib.util
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
import numpy as np, torch
from PIL import Image
from IPython.display import display

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Working directory:', ROOT)
print('Device:', DEVICE, '| PyTorch:', torch.__version__)
print('Free disk GiB:', round(shutil.disk_usage(ROOT).free / 2**30, 1))
if shutil.disk_usage(ROOT).free < 15 * 2**30:
    raise RuntimeError('At least 15 GiB of free space is needed for the source archives. Free Colab disk space and rerun this cell.')

RUN_DOWNLOADS = True
RUN_BASELINES = True
RUN_POOLED_SEGMENTER = True
RUN_FINETUNING = True
RUN_VISUALS_AND_EXPORT = True
FORCE_RETRAIN = False

def run_script(name, required_output=None):
    target = ROOT / 'tools' / name
    if required_output and (ROOT / required_output).exists() and not FORCE_RETRAIN:
        print('Already complete:', required_output)
        return
    print('RUN:', name, flush=True)
    subprocess.run([sys.executable, str(target)], cwd=ROOT, check=True)

print('Setup complete. Continue to Step 1.')''')

stage("Step 1 — Download the five dataset sources", "The commands fetch all archives directly. Interrupted downloads resume. `data/raw/` and `data/provisional/` are created in this fresh runtime; no upload is needed.",
      ["tools/download_verified.py", "tools/download_drishti_kaggle_inventory.py", "tools/verify_figshare_archives.py"],
      r'''# Download and verify the original source archives plus the chosen DRISHTI mirror.
if RUN_DOWNLOADS:
    for name in ('acrima', 'rim_one_r3', 'origa_candidate', 'refuge_candidate'):
        subprocess.run([sys.executable, 'tools/download_verified.py', name], check=True)
    run_script('download_drishti_kaggle_inventory.py')
    run_script('verify_figshare_archives.py')
else:
    print('Downloads skipped by configuration; checking local runtime files.')
for name in ('ACRIMA_Database.zip', 'RIM-ONE-r3.zip', 'ORIGA_candidate_figshare.zip', 'REFUGE_figshare.rar'):
    assert (ROOT / 'data/raw' / name).is_file(), name
assert (ROOT / 'data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip').is_file()
print('Step 1 complete: five source archives are present.')''')

stage("Step 2 — Audit labels, masks, splits, and provenance", "The REFUGE RAR needs a RAR-capable extractor. This cell installs the free `bsdtar` utility if Linux does not already have one. Audits stop on missing or mismatched members; the DRISHTI source remains explicitly marked as a mirror.",
      ["tools/audit_acrima.py", "tools/audit_rim_one_r3.py", "tools/audit_origa_refuge.py", "tools/audit_refuge_original.py", "tools/audit_drishti_mirror.py", "tools/audit_split_duplicates.py"],
      r'''# Build every manifest directly from the downloaded archives.
if os.name != 'nt' and shutil.which('bsdtar') is None:
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'libarchive-tools'], check=True)
for name in ('audit_acrima.py', 'audit_rim_one_r3.py', 'audit_origa_refuge.py', 'audit_refuge_original.py', 'audit_drishti_mirror.py'):
    run_script(name)
for filename in ('acrima_manifest.csv', 'rim_one_r3_manifest.csv', 'origa_manifest.csv', 'refuge_manifest.csv', 'drishti_gs1_kaggle_mirror_manifest.csv', 'refuge_test_labels_official.csv'):
    assert (ROOT / 'data/audit' / filename).is_file(), filename
print('Step 2 complete: five manifests and official REFUGE test labels are verified.')''')

md("""### Visual dataset check

These sheets let you inspect image quality and annotation alignment before training. For ORIGA and REFUGE the mask values are `0=background, 1=disc rim, 2=cup`; therefore the **disc is `(mask > 0)`**, and the **cup is `(mask == 2)`**. ACRIMA has no masks, so it must not receive Dice or IoU scores.""")
code(r'''for name in ('acrima_samples.jpg', 'rim_sample.jpg', 'origa_sample.png', 'refuge_sample.png', 'drishti_gs1_kaggle_mirror_samples.jpg'):
    path = ROOT / 'data/audit' / name
    if path.exists():
        print(name)
        display(Image.open(path))''')

stage("Step 3 — Baseline classification and RIM-ONE segmentation", "Train the small, reproducible baselines first. These produce frozen ResNet-18 features, a compact RIM-ONE U-Net, held-out metrics, checkpoints, and an exact duplicate audit. They are *baselines*, not the final model claim.",
      ["models/__init__.py", "models/unet_small.py", "tools/train_acrima_baseline.py", "tools/package_acrima_classifier.py", "tools/train_rim_segmentation.py", "tools/evaluate_rim_segmentation.py", "tools/evaluate_acrima_on_rim.py", "tools/evaluate_acrima_on_repackaged.py"],
      r'''if RUN_BASELINES:
    run_script('train_acrima_baseline.py', 'outputs/acrima_resnet18_frozen/metrics.json')
    run_script('package_acrima_classifier.py', 'outputs/acrima_resnet18_frozen/resnet18_classifier.pt')
    run_script('train_rim_segmentation.py', 'outputs/rim_unet_baseline/metrics.json')
    run_script('evaluate_rim_segmentation.py', 'outputs/rim_unet_baseline/test_predictions.csv')
    run_script('evaluate_acrima_on_rim.py', 'outputs/acrima_resnet18_frozen/rim_external_metrics.json')
    run_script('evaluate_acrima_on_repackaged.py', 'outputs/acrima_resnet18_frozen/refuge_val_external_metrics.json')
    run_script('audit_split_duplicates.py')
print('Step 3 complete: baseline metrics and checkpoints are saved.')''')

stage("Step 4 — Four-dataset pooled classification comparison", "A pooled frozen-feature classifier uses full-image letterboxing and balances dataset/class groups. ORIGA B and REFUGE validation remain held out. REFUGE official test labels provide an independent final check. The same-image chart prevents unfair comparisons between different test sets.",
      ["tools/train_pooled_frozen_baseline.py", "tools/evaluate_pooled_refuge_official_test.py", "tools/compare_classifiers_common_splits.py"],
      r'''if RUN_BASELINES:
    run_script('train_pooled_frozen_baseline.py', 'outputs/pooled_four_dataset_frozen/metrics.json')
    run_script('evaluate_pooled_refuge_official_test.py', 'outputs/pooled_four_dataset_frozen/refuge_official_test_metrics.json')
    run_script('compare_classifiers_common_splits.py', 'outputs/classifier_common_split_comparison.json')
comparison = json.loads((ROOT / 'outputs/classifier_common_split_comparison.json').read_text())
for dataset, row in comparison['results'].items():
    print(dataset, 'AUC:', round(row['ACRIMA-only frozen']['auc'], 3), '->', round(row['Pooled four-dataset frozen']['auc'], 3))
display(Image.open(ROOT / 'outputs/classifier_common_split_comparison.png'))
print('Step 4 complete: fair held-out comparison saved.')''')

stage("Step 5 — Disc/cup segmentation on mask datasets", "Train a pooled two-channel U-Net on RIM-ONE, ORIGA, and REFUGE. The notebook version uses CUDA when available. The first mask channel is the **whole optic disc** (rim plus cup); the second is cup. Evaluate on fixed held-out images and save per-image Dice/IoU and best/worst overlays. DRISHTI mirror segmentation is trained and reported separately.",
      ["tools/train_pooled_segmentation.py", "tools/render_pooled_segmentation.py", "tools/train_drishti_segmentation.py", "tools/evaluate_drishti_mirror.py"],
      r'''if RUN_POOLED_SEGMENTER:
    run_script('train_pooled_segmentation.py', 'outputs/pooled_segmentation/metrics.json')
    run_script('render_pooled_segmentation.py', 'outputs/pooled_segmentation/test_predictions.csv')
    run_script('train_drishti_segmentation.py', 'outputs/drishti_gs1_mirror/metrics.json')
    run_script('evaluate_drishti_mirror.py', 'outputs/drishti_gs1_mirror/test_predictions.csv')
for path in ('outputs/pooled_segmentation/metrics.json', 'outputs/drishti_gs1_mirror/metrics.json'):
    result = json.loads((ROOT / path).read_text())
    print(path, json.dumps(result.get('test_metrics', result), indent=2)[:1500])
print('Step 5 complete: segmentation metrics and visual examples saved.')''')

stage("Step 6 — Matched whole-image versus predicted-disc crop", "The segmentation model predicts disc locations for **train, validation, and test** classification images. The crop model never uses a test ground-truth mask. Compare both ResNet-18 feature classifiers on exactly the same RIM-ONE test images, with the same candidate regularization values.",
      ["tools/train_rim_whole_vs_crop.py", "tools/render_rim_crop_comparison.py"],
      r'''if RUN_BASELINES:
    run_script('train_rim_whole_vs_crop.py', 'outputs/rim_whole_vs_crop/metrics.json')
    run_script('render_rim_crop_comparison.py', 'outputs/rim_whole_vs_crop/comparison.png')
print(json.loads((ROOT / 'outputs/rim_whole_vs_crop/metrics.json').read_text())['protocol'])
display(Image.open(ROOT / 'outputs/rim_whole_vs_crop/comparison.png'))
print('Step 6 complete: matched crop experiment saved.')''')

stage("Step 7 — DRISHTI mirror classification and pooled fine-tuning", "Train a separate DRISHTI mirror classifier and fine-tune a pooled ResNet-18 across all five source names. The mirror provenance remains visible in every result. The fine-tuned model uses ImageNet initialization, dataset/class-balanced sampling, a fixed seed, saved validation history, and held-out test predictions. Use a free Colab GPU for this stage.",
      ["tools/train_drishti_classifier.py", "tools/train_pooled_finetuned_classifier.py"],
      r'''if RUN_FINETUNING:
    if DEVICE.type != 'cuda':
        raise RuntimeError('Choose Runtime > Change runtime type > GPU in Colab, then rerun Step 0 and this stage.')
    run_script('train_drishti_classifier.py', 'outputs/drishti_gs1_mirror_classifier/metrics.json')
    run_script('train_pooled_finetuned_classifier.py', 'outputs/pooled_finetuned_classifier/metrics.json')
fine = json.loads((ROOT / 'outputs/pooled_finetuned_classifier/metrics.json').read_text())
print('Fine-tuning device:', fine['device'])
for dataset, metrics in fine['test_metrics'].items():
    print(dataset, 'n=', metrics['n_test'], 'AUC=', metrics['auc'], 'F1=', metrics['f1'])
print('Step 7 complete: trained weights and held-out predictions saved.')''')

stage("Step 8 — XAI, architecture, and figures", "Generate predicted-class Grad-CAM panels, segmentation examples, confusion/ROC figures, and a baseline architecture diagram. A heatmap shows the model's sensitivity; it does not prove clinical reasoning.",
      ["tools/render_gradcam.py", "tools/render_pooled_gradcam.py", "tools/render_rim_predictions.py", "tools/render_results.py", "tools/render_baseline_architecture.py"],
      r'''if RUN_VISUALS_AND_EXPORT:
    for name in ('render_gradcam.py', 'render_pooled_gradcam.py', 'render_rim_predictions.py', 'render_results.py', 'render_baseline_architecture.py'):
        run_script(name)
for path in ('outputs/architecture_baselines.png', 'outputs/acrima_resnet18_frozen/comparison.png'):
    display(Image.open(ROOT / path))
print('Step 8 complete: visual explanations and architecture figure saved.')''')

stage("Step 9 — Export measured results for the later website", "The export includes only numbers produced by this run. It preserves per-image predictions, model/dataset/split keys, comparison data, and paths to local visual assets. Never fill missing combinations with invented metrics.",
      ["tools/export_results.py", "tools/validate_website_export.py", "website/index.html", "website/styles.css", "website/app.js"],
      r'''if RUN_VISUALS_AND_EXPORT:
    run_script('export_results.py')
    run_script('validate_website_export.py')
    shutil.copy2(ROOT / 'outputs/website_results_partial.json', ROOT / 'results/website_results_partial.json')
payload = json.loads((ROOT / 'results/website_results_partial.json').read_text())
print('Measured combinations:', len(payload['experiments']))
print('Datasets represented:', sorted({row['dataset'] for row in payload['experiments']}))
print('Step 9 complete: website JSON written to results/website_results_partial.json.')''')

md("""## Step 10 — Make a report draft and downloadable package

This report text is generated from the metrics saved in **this run**. Review it with your team and make the final document 2–3 pages. The ZIP contains code outputs, metrics, plots, and trained checkpoints; it excludes raw dataset archives and DRISHTI mirror images. Keep raw images private under their source terms.""")
code(r'''# Generate a concise report draft from measured files, then package submission artifacts.
pooled = json.loads((ROOT / 'outputs/pooled_four_dataset_frozen/metrics.json').read_text())
seg = json.loads((ROOT / 'outputs/pooled_segmentation/metrics.json').read_text())
fine = json.loads((ROOT / 'outputs/pooled_finetuned_classifier/metrics.json').read_text())
official = json.loads((ROOT / 'outputs/pooled_four_dataset_frozen/refuge_official_test_metrics.json').read_text())
lines = ['# Glaucoma Image Classification, Disc/Cup Segmentation and Explainable AI', '',
         '## Introduction', 'We evaluated retinal fundus models across ACRIMA, RIM-ONE r3, ORIGA, REFUGE, and a DRISHTI-GS1 Kaggle mirror. The DRISHTI results are mirror-based and must be labeled as such.', '',
         '## Methodology', 'Fixed training, validation and test assignments were recorded before model fitting. ACRIMA has classification labels only. Segmentation predicts optic disc and cup; classifiers use full images or predicted disc crops. Test masks were not used to crop classification images. The source archives and masks were audited before training.', '',
         '## Architecture', 'A frozen or fine-tuned ResNet-18 predicts glaucoma; a two-channel U-Net segments disc and cup. The architecture diagram is saved at outputs/architecture_baselines.png. Grad-CAM targets the predicted class.', '',
         '## Results', '| Dataset | Pooled frozen AUC | Fine-tuned AUC |', '|---|---:|---:|']
for dataset, old in pooled['test_metrics'].items():
    new = fine['refuge_test_metrics_by_source_split']['val'] if dataset == 'REFUGE' else fine['test_metrics'].get(dataset, {})
    lines.append(f"| {dataset} | {old['auc']:.3f} | {new.get('auc', float('nan')):.3f} |")
lines += ['', f"Independent REFUGE official-test AUC for the pooled frozen model: {official['auc']:.3f} (n={official['n_test']}).", '',
          f"Independent REFUGE official-test AUC for the fine-tuned model: {fine['refuge_test_metrics_by_source_split']['test']['auc']:.3f} (n={fine['refuge_test_metrics_by_source_split']['test']['n_test']}).", '',
          '### Segmentation', '| Dataset | Disc Dice | Cup Dice |', '|---|---:|---:|']
for row in seg['test_metrics']:
    lines.append(f"| {row['dataset']} | {row['test_disc_dice']:.3f} | {row['test_cup_dice']:.3f} |")
lines += ['', '## Conclusion', 'Compare whole-image and predicted-disc-crop performance on the same held-out images. Discuss domain shift, class imbalance, the small RIM-ONE and DRISHTI samples, and third-party DRISHTI mirror provenance. These research models are not clinical devices.', '']
(ROOT / 'report/glaucoma_project_report_generated.md').write_text('\n'.join(lines), encoding='utf-8')
package = ROOT / 'glaucoma_submission_artifacts.zip'
with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for folder in ('outputs', 'results', 'report', 'website'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file(): z.write(path, path.relative_to(ROOT))
print('Report:', ROOT / 'report/glaucoma_project_report_generated.md')
print('Artifact ZIP:', package, '| MiB:', round(package.stat().st_size / 2**20, 1))
print('Everything above was generated from this notebook. Dataset archives remain outside the package.')''')

md("""### Optional: download the artifact ZIP from Colab

This is an **output download** after training, not a dataset upload. If the ZIP is large, copy it to your free Google Drive instead. The raw dataset archives are not included.""")
code(r'''# Uncomment only when you want the artifact ZIP on your computer.
# from google.colab import files
# files.download(str(ROOT / 'glaucoma_submission_artifacts.zip'))''')

nbformat.validate(nb)
nbformat.write(nb, TARGET)
print(f"Wrote {TARGET} with {len(nb.cells)} cells ({TARGET.stat().st_size:,} bytes)")
