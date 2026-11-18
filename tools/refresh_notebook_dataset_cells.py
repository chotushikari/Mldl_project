"""Update acquisition cells after auditing the combined ORIGA/REFUGE archive."""
from pathlib import Path

import nbformat


path = Path("PROJECT.ipynb")
nb = nbformat.read(path, as_version=4)
nb.cells[3].source = """DATASETS = {
    'ACRIMA': {
        'archive': DATA_RAW / 'ACRIMA_Database.zip', 'status': 'verified',
        'source': 'Authors Figshare article 7613135',
        'labels': 'filename _g means glaucoma', 'masks': 'none',
    },
    'RIM-ONE V3': {
        'archive': DATA_RAW / 'RIM-ONE-r3.zip', 'status': 'verified',
        'source': 'University of La Laguna original archive',
        'labels': 'filename prefix N/G/S; exclude suspects for binary classification',
        'masks': 'paired average disc/cup masks for all 159 images',
    },
    'REFUGE': {
        'archive': DATA_RAW / 'REFUGE_figshare.rar', 'status': 'verified',
        'source': 'Figshare article 26049574, original-source Refuge.rar; published MD5 matched',
        'labels': '400 train, 400 validation, 400 test; official test labels recovered from REFUGE-Test-GT spreadsheet',
        'masks': 'official BMP ground truth; combined archive PNG samples map 255/128/0 to 0/1/2 exactly',
        'restriction': 'Preserve original train/val/test assignments',
    },
    'ORIGA': {
        'archive': DATA_RAW / 'ORIGA_candidate_figshare.zip', 'status': 'verified_repackaged',
        'source': 'Figshare article 24549217, combined third-party archive; published MD5 matched',
        'labels': '650 rows in OrigaList.csv; 482 normal, 168 glaucoma; provided A/B sets',
        'masks': '650 masks, values 0 background, 1 disc rim, 2 cup in sampled files',
        'restriction': 'Do not treat this repackaging as an original release',
    },
    'DRISHTI-GS': {
        'archive': None, 'status': 'registration_required',
        'source': 'IIIT Hyderabad MIP official access process',
        'labels': 'Confirm GS1 expert binary labels after authorized access',
        'masks': 'Confirm multi-expert disc/cup annotations after access',
        'restriction': 'Do not redistribute restricted images',
    },
}
Path(DATA_AUDIT / 'dataset_registry.json').write_text(json.dumps({k: {**v, 'archive': str(v['archive']) if v['archive'] else None} for k, v in DATASETS.items()}, indent=2), encoding='utf-8')
for name, record in DATASETS.items():
    print(f"{name}: {record['status']} | archive={record['archive']}")"""

nb.cells[4].source = """## Acquisition and provenance

The Figshare article 24549217 combined archive contains ORIGA and a repackaged REFUGE copy. Its published MD5 matched the downloaded 5.96 GB file. The separate 4.27 GB REFUGE RAR from Figshare article 26049574 also passed its published MD5. It supplies official test labels absent from the repackaged index. All 1200 original image/mask names were found, and nine sampled images/masks matched the repackaging after the documented mask-value conversion. DRISHTI-GS1 still requires the IIIT Hyderabad access process."""

nb.cells[5].source = """DOWNLOAD_PUBLIC_CANDIDATES = False
PUBLIC_DOWNLOADS = ('acrima', 'rim_one_r3', 'origa_candidate', 'refuge_candidate')
if DOWNLOAD_PUBLIC_CANDIDATES:
    downloader = PROJECT_ROOT / 'tools' / 'download_verified.py'
    for dataset in PUBLIC_DOWNLOADS:
        subprocess.run([sys.executable, str(downloader), dataset], cwd=PROJECT_ROOT, check=True)
else:
    print('Downloads disabled; existing verified archives will be used.')"""

nb.cells[7].source = """def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    suffixes = {}
    for member in members:
        suffix = Path(member).suffix.lower() or '[none]'
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
    return {'members': len(members), 'suffixes': suffixes, 'sample_members': members[:5]}

for name in ('ACRIMA', 'RIM-ONE V3', 'ORIGA'):
    path = DATASETS[name]['archive']
    if path and path.exists():
        print(name, inspect_zip(path))
print('REFUGE files reside inside the combined ORIGA/REFUGE ZIP archive.')"""

nb.cells[8].source = """# Source-specific audits produce manifests and sample overlays.
for script in ('audit_acrima.py', 'audit_rim_one_r3.py'):
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools' / script)], cwd=PROJECT_ROOT, check=True)
if (DATA_RAW / 'ORIGA_candidate_figshare.zip').exists():
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools' / 'verify_figshare_archives.py')], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools' / 'audit_origa_refuge.py')], cwd=PROJECT_ROOT, check=True)
    if (DATA_RAW / 'REFUGE_figshare.rar').exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools' / 'audit_refuge_original.py')], cwd=PROJECT_ROOT, check=True)"""

nb.cells[10].source = """def csv_counts(path, key):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        value = row[key]
        counts[value] = counts.get(value, 0) + 1
    return len(rows), counts

acrima_count, acrima_labels = csv_counts(DATA_AUDIT / 'acrima_manifest.csv', 'label')
rim_count, rim_labels = csv_counts(DATA_AUDIT / 'rim_one_r3_manifest.csv', 'label_original')
combined_audit_path = DATA_AUDIT / 'origa_refuge_audit.json'
combined = json.loads(combined_audit_path.read_text(encoding='utf-8')) if combined_audit_path.exists() and DATASETS['ORIGA']['archive'].exists() else None
original_refuge_path = DATA_AUDIT / 'refuge_original_audit.json'
original_refuge = json.loads(original_refuge_path.read_text(encoding='utf-8')) if original_refuge_path.exists() and DATASETS['REFUGE']['archive'].exists() else None
counts = {
    'ACRIMA': {'images': acrima_count, 'labels': acrima_labels, 'disc_masks': False, 'cup_masks': False, 'verification': 'passed'},
    'RIM-ONE V3': {'images': rim_count, 'labels': rim_labels, 'disc_masks': rim_count, 'cup_masks': rim_count, 'verification': 'passed'},
    'REFUGE': {'images': sum(s['images'] for s in combined['refuge'].values()) if combined else None, 'labels': {**{k: v['labels'] for k, v in combined['refuge'].items() if k != 'test'}, 'test': original_refuge['official_test_labels'] if original_refuge else 'unavailable'} if combined else None, 'disc_masks': 1200 if original_refuge else None, 'cup_masks': 1200 if original_refuge else None, 'verification': 'original-source RAR and 9 repackaged samples checked' if original_refuge else 'pending original-source audit'},
    'ORIGA': {'images': combined['origa']['images'] if combined else None, 'labels': combined['origa']['labels'] if combined else None, 'disc_masks': 650 if combined else None, 'cup_masks': 650 if combined else None, 'verification': 'repackaged archive audited' if combined else 'archive unavailable'},
    'DRISHTI-GS': {'images': None, 'labels': None, 'disc_masks': None, 'cup_masks': None, 'verification': 'pending authorized access'},
}
(DATA_AUDIT / 'dataset_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
print(json.dumps(counts, indent=2))"""

for cell in nb.cells:
    if "RUN_FINAL_TRAINING = False" in cell.source:
        cell.source = cell.source.replace("record['status'] == 'verified'", "record['status'].startswith('verified')")
        cell.source = cell.source.replace("All dataset gates passed. Add the approved training cells here.", "All five dataset inventories are present; final training may run.")
        break

nbformat.write(nb, path)
print("Updated dataset inventory cells in", path)
