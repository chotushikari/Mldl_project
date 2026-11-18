"""Render measured classification results for the later report and website."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

base=Path('outputs/acrima_resnet18_frozen')
fig, axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
for col, (name, prediction_file, metrics_file) in enumerate((
    ('ACRIMA held-out','test_predictions.csv','metrics.json'),
    ('RIM-ONE external','rim_external_predictions.csv','rim_external_metrics.json'))):
    rows=list(csv.DictReader((base/prediction_file).open(encoding='utf-8')))
    y=np.array([int(r['true_label']) for r in rows]); p=np.array([float(r['probability_glaucoma']) for r in rows]); pred=np.array([int(r['predicted_label']) for r in rows])
    m=json.loads((base/metrics_file).read_text())
    cm=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]])
    ConfusionMatrixDisplay(cm,display_labels=['Normal','Glaucoma']).plot(ax=axes[0,col],colorbar=False,cmap='Blues')
    axes[0,col].set_title(f'{name}: confusion matrix (n={len(y)})')
    RocCurveDisplay.from_predictions(y,p,ax=axes[1,col],name=name)
    axes[1,col].plot([0,1],[0,1],'--',color='gray',linewidth=1)
    axes[1,col].set_title(f'{name}: ROC curve')
fig.suptitle('Frozen ResNet-18 feature baseline: internal vs external evaluation',fontsize=14)
fig.savefig(base/'comparison.png',dpi=160)
print(base/'comparison.png')
