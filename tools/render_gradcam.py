"""Grad-CAM for the measured frozen ResNet-18 + logistic classifier."""
import csv
import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights, resnet18

torch.set_num_threads(4)
root=Path('outputs/acrima_resnet18_frozen'); out=root/'gradcam'; out.mkdir(exist_ok=True)
manifest={r['image_id']:r for r in csv.DictReader(Path('data/audit/acrima_manifest.csv').open(encoding='utf-8'))}
predictions=list(csv.DictReader((root/'test_predictions.csv').open(encoding='utf-8')))
groups={'true_positive':[], 'true_negative':[], 'false_positive':[], 'false_negative':[]}
for r in predictions:
    truth=int(r['true_label']); pred=int(r['predicted_label'])
    key=('true_' if truth==pred else 'false_')+('positive' if pred else 'negative')
    groups[key].append(r)
chosen=[r for key in groups for r in groups[key][:2]]
clf=joblib.load(root/'classifier.joblib')
model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc=torch.nn.Linear(512,1)
with torch.no_grad():
    model.fc.weight.copy_(torch.from_numpy(clf.coef_).float()); model.fc.bias.copy_(torch.from_numpy(clf.intercept_).float())
model.eval(); transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
acts=[]; grads=[]
def capture(_m,_i,o):
    acts.append(o)
    o.register_hook(lambda grad: grads.append(grad))
model.layer4[-1].register_forward_hook(capture)
for r in chosen:
    with zipfile.ZipFile('data/raw/ACRIMA_Database.zip') as z:
        with Image.open(io.BytesIO(z.read(manifest[r['image_id']]['member']))) as source:
            image=source.convert('RGB')
    x=transform(image).unsqueeze(0); acts.clear(); grads.clear(); model.zero_grad()
    logit=model(x)[0,0]
    target=logit if int(r['predicted_label'])==1 else -logit
    target.backward()
    a=acts[-1][0].detach(); g=grads[-1][0].detach()
    cam=torch.relu((g.mean((1,2))[:,None,None]*a).sum(0)); cam=cam-cam.min(); cam=cam/(cam.max()+1e-9)
    heat=Image.fromarray(np.uint8(cam.numpy()*255)).resize((224,224),Image.Resampling.BILINEAR)
    crop=ResNet18_Weights.IMAGENET1K_V1.transforms()(image) # display the same center-cropped view
    normalized=crop.numpy().transpose(1,2,0)
    rgb=np.uint8(np.clip((normalized*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255))
    view=Image.fromarray(rgb)
    overlay=Image.new('RGB',(224,224)); overlay.paste((255,0,0),mask=heat)
    merged=Image.blend(view,overlay,.4)
    panel=Image.new('RGB',(448,260),'white'); panel.paste(view,(0,0)); panel.paste(merged,(224,0))
    label=f"id {r['image_id']} | true {r['true_label']} | pred {r['predicted_label']} | p={float(r['probability_glaucoma']):.3f} | target={'glaucoma' if int(r['predicted_label']) else 'normal'}"
    ImageDraw.Draw(panel).text((8,232),label,fill='black')
    panel.save(out/f"{r['image_id']}.png")
print(f'Wrote {len(chosen)} Grad-CAM panels to {out}')
