"""Render held-out RIM-ONE images with true and predicted disc/cup masks."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet

torch.set_num_threads(4)
root=Path('outputs/rim_unet_baseline'); out=root/'examples'; out.mkdir(exist_ok=True)
scores=list(csv.DictReader((root/'test_predictions.csv').open(encoding='utf-8')))
scores.sort(key=lambda r:(float(r['disc_dice'])+float(r['cup_dice']))/2)
selected=[('worst',r) for r in scores[:4]]+[('best',r) for r in scores[-4:]]
manifest={r['image']:r for r in csv.DictReader(Path('data/audit/rim_one_r3_manifest.csv').open(encoding='utf-8'))}
model=UNet(); model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=True)); model.eval()
def overlay(image,disc,cup):
    base=np.asarray(image).copy(); base[disc]=.65*base[disc]+.35*np.array([0,220,100]); base[cup]=.55*base[cup]+.45*np.array([255,180,0])
    return Image.fromarray(base.astype('uint8'))
with zipfile.ZipFile('data/raw/RIM-ONE-r3.zip') as z, torch.no_grad():
    for quality, score in selected:
        name=score['image']
        row=manifest[name]
        image=Image.open(io.BytesIO(z.read(name))).convert('RGB'); image=image.crop((0,0,image.width//2,image.height)).resize((256,256))
        masks=[]
        for key in ('disc_mask','cup_mask'):
            mask=Image.open(io.BytesIO(z.read(row[key]))).convert('L'); mask=mask.crop((0,0,mask.width//2,mask.height)).resize((256,256),Image.Resampling.NEAREST)
            masks.append(np.asarray(mask)>127)
        x=torch.from_numpy(np.asarray(image).copy().transpose(2,0,1)).float().unsqueeze(0)/255
        pred=(model(x).sigmoid()[0].numpy()>.5)
        panel=Image.new('RGB',(768,280),'white'); panel.paste(image,(0,0)); panel.paste(overlay(image,*masks),(256,0)); panel.paste(overlay(image,*pred),(512,0))
        ImageDraw.Draw(panel).text((8,260),'Original',fill='black'); ImageDraw.Draw(panel).text((264,260),'Ground truth',fill='black'); ImageDraw.Draw(panel).text((520,260),f"Prediction D={float(score['disc_dice']):.2f} C={float(score['cup_dice']):.2f}",fill='black')
        panel.save(out/(quality+'_'+Path(name).stem+'.png'))
(out/'selection.json').write_text(json.dumps([{'quality':quality,**score} for quality,score in selected],indent=2),encoding='utf-8')
print(f'Wrote {len(selected)} best/worst held-out prediction panels to {out}')
