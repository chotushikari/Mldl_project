"""Train and evaluate a compact disc/cup U-Net on original RIM-ONE r3."""
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)
SIZE = 256
rows = list(csv.DictReader(Path('data/audit/rim_one_r3_manifest.csv').open(encoding='utf-8')))
labels = [r['label_original'] for r in rows]
idx = np.arange(len(rows))
trainval, test = train_test_split(idx, test_size=.2, stratify=labels, random_state=SEED)
train, val = train_test_split(trainval, test_size=.1875, stratify=np.array(labels)[trainval], random_state=SEED)
out = Path('outputs/rim_unet_baseline'); out.mkdir(parents=True, exist_ok=True)
with (out/'split.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['image','split']); w.writerows((rows[i]['image'],s) for s,a in [('train',train),('val',val),('test',test)] for i in a)

class RIM(Dataset):
    def __init__(self, indices, augment=False):
        self.indices=list(indices); self.augment=augment
        with zipfile.ZipFile('data/raw/RIM-ONE-r3.zip') as z:
            self.data=[]
            for i in self.indices:
                r=rows[i]
                im=Image.open(io.BytesIO(z.read(r['image']))).convert('RGB')
                im=im.crop((0,0,im.width//2,im.height)).resize((SIZE,SIZE))
                masks=[]
                for key in ('disc_mask','cup_mask'):
                    mask=Image.open(io.BytesIO(z.read(r[key]))).convert('L')
                    mask=mask.crop((0,0,mask.width//2,mask.height)).resize((SIZE,SIZE),Image.Resampling.NEAREST)
                    masks.append(np.asarray(mask)>127)
                self.data.append((np.asarray(im).copy(),np.stack(masks).astype('float32')))
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        im,mask=self.data[i]; im=im.copy(); mask=mask.copy()
        if self.augment and random.random()<.5: im=im[:,::-1].copy(); mask=mask[:,:,::-1].copy()
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/255,torch.from_numpy(mask)

def scores(model,ds):
    model.eval(); totals=np.zeros((2,2)); count=0
    with torch.no_grad():
        for x,y in DataLoader(ds,batch_size=4):
            p=(model(x).sigmoid()>.5).numpy(); g=y.numpy()>0
            for k in range(2):
                a=p[:,k]; b=g[:,k]; inter=(a&b).sum(axis=(1,2)); ps=a.sum(axis=(1,2)); gs=b.sum(axis=(1,2)); union=(a|b).sum(axis=(1,2))
                totals[k,0]+=((2*inter+1e-6)/(ps+gs+1e-6)).sum(); totals[k,1]+=((inter+1e-6)/(union+1e-6)).sum()
            count+=len(x)
    return totals/count

train_ds=RIM(train,True); val_ds=RIM(val); test_ds=RIM(test)
model=UNet(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); bce=nn.BCEWithLogitsLoss()
best=-1; history=[]
for epoch in range(12):
    model.train(); losses=[]
    for x,y in DataLoader(train_ds,batch_size=4,shuffle=True):
        logits=model(x); prob=logits.sigmoid(); dice=1-((2*(prob*y).sum((0,2,3))+1)/((prob+y).sum((0,2,3))+1)).mean(); loss=bce(logits,y)+dice
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    v=scores(model,val_ds); rank=v[:,0].mean(); history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_disc_dice':float(v[0,0]),'val_cup_dice':float(v[1,0])})
    if rank>best: best=rank; torch.save(model.state_dict(),out/'best.pt')
    print(history[-1],flush=True)
model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); result=scores(model,test_ds)
metrics={'dataset':'RIM-ONE r3','split_seed':SEED,'n_train':len(train),'n_val':len(val),'n_test':len(test),'test_disc_dice':float(result[0,0]),'test_disc_iou':float(result[0,1]),'test_cup_dice':float(result[1,0]),'test_cup_iou':float(result[1,1])}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)); (out/'history.json').write_text(json.dumps(history,indent=2)); print(json.dumps(metrics,indent=2),flush=True)
