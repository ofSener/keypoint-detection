#!/usr/bin/env python3
"""
train.py – uçtan-uca keypoint heatmap + descriptor eğitimi
----------------------------------------------------------------
Örnek:
  python train.py train \
    --data ~/projects/keypoint/data/processed \
    --bs 8 --ep 1 --lr 1e-4
----------------------------------------------------------------
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Fix MPS grid_sampler issue

import argparse, pathlib, random, math, warnings
from typing import List, Dict, Any

import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
from multiprocessing import cpu_count


# ---------------------------------------------------------------- utils
def device_auto() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")  # Re-enabled for speed
    return torch.device("cpu")

def seed_all(s: int = 42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- dataset
class KeypointDataset(Dataset):
    """PNG + .npz (xy,desc).  Eksik .npz varsa SIFT anında üretir."""
    def __init__(self, root, split="train", val_ratio=0.1, max_kp=500, img_size=(480, 640), max_samples=None):
        self.root = pathlib.Path(root).expanduser()
        self.lbl  = pathlib.Path(str(self.root).replace('/processed', '/kp_desc'))  # Fixed path
        self.max_kp = max_kp
        self.img_size = img_size  # (height, width)
        
        # Image transformations - resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        imgs = sorted(self.root.glob("**/*.png"))
        if not imgs: raise RuntimeError(f"no PNG in {self.root}")
        
        # Limit dataset size for faster training
        if max_samples:
            imgs = imgs[:max_samples]
            print(f"[Dataset] Limited to {max_samples} images")
        
        seed_all(0); random.shuffle(imgs)
        cut = int(len(imgs)*(1-val_ratio))
        self.files = imgs[:cut] if split=="train" else imgs[cut:]

        miss = sum(1 for p in self.files
                   if not (self.lbl/p.relative_to(self.root)).with_suffix(".npz").exists())
        if miss:
            print(f"[Dataset] {miss}/{len(self.files)} images lack .npz – SIFT on-the-fly")

    # helpers
    @staticmethod
    def _heat(xy, shape):
        h = np.zeros(shape, np.float32)
        if len(xy): h[xy[:,1], xy[:,0]] = 1
        return cv2.GaussianBlur(h,(0,0),2)

    def _load_npz(self, path):
        d=np.load(path); return d["xy"].astype(np.int32), d["d"].astype(np.float32)/512.

    def _sift(self, gray):
        sift=cv2.SIFT_create(nfeatures=self.max_kp)
        kps,des=sift.detectAndCompute(gray,None)
        if des is None: return np.empty((0,2),np.int32), np.empty((0,128),np.float32)
        kps,des=kps[:self.max_kp],des[:self.max_kp]
        xy=np.round([[kp.pt[0],kp.pt[1]] for kp in kps]).astype(np.int32)
        return xy, des/512.
        
    def _scale_keypoints(self, xy, orig_shape, new_shape):
        """Scale keypoint coordinates from original image size to new size"""
        if len(xy) == 0:
            return xy
        scale_x = new_shape[1] / orig_shape[1]  # width scaling
        scale_y = new_shape[0] / orig_shape[0]  # height scaling
        xy_scaled = xy.copy().astype(np.float32)
        xy_scaled[:, 0] *= scale_x
        xy_scaled[:, 1] *= scale_y
        return np.round(xy_scaled).astype(np.int32)

    # main
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p=self.files[idx]
        npz=(self.lbl/p.relative_to(self.root)).with_suffix(".npz")

        img_bgr=cv2.imread(str(p));  # BGR uint8
        if img_bgr is None: raise FileNotFoundError(p)
        orig_shape = img_bgr.shape[:2]  # (height, width)
        gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

        xy,des = self._load_npz(npz) if npz.exists() else self._sift(gray)
        # if not npz.exists(): warnings.warn(f"SIFT on-the-fly {p}")  # Too much spam, commented out

        # Scale keypoints to match the resized image
        xy_scaled = self._scale_keypoints(xy, orig_shape, self.img_size)
        
        # Create heatmap at target size
        heat=self._heat(xy_scaled, self.img_size)
        
        # Apply transformations to image
        img_tensor = self.transform(img_rgb)
        
        return {
            "image"  : img_tensor,
            "heat_gt": torch.from_numpy(heat).unsqueeze(0),
            "kps"    : xy_scaled.copy(),                      # np.ndarray (N,2)
            "descs"  : des.astype(np.float32).copy()   # (N,128)
        }


# -------------------------------------------------------------- model
class KPNet(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        b=resnet18(weights="IMAGENET1K_V1")
        self.enc=nn.Sequential(*list(b.children())[:-2])
        self.kp_head=nn.Sequential(
            nn.Conv2d(512,256,3,padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=32,mode="bilinear",align_corners=False),
            nn.Conv2d(256,1,1))
        self.desc_head=nn.Conv2d(512,dim,1)

    def forward(self,x):
        f=self.enc(x)
        return self.kp_head(f), F.normalize(self.desc_head(f),p=2,dim=1)


# -------------------------------------------------------------- losses
def heat_loss(p,g): return F.binary_cross_entropy_with_logits(p,g)

def sample_desc(m: torch.Tensor, xy: np.ndarray):
    if xy.size==0: return torch.empty(0,m.size(0),device=m.device)
    xs=torch.tensor(xy[:,0],device=m.device)
    ys=torch.tensor(xy[:,1],device=m.device)
    H,W=m.shape[1:]
    grid=torch.stack(((xs/(W-1))*2-1,(ys/(H-1))*2-1),1).view(1,1,-1,2)
    return F.grid_sample(m.unsqueeze(0),grid,align_corners=True).squeeze().t()

def desc_loss(pred,gt):
    if pred.numel()==0: return torch.tensor(0.,device=pred.device)
    return F.mse_loss(pred,gt)


# -------------------------------------------------------------- collate fn
def collate(batch: List[Dict[str,Any]]):
    imgs   = torch.stack([b["image"] for b in batch])
    heats  = torch.stack([b["heat_gt"] for b in batch])
    kps    = [b["kps"]   for b in batch]   # list of np arrays (N_i,2)
    descs  = [torch.from_numpy(b["descs"]).float() for b in batch]   # list of tensors (N_i,128)
    return {"image":imgs, "heat_gt":heats, "kps":kps, "descs":descs}


# -------------------------------------------------------------- epoch
def run_epoch(net,loader,opt,dev,train=True):
    net.train() if train else net.eval(); tot=0.
    for b in tqdm(loader,desc="train" if train else "val",leave=False):
        img=b["image"].to(dev); h_gt=b["heat_gt"].to(dev)
        kps,des=b["kps"],b["descs"]
        if train: opt.zero_grad()
        h_log,d_map=net(img); lh=heat_loss(h_log,h_gt); ld=torch.zeros(1,device=dev)
        for i in range(img.size(0)):
            ld+=desc_loss(sample_desc(d_map[i],kps[i]),
                          des[i].to(dev))
        loss=lh+0.1*ld
        if train: loss.backward(); opt.step()
        tot+=loss.item()*img.size(0)
    return tot/len(loader.dataset)


# -------------------------------------------------------------- cli train
def train_cli(a):
    seed_all(); dev=device_auto()
    print(f"Using device: {dev}")
    
    tr=KeypointDataset(a.data,"train", max_samples=a.max_samples)
    vl=KeypointDataset(a.data,"val", max_samples=a.max_samples)
    
    # Use maximum CPU cores for data loading
    num_workers = min(cpu_count(), 8)  # Use all available cores
    
    ltr=DataLoader(tr,a.bs,shuffle=True,collate_fn=collate,num_workers=num_workers,pin_memory=True)
    lvl=DataLoader(vl,a.bs,shuffle=False,collate_fn=collate,num_workers=num_workers,pin_memory=True)
    
    net=KPNet(a.dim).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr, weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.ep)
    best=math.inf; ck=pathlib.Path(a.ckpt).expanduser(); ck.mkdir(exist_ok=True)

    print(f"Training set: {len(tr)} images, {len(ltr)} batches")
    print(f"Validation set: {len(vl)} images, {len(lvl)} batches")
    print(f"Batch size: {a.bs}, Workers: {num_workers}")

    for e in range(1,a.ep+1):
        print(f"\nEpoch {e}/{a.ep}")
        tl=run_epoch(net,ltr,opt,dev,True)
        vl_loss=run_epoch(net,lvl,opt,dev,False)
        sch.step(); print(f"train={tl:.4f} val={vl_loss:.4f}")
        if vl_loss<best:
            best=vl_loss; torch.save({"model":net.state_dict(),"ep":e}, ck/"best.pt")
            print("✅ saved best")

# -------------------------------------------------------------- main
def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train")
    t.add_argument("--data",default="~/projects/keypoint/data/processed")
    t.add_argument("--bs",type=int,default=32)
    t.add_argument("--ep",type=int,default=10)
    t.add_argument("--lr",type=float,default=1e-4)
    t.add_argument("--dim",type=int,default=128)
    t.add_argument("--ckpt",default="checkpoints")
    t.add_argument("--max_samples",type=int,default=None)
    args=p.parse_args(); args.data=pathlib.Path(args.data).expanduser()
    if args.cmd=="train": train_cli(args)

if __name__=="__main__":
    main()
