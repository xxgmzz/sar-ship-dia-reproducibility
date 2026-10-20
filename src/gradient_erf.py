# -*- coding: utf-8 -*-
"""#30 最终训练模型梯度 ERF（HRSID seed42, layer2 注入层），baseline vs DIA。
- N=64 张有船验证图（均匀采样），letterbox 640；
- 每张图在 layer2 特征图上取 3x3 网格位置（中心区域），分别反传并平移对齐到图像中心，
  得到与内容位置无关、中心对齐的 ERF；
- 量化 cross_ratio（十字轴能量占比）、R90 紧凑度，做同图配对；
- 出图：baseline ERF / DIA ERF / 差值 / 中心轴剖面。"""
import sys, os, glob, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.data.augment import LetterBox

class StripConv(nn.Module):
    def __init__(self, c1, c2, k=5, stride=1, act=True):
        super().__init__()
        self.conv_h = nn.Conv2d(c1, c1, kernel_size=(1,k), stride=stride, padding=(0,k//2), groups=c1, bias=False)
        self.bn_h = nn.BatchNorm2d(c1)
        self.conv_v = nn.Conv2d(c1, c1, kernel_size=(k,1), stride=stride, padding=(k//2,0), groups=c1, bias=False)
        self.bn_v = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        x = self.act(self.bn_h(self.conv_h(x)))
        x = self.act(self.bn_v(self.conv_v(x)))
        return self.act(self.bn_pw(self.pw(x)))

class DualBranchBottleneck(nn.Module):
    def __init__(self, c, shortcut=True, k=5):
        super().__init__(); self.shortcut = shortcut; hc = c//2
        self.branch_origin = Conv(c, hc, 3, 1)
        self.branch_strip = StripConv(c, hc, k=k)
    def forward(self, x):
        out = torch.cat([self.branch_origin(x), self.branch_strip(x)], 1)
        return x + out if self.shortcut else out

class C2f_DualBranch(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super().__init__(); self.c = int(c2*e)
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n)*self.c, c2, 1)
        self.m = nn.ModuleList(DualBranchBottleneck(self.c, shortcut=shortcut, k=k) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).split((self.c,self.c),1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

for _n in ["StripConv","DualBranchBottleneck","C2f_DualBranch"]:
    setattr(sys.modules["__main__"], _n, getattr(sys.modules[__name__], _n))

DEVICE="cuda:0"; IMG=640; N_IMG=64; STRIDE=4
VAL_DIR=r"D:\SAR_Det_Research\hrsid_experiments\data\images\val"
LAB_DIR=r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val"
OUT_PNG=r"D:\SAR_Det_Research\draft_paper\Figure_gradient_erf.png"
OUT_CSV=r"D:\SAR_Det_Research\hrsid_experiments\gradient_erf_metrics.csv"
W={"baseline":r"D:\SAR_Det_Research\multiseed_experiments\HRSID_baseline_seed42\weights\best.pt",
   "DIA":r"D:\SAR_Det_Research\multiseed_experiments\HRSID_dia_l2_k3_seed42\weights\best.pt"}

def preprocess(path):
    img=cv2.imread(path)
    lb=LetterBox(IMG,auto=False,stride=32)(image=img)
    rgb=cv2.cvtColor(lb,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()[None]/255.0

def erf_aligned(m, x, grid):
    """grid: list of (fy,fx) feature positions; 返回平移对齐平均的 |grad| 640x640。"""
    acc=np.zeros((IMG,IMG),np.float32)
    for (fy,fx) in grid:
        xx=x.clone().to(DEVICE).requires_grad_(True)
        h=xx
        for i in range(3):
            h=m.model[i](h)
        target=h[:, :, fy, fx].sum()
        m.zero_grad(set_to_none=True)
        target.backward()
        g=xx.grad[0].abs().mean(0).detach().cpu().numpy()
        # 该神经元输入中心约 (fy*STRIDE+2, fx*STRIDE+2)（stride4 + 中心偏移）
        cy, cx = fy*STRIDE+2, fx*STRIDE+2
        dy, dx = IMG//2-cy, IMG//2-cx
        gs=np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        acc+=gs
    return acc/len(grid)

def smooth(g,sigma=2.5):
    k=int(sigma*6)|1
    return cv2.GaussianBlur(g,(k,k),sigma)

def metrics(g, R=36, axis_half=3):
    c=IMG//2
    z=g[c-R:c+R+1, c-R:c+R+1].copy(); z/=z.sum()
    n=2*R+1; yy,xx=np.mgrid[0:n,0:n]-R
    axis=(np.abs(yy)<=axis_half)|(np.abs(xx)<=axis_half)
    corner=(np.abs(yy)>axis_half)&(np.abs(xx)>axis_half)
    cr=z[axis].sum()/(z[axis].sum()+z[corner].sum())
    # 沿水平轴的半高宽（剖面，归一到峰值）
    prof=z[R, :].copy(); prof/=prof.max()
    above=np.where(prof>=0.5)[0]
    fwhm=(above[-1]-above[0]+1) if len(above) else 0
    return cr, fwhm

# 有船图，均匀采样
imgs=[]
for f in sorted(glob.glob(os.path.join(VAL_DIR,"*"))):
    if f.lower().endswith((".jpg",".png",".jpeg",".tif")):
        lab=os.path.join(LAB_DIR, os.path.splitext(os.path.basename(f))[0]+".txt")
        if os.path.exists(lab) and os.path.getsize(lab)>0:
            imgs.append(f)
if len(imgs)>N_IMG:
    idx=np.linspace(0,len(imgs)-1,N_IMG).astype(int)
    imgs=[imgs[i] for i in idx]
print("images:",len(imgs))

# layer2 特征 160x160；中心 3x3 网格（特征间距 10），对应输入间距 40
grid=[(fy,fx) for fy in (70,80,90) for fx in (70,80,90)]

per_img={"baseline":[], "DIA":[]}
agg={}
for name,p in W.items():
    m=YOLO(p).model.to(DEVICE).eval()
    acc=np.zeros((IMG,IMG),np.float32)
    for j,f in enumerate(imgs):
        x=preprocess(f)
        g=erf_aligned(m,x,grid)
        acc+=g
        per_img[name].append(metrics(smooth(g))[0])
    agg[name]=smooth(acc/len(imgs))
    np.save(rf"D:\SAR_Det_Research\hrsid_experiments\_erf_agg_{name}.npy", agg[name])
    print(name,"done")

cr_b=[metrics(smooth(np.load if False else agg["baseline"]))[0]]  # placeholder
mb,fb=metrics(agg["baseline"]); md,fd=metrics(agg["DIA"])
pb=np.array(per_img["baseline"]); pd=np.array(per_img["DIA"])
diff=pd-pb
# 配对 t（内置 betai）
def tpair(d):
    n=len(d); mean=d.mean(); sd=d.std(ddof=1); t=mean/(sd/np.sqrt(n)); df=n-1
    return mean,sd,t
mean,sd,t=tpair(diff)
print(f"cross_ratio baseline={mb:.4f} DIA={md:.4f} | per-image paired diff={mean:.4f}±{sd:.4f} t={t:.3f}")
print(f"FWHM baseline={fb} DIA={fd}")
with open(OUT_CSV,"w",newline="",encoding="utf-8-sig") as fp:
    w=csv.writer(fp); w.writerow(["image","cross_ratio_baseline","cross_ratio_dia"])
    for i,f in enumerate(imgs):
        w.writerow([os.path.basename(f),f"{pb[i]:.5f}",f"{pd[i]:.5f}"])
    w.writerow(["AGG_mean",f"{pb.mean():.5f}",f"{pd.mean():.5f}"])
    w.writerow(["paired_diff",f"mean={mean:.5f}",f"sd={sd:.5f} t={t:.3f} n={len(imgs)}"])

# ---- 出图 ----
R=36; c=IMG//2
def nrm(z):
    zz=z[c-R:c+R+1,c-R:c+R+1].copy(); return zz/zz.max()
zb,zd=nrm(agg["baseline"]),nrm(agg["DIA"])
fig,ax=plt.subplots(1,4,figsize=(14,4.2))
for a,z,t in [(ax[0],zb,"(a) Baseline (isotropic C2f)"),(ax[1],zd,"(b) DIA-Conv (isotropic + cross strips)")]:
    im=a.imshow(z,cmap="viridis",vmin=0,vmax=1,extent=[-R,R,-R,R],origin="lower")
    a.set_title(t,fontsize=10); a.set_xlabel("horizontal offset (px)"); a.set_ylabel("vertical offset (px)")
    fig.colorbar(im,ax=a,fraction=0.046)
im=ax[2].imshow(zd-zb,cmap="RdBu_r",vmin=-0.45,vmax=0.45,extent=[-R,R,-R,R],origin="lower")
ax[2].set_title("(c) DIA − Baseline",fontsize=10); ax[2].set_xlabel("horizontal offset (px)"); ax[2].set_ylabel("vertical offset (px)")
fig.colorbar(im,ax=ax[2],fraction=0.046)
xx=np.arange(-R,R+1)
for a in (zb,zd): pass
hb=agg["baseline"][c,c-R:c+R+1]; hb/=hb.sum(); hd=agg["DIA"][c,c-R:c+R+1]; hd/=hd.sum()
vb=agg["baseline"][c-R:c+R+1,c]; vb/=vb.sum(); vd=agg["DIA"][c-R:c+R+1,c]; vd/=vd.sum()
ax[3].plot(xx,hb,color="#1f77b4",ls="-",lw=1.8,label="Base horiz.")
ax[3].plot(xx,vb,color="#1f77b4",ls=":",lw=1.8,label="Base vert.")
ax[3].plot(xx,hd,color="#d62728",ls="-",lw=1.8,label="DIA horiz.")
ax[3].plot(xx,vd,color="#d62728",ls=":",lw=1.8,label="DIA vert.")
ax[3].set_xlim(-R,R); ax[3].set_xlabel("offset (px)"); ax[3].set_ylabel("normalized |gradient|")
ax[3].set_title("(d) Center-axis profiles",fontsize=10); ax[3].legend(fontsize=7.5,loc="upper right"); ax[3].grid(alpha=.3)
plt.tight_layout()
plt.savefig(OUT_PNG,dpi=180,bbox_inches="tight")
print("saved",OUT_PNG)
