"""按场景类别比较模拟与真实 800T 扫描的统计量 (同一套指标)。"""
import sys, glob, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
from PIL import Image
SP="/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
sys.path.insert(0, SP); from ref_labels import OURS
from ref3_labels import REF3 as REF
refs=open(SP+"/reflist.txt").read().split("\n")
def metrics(path):
    im8=np.asarray(Image.open(path).convert("RGB").resize((480,320))).astype(float)
    lin=colour.cctf_decoding(im8/255.,"sRGB")
    lab=colour.XYZ_to_Lab(colour.sRGB_to_XYZ(lin,apply_cctf_decoding=False))
    L,a,b=lab[...,0],lab[...,1],lab[...,2]; v=im8.mean(-1)
    def ab(m): return (a[m].mean(),b[m].mean()) if m.sum()>200 else (np.nan,np.nan)
    dk=L<=np.percentile(L,5); md=(L>30)&(L<60); hi=(L>75)&(im8.max(-1)<250)
    C=np.hypot(a,b)
    return [np.percentile(v,1),np.percentile(v,50),*ab(dk),*ab(md),*ab(hi),np.nanmean(np.where(md,C,np.nan)),np.percentile(L,95)-np.percentile(L,5)]
NAMES=["p1","p50","暗a","暗b","中a","中b","亮a","亮b","中彩度","L范围"]
def show(tag,rows):
    r=np.array(rows); med=np.nanmedian(r,0); q1=np.nanpercentile(r,25,0); q3=np.nanpercentile(r,75,0)
    print(f"  {tag:5s} n={len(r):2d} "+"  ".join(f"{n} {m:5.1f}[{a:4.0f},{c:4.0f}]" for n,m,a,c in zip(NAMES,med,q1,q3)))
outdir=sys.argv[1]
ours=glob.glob(SP+f"/{outdir}/*.jpg")
for cat in REF:
    print(cat)
    show("真实",[metrics(p) for p in REF[cat]])
    sel=[f for f in ours if any(f.split("/")[-1].startswith(p) for p in OURS[cat])]
    show("模拟",[metrics(f) for f in sel])

# 综合误差: 每类每项 |模拟中位 − 真实中位| / 真实四分位距, 求和 (越小越接近)
def score(outdir):
    tot=0; parts={}
    for cat in REF:
        r=np.array([metrics(p) for p in REF[cat]]); o=np.array([metrics(f) for f in glob.glob(SP+f"/{outdir}/*.jpg") if any(f.split("/")[-1].startswith(p) for p in OURS[cat])])
        iqr=np.maximum(np.nanpercentile(r,75,0)-np.nanpercentile(r,25,0),1.0)
        e=np.abs(np.nanmedian(o,0)-np.nanmedian(r,0))/iqr
        parts[cat]=dict(zip(NAMES,e.round(2))); tot+=e.sum()
    return tot, parts
t,p=score(outdir)
print("综合误差", round(t,2))
for c,v in p.items(): print(" ",c, " ".join(f"{k}:{x}" for k,x in v.items() if x>0.5))
