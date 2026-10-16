"""综合误差 (调参集 A / 验证集 B / 全部), 模拟按 ours_labels 的明确分类。用法: score2.py 输出目录"""
import sys, glob, os, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
from PIL import Image
SP="/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
sys.path.insert(0,SP)
from ours_labels import CAT
from ref3_split import REFA, REFB
from ref3_labels import REF3
src=open(SP+"/compare_cat3.py").read().split("outdir=sys.argv[1]")[0]; g={}; exec(src,g); metrics,NAMES=g["metrics"],g["NAMES"]
_cache={}
def M(p):
    if p not in _cache: _cache[p]=metrics(p)
    return _cache[p]
def silhouette(p):
    im=np.asarray(Image.open(p).convert("RGB").resize((240,160)))/255.
    return (colour.XYZ_to_Lab(colour.sRGB_to_XYZ(im))[...,0]<5).mean()>0.30
def score(sim, ref):
    t=0; per={}
    for c in ref:
        R=np.array([M(p) for p in ref[c] if not silhouette(p)]); S=np.array([M(p) for p in sim.get(c,[]) if not silhouette(p)])
        if len(S)==0: continue
        iqr=np.maximum(np.nanpercentile(R,75,0)-np.nanpercentile(R,25,0),1.0)
        e=np.abs(np.nanmedian(S,0)-np.nanmedian(R,0))/iqr; per[c]=e; t+=e.sum()
    return t, per
if __name__=="__main__":
    d=sys.argv[1]
    sim={}
    for f in glob.glob(SP+f"/{d}/*.jpg"):
        c=CAT.get(os.path.basename(f)[:-4]); 
        if c: sim.setdefault(c,[]).append(f)
    for nm,ref in (("A(调参)",REFA),("B(验证)",REFB),("全部",REF3)):
        t,per=score(sim,ref)
        print(f"{nm:8s} {t:6.2f}  "+"  ".join(f"{c}:{v.sum():.2f}" for c,v in per.items()))
    t,per=score(sim,REF3)
    for c,v in per.items(): print("   ",c," ".join(f"{n}:{x:.2f}" for n,x in zip(NAMES,v) if x>0.5))
