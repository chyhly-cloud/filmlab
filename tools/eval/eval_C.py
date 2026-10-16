"""按 tools/eval/CRITERIA.md 在测试集 C 上判定。用法: eval_C.py 模拟输出目录"""
import sys, glob, os, csv, re, numpy as np, warnings; warnings.filterwarnings("ignore")
SP="/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
g={}; exec(open(SP+"/score2.py").read().split('if __name__')[0],g)
M,sil,REF3,CAT,NAMES=g["M"],g["silhouette"],g["REF3"],g["CAT"],g["NAMES"]
# 只排除作者明确声明调色的帧 (采集时已排除; 注释里 "no editing mentioned" / "unedited" 之类不算)
bad=re.compile(r"\bedited in\b|colou?r[- ]corrected in|lightroom|\bpreset|negative lab|\bNLP\b",re.I)
C={}
for r in csv.DictReader(open(SP+"/refC/manifest.tsv"),delimiter="\t"):
    if bad.search(r.get("notes","")): continue
    cat={"daylight":"day"}.get(r["category"],r["category"])
    p=SP+"/refC/img/"+r["filename"]
    if os.path.exists(p): C.setdefault(cat,[]).append(p)
sim={}
for f in glob.glob(SP+f"/{sys.argv[1]}/*.jpg"):
    c=CAT.get(os.path.basename(f)[:-4])
    if c: sim.setdefault(c,[]).append(f)
arr=lambda L:np.array([M(p) for p in L if not sil(p)])
tot={"模拟":0,"ref3":0}; out={"模拟":0,"ref3":0}
print("测试集 C:", {c:len(v) for c,v in C.items()})
for c in ("night","dusk","indoor","day"):
    Cm=arr(C[c]); q1,q3=np.nanpercentile(Cm,25,0),np.nanpercentile(Cm,75,0); iqr=np.maximum(q3-q1,1.0); med=np.nanmedian(Cm,0)
    for nm,L in (("模拟",sim[c]),("ref3",REF3[c])):
        X=np.nanmedian(arr(L),0); e=np.abs(X-med)/iqr; tot[nm]+=e.sum()
        o=[n for n,x,a,b in zip(NAMES,X,q1,q3) if not (a<=x<=b)]; out[nm]+=len(o)
        print(f"  {c:6s} {nm:4s} 误差 {e.sum():5.2f}  四分位外: {o}")
print("综合误差: 模拟 %.2f  ref3 %.2f   四分位外项数: 模拟 %d  ref3 %d" % (tot["模拟"],tot["ref3"],out["模拟"],out["ref3"]))
print("判定1 (整体):", "通过" if tot["模拟"]<=tot["ref3"] else "未通过", "  判定2 (逐项):", "通过" if out["模拟"]<=out["ref3"] else "未通过")
