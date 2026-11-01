import sys, warnings, numpy as np
warnings.filterwarnings("ignore"); sys.path[:0]=["/home/user/filmlab"]
from filmlab import generation, materials
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.io import srgb_encode
from PIL import Image
generation.use("t22_bxyz")
W,Hh=1400,500; px=36000/W
STOPS=[6,9,12,15]; bg=0.184/2**4
col=np.array([1.0,0.80,0.55])      # 钨丝/暖光灯的线性 RGB (日光白平衡下)
img=np.full((Hh,W,3),bg)*col
yy,xx=np.mgrid[:Hh,:W]; cents=[]
for i,s in enumerate(STOPS):
    cx,cy=175+i*350,250; m=np.hypot(yy-cy,xx-cx)<=3.0
    img[m]=0.184*2**s*col; cents.append((s,cx,cy))
r=recipe("cinestill_800t")
res={}
for name,film in (("no_remjet",r["film"]),("remjet",materials.emulsion("kodak_vision3_500t"))):
    eng=Engine(Params(**{**r,"film":film,"timing":"frame","roll":None,"frame":0,"grain":0}))
    tn={}; neg=eng.develop_negative(img,trace=tn); out=np.clip(eng.scan(neg),0,None)
    res[name]=(tn["total_exposure"],out)
    Image.fromarray((np.clip(srgb_encode(np.clip(out,0,1)),0,1)*255+.5).astype(np.uint8)).save(f"{sys.argv[1]}_{name}.png")
Hn,On=res["no_remjet"]; Hr,Or=res["remjet"]
mid=np.array([1,1,1.0])  # 以中灰曝光为 1 的单位? 打印 trace 中灰水平
print("px µm",round(px,1),"背景曝光",np.round(np.median(Hn[50:60,50:60],(0,1)),4))
rr=np.hypot(yy,xx)
for s,cx,cy in cents:
    d=np.hypot(yy-cy,xx-cx)*px
    print(f"\n+{s} 档灯 (核心半径 77µm)")
    print("  r(µm) | 光晕曝光 R G B (除碳−带碳, 背景曝光为单位) | 成片 sRGB 除碳 | 成片 sRGB 带碳")
    bgH=np.median(Hn[50:60,50:60],(0,1))
    for R in (100,150,200,250,300,400,500,700,1000,1500,2000):
        m=(d>=R-px/2)&(d<R+px/2)
        dh=(Hn[m]-Hr[m]).mean(0)/bgH
        sn=srgb_encode(np.clip(On[m].mean(0),0,1)); sr=srgb_encode(np.clip(Or[m].mean(0),0,1))
        print(f"  {R:5d} | {np.round(dh,2)} | {np.round(sn*255).astype(int)} | {np.round(sr*255).astype(int)}")
