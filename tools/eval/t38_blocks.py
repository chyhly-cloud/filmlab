"""T38 诊断: 夜景 RAW 在 load_raw 层面每块的观测量与抽到的 flux (t31_base / t37_base / t38_base, 同一随机数),
以及"相同的灯"的一致性: 招牌 city_61360 那对灯 (裁切图中的两盏) 与河边 city_40095 桥上灯排 (y≈586, x 130–720)。
用法: python3 t38_blocks.py 输出.json RAW…  或  python3 t38_blocks.py --report 输出.json      环境变量 GENS (默认 t31_base,t37_base,t38_base)"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import generation, rawio

gens = os.environ.get("GENS", "t31_base,t37_base,t38_base").split(",")


def report(res):
    """相同的灯: 招牌那对 (city_61360, 行 985–1030、列 380–520 的灯具块) 与桥上灯排 (city_40095, 行 575–600、列 130–720 的灯具块)。"""
    print("#### 招牌那对 (city_61360)")
    d = res["city_61360"]
    for a in zip(*(d[g] for g in gens)):
        c = a[-1]
        if 985 <= c["cy"] <= 1030 and 380 <= c["cx"] <= 520 and c["cls"] == "lamp":
            o = c.get("obs", {})
            print(f"  ({c['cy']:.0f}, {c['cx']:.0f}) r {c['r_um']:.0f} µm  bg {o.get('bg', float('nan')):+.2f}  skirt {o.get('skirt', float('nan')):+.2f}  "
                  + "  ".join(f"{g} flux {b['flux']:.2f}" for g, b in zip(gens, a)))
    print("#### 桥上灯排 (city_40095)")
    d = res["city_40095"]
    sel = [a for a in zip(*(d[g] for g in gens)) if 575 <= a[-1]["cy"] <= 600 and 130 <= a[-1]["cx"] <= 720 and a[-1]["cls"] == "lamp"]
    for i, g in enumerate(gens):
        f = np.array([a[i]["flux"] for a in sel])
        print(f"  {g}: 灯 {len(f)}, flux 中位 {np.median(f):.2f} [{np.percentile(f, 25):.2f}, {np.percentile(f, 75):.2f}], 标准差 {f.std():.2f}, "
              f"> 3 档的比例 {np.mean(f > 3):.2f}")
    bg = np.array([a[-1]["obs"]["bg"] for a in sel]) if "obs" in sel[0][-1] else None
    if bg is not None:
        print(f"  桥上灯 bg 中位 {np.median(bg):+.2f} [{np.percentile(bg, 25):+.2f}, {np.percentile(bg, 75):+.2f}]")


if len(sys.argv) == 3 and sys.argv[1] == "--report":
    report(json.load(open(sys.argv[2])))
    sys.exit(0)
res = {}
for f in sys.argv[2:]:
    s = Path(f).stem; res[s] = {}
    for g in gens:
        generation.use(g)
        rawio.STOPS_LOG = []
        rgb = rawio.load_raw(f, max_size=1600)
        um = 36000.0 / max(rgb.shape[:2])
        res[s][g] = [dict(b, r_um=b["r_eq"] * um) for b in rawio.STOPS_LOG]
        res[s]["shape"] = list(rgb.shape[:2])
        rawio.STOPS_LOG = None
json.dump(res, open(sys.argv[1], "w"))
if "city_61360" in res and "city_40095" in res:
    report(res)
