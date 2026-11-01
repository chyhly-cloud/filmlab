"""T19: 在网上公开的 CineStill 800T 扫描 (参考集, 见 docs/tasks/T19_REFERENCE_HALATION.md) 上测光晕统计量。
用法: python3 t19_reference.py 参考目录(含 curated.tsv) 输出.json"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t19_halo_stats as hs

ref, dst = Path(sys.argv[1]), sys.argv[2]
per_img, allp = {}, {s: [] for s in hs.SIZES}
for line in open(ref / "curated.tsv"):
    f, url, page, fmt = line.rstrip("\n").split("\t")
    im = np.asarray(Image.open(ref / f).convert("RGB")) / 255.0
    um = hs.px_um(im.shape, fmt)
    if um is None:
        continue
    pr = hs.profiles(hs.srgb_to_lin(im), um)
    per_img[f] = {"url": url, "um_px": um, "n": {f"{a}-{b}": len(v) for (a, b), v in pr.items()}}
    for s, v in pr.items():
        if v:                                         # 每张图每组取中位, 避免灯多的图主导
            allp[s].append(np.nanmedian(np.array(v), 0))
res = {"images": per_img, "summary": {}}
for s, v in allp.items():
    if v:
        sm = hs.summarize(v)
        res["summary"][f"{s[0]}-{s[1]}"] = {k: (np.round(x, 4).tolist() if hasattr(x, "tolist") else x) for k, x in sm.items()}
json.dump(res, open(dst, "w"), indent=1)
print("µm 环:", hs.BINS[1:].tolist())
for k, v in res["summary"].items():
    print(f"灯芯半径 {k} µm  图数 {v['n']}")
    print("   r  ", v["r"]); print("   Δr ", v["dr"]); print("   b  ", v["b"]); print("   Y  ", v["Y"])
