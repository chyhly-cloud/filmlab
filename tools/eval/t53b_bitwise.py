"""T53b: HEAD 与改动后的整条链逐位核对 (T51 / T53 做法: 合成灯光图; docs/tasks/T53b_SILVER_BLOCK_BOUNDARY.md)。
HEAD 在 git worktree 里跑 (仓库根 = worktree), 改动后在本仓库跑; 再用 t53b_bitcmp.py 比较。
用法: python3 -I t53b_bitwise.py 仓库根 输出目录 组名 [进程数]
组: small (合成图 192×128; 画面宽 0.384 / 0.9 / 2 / 36 mm → 像素 2 / 4.7 / 10.4 / 187.5 µm)
    | full (3032×2020 合成图, 35 mm 整帧像素 11.87 µm; 半格 7.92 µm)
每组情形输出 线性 sRGB float32 .npy。"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from filmlab import generation, grain
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
group = sys.argv[3]
grain.SILVER_WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 1


def scene(h, w):
    """合成灯光图: 横向 −6 … +4 EV 的对数斜坡 (18% 灰为 0 EV), 纵向冷暖渐变, 几盏亮灯 (+6 … +9 EV) 与暗区。"""
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    ev = -6 + 10 * x + 0.8 * np.sin(7 * y + 3 * x)
    base = 0.18 * 2.0 ** ev
    tint = np.stack([1 + 0.3 * (y - 0.5), np.ones_like(y), 1 - 0.3 * (y - 0.5)], -1)
    img = base[..., None] * tint
    for cy, cx, r, e in ((0.25, 0.2, 0.03, 9), (0.6, 0.45, 0.05, 7), (0.8, 0.8, 0.02, 8), (0.4, 0.7, 0.08, 6)):
        d = np.hypot((y - cy) * h, (x - cx) * w) / (r * max(h, w))
        img += (0.18 * 2.0 ** e * np.exp(-d ** 2))[..., None] * np.array([1.0, 0.85, 0.6])
    img[int(0.1 * h):int(0.2 * h), int(0.85 * w):int(0.95 * w)] = 0.18 * 2.0 ** -7
    return img


BW = ["kodak_trix_400", "kodak_trix_400@rodinal", "kodak_trix_400@xtol", "ilford_hp5_plus", "kodak_tmax_400",
      "kodak_tmax_100", "kodak_plusx_125", "kodak_trix_400+w25"]
if group == "small":
    gens_bw = ["t47_bw", "t48_bw", "t49_bw", "t50_bw", "t51_bw"]
    cases = []
    for g in gens_bw:
        for rec in (["kodak_trix_400", "kodak_tmax_400", "kodak_plusx_125", "bw_pan"] if g == "t47_bw" else
                    ["kodak_trix_400", "kodak_tmax_400", "ilford_hp5_plus"] if g in ("t48_bw", "t49_bw") else BW):
            if "@" in rec and g not in ("t50_bw", "t51_bw"):
                continue
            for fw in (0.384, 0.9, 2.0, 36.0):
                if "@" in rec and fw == 36.0:      # 成团显影液在 187 µm 像素上逐颗枚举显影晶体 (每行数千万颗), 内存不够; 与本改动无关
                    continue
                for gr in (0.0, 1.0):
                    cases.append((g, rec, fw, gr))
    for rec in ("kodak_trix_400+g0.5", "kodak_trix_400+g2", "kodak_trix_400+g3", "kodak_trix_400+fhalf",
                "kodak_portra_400", "kodak_portra_400+g2"):
        for fw in (0.384, 0.9, 2.0):
            cases.append(("t51_bw", rec, fw, 1.0))
    for g in ("current", "next", "t46_base", "t51_base"):
        for rec in ("kodak_portra_400", "kodak_vision3_500t"):
            for gr in (0.0, 1.0):
                cases.append((g, rec, 0.9, gr))
    img = scene(128, 192)
else:
    cases = [("t51_bw", r, 36.0, 1.0) for r in
             ("kodak_trix_400", "kodak_trix_400@rodinal", "ilford_hp5_plus", "kodak_tmax_400", "kodak_trix_400+g2",
              "kodak_trix_400+g3", "kodak_trix_400+fhalf")] + [("t50_bw", "kodak_trix_400", 36.0, 1.0)]
    img = scene(2020, 3032)
log = {}
for g, rec, fw, gr in cases:
    name = f"{g}__{rec}__fw{fw:g}__g{gr:g}"
    f = dst / f"{name}.npy"
    if f.exists():
        continue
    t = time.time()
    generation.use(g)
    try:
        p = {**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": gr}
        if "+f" not in rec:
            p["frame_width_mm"] = fw
        eng = Engine(Params(**p))
        o = eng.render(img.copy())
    except Exception as ex:                      # 记录 (旧 / 新同样报错则不算差异)
        log[name] = f"error: {type(ex).__name__}: {ex}"
        print(name, log[name], flush=True)
        continue
    np.save(f, np.asarray(o, np.float32))
    log[name] = dict(px_um=float(eng.px_um), sec=time.time() - t)
    print(name, f"px={eng.px_um:.2f}µm", f"{time.time() - t:.1f}s", flush=True)
json.dump(log, open(dst / f"log_{group}.json", "w"), indent=1)
