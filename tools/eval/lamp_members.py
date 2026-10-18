"""冷光缺口定位: 一个均匀区域 (相机测得的照明色) 在每一款候选灯下, 胶片 + 固定扫描 (锁定颜色, 无输出调整) 印成什么颜色。

相机只给出照明的 RGB 色度; 同色度的不同灯 (同色异谱) 对胶片三层的曝光不同。
若与相机色度一致的灯都印不出真实扫描的颜色 → 缺口在胶片 / 扫描物理; 若有的能、有的不能 → 光源识别 (信息极限)。

用法: python3 lamp_members.py 场景 [x0 x1 y0 y1 (相对坐标, 取样区域)]
"""
import sys, zlib
import numpy as np
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
from filmlab import generation, lights
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
import colour

LOCK = {"scan_neutral": "film", "scan_color_strength": 0.0, "output_conditioning": None}


def lab(rgb):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1), apply_cctf_decoding=False))


def region(img, box):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = box
    return np.median(img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].reshape(-1, 3), 0)


def print_color(rgb_patch, member, cls, extra=LOCK):
    k = [c.key for c in lights.classes()].index(cls)
    lights.class_weights = lambda ill, *a, **kw: np.eye(len(lights.classes()))[k] * np.ones(ill.shape[:-1] + (1,))
    lights.pick_members = lambda seed: {c.key: (member if c.key == cls else c.members[0]) for c in lights.classes()}
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 0, **extra}))
    img = np.ones((48, 48, 3)) * rgb_patch
    return lab(eng.render(img)[16:32, 16:32].reshape(-1, 3).mean(0))


if __name__ == "__main__":
    import oracle_timing as ot, pair_compare as pc
    from PIL import Image
    sid = sys.argv[1]
    box = tuple(map(float, sys.argv[2:6])) if len(sys.argv) > 5 else (0.35, 0.65, 0.35, 0.65)
    generation.use("next")
    cw, pm = lights.class_weights, lights.pick_members
    rawp, scan, _ = ot.sources(sid)
    rgb = pc.load_digital(rawp, 800)[0]
    c = region(rgb, box)
    ill = np.log(c[[0, 2]] / c[1])
    patch = 0.18 * c / (c @ [0.2126, 0.7152, 0.0722])
    real = Image.open(scan).convert("RGB"); real.thumbnail((800, 800))
    print(f"{sid}: 相机照明色度 log(R/G) {ill[0]:+.3f} log(B/G) {ill[1]:+.3f}")
    print(f"自动识别 (锁定颜色, 无输出调整): Lab {np.round(print_color.__wrapped__(patch) if hasattr(print_color, '__wrapped__') else 0, 1)}" if False else "")
    lights.class_weights, lights.pick_members = cw, pm
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 0, **LOCK}))
    print(f"  自动识别 → Lab {np.round(lab(eng.render(np.ones((48, 48, 3)) * patch)[16:32, 16:32].reshape(-1, 3).mean(0)), 1)}"
          f"   光源 {', '.join(f'{k}({n}) {v:.0%}' for k, (n, v) in eng.light_mix.items() if v >= 0.01)}")
    rows = []
    for cl in lights.classes():
        for name, spec in cl.members:
            d = float(np.hypot(*(np.log(lights.light_rgb(spec)[[0, 2]] / lights.light_rgb(spec)[1]) - ill)))
            L = print_color(patch, (name, spec), cl.key)
            rows.append((d, cl.key, name, L))
    lights.class_weights, lights.pick_members = cw, pm
    print(f"  {'灯':34s} 与相机色度距离   印出 L* a* b*")
    for d, k, n, L in sorted(rows):
        print(f"  {k:12s} {n:28s} {d:6.3f}   {L[0]:5.1f} {L[1]:+6.1f} {L[2]:+6.1f}")
