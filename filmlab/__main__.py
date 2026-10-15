"""命令行:  python -m filmlab 照片.NEF -p kodak_portra_400 -o out.jpg
         python -m filmlab --list"""
import argparse
import time

from .io import save_srgb
from .chemistry import PROCESSES
from .engine import Engine, Params
from .recipes import RECIPES, recipe
from .rawio import auto_expose, load_raw
from .stocks import list_stocks


def main():
    ap = argparse.ArgumentParser(prog="filmlab", description="RAW → 物理胶片模拟")
    ap.add_argument("raw", nargs="?")
    ap.add_argument("-p", "--recipe", default="kodak_portra_400", help="配方或胶片名 (--list 查看)")
    ap.add_argument("-o", "--out", default="film.jpg")
    ap.add_argument("--size", type=int, default=3000, help="长边像素")
    ap.add_argument("--ev", type=float, default=0.0, help="相机曝光补偿")
    ap.add_argument("--paper", help="相纸 (负片印放用)")
    ap.add_argument("--process", help="冲洗工艺: " + " / ".join(PROCESSES))
    ap.add_argument("--push", type=float, help="推 (+) / 拉 (−) 档数, 同时按提高后的 EI 曝光")
    ap.add_argument("--bleach", type=float, help="漂白效率 0..1 (0 = 漂白旁路)")
    ap.add_argument("--points", type=float, nargs=3, default=(0, 0, 0), metavar=("R", "G", "B"),
                    help="在自动光号上加减印放光号")
    ap.add_argument("--illuminant", help="拍摄现场光源, 例: D55 / D65 / T / BB2800")
    ap.add_argument("--format", type=float, default=36.0, help="底片宽度 mm (35mm=36, 6x7=70)")
    ap.add_argument("--grain", type=float, help="颗粒倍数")
    ap.add_argument("--wb", default="daylight", choices=["daylight", "camera", "auto"],
                    help="RAW 白平衡: daylight 保留场景光色 (推荐, 像胶片一样); camera 用相机白平衡")
    ap.add_argument("--timing", choices=["frame", "gray", "standard"], help="冲印店定光方式")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not a.raw:
        print("配方:", *RECIPES, sep="\n  ")
        print("冲洗工艺:", *[f"{k}  {v.note}" for k, v in PROCESSES.items()], sep="\n  ")
        for k, v in list_stocks().items():
            print(f"{k}:", *v, sep="\n  ")
        return

    kw = recipe(a.recipe)
    kw["exposure_ev"] = kw.get("exposure_ev", 0.0) + a.ev
    kw.update(printer_points=tuple(a.points), frame_width_mm=a.format)
    if a.wb != "daylight" and "scene_illuminant" not in kw and a.illuminant is None:
        kw["scene_illuminant"] = None      # 输入已被白平衡"拉正" → 按胶片自己的平衡光源
    if a.push is not None:
        kw["push"] = a.push
        kw["exposure_ev"] -= a.push
    for k, v in [("paper", a.paper), ("scene_illuminant", a.illuminant), ("process", a.process),
                 ("bleach", a.bleach), ("grain", a.grain), ("timing", a.timing)]:
        if v is not None:
            kw[k] = v

    t = time.time()
    rgb = auto_expose(load_raw(a.raw, max_size=a.size, white_balance=a.wb))
    eng = Engine(Params(**kw))
    save_srgb(eng.render(rgb), a.out)
    print(eng.describe())
    print(f"→ {a.out}  ({rgb.shape[1]}×{rgb.shape[0]}, {time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
