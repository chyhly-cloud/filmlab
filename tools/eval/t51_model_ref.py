"""T51: 模型的胶片侧颗粒参照 —— 均匀灰场经整条模型链 (颗粒开), 在细像素上渲染, 不含扫描仪光学 (scan_psf 关)。

分析时 (t51_fit.py) 再加扫描仪的光学点扩散与传感器孔径 (按冲印店像素积分), 得到"未经冲印店软件"的输出颗粒;
真实扫描与它的功率谱之比就是冲印店的空间处理。
用法: python3 -I t51_model_ref.py 代际 配方 细像素µm 边长像素 输出.npz [曝光档 ...]
输出: 每档 (相对 18% 灰的档数) 的线性 sRGB (float32)。定光 timing="gray" (固定), roll=None, 单一光源。
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main():
    sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
    from filmlab import engine, generation, grain
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    gen, rec, fine, n, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    levels = [float(v) for v in sys.argv[6:]] or [-4.0, -2.0, 0.0, 2.0]
    generation.use(gen)
    grain.SILVER_WORKERS = 1
    corr = tuple(c for c in engine.DEFAULT_CORRECTIONS if c != "scan_psf")
    out = {}
    for k in levels:
        t = time.time()
        p = Params(**{**recipe(rec), "timing": "gray", "roll": None, "frame": 0, "grain": 1.0, "scene_lights": None,
                      "corrections": corr, "frame_width_mm": n * fine / 1000.0})
        eng = Engine(p)
        rgb = np.full((n, n, 3), 0.18 * 2.0 ** k)
        o = eng.render(rgb).astype(np.float32)
        out[f"ev{k:+.1f}"] = o
        print(rec, k, f"{eng.px_um:.3f} µm/px", np.round(o.mean((0, 1)), 4), f"{time.time() - t:.0f} s", flush=True)
    np.savez_compressed(dst, **out)


if __name__ == "__main__":
    main()
