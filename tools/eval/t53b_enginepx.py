"""T53b: 整帧像素 (35 mm 36000/3032 µm、半格 24000/3032 µm) 上的 render_silver, 引擎默认 sub=16、rows_per_task=16。
设置同 t53_validate.engine_px_checks (1024×256, seed 3, 净 D 0.682)。HEAD 与改动后各跑一次, 比较见文档第 3.2 节。
用法: python3 -I t53b_enginepx.py 仓库根 输出.npz
      python3 -I t53b_enginepx.py cmp HEAD.npz 新.npz 输出.json"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")


def compare(fa, fb, dst):
    a, b = np.load(fa), np.load(fb)
    out = {}
    for k in a.files:
        A, B = a[k], b[k]
        r = dict(bitwise_equal=bool(np.array_equal(A, B)))
        if not r["bitwise_equal"]:
            dd = -np.log10(B) + np.log10(A)
            edge = np.isin(np.arange(A.shape[0]) % 16, (0, 15))
            ch = A != B
            r.update(frac_px_changed=float(ch.mean()), frac_changed_on_block_edge_rows=float(ch[edge].sum() / ch.sum()),
                     mean_dD=float(dd.mean()), mean_dD_block_edge_rows=float(dd[edge].mean()),
                     max_abs_dD=float(np.abs(dd).max()),
                     field_dD=float(np.log10(A.mean()) - np.log10(B.mean())))
        out[k] = r
        print(k, r)
    json.dump(out, open(dst, "w"), indent=1)


if sys.argv[1] == "cmp":
    compare(*sys.argv[2:5])
    sys.exit()
sys.path.insert(0, sys.argv[1])
from filmlab import generation, grain, materials  # noqa: E402
generation.use("t51_bw")
out = {}
for key in ("kodak_trix_400", "kodak_trix_400@rodinal", "kodak_tmax_400"):
    sp = materials.silver_spec(materials.bw_silver_film(key))
    lhs = np.linspace(-4.0, 0.0, 2001)
    Hs = 10.0 ** np.interp(0.682, grain.silver_curve(sp, lhs) - grain.silver_density(sp, 0.0), lhs)
    for k in ((1.0, 2.0, 3.0) if key == "kodak_trix_400" else (1.0,)):
        for px in (36000 / 3032, 24000 / 3032):
            s_ = grain.scale_silver(sp, k)
            T = grain.render_silver(s_, np.full((1024, 256), Hs), px, seed=3, workers=1)
            out[f"{key}|k{k:g}|px{px:.2f}"] = T
            print(key, k, round(px, 2), flush=True)
np.savez(sys.argv[2], **out)
