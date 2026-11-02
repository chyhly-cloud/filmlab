"""T29: 测量表。真实 800T / 真实对照 / 现基准 t22_bxyz / remjet 底图 / ② / ③, 同一套 t24 算法 (t29_measure)。
用法: python3 t29_report.py 800T参考目录 对照目录 t22缓存目录 t29输出目录 参数.json"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t29_measure as M  # noqa: E402
from t29_overlay import SCENES  # noqa: E402

COLS = [100, 150, 200, 300, 400, 500, 700, 1000]
IX = [M.BINS.index(c) for c in COLS]
REAL_HALO = np.array([0.186, 0.270, 0.256, 0.198, 0.091, 0.047, 0.031, 0.014])   # T25/T26 (60–150 µm 灯)


def row(lab, a, fmt="6.3f"):
    return f"| {lab} | " + " | ".join(format(x, fmt) if np.isfinite(x) else "—" for x in a) + " |"


def head(first):
    return f"| {first} | " + " | ".join(str(c) for c in COLS) + " |\n" + "|---" * (len(COLS) + 1) + "|"


def main(ref, ctl, cache, out, par_json):
    par = json.load(open(par_json))
    cache, out = Path(cache), Path(out)
    files = lambda d, suf: M.merge([M.lamps(M.load_lin(d / f"{s}_{suf}.png")) for s in SCENES])
    D = {"真实 800T": M.measure_dir(ref), "真实对照 (带 remjet)": M.measure_dir(ctl),
         "① 现基准 t22_bxyz": files(cache, "t22_bxyz_0"), "底图 remjet": files(cache, "t22_bxyz_0_remjet")}
    names = {"dv150": "② 高斯 σ150", "dv250": "② 高斯 σ250", "hyb": "③ 混合", "hyb_pc": "③' 混合·管线色"}
    for v in par["versions"]:
        D[names.get(v, v)] = files(out, f"t29_{v}")
    S = {cls: {n: M.summary(A[cls]) for n, A in D.items()} for cls in ("60-150", "0-60")}

    for cls in ("60-150", "0-60"):
        print(f"\n## 灯芯 {cls} µm：叠加成片（总量）超出背景\n")
        print("灯数：" + "，".join(f"{n} {S[cls][n]['n']}" for n in D))
        for key, lab, fmt in (("R", "R", "6.3f"), ("GR", "G/R", "6.2f"), ("BG", "B/G", "6.2f")):
            print(f"\n{lab}\n\n" + head("距灯芯边缘 µm"))
            for n in D:
                print(row(n, S[cls][n][key][IX], fmt))

    for cls in ("60-150", "0-60"):
        real = S[cls]["真实 800T"]["R"][IX] - S[cls]["真实对照 (带 remjet)"]["R"][IX]
        print(f"\n## 光晕部分（版本 − 对照/底图），灯芯 {cls} µm\n\n" + head("R"))
        print(row("真实 800T − 对照", real))
        H = {}
        for n in D:
            if n.startswith(("①", "②", "③")):
                H[n] = S[cls][n]["R"][IX] - S[cls]["底图 remjet"]["R"][IX]
                print(row(n, H[n]))
        print("\n" + head("绝对值 / 真实"))
        for n, h in H.items():
            print(row(n, h / real, "6.2f"))
        rn = real / np.nanmax(real)
        print("\n" + head("归一形状 / 真实形状") + "")
        print(row("真实归一形状", rn, "6.2f"))
        for n, h in H.items():
            hn = h / np.nanmax(h)
            print(row(n, hn / rn, "6.2f") + f"  形状 RMS {np.sqrt(np.nanmean((hn - rn) ** 2)):.3f}，700/峰 {hn[COLS.index(700)]:.2f}（真实 {rn[COLS.index(700)]:.2f}），峰位 {COLS[int(np.nanargmax(h))]} µm")
        print("\n光晕部分颜色（中位差之比）G/R、B/G：")
        for n in list(H) + ["真实"]:
            a = S[cls][n if n != "真实" else "真实 800T"]
            b = S[cls]["底图 remjet" if n != "真实" else "真实对照 (带 remjet)"]
            dR, dG, dB = (a[c][IX] - b[c][IX] for c in "RGB")
            np.seterr(all="ignore")
            print(f"  {n}: G/R " + " ".join(f"{x:5.2f}" for x in dG / dR) + " | B/G " + " ".join(f"{x:5.2f}" for x in dB / dG))

    print("\n## 环的可见度：每灯 (150–300 µm 环 R 超出背景) ÷ (700–1000 µm)\n")
    print("| 版本 | 60–150：逐灯中位 | 中位数之比 | 有效/分母过小 | 0–60：逐灯中位 | 中位数之比 | 有效/分母过小 |\n|---|---|---|---|---|---|---|")
    for n, A in D.items():
        a, b = M.ring_visibility(A["60-150"]), M.ring_visibility(A["0-60"])
        print(f"| {n} | {a[0]:.2f} | {a[1]:.2f} | {a[2]}/{a[3]} | {b[0]:.2f} | {b[1]:.2f} | {b[2]}/{b[3]} |")


def area_report(cache, out, par):
    """成片里被加亮的面积, 以及源 (超过阈值的像素) 中来自大面积亮区 (等效半径 > 400 µm, 非点光源) 的比例。"""
    from scipy import ndimage as ndi
    import t29_overlay as O
    print("\n## 被加亮的面积（相对 remjet 底图，R 通道 8 位码值）与源的构成\n")
    vers = ["①"] + list(par["versions"])
    print("| 场景 | 源中大面积亮区占比 | " + " | ".join(f"{v} ΔR≥10 / ≥40 码值面积 %" for v in vers) + " |")
    print("|---" * (len(vers) + 2) + "|")
    for s in SCENES:
        b8 = np.asarray(M.Image.open(cache / f"{s}_t22_bxyz_0_remjet.png").convert("RGB")).astype(int)
        base = M.hs.srgb_to_lin(b8 / 255.0)
        um = O.px_um(base.shape)
        src = O.source(base, par["T"])
        lab, n = ndi.label(src > 0)
        area = np.bincount(lab.ravel())
        big = np.sqrt(area / np.pi) * um > 400
        big[0] = False
        frac_big = src[big[lab]].sum() / max(src.sum(), 1e-9)
        cells = []
        for v in vers:
            f = cache / f"{s}_t22_bxyz_0.png" if v == "①" else out / f"{s}_t29_{v}.png"
            d = np.asarray(M.Image.open(f).convert("RGB")).astype(int)[..., 0] - b8[..., 0]
            cells.append(f"{(d >= 10).mean() * 100:.1f} / {(d >= 40).mean() * 100:.1f}")
        print(f"| {s} | {frac_big * 100:.0f}% | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:6])
    area_report(Path(sys.argv[3]), Path(sys.argv[4]), json.load(open(sys.argv[5])))
