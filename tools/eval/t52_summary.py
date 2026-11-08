"""T52 汇总: 场景帧、模型均匀灰、真实 RPL 的颗粒 RMS 与可见度 → tools/eval/t52/summary.json 与 Markdown 表 (stdout)。
用法: python3 -I t52_summary.py
"""
import gzip
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "t52"


def interp(pts, code):
    """pts: [(码值, 值)] → 按码值线性插值 (不外推)。"""
    pts = sorted(pts)
    c = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
    if code < c[0] or code > c[-1]:
        return float("nan")
    return float(np.interp(code, c, v))


def main():
    uni = json.load(open(D / "uniform_model.json"))
    real = json.load(gzip.open(D / "real_rpl.json.gz", "rt"))
    sc = json.load(open(D / "scenes.json")) if (D / "scenes.json").exists() else {}
    out = {}

    # ---- 1. 模型均匀灰: 每个情形在中间调 (EV 0, 码值约 115; EV +1, 约 163)
    cases = sorted({k.split("/")[0] for k in uni})
    rows = []
    for c in cases:
        for ev in ("-1", "+0", "+1"):
            r = uni.get(f"{c}/{ev}")
            if r is None or "native_visibility_100pct" not in r:
                continue
            rows.append(dict(case=c, ev=ev, code=r["native"]["code"], nat_px=r["native"]["s_px"], nat_hp=r["native"]["s_hp"],
                             ph_px=r["phone"]["s_px"], ph_hp=r["phone"]["s_hp"], R_phone=r["phone_visibility"]["R_max"],
                             R_100=r["native_visibility_100pct"]["R_max"],
                             R_bands=[b["R"] for b in r["phone_visibility"]["bands"]]))
    out["uniform"] = rows
    print("\n### 模型均匀灰 (t51_bw, 固定定光)\n")
    print("| 情形 | EV | 码值 | 原尺寸 σ_px / σ_hp | 手机 σ_px / σ_hp | 手机 R (4 带) | 原尺寸 100% R |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r['ev']} | {r['code']:.0f} | {r['nat_px']:.1f} / {r['nat_hp']:.1f} | {r['ph_px']:.2f} / {r['ph_hp']:.2f} | "
              f"{' '.join(f'{x:.1f}' for x in r['R_bands'])} | {r['R_100']:.1f} |")

    # ---- 2. 真实 RPL 对模型 645 (同码值)
    def model_pts(case, scale, key):
        return [(uni[k][scale]["code"], uni[k][scale][key]) for k in uni if k.startswith(case + "/")]

    print("\n### 真实 RPL (645, Noritsu) 对模型 645 (同码值; 每档 = 各帧该码值档的中位, 再取帧间中位)\n")
    print("| 片 | 尺寸 | 码值档 | 帧数 | 真实 σ_px / σ_hp | 模型 σ_px / σ_hp | 比值 σ_px / σ_hp | 真实手机 R |")
    print("|---|---|---|---|---|---|---|---|")
    anchor = []
    for film, case in (("6_KODAK400TX", "645_trix_xtol"), ("7_ILFORDHP5PLUS", "645_hp5_d76"), ("8_ILFORDDELTA3200PRO", None)):
        for scale in ("native", "phone"):
            for rng in ([80, 120], [120, 160], [160, 200]):
                vals = []
                for k, r in real.items():
                    if not k.startswith(film + "/"):
                        continue
                    for lv in r[scale]["levels"]:
                        if lv["range"] == rng:
                            vals.append((lv["code"], lv["s_px"], lv["s_hp"], k))
                if len(vals) < 2:
                    continue
                a = np.array([v[:3] for v in vals])
                code, spx, shp = np.median(a, 0)
                R = [real[v[3]]["phone"].get("visibility", {}).get("R_max") for v in vals] if scale == "phone" else []
                R = [x for x in R if x is not None]
                row = dict(film=film, scale=scale, range=rng, n_frames=len(vals), code=float(code), real_px=float(spx),
                           real_hp=float(shp), R_phone_frames_median=float(np.median(R)) if R else None)
                if case:
                    mp = np.median([interp(model_pts(case, scale, "s_px"), v[0]) for v in vals])
                    mh = np.median([interp(model_pts(case, scale, "s_hp"), v[0]) for v in vals])
                    row.update(model_px=float(mp), model_hp=float(mh), ratio_px=float(spx / mp), ratio_hp=float(shp / mh))
                anchor.append(row)
                m = (f"{row['model_px']:.2f} / {row['model_hp']:.2f} | {row['ratio_px']:.2f} / {row['ratio_hp']:.2f}"
                     if case else "— | —")
                print(f"| {film.split('_', 1)[1]} | {scale} | {rng[0]}–{rng[1]}（{code:.0f}） | {len(vals)} | {spx:.2f} / {shp:.2f} | {m} | "
                      f"{row['R_phone_frames_median'] if row['R_phone_frames_median'] is None else round(row['R_phone_frames_median'], 1)} |")
    out["real_anchor"] = anchor

    # 每帧的手机尺寸中间调 (80–180) 与 R
    per = []
    for k, r in real.items():
        m = r["phone"]["mid"]
        if m.get("n", 0) >= 3:
            per.append(dict(frame=k, code=m["code"], s_px=m["s_px"], s_hp=m["s_hp"], n=m["n"],
                            R=r["phone"].get("visibility", {}).get("R_max"),
                            R_bands=[b["R"] for b in r["phone"].get("visibility", {}).get("bands", [])],
                            native_mid=r["native"]["mid"]))
    out["real_frames_phone_mid"] = per
    print("\n### 真实 RPL 每帧 (手机尺寸, 中间调 80–180)\n")
    print("| 帧 | 块数 | 码值 | σ_px / σ_hp | R (4 带) |")
    print("|---|---|---|---|---|")
    for p in per:
        print(f"| {p['frame']} | {p['n']} | {p['code']:.0f} | {p['s_px']:.2f} / {p['s_hp']:.2f} | {' '.join(f'{x:.1f}' for x in p['R_bands'])} |")

    # ---- 3. 场景帧
    print("\n### 场景帧 (整张; 均匀块, 中间调 80–180)\n")
    print("| 帧 | µm/像素 | 原尺寸 块数 / 码值 / σ_px / σ_hp | 手机 块数 / 码值 / σ_px / σ_hp | 手机 R (4 带) | 原尺寸 100% R | 颗粒关同位置 σ_hp 原 / 手机 |")
    print("|---|---|---|---|---|---|---|")
    srows = []
    for k, r in sc.items():
        n, p = r["native"]["mid"], r["phone"]["mid"]
        if n.get("n", 0) < 3 or p.get("n", 0) < 3:
            print(f"| {k} | {r['meta']['px_um']:.2f} | {n.get('n')} | {p.get('n')} | | | |")
            continue
        v = r["phone"].get("visibility", {})
        g0n, g0p = r["native"].get("grain_off_same_tiles", {}), r["phone"].get("grain_off_same_tiles", {})
        row = dict(frame=k, px_um=r["meta"]["px_um"], native=n, phone={a: b for a, b in p.items()},
                   R_phone=v.get("R_max"), R_bands=[b["R"] for b in v.get("bands", [])],
                   R_100=r["native"].get("visibility_100pct", {}).get("R_max"),
                   g0_native_hp=g0n.get("s_hp"), g0_phone_hp=g0p.get("s_hp"))
        srows.append(row)
        g0 = f"{g0n['s_hp']:.2f} / {g0p['s_hp']:.2f}" if "s_hp" in g0n and "s_hp" in g0p else ""
        print(f"| {k} | {row['px_um']:.2f} | {n['n']} / {n['code']:.0f} / {n['s_px']:.1f} / {n['s_hp']:.1f} | "
              f"{p['n']} / {p['code']:.0f} / {p['s_px']:.2f} / {p['s_hp']:.2f} | {' '.join(f'{x:.1f}' for x in row['R_bands'])} | "
              f"{row['R_100']:.1f} | {g0} |")
    out["scenes"] = srows
    json.dump(out, open(D / "summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
