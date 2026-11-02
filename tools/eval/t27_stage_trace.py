"""T27: 暗背景灯的分级追踪 (与 t19_model.py 同流程: 代际 t22_bxyz, STOPS=0, 1600 像素, roll=None, grain=0)。

render  目录 RAW…      每个 RAW 渲染除碳 800T 与带 remjet 两次, 保存 (按成片灯芯测量所需的) 各级数组:
                       raw  = 不降噪、不重建的 RAW 线性 (load_raw(highlights="clip", denoise_raw=False))
                       pre  = 降噪后、截断重建前; reco = 重建后 (= 测光前的 load_raw 输出); clip = 截断子像素比例
                       direct / scattered / total = Engine.develop_negative 的 trace (红绿蓝三层曝光)
                       Hr / Hp = raw / pre 经与 direct 同一逐点曝光映射 (Engine._expose_mixed) 的三层曝光
                       成片 PNG 与 t19_model.py 缓存逐像素相同 (已核对)。
profile 目录           按成片灯芯 (三通道 ≥ 0.93, 1.5–2.5 mm 背景 < 0.02, 同 t24_dist_colour) 分档,
                       输出各级红层曝光剖面 / 白阈值曝光 (成片 R 达到 0.93 的红层总曝光, 每场景由像素关系求得),
                       以及成片 R 经同一曲线换回的曝光。
flux    目录           截断块 (dilate(clip>0.25), 同 reconstruct_highlights) 的重建光通量与块外 ρ=6–12 px 实测裙边之比,
                       以及未截断紧凑亮点给出的 PSF 尾 T(ρ) = 裙边/光通量。
环境变量 NPROC (默认 4)。"""
import json, os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]
BINS = [25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
LUMA = np.array([0.2126, 0.7152, 0.0722])


def _render(a):
    out, f, remjet = a
    from filmlab import generation, rawio, materials
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.io import srgb_encode
    from PIL import Image
    generation.use("t22_bxyz")
    rawio.SOURCE_STOPS = 0.0
    cap, orig = {}, rawio.reconstruct_highlights

    def wrapped(rgb, clip_frac, rng, l_clip=None, source_stops=0.0):
        cap["pre"], cap["clip"] = rgb.copy(), clip_frac.copy()
        return orig(rgb, clip_frac, rng, l_clip, source_stops)
    rawio.reconstruct_highlights = wrapped
    ex = rawio.exif_exposure(f)
    rgb = rawio.load_raw(f, max_size=1600)
    rawio.reconstruct_highlights = orig
    raw = rawio.load_raw(f, max_size=1600, highlights="clip", denoise_raw=False)
    m, _ = rawio.film_metering(rgb, rawio.clip_luminance(f), rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),
                               t_original=ex[0] if ex else None)
    k = float(np.nanmedian(m[rgb > 1e-3] / rgb[rgb > 1e-3]))
    extra = {"film": materials.emulsion("kodak_vision3_500t")} if remjet else {}
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0, **extra}))
    tr = {}
    neg = eng.develop_negative(m, trace=tr)
    o = eng.scan(neg.state)
    ev = 2.0 ** eng.p.exposure_ev
    Hp = eng._expose_mixed(cap.get("pre", rgb).astype(np.float64) * k * ev)[0]
    Hr = eng._expose_mixed(raw.astype(np.float64) * k * ev)[0]
    tag = Path(f).stem + ("_remjet" if remjet else "")
    Image.fromarray((np.clip(srgb_encode(np.clip(o, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)).save(Path(out) / f"{tag}.png")
    f32 = lambda x: np.asarray(x, np.float32)
    np.savez(Path(out) / f"{tag}.npz", pre=f32(cap.get("pre", rgb)), reco=f32(rgb), clip=f32(cap.get("clip", np.zeros(rgb.shape[:2]))),
             direct=f32(tr["direct_exposure"]), scattered=f32(tr["scattered_exposure"]), total=f32(tr["total_exposure"]),
             Hr=f32(Hr), Hp=f32(Hp), k=k)
    return tag


def _curve(s, o):
    """成片 R 对某级红层曝光的中位曲线; 返回 (log10 曝光, 单调化的成片值, 白阈值曝光 = 最后一次从下方穿过 0.93)。"""
    ok = s > 0
    lb = np.linspace(np.log10(np.percentile(s[ok], 1)), np.log10(s.max()), 150)
    idx = np.digitize(np.log10(np.maximum(s, 1e-9)), lb)
    med = np.array([np.median(o[idx == i]) if (idx == i).sum() > 30 else np.nan for i in range(1, len(lb))])
    x = lb[:-1] + np.diff(lb) / 2
    g = np.isfinite(med); x, med = x[g], med[g]
    below = np.nonzero(med < 0.93)[0]
    i = min(int(below[-1] + 1) if len(below) else 0, len(x) - 1)
    medm = np.maximum.accumulate(np.where(np.arange(len(med)) < i, np.minimum(med, 0.929), med))
    return x, medm, 10 ** x[i]


def profile(d):
    from PIL import Image
    from scipy import ndimage as ndi
    import t19_halo_stats as hs
    res = {}
    for npz in sorted(Path(d).glob("*.npz")):
        tag = npz.stem; sc = tag.replace("_remjet", "")
        z = np.load(npz)
        im = hs.srgb_to_lin(np.asarray(Image.open(Path(d) / f"{tag}.png").convert("RGB")) / 255.0)
        st = {"raw": z["Hr"][..., 0], "pre": z["Hp"][..., 0], "direct": z["direct"][..., 0],
              "scattered": z["scattered"][..., 0], "total": z["total"][..., 0]}
        oR = im[..., 0]
        core = im.min(-1) >= 0.93
        nc = ~ndi.binary_dilation(core)
        x, med, ew = _curve(st["total"][nc], oR[nc])
        dist, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
        um = 36000.0 / max(im.shape[:2]); dist *= um
        lab, n = ndi.label(core); own = lab[iy, ix]; idx = np.arange(1, n + 1)
        rad = np.sqrt(ndi.sum(core, lab, idx) / np.pi) * um
        ring = (dist > 1500) & (dist < 2500)
        bgo = np.stack([np.asarray(ndi.median(im[..., c], own * ring, index=idx), float) for c in range(3)], 1)
        ok = np.where(np.all(np.nan_to_num(bgo, nan=1) < 0.02, 1))[0] + 1
        sel = np.isin(own, ok) & (dist <= BINS[-1]) & ~core
        L, bb = own[sel], np.searchsorted(BINS, dist[sel])
        rs = np.isin(own, ok) & ring
        key = L * 16 + bb; order = np.argsort(key, kind="stable"); key = key[order]
        uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
        prof = {}
        for s, a in list(st.items()) + [("out", oR)]:
            bg = np.asarray(ndi.median(a, own * rs, index=idx))
            v = a[sel][order]
            for kk, s0, c in zip(uk, start, cnt):
                l, bi = divmod(int(kk), 16)
                if c >= 3 and bi < len(BINS):
                    prof.setdefault(l, {}).setdefault(s, np.full(len(BINS), np.nan))[bi] = np.median(v[s0:s0 + c]) - bg[l - 1]
                    prof[l].setdefault("_bg_" + s, bg[l - 1])
        rows = []
        for l, p in prof.items():
            r = {"r": float(rad[l - 1]), "scene": sc}
            for s in st:
                r[s] = (p[s] / ew).tolist()
            bo = p["_bg_out"]
            r["out"] = p["out"].tolist()
            r["out2E"] = (10 ** np.interp(p["out"] + bo, med, x) / ew - 10 ** np.interp(bo, med, x) / ew).tolist()
            rows.append(r)
        res[tag] = {"E_white": float(ew), "lamps": rows}
    json.dump(res, open(Path(d) / "profile.json", "w"))
    cls = lambda r: "0-60" if r < 60 else "60-150" if r < 150 else None
    I = [2, 3, 4, 5, 6]
    for rj in (True, False):
        print(f"\n== {'带 remjet' if rj else '除碳 800T'}: 红层曝光 / 白阈值曝光 (超出背景), 距灯芯边缘 100 150 200 300 400 µm")
        for c in ("0-60", "60-150"):
            for ex in (None, "city_61360"):
                Ls = [l for t, v in res.items() if t.endswith("_remjet") == rj for l in v["lamps"]
                      if cls(l["r"]) == c and (ex is None or l["scene"] != ex)]
                if not Ls:
                    continue
                print(f"[灯芯 {c} µm{' 不含 ' + ex if ex else ''}] 灯 {len(Ls)}")
                for s in ("raw", "pre", "direct", "scattered", "total", "out2E", "out"):
                    P = np.nanmedian(np.array([l[s] for l in Ls]), 0)
                    lab_ = {"out2E": "成片→曝光", "out": "成片线性R"}.get(s, s)
                    print(f"   {lab_:10s}" + " ".join(f"{P[i]:.3f}" for i in I))


def flux(d):
    from scipy import ndimage as ndi
    RHO = (6, 8, 10, 12)
    rows, psf = [], {}
    for npz in sorted(Path(d).glob("*.npz")):
        if npz.stem.endswith("_remjet"):
            continue
        z = np.load(npz); sc = npz.stem
        k = float(z["k"])
        pre, reco, clip = z["pre"] @ LUMA, z["reco"] @ LUMA, z["clip"]
        mask = ndi.binary_dilation(clip > 0.25)
        lab, n = ndi.label(mask)
        dd, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
        own = lab[iy, ix]; h, w = pre.shape
        for kk, sl in enumerate(ndi.find_objects(lab), 1):
            if sl[0].start < 20 or sl[1].start < 20 or sl[0].stop > h - 20 or sl[1].stop > w - 20:
                continue
            y0, y1, x0, x1 = sl[0].start - 18, sl[0].stop + 18, sl[1].start - 18, sl[1].stop + 18
            blob = lab[y0:y1, x0:x1] == kk; area = int(blob.sum())
            cy, cx = ndi.center_of_mass(blob)
            yy, xx = np.mgrid[y0:y1, x0:x1]; rr = np.hypot(yy - cy - y0, xx - cx - x0)
            mine = own[y0:y1, x0:x1] == kk; p = pre[y0:y1, x0:x1]
            bgm = mine & (rr > 14) & (rr <= 17)
            if bgm.sum() < 5:
                continue
            bg = np.median(p[bgm])
            sk = []
            for r_ in RHO:
                q = mine & (np.abs(rr - r_) <= 0.5) & (dd[y0:y1, x0:x1] >= 2)
                sk.append(float(np.median(p[q]) - bg) if q.sum() >= 4 else np.nan)
            rows.append(dict(scene=sc, area=area, req=float(np.sqrt(area / np.pi)), bg=float(bg), sk=sk,
                             F_reco=float(reco[y0:y1, x0:x1][blob].sum() - bg * area), F_clip=float(p[blob].sum() - bg * area),
                             clipmax=float(np.percentile(p[blob], 90))))
        cl = np.median(pre[clip > 0.9]) if (clip > 0.9).any() else pre.max()
        dc = ndi.distance_transform_edt(~(clip > 0))
        pk = (pre == ndi.maximum_filter(pre, 7)) & (pre > 0.2 * cl) & (pre < 0.9 * cl) & (dc > 20)
        yy, xx = np.mgrid[-17:18, -17:18]; rr = np.hypot(yy, xx); T = []
        for y, x in zip(*np.nonzero(pk)):
            if y < 18 or x < 18 or y >= h - 18 or x >= w - 18:
                continue
            win = pre[y - 17:y + 18, x - 17:x + 18]; bg = np.median(win[(rr > 14) & (rr <= 17)]); v = win - bg
            if bg > 0.05 * v[17, 17] or (v[rr <= 4] > 0.5 * v[17, 17]).sum() > 9 or (win[rr <= 13] > win[17, 17]).any():
                continue
            F = v[rr <= 3].sum()
            T.append([np.median(v[np.abs(rr - r_) <= 0.5]) / F for r_ in RHO])
        psf[sc] = np.median(np.array(T), 0).tolist() if len(T) >= 5 else None
        print(f"{sc}: 未截断紧凑亮点 {len(T)} 个, T(ρ=6,8,10,12) = {psf[sc]}")
    json.dump({"blobs": rows, "T": psf}, open(Path(d) / "flux.json", "w"))
    Ts = [np.array(t) for t in psf.values() if t is not None]
    Tg = np.exp(np.mean(np.log(Ts), 0)) if Ts else None
    print("\n截断块: 重建光通量 / 截断电平下光通量 (档), 以及按 T(ρ) 由裙边推算的光通量 / 截断电平下光通量 (档) [T 取上面各场景的几何平均]")
    for sc in sorted({r["scene"] for r in rows}):
        for lo, hi in ((0, 3), (3, 7)):
            S = [r for r in rows if r["scene"] == sc and lo < r["req"] <= hi and r["F_reco"] > 0 and r["bg"] < 0.05 * r["clipmax"]]
            if len(S) < 3:
                continue
            rc = np.median(np.log2([r["F_reco"] / r["F_clip"] for r in S]))
            line = f"  {sc:12s} 块半径 {lo}-{hi} px  块数 {len(S):4d}  重建/截断 {rc:+5.1f}"
            if Tg is not None:
                A = np.array([r["sk"][:3] for r in S])
                g = np.nanmedian(np.log2(np.clip(A / Tg[:3], 1e-12, None) / np.array([r["F_clip"] for r in S])[:, None]), 1)
                line += f"   裙边推算/截断 {np.nanmedian(g):+5.1f}"
            print(line)


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "render":
        import multiprocessing as mp
        Path(d).mkdir(parents=True, exist_ok=True)
        with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as p:
            print(p.map(_render, [(d, f, rj) for rj in (False, True) for f in sys.argv[3:]], chunksize=1))
    elif cmd == "profile":
        profile(d)
    elif cmd == "flux":
        flux(d)
