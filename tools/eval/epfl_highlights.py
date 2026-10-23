"""Fixed-negative comparison against EPFL scans; see docs/EPFL_HIGHLIGHT_VALIDATION.md.

FILMLAB_DATA=/workspace/filmlab-data python -I tools/eval/epfl_highlights.py --out DIR
No curve/colour fitting. Multiprocessing uses spawn. External images are data only.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools/eval"))
LUMA = np.array([.2126, .7152, .0722])
EDGES = np.arange(-6, 6.01, .5)
NIGHTS = [f"R1-{i:02d}-{i + 1}" for i in range(8)]


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summaries(ev, measured, real, predictions):
    import colour
    from filmlab.spectral import srgb_encode
    from scipy.ndimage import gaussian_filter
    arrays = {"real": real, **{k: np.clip(v, 0, 1) for k, v in predictions.items()}}
    logs, labs, derivatives = {}, {}, {}
    weights = gaussian_filter(measured.astype(float), 2)
    for name, rgb in arrays.items():
        y = rgb @ LUMA
        logs[name] = np.log2(np.maximum(y, 1e-6))
        labs[name] = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.maximum(rgb, 0), apply_cctf_decoding=False))
        smooth_y = gaussian_filter(np.where(measured, y, 0), 2) / np.maximum(weights, 1e-9)
        derivatives[name] = np.log2(np.maximum(smooth_y, 1e-6))
    valid = measured & np.isfinite(ev) & np.isfinite(real).all(-1)
    local = {name: [] for name in arrays}
    local_ev = []
    for axis in (0, 1):
        a = (slice(None, -4), slice(None)) if axis == 0 else (slice(None), slice(None, -4))
        b = (slice(4, None), slice(None)) if axis == 0 else (slice(None), slice(4, None))
        de = np.abs(ev[b] - ev[a])
        keep = valid[a] & valid[b] & (weights[a] > .999) & (weights[b] > .999) & (de >= .125) & (de <= 1)
        local_ev.append(((ev[a] + ev[b]) / 2)[keep])
        for name in arrays:
            local[name].append((np.abs(derivatives[name][b] - derivatives[name][a]) / np.maximum(de, 1e-9))[keep])
    lev = np.concatenate(local_ev)
    local = {k: np.concatenate(v) for k, v in local.items()}
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mask = valid & (ev >= lo) & (ev < hi)
        n = int(mask.sum())
        item = {"EV": [float(lo), float(hi)], "pixels": n, "status": "ok" if n >= 200 else "insufficient"}
        if n >= 200:
            item["paths"] = {}
            lm = (lev >= lo) & (lev < hi)
            for name, rgb in arrays.items():
                lab = labs[name][mask]
                entry = {"median_log2_Y": float(np.median(logs[name][mask])),
                         "median_Lab": np.median(lab, 0).tolist(),
                         "local_pairs": int(lm.sum()),
                         "local_gain": float(np.median(local[name][lm])) if lm.sum() >= 200 else None}
                if name != "real":
                    reference = labs["real"][mask]
                    entry["median_dLab"] = np.median(lab - reference, 0).tolist()
                    entry["median_dE00"] = float(np.median(colour.delta_E(lab, reference, method="CIE 2000")))
                    c, cr = np.linalg.norm(lab[:, 1:], axis=-1), np.linalg.norm(reference[:, 1:], axis=-1)
                    chromatic = (c > 5) & (cr > 5)
                    entry["median_chroma_ratio"] = float(np.median(c[chromatic] / cr[chromatic])) if chromatic.sum() >= 50 else None
                    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]) - np.arctan2(reference[:, 2], reference[:, 1]))
                    entry["median_abs_hue_error"] = float(np.median(np.abs((hue[chromatic] + 180) % 360 - 180))) if chromatic.sum() >= 50 else None
                    codes = np.floor(srgb_encode(rgb[mask]) * 255 + .5)
                    entry["all_codes_ge_246_fraction"] = float((codes >= 246).all(-1).mean())
                    entry["any_float_channel_gt_1_fraction"] = float((predictions[name][mask] > 1).any(-1).mean())
                item["paths"][name] = entry
        rows.append(item)
    return rows


def sheet(real_srgb, predictions, sat, destination):
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import label, center_of_mass
    from filmlab.spectral import srgb_encode
    arrays = {"Real Frontier scan": real_srgb,
              **{k: srgb_encode(v) for k, v in predictions.items()}}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    cc, count = label(sat)
    crop = None
    if count:
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        cy, cx = center_of_mass(sat, cc, int(np.argmax(sizes)))
        cy, cx = int(cy), int(cx)
        y0, x0 = max(0, min(cy - 160, sat.shape[0] - 320)), max(0, min(cx - 160, sat.shape[1] - 320))
        crop = (y0, x0)
    for kind in ("full", "source") if crop else ("full",):
        tiles = []
        for name, arr in arrays.items():
            if kind == "source":
                y, x = crop; arr = arr[y:y + 320, x:x + 320]
            im = Image.fromarray(np.floor(np.clip(np.nan_to_num(arr), 0, 1) * 255 + .5).astype(np.uint8))
            im.thumbnail((400, 400))
            tile = Image.new("RGB", (400, 430), (20, 20, 20))
            tile.paste(im, (0, 30)); ImageDraw.Draw(tile).text((5, 5), name, font=font, fill="white")
            tiles.append(tile)
        result = Image.new("RGB", (400 * len(tiles), 430))
        for i, tile in enumerate(tiles): result.paste(tile, (400 * i, 0))
        result.save(destination.with_name(destination.name + f"_{kind}.jpg"), quality=94)
    return crop


def job(args):
    sid, output = args
    import colour, rawpy
    from scipy.ndimage import binary_dilation
    from skimage.transform import warp, ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    import oracle_timing as ot
    generation.use("next_sensor")
    manifest = ot.MAN[sid]
    rawpath, scanpath, bias = ot.sources(sid)
    # Verify before image parsing, even when files were restored earlier.
    for path, key in ((rawpath, "raw_sha256"), (scanpath, "film_sha256")):
        if digest(path) != manifest[key]: raise ValueError(f"SHA256 mismatch: {path}")
    H, real_srgb = ot.geometry(sid)
    with np.load(Path(ot.GEO) / f"geo_{sid}.npz", allow_pickle=False) as geo:
        geometry_method = str(geo["method"]) if "method" in geo else "ORB_or_SIFT_fixed_seed_0"
    with rawpy.imread(rawpath) as raw:
        sat = rawio.sensor_saturation(raw, per_channel=True)
    rgb = rawio.load_raw(rawpath, max_size=1400)
    seed = zlib.crc32(sid.encode()) % 1000
    metered, note = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    engine = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1,
                             "frame": seed % 36, "grain": 0, "output_conditioning": None}))
    negative = engine.develop_negative(metered)
    trace = {}; baseline = engine.scan(negative, trace=trace)
    sc, t = engine.scanner, trace["timing"]
    paths = {"exp_plain": baseline}
    sc.timing = replace(t, x_knee=None, x_white=None)
    paths["paper_plain"] = sc.to_rgb(trace["Dn"])
    for name in ("exp", "paper"):
        sc.timing = t if name == "exp" else replace(t, x_knee=None, x_white=None)
        sc.conditioning = "lab"
        paths[name + "_lab"] = sc.to_rgb(trace["Dn"])
    transform = ProjectiveTransform(H)
    wp = lambda a, fill=np.nan: warp(a, transform.inverse, output_shape=real_srgb.shape[:2], order=1, cval=fill)
    if sat is None:
        measured = np.ones(rgb.shape[:2], bool)
        saturation = np.zeros(rgb.shape[:2], bool)
    else:
        step = max(1, int(round(max(sat.shape[:2]) / 1400)))
        saturation = rawio._downsample(sat.any(-1).astype(float), step) > 0
        if saturation.shape != rgb.shape[:2]: raise ValueError(f"Sensor/output shape mismatch: {sid}")
        measured = ~binary_dilation(saturation, iterations=2)
    ev = wp(np.log2(np.maximum(metered @ LUMA, 1e-6) / .184))
    valid = (wp(measured.astype(float), 0) > .999) & np.isfinite(ev)
    aligned = {k: wp(v) for k, v in paths.items()}
    valid &= np.isfinite(aligned["exp_plain"]).all(-1)
    real = colour.models.eotf_sRGB(real_srgb)
    rows = summaries(ev, valid, real, aligned)
    output = Path(output)
    np.savez_compressed(output / f"{sid}_aligned.npz", ev=ev.astype(np.float32), measured=valid,
                        real_srgb=real_srgb.astype(np.float32),
                        **{k: v.astype(np.float32) for k, v in aligned.items()})
    crop = sheet(real_srgb, aligned, wp(saturation.astype(float), 0) > .5, output / sid)
    result = {"scene": sid, "metering": note, "raw_sha256": manifest["raw_sha256"],
              "film_sha256": manifest["film_sha256"], "geometry": H.tolist(),
              "geometry_method": geometry_method, "source_crop_yx": crop,
              "valid_pixels": int(valid.sum()), "timing": {"K": t.K.tolist(), "s": t.s.tolist(),
              "M": [a.tolist() for a in t.M], "x_knee": t.x_knee, "x_white": t.x_white}, "bins": rows}
    if not any(r["status"] == "ok" for r in rows): raise ValueError(f"No valid bins: {sid}")
    (output / f"{sid}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return sid, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", default=NIGHTS)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if not 1 <= args.jobs <= 4: ap.error("--jobs must be between 1 and 4")
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        results = {}
        for sid, result in pool.imap_unordered(job, [(s, str(args.out.resolve())) for s in args.scenes]):
            results[sid] = result; print(f"{sid}: {result['valid_pixels']} measured aligned pixels", flush=True)
    (args.out / "results.json").write_text(json.dumps({"scope": "preregistered existing EPFL pairs, not a new holdout",
        "generation": "next_sensor", "shared_timing": True, "results": results}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
