"""Native TIFF crops vs native half-size RAW renders, with frozen prior timing.

No curve, colour, exposure or geometry fitting. Coordinates refer to the earlier
1600-pixel scan. All micrometres assume a 36 mm scan width (crop scale uncalibrated).
"""
import argparse
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import sys
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools/eval"))
LUMA = np.array([.2126, .7152, .0722])
BINS = (0, 50, 100, 150, 200, 300, 500, 800)


def pixel_scaling(sx, sy):
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])


def crop_warp(transform):
    """Bind one crop's transform; different ROIs must never share a loop closure."""
    from skimage.transform import warp
    def apply(a, order=1):
        return warp(a, transform.inverse, output_shape=(512,512), order=order, cval=np.nan)
    return apply


def region_statistics(images, core, measured, valid, um):
    import colour
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    if not core.any(): return {"status": "no_sensor_core_near_selected_light"}
    distance = distance_transform_edt(~core) * um
    margin = int(np.ceil(300 / um)) + 1
    interior = np.zeros(core.shape, bool); interior[margin:-margin, margin:-margin] = True
    masks = {"core_prior_constrained": core & valid & interior,
             "surround_0_300_all": (distance > 0) & (distance <= 300) & valid & interior,
             "surround_0_300_measured": (distance > 0) & (distance <= 300) & measured & valid & interior}
    masks.update({f"ring_{lo}_{hi}": (distance > lo) & (distance <= hi) & valid & interior
                  for lo, hi in zip(BINS[:-1], BINS[1:])})
    labs = {k: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(v, 0, 1), apply_cctf_decoding=False)) for k, v in images.items()}
    stats = {}
    for name, mask in masks.items():
        row = {"pixels": int(mask.sum()), "measured_fraction": float(measured[mask].mean()) if mask.any() else None}
        if mask.sum() >= 30:
            row["paths"] = {}
            for key, rgb in images.items():
                v = np.clip(rgb, 0, 1)
                y = v @ LUMA
                # Same nominal 25 um low pass, 50 um separation; no sharpening.
                logy = np.log2(np.maximum(gaussian_filter(np.nan_to_num(y), 25 / um), 1e-6))
                shift = max(1, int(round(50 / um)))
                use = mask[:, :-shift] & mask[:, shift:]
                gradient = np.abs(logy[:, shift:] - logy[:, :-shift])
                lr = np.log(np.maximum(v[..., 0], 1e-4) / np.maximum(v[..., 1], 1e-4))
                entry = {"median_Y": float(np.median(y[mask])), "median_Lab": np.median(labs[key][mask], 0).tolist(),
                         "logY_p90_p10": float(np.ptp(np.percentile(logy[mask], [10, 90]))),
                         "contrast_50um": float(np.median(gradient[use])) if use.sum() >= 30 else None,
                         "log_R_over_G": float(np.median(lr[mask]))}
                if key != "real":
                    entry["dLab"] = np.median(labs[key][mask] - labs["real"][mask], 0).tolist()
                    entry["dE00"] = float(np.median(colour.delta_E(labs[key][mask], labs["real"][mask], method="CIE 2000")))
                row["paths"][key] = entry
        stats[name] = row
    footprints = {}
    for level in (.6, .8, .9):
        use = valid & interior & (distance <= 300)
        footprints[str(level)] = {k: int(((np.clip(v, 0, 1) @ LUMA >= level) & use).sum()) for k, v in images.items()}
    return {"status": "ok", "regions": stats, "bright_area_pixels_at_Y": footprints,
            "warning": "core includes sensor-flagged samples; ratios describe output appearance, not isolated physical halation"}


def make_sheet(images, path, um, model_um, name):
    from PIL import Image, ImageDraw, ImageFont
    from filmlab.spectral import srgb_encode
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for mode in ("lab", "plain"):
        names = [("Real scan", "real"), ("Exponential", f"exp_{mode}"), ("Paper only", f"paper_{mode}")]
        panels = []
        for label, key in names:
            a = np.floor(srgb_encode(np.nan_to_num(images[key])) * 255 + .5).astype(np.uint8)
            im = Image.fromarray(a)
            im.save(path.with_name(path.name + f"_{key}.png"))
            # Integer nearest-neighbour display magnification adds no new detail.
            im = im.resize((1024, 1024), Image.Resampling.NEAREST)
            panel = Image.new("RGB", (1024, 1100), (20, 20, 20));panel.paste(im, (0, 76))
            draw = ImageDraw.Draw(panel);draw.text((12, 8), f"{name} | {label} | {mode}", font=font, fill="white")
            draw.text((12, 38), f"TIFF {um:.2f} um/px; model {model_um:.2f} um/px; display 2x", font=font, fill="white")
            length = int(round(200 / um * 2));draw.line((30, 1060, 30 + length, 1060), fill="white", width=5)
            draw.text((30, 1028), "200 um (nominal)", font=font, fill="white");panels.append(panel)
        sheet = Image.new("RGB", (3072, 1100))
        for i, panel in enumerate(panels):sheet.paste(panel, (1024 * i, 0))
        sheet.save(path.with_name(path.name + f"_{mode}_compare.png"))


def job(args):
    sid, regions, output, cache = args
    import colour, rawpy
    from PIL import Image
    from scipy.ndimage import label, binary_dilation, distance_transform_edt
    from skimage.transform import ProjectiveTransform, warp
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from epfl_highlights import digest
    import oracle_timing as ot
    generation.use("next_sensor")
    rawpath, scanpath, bias = ot.sources(sid)
    for p, field in ((rawpath, "raw_sha256"), (scanpath, "film_sha256")):
        if digest(p) != ot.MAN[sid][field]:raise ValueError(f"hash mismatch: {p}")
    old = json.loads((Path(cache) / f"{sid}.json").read_text())
    with rawpy.imread(rawpath) as raw:sat = rawio.sensor_saturation(raw, per_channel=True)
    rgb = rawio.load_raw(rawpath, max_size=3000)
    if sat is None:sat = np.zeros(rgb.shape, bool)
    if sat.shape != rgb.shape:raise ValueError("Expected native half-size Bayer output")
    seed = zlib.crc32(sid.encode()) % 1000
    m, note = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1,
                          "frame": seed % 36, "grain": 0, "output_conditioning": None}))
    neg = eng.develop_negative(m)
    dn = neg.measure(eng.scanner.normalized_density)
    t = old["timing"]
    timing = PrintTiming(K=np.array(t["K"]), s=np.array(t["s"]), M=tuple(np.array(a) for a in t["M"]),
                         x_knee=t["x_knee"], x_white=t["x_white"])
    sc = eng.scanner
    old_real = np.load(Path(ot.GEO) / f"geo_{sid}.npz")["real"]
    with Image.open(scanpath) as im: real = colour.models.eotf_sRGB(np.asarray(im.convert("RGB"), float) / 255)
    h, w = real.shape[:2]
    step = max(1, int(round(max(rgb.shape[:2]) / 1400)))
    H = pixel_scaling(w / old_real.shape[1], h / old_real.shape[0]) @ np.array(old["geometry"]) @ pixel_scaling(1 / step, 1 / step)
    output = Path(output);items = []
    for r in regions:
        x, y = np.array(r["center"]) * [w / old_real.shape[1], h / old_real.shape[0]]
        x0, y0 = int(round(x)) - 256, int(round(y)) - 256
        if min(x0, y0) < 0 or x0 + 512 > w or y0 + 512 > h:raise ValueError("Crop outside scan")
        local = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]]) @ H
        transform = ProjectiveTransform(local)
        wp = crop_warp(transform)
        rawcore = wp(sat.any(-1).astype(float), 0) > .5
        measured = wp((~binary_dilation(sat.any(-1), iterations=1)).astype(float)) > .999
        cc, n = label(rawcore)
        core = np.zeros(rawcore.shape, bool)
        if n:
            distances = distance_transform_edt(~rawcore, return_distances=False, return_indices=True)
            cy, cx = distances[:, 256, 256];chosen = cc[cy,cx]
            # Do not invent a saturated core if no source is near the marked lamp.
            if np.hypot(cy-256,cx-256) * 36000 / w <= 150:core = cc == chosen
        realcrop = real[y0:y0+512,x0:x0+512]
        items.append({"region": r, "origin_xy": [x0,y0], "images":{"real":realcrop},
                      "warp":wp,"core":core,"measured":measured})
    for mode in ("plain", "lab"):
        sc.conditioning = None if mode == "plain" else "lab"
        for variant in ("exp", "paper"):
            sc.timing = timing if variant == "exp" else replace(timing, x_knee=None,x_white=None)
            rendered = sc.to_rgb(dn)
            for item in items:item["images"][variant + "_" + mode] = item["warp"](rendered)
            del rendered
    results = []
    for item in items:
        name = sid + "_" + item["region"]["name"]
        valid = np.isfinite(item["images"]["exp_lab"]).all(-1)
        images = item["images"]
        stats = region_statistics(images,item["core"],item["measured"],valid,36000/w)
        make_sheet(images,output/name,36000/w,36000/rgb.shape[1],name)
        np.savez_compressed(output/(name+".npz"),core=item["core"],measured=item["measured"],
                            **{k:v.astype(np.float32) for k,v in images.items()})
        results.append({"name":name,"center_thumb":item["region"]["center"],"crop_xy":item["origin_xy"],
                        "scan_um_per_px":36000/w,"model_um_per_px":36000/rgb.shape[1],"statistics":stats})
    report = {"scene":sid,"model_size":list(rgb.shape[:2]),"metering":note,"frozen_timing":t,"regions":results,
              "warning":"native half-size rerender; geometry is globally fixed; local alignment and 36 mm crop scale remain uncertain"}
    (output/f"{sid}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True);ap.add_argument("--jobs",type=int,default=2)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    regions=json.loads(args.regions.read_text())["regions"]
    scenes=sorted({r["scene"] for r in regions})
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        results=[]
        for r in pool.imap_unordered(job,[(s,[r for r in regions if r["scene"]==s],str(args.out),str(args.cache)) for s in scenes]):
            results.append(r);print(r["scene"],"done",flush=True)
    (args.out/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
