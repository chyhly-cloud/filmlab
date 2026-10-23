"""Separate floating shoulder/paper response from clipping and quantization.

Reuses highlight_transfer.py's saved exposure wedge, with no new timing fit:
  python -I tools/eval/shoulder_ablation.py --cache DIR --out DIR [--plot]
This tests the mapping, not agreement with real CineStill scans.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import generation, spectral
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.scanning.paper import OUTPUT_BLACK
from filmlab.scanning.timing import SHOULDER_WHITE

LUMA = np.array([0.2126, 0.7152, 0.0722])


def code_counts(rgb):
    return {str(bits): len(np.unique(np.floor(spectral.srgb_encode(rgb) * ((1 << bits) - 1) + .5)
                                    .astype(np.uint16), axis=0)) for bits in (8, 16)}


def audit(cache):
    metadata = json.loads((cache / "report.json").read_text())
    generation.use(metadata["generation"])
    engine = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0,
                              "output_conditioning": None}))
    paper = engine.scanner.digital_paper
    with np.load(cache / "wedge_frame.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    ev = data["input_ev"]
    xe0, xe1 = data["paper_exposure_linear"][0], data["paper_exposure"][0]
    # This control directly removes ONLY the extra shoulder from the same xe.
    def render(xe):
        return paper.reference_print(paper.density(xe))
    base, candidate = render(xe1), render(xe0)
    if not np.array_equal(np.maximum(base, 0), data["output_rgb"][0]):
        raise ValueError("Cache is not the expected unconditioned frame wedge")
    # Paper local slopes are measured where each path actually enters the paper.
    # The shoulder shifts this operating point, so effects cannot be subtracted
    # as independent percentage contributions.
    h = 1e-5
    def paper_slope(xe):
        return (paper.density(xe + h) - paper.density(xe - h)) / (2 * h)
    slopes = {"paper_slope_after_shoulder": paper_slope(xe1),
              "paper_slope_without_shoulder": paper_slope(xe0)}
    curves = {"exponential": base, "paper_only": candidate}
    rows = []
    for value in (0, 2, 4, 6, 8, 10):
        i = int(np.argmin(abs(ev - value)))
        sh = float(data["shoulder_slope"][0, i, 1])
        ps = float(slopes["paper_slope_after_shoulder"][i, 1])
        row = {"ev": float(ev[i]), "shoulder_slope_G": sh,
               "paper_slope_after_shoulder_G": ps,
               "combined_density_slope_G": sh * ps,
               "paper_only_density_slope_G": float(slopes["paper_slope_without_shoulder"][i, 1])}
        for name, rgb in curves.items():
            row[name] = {"Y": float(rgb[i] @ LUMA), "rgb_float": rgb[i].tolist(),
                         "code8": np.floor(spectral.srgb_encode(rgb[i]) * 255 + .5).astype(int).tolist()}
        rows.append(row)
    # Compute the actual asymptote through paper dye spectra and black compensation,
    # not just srgb_encode(0.93). The output white is slightly chromatic by design.
    xw = next(w["x_white"] for w in metadata["wedge"] if w["timing"] == "frame")
    dp_limit = paper.density(np.full((1, 3), xw))
    limit = paper.reference_print(dp_limit)[0]
    engine.scanner.conditioning = "lab"
    conditioned_limit = engine.scanner._condition(limit[None])[0]
    black_compensated = OUTPUT_BLACK + (1 - OUTPUT_BLACK) * (SHOULDER_WHITE - paper.k) / (1 - paper.k)
    report = {"scope": "fixed cached negative/timing; synthetic, no real-scan validation",
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cache_tool_sha256": metadata.get("tool_sha256"),
              "generation": metadata["generation"], "source_versions": metadata.get("versions"),
              "asymptote": {"assumed_reflectance": SHOULDER_WHITE,
                            "naive_code8": float(spectral.srgb_encode(np.array(SHOULDER_WHITE)) * 255),
                            "paper_density": dp_limit[0].tolist(), "actual_rgb": limit.tolist(),
                            "actual_Y": float(limit @ LUMA), "black_compensated_scalar": black_compensated,
                            "actual_code8": np.floor(spectral.srgb_encode(limit) * 255 + .5).astype(int).tolist(),
                            "with_existing_lab_conditioning_rgb": conditioned_limit.tolist(),
                            "with_existing_lab_conditioning_code8":
                            np.floor(spectral.srgb_encode(conditioned_limit) * 255 + .5).astype(int).tolist()},
              "rows": rows, "bands": {}}
    for lo, hi in ((2, 4), (4, 6), (6, 8), (8, 10)):
        a, b = [int(np.argmin(abs(ev - v))) for v in (lo, hi)]
        band = {}
        for name, rgb in curves.items():
            y = rgb[a:b + 1] @ LUMA
            band[name] = {"output_stops": float(np.log2(y[-1] / y[0])),
                          "delta_Y": float(y[-1] - y[0]),
                          "distinct_float_RGB": len(np.unique(rgb[a:b + 1], axis=0)),
                          "distinct_display_RGB": code_counts(rgb[a:b + 1]),
                          "any_channel_gt_1": int((rgb[a:b + 1] > 1).any(-1).sum()),
                          "all_channels_ge_1": int((rgb[a:b + 1] >= 1).all(-1).sum()),
                          "negative_Y_steps": int((np.diff(y) < -1e-8).sum())}
        report["bands"][f"{lo}:{hi}"] = band
    return report, {"ev": ev, "exponential_rgb": base, "paper_only_rgb": candidate,
                    **slopes, "shoulder_slope": data["shoulder_slope"][0]}


def colour_stress():
    """Measured reflectance spectra test colour side effects, not real-scan fidelity.

    Both renders share one negative and the SAME K/s/M derived from the gray wedge.
    No matrix or tint is fitted or changed. Below-knee samples must be identical.
    """
    import colour
    from filmlab.negative import Negative
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None,
                          "grain": 0, "output_conditioning": None}))
    ev = np.linspace(-6, 10, 129)
    reference = eng._expose_lin(np.full((1, 1, 3), .184), eng.M_ref)
    logh = eng._log_exposure(reference * (2.0 ** ev)[None, :, None])
    neg = Negative(eng.e, eng._develop(logh), eng._transmittance, log_exposure=logh)
    eng.scan(neg)
    timing = eng.scanner.timing
    sens = 10.0 ** eng.e.stock.log_sensitivity
    illum_ref = spectral.illuminant("BB3200")
    scale = reference[0, 0, 1] / (.184 * (illum_ref @ sens)[1])
    patches = colour.SDS_COLOURCHECKERS["BabelColor Average"]
    reflectance = np.stack([sd.copy().align(colour.SpectralShape(380, 780, 5)).values for sd in patches.values()])
    rows = []
    for lamp in ("BB3200", "D65", "FL2"):
        illum = spectral.illuminant(lamp)
        # Match photopic illumination, not output brightness.
        cmf_y = spectral.cmfs()[:, 1]
        illum = illum * ((illum_ref @ cmf_y) / (illum @ cmf_y))
        exposure = ((reflectance * illum) @ sens) * scale
        logh = eng._log_exposure(exposure[:, None, :] * (2.0 ** ev)[None, :, None])
        neg = Negative(eng.e, eng._develop(logh), eng._transmittance)
        dn = neg.measure(eng.scanner.normalized_density)
        eng.scanner.timing = timing
        a = eng.scanner.to_rgb(dn)
        eng.scanner.timing = replace(timing, x_knee=None, x_white=None)
        b = eng.scanner.to_rgb(dn)
        linear_x = eng.scanner.timing.paper_exposure(dn)
        unchanged = (linear_x >= timing.x_knee).all(-1)
        if not np.array_equal(a[unchanged], b[unchanged]):
            raise RuntimeError("Ablation changed pixels outside the active shoulder")
        lab = lambda rgb: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.maximum(rgb, 0), apply_cctf_decoding=False))
        la, lb = lab(a), lab(b)
        ca, cb = np.linalg.norm(la[..., 1:], axis=-1), np.linalg.norm(lb[..., 1:], axis=-1)
        hue_a = np.degrees(np.arctan2(la[..., 2], la[..., 1]))
        hue_b = np.degrees(np.arctan2(lb[..., 2], lb[..., 1]))
        hue_difference = np.abs((hue_b - hue_a + 180) % 360 - 180)
        for lo, hi in ((-6, 0), (0, 2), (2, 4), (4, 6), (6, 10)):
            mask = np.broadcast_to(((ev >= lo) & (ev < hi))[None], unchanged.shape)
            ab = np.linalg.norm(lb[..., 1:] - la[..., 1:], axis=-1)
            hue_mask = mask & (ca > 3) & (cb > 3)
            rows.append({"illuminant": lamp, "EV_band": [lo, hi], "samples": int(mask.sum()),
                         "outside_shoulder": int((mask & unchanged).sum()),
                         "median_chroma_exponential": float(np.median(ca[mask])),
                         "median_chroma_paper_only": float(np.median(cb[mask])),
                         "median_delta_ab": float(np.median(ab[mask])),
                         "hue_samples_C_gt_3": int(hue_mask.sum()),
                         "median_abs_hue_change_degrees": float(np.median(hue_difference[hue_mask])) if hue_mask.any() else None,
                         "median_delta_L": float(np.median((lb - la)[..., 0][mask]))})
    eng.scanner.timing = timing
    return {"scope": "24 measured ColorChecker reflectances x 3 illuminants; synthetic scan colour differences, not accuracy",
            "rows": rows}


def raw_workflow(cache):
    """Reuse the actual decoded RAW from the first audit; no real film counterpart."""
    path = cache / "raw_stages.npz"
    if not path.exists():
        return {"status": "not_run", "reason": "no cached RAW stages"}
    with np.load(path, allow_pickle=False) as data:
        rgb, measured = data["metered_rgb"], data["measured"]
        expected = data["output_rgb"]
    # highlight_transfer.inspect_raw saved the default lab conditioning. Keep
    # that original setting here as well; the wedge separately uses None.
    p = Params(**{**recipe("cinestill_800t"), "timing": "frame", "grain": 0,
                  "roll": None, "output_conditioning": "lab"})
    engine = Engine(p)
    neg = engine.develop_negative(rgb)
    trace = {}
    baseline = engine.scan(neg, trace=trace)
    if not np.array_equal(baseline, expected):
        raise RuntimeError("Cached RAW baseline is not reproducible")
    candidate = Engine(replace(p, scan_shoulder="paper_only")).scan(neg)
    return {"status": "passed", "scope": "real digital RAW, same negative and original lab conditioning; no paired film scan or colour-accuracy claim",
            "pixels": int(measured.size), "fully_measured_pixels": int(measured.sum()),
            "changed_measured_pixels": int(np.any(baseline[measured] != candidate[measured], -1).sum()),
            "max_abs_RGB_change_measured": float(np.max(np.abs(baseline[measured] - candidate[measured]))) if measured.any() else None}


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for name in ("exponential", "paper_only"):
        rgb = data[name + "_rgb"]
        y = rgb @ LUMA
        axes[0].plot(data["ev"], y, label=name)
        axes[1].plot(data["ev"], np.gradient(y, data["ev"]), label=name)
        axes[2].plot(data["ev"], spectral.srgb_encode(rgb)[:, 1] * 255, label=name + " float")
        axes[2].step(data["ev"], np.floor(spectral.srgb_encode(rgb)[:, 1] * 255 + .5), alpha=.5)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_xlabel("Input exposure (EV)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_title("Fixed timing: active shoulder sets the ceiling")
    axes[0].set_ylabel("Linear output Y")
    axes[1].set_title("Floating response BEFORE quantization")
    axes[1].set_ylabel("dY / dEV")
    axes[2].set_title("Quantization adds discrete plateaus")
    axes[2].set_ylabel("sRGB green code")
    fig.suptitle("Synthetic ablation: removing the extra shoulder does not remove paper whitening")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report, data = audit(args.cache)
    report["colour_stress"] = colour_stress()
    report["raw_workflow"] = raw_workflow(args.cache)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    np.savez(args.out / "curves.npz", **data)
    if args.plot:
        plot(data, args.out / "shoulder_ablation.png")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
