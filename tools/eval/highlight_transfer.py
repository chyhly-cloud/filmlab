"""Audit highlight information loss without fitting brightness or changing presets.

Run from any directory (also with python -I):
    python tools/eval/highlight_transfer.py --out /workspace/filmlab-audit
    python tools/eval/highlight_transfer.py --out DIR --raw INPUT.NEF

The synthetic wedge tests the mapping, not agreement with a real film scan.
Optional plots require matplotlib. RAW inspection requires a Bayer CFA and uses
the existing sensor-pileup detector, not display RGB >= 1, to classify saturation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import generation, rawio, spectral
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe

LUMA = np.array([0.2126, 0.7152, 0.0722])


def stages(engine, negative):
    """Reconstruct the scan stages and verify they match the public render path.

    The alternative without the extra shoulder uses the SAME negative, timing,
    paper, and output conditioning. It is an ablation, not a proposed correction.
    """
    trace = {}
    output = engine.scan(negative, trace=trace)
    sc, dn = engine.scanner, trace["Dn"]
    timing, paper = sc.timing, sc.digital_paper
    dm = dn if timing.M is None else timing.M[0] + timing.M[1] * dn
    xe_linear = timing.K - timing.s * dm
    xe = timing.paper_exposure(dn)
    dp = paper.density(xe)
    paper_rgb = paper.reference_print(dp).reshape(dn.shape)
    reconstructed = sc._condition(paper_rgb.reshape(-1, 3)).reshape(dn.shape)
    if not np.array_equal(output, reconstructed):
        raise RuntimeError("Stage reconstruction differs from Engine.scan")
    no_shoulder_dp = paper.density(xe_linear)
    no_shoulder = sc._reference_print(no_shoulder_dp, paper.k)
    encoded = np.rint(spectral.srgb_encode(output) * 255).astype(np.uint8)
    if timing.x_knee is None:
        shoulder_slope = np.ones_like(xe)
    else:
        width = max(timing.x_knee - timing.x_white, 1e-3)
        shoulder_slope = np.exp(-np.maximum(timing.x_knee - xe_linear, 0) / width)
    return {
        "input_logH": negative.log_exposure, "dye": negative.dye,
        "Dn": dn, "paper_exposure_linear": xe_linear, "paper_exposure": xe,
        "shoulder_slope": shoulder_slope, "paper_density": dp,
        "paper_rgb": paper_rgb, "output_rgb": output, "encoded_u8": encoded,
        "without_extra_shoulder_rgb": no_shoulder,
    }


def wedge(timing, out):
    ev = np.linspace(-6, 10, 129)
    eng = Engine(Params(**{
        **recipe("cinestill_800t"), "timing": timing, "roll": None, "grain": 0,
        "scene_illuminant": None, "scene_lights": None, "output_conditioning": None,
    }))
    # Isolate the pointwise transfer: no spatial scatter, halation, or grain.
    # Input neutral is relative to the film's design illuminant, not a RAW pixel.
    from filmlab.negative import Negative
    # Scale a fixed spectral exposure, rather than recomputing a reflectance
    # spectrum for every brightness. Only film + scan tone transfer is at issue.
    reference = eng._expose_lin(np.full((1, 1, 3), 0.184), eng.M)
    logh = eng._log_exposure(reference * (2.0 ** ev)[None, :, None])
    negative = Negative(eng.e, eng._develop(logh), eng._transmittance, log_exposure=logh)
    data = stages(eng, negative)
    np.savez(out / f"wedge_{timing}.npz", input_ev=ev, **data)
    rows = []
    for value in (0, 1, 2, 3, 4, 5, 6, 8, 10):
        i = int(np.argmin(abs(ev - value)))
        rows.append({"input_ev": value,
                     "Dn_green": float(data["Dn"][0, i, 1]),
                     "paper_density_green": float(data["paper_density"][0, i, 1]),
                     "output_Y": float(data["output_rgb"][0, i] @ LUMA),
                     "without_extra_shoulder_Y": float(data["without_extra_shoulder_rgb"][0, i] @ LUMA),
                     "shoulder_slope_green": float(data["shoulder_slope"][0, i, 1]),
                     "encoded_rgb": data["encoded_u8"][0, i].tolist()})
    return {"timing": timing, "K": eng.scanner.K.tolist(), "s": eng.scanner.s.tolist(),
            "x_knee": eng.scanner.x_knee, "x_white": eng.scanner.x_white,
            "rows": rows}, ev, data


def inspect_raw(path, out, size):
    import rawpy
    with rawpy.imread(str(path)) as raw:
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            raise ValueError("RAW classification requires Bayer CFA data; stacked/sRAW and X-Trans are unsupported")
        sat = rawio.sensor_saturation(raw, per_channel=True)
        h, w = raw.raw_image_visible.shape
        shape = (h // 2, w // 2)
        if raw.sizes.flip in (5, 6):
            shape = shape[::-1]
        if sat is None:
            sat = np.zeros((*shape, 3), dtype=bool)
    # Keep measured data: no randomized reconstruction and no denoising.
    rgb = rawio.load_raw(path, max_size=size, highlights="clip", denoise_raw=False, clip_detect="sensor")
    step = max(1, int(round(max(shape) / size)))
    fraction = rawio._downsample(sat.astype(float), step)
    if fraction.shape != rgb.shape:
        raise RuntimeError("Sensor mask and LibRaw output shapes do not match")
    clean = (fraction == 0).all(-1)
    full = rawio._downsample(sat.all(-1).astype(float), step) == 1
    # The remaining blocks include partial-channel saturation and spatial mixtures.
    # Do not call a downsampled block with any saturation 'all channels lost'.
    # Also, the existing detector combines two greens with ANY: all three RGB
    # flags do not establish that every individual CFA sample has saturated.
    mixed = ~clean & ~full
    m, note = rawio.film_metering(rgb, rng=np.random.default_rng(0), bias_ev=0.0)
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "grain": 0, "roll": None}))
    data = stages(eng, eng.develop_negative(m))
    np.savez(out / "raw_stages.npz", decoded_rgb=rgb, metered_rgb=m,
             saturation_fraction=fraction, measured=clean, **data)
    groups = {}
    for name, mask in (("measured", clean), ("partial_or_spatial_mix", mixed), ("all_RGB_flags_entire_block", full)):
        v = data["output_rgb"][mask]
        groups[name] = {"pixels": int(mask.sum()),
                        "all_RGB_ge_1": int((v >= 1).all(-1).sum()),
                        "any_RGB_gt_1": int((v > 1).any(-1).sum()),
                        "all_RGB_ge_0.9": int((v >= 0.9).all(-1).sum())}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "metering": note,
            "detector": "existing sensor_saturation heuristic; not an independently calibrated sensor ceiling; green flags combine both greens with ANY",
            "half_size_counts": {"unsaturated": int((~sat.any(-1)).sum()),
                                 "partial_channels": int((sat.any(-1) & ~sat.all(-1)).sum()),
                                 "all_RGB_flags": int(sat.all(-1).sum())},
            "output_groups": groups}


def plot(results, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for timing, ev, data in results:
        axes[0].plot(ev, data["Dn"][0, :, 1], label=timing)
        axes[1].plot(ev, data["output_rgb"][0] @ LUMA, label=timing)
        axes[2].plot(ev, data["encoded_u8"][0, :, 1], label=timing)
        if timing == "frame":
            ablation = data["without_extra_shoulder_rgb"][0]
            axes[1].plot(ev, ablation @ LUMA, "k--", label="frame, extra shoulder off")
            axes[2].plot(ev, np.rint(spectral.srgb_encode(ablation)[:, 1] * 255), "k--",
                         label="frame, extra shoulder off")
    for ax in axes:
        ax.set_xlabel("Input exposure (EV relative to 0.184)")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_title("Negative still records increasing exposure")
    axes[0].set_ylabel("Normalized green density")
    axes[1].set_title("Paper + optional frame shoulder")
    axes[1].set_ylabel("Linear output luminance Y")
    axes[2].set_title("Distinct values collapse in 8-bit export")
    axes[2].set_ylabel("sRGB green code (0-255)")
    fig.suptitle("Controlled neutral wedge — mapping diagnostic, no real-film validation")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generation", choices=generation.GENERATIONS, default="next")
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.size < 1:
        ap.error("--size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    generation.use(args.generation)
    report = {"generation": args.generation,
              "scope": "controlled transfer test, not EPFL/SillyStill validation; no fitted parameters",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "colour-science", "rawpy", "pillow")},
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "wedge": []}
    results = []
    for timing in ("standard", "gray", "frame"):
        summary, ev, data = wedge(timing, args.out)
        report["wedge"].append(summary)
        results.append((timing, ev, data))
    if args.raw:
        report["raw"] = inspect_raw(args.raw.resolve(), args.out, args.size)
    if args.plot:
        plot(results, args.out / "highlight_transfer.png")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
