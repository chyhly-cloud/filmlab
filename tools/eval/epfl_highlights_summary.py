"""Equal-scene summaries of epfl_highlights.py output; no fitting or rerendering."""
import argparse
import json
from pathlib import Path

import numpy as np

PATHS = ("real", "exp_plain", "paper_plain", "exp_lab", "paper_lab")


def summarize(source):
    results = json.loads(source.read_text())["results"]
    bins = []
    for i in range(24):
        rows = [s["bins"][i] for s in results.values() if s["bins"][i]["status"] == "ok"]
        row = {"center_EV": -5.75 + .5 * i, "scenes": len(rows), "paths": {}}
        for path in PATHS:
            values = [r["paths"][path] for r in rows]
            if not values: continue
            gain = [v["local_gain"] for v in values if v["local_gain"] is not None]
            entry = {"log2_Y": float(np.median([v["median_log2_Y"] for v in values])),
                     "local_gain": float(np.median(gain)) if gain else None, "local_gain_scenes": len(gain)}
            if path != "real":
                entry["dE00"] = float(np.median([v["median_dE00"] for v in values]))
            row["paths"][path] = entry
        bins.append(row)
    spans = {}
    for sid, scene in sorted(results.items()):
        by_ev = {b["EV"][0]: b["paths"] for b in scene["bins"] if b["status"] == "ok"}
        if 1.5 in by_ev and 3.5 in by_ev:
            spans[sid] = {k: by_ev[3.5][k]["median_log2_Y"] - by_ev[1.5][k]["median_log2_Y"] for k in PATHS}
    colour = {}
    for path in PATHS[1:]:
        values = []
        for scene in results.values():
            rows = [b["paths"][path] for b in scene["bins"] if b["status"] == "ok" and 2 <= b["EV"][0] < 4]
            if len(rows) == 4:
                values.append([np.median([r["median_dE00"] for r in rows]),
                               np.median([r["median_dLab"][0] for r in rows]),
                               np.median([r["median_dLab"][2] for r in rows])])
        colour[path] = {"scenes": len(values), "median_dE00_dL_db": np.median(values, 0).tolist() if values else None}
    return {"scope": "equal-scene medians, existing eight EPFL nights; not a held-out score",
            "scene_count": len(results), "bins": bins,
            "high_span_centers_EV": [1.75, 3.75], "high_span_per_scene": spans,
            "median_high_span": {k: float(np.median([v[k] for v in spans.values()])) for k in PATHS} if spans else {},
            "colour_EV_2_to_4": colour}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    colours = {"real": "black", "exp_plain": "tab:blue", "paper_plain": "tab:orange",
               "exp_lab": "tab:green", "paper_lab": "tab:red"}
    for name in PATHS:
        rows = [b for b in report["bins"] if b["scenes"] >= 4 and name in b["paths"]]
        x = [r["center_EV"] for r in rows]
        if name in ("real", "exp_plain", "paper_plain"):
            axes[0].plot(x, [r["paths"][name]["log2_Y"] for r in rows], color=colours[name], label=name)
            axes[1].plot(x, [r["paths"][name]["local_gain"] for r in rows], color=colours[name], label=name)
        if name != "real":
            axes[2].plot(x, [r["paths"][name]["dE00"] for r in rows],
                         color=colours[name], linestyle="--" if name.endswith("lab") else "-", label=name)
    for ax in axes:
        ax.set_xlabel("Measured RAW exposure bin (EV)")
        ax.set_xlim(-2, 5); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_title("Tone progression (no brightness matching)")
    axes[0].set_ylabel("Median log2 output luminance")
    axes[1].set_title("Local contrast, fixed input neighbours")
    axes[1].set_ylabel("Median |delta log2 Y| / |delta input EV|")
    axes[2].set_title("Colour error to the same real pixels")
    axes[2].set_ylabel("Median Delta E 2000")
    fig.suptitle("Eight restored EPFL pairs: same negative and timing for both shoulder paths")
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    report = summarize(args.results)
    destination = args.results.with_name("summary.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if args.plot: plot(report, args.results.with_name("summary.png"))
    print(json.dumps({k: v for k, v in report.items() if k != "bins"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
