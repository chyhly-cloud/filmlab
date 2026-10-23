"""One frozen neutral-response comparison; no material/process fitting.

The source bounds below are deliberately coarse MANUAL readings of the
1586 x 1568 inline chart. They are not automated raster extraction, raw
sensitometer samples, confidence intervals, or density-instrument error bars.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Crossing brackets include uncertainty in the visible low-exposure plateau.
# Columns: visible plateau D range, camera-stop brackets for plateau+.3D
# and plateau+1.3D. These are deliberately broad visual bounds.
READINGS = {
    'R': ([.20, .25], [-3.5, -2.0], [2.0, 4.0]),
    'G': ([.55, .65], [-4.5, -3.0], [1.0, 3.0]),
    'B': ([.85, .90], [-4.0, -3.0], [1.0, 2.5]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_npz', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a = np.load(args.model_npz)
    x = a['relative_log10_exposure']
    d = a['approximate_StatusM_total']
    n = d - d[0]
    assert np.all(np.diff(x) > 0) and np.all(np.diff(n, axis=0) >= -1e-10)
    results = {}
    for j, c in enumerate('RGB'):
        plateau, low, high = READINGS[c]
        # Both axis conventions are retained: 16 stops versus five log units.
        minimum = 1 / ((high[1] - low[0]) * (5 / 16))
        maximum = 1 / ((high[0] - low[1]) * np.log10(2))
        mask = (n[:, j] >= .3) & (n[:, j] <= 1.3)
        assert mask.sum() > 5 and n[-1, j] > 1.3
        results[c] = {
            'visible_plateau_D_manual_range': plateau,
            'crossing_plus_0p3D_camera_stops_manual_range': low,
            'crossing_plus_1p3D_camera_stops_manual_range': high,
            'chart_secant_manual_envelope_D_per_log10H': [minimum, maximum],
            'model_visible_low_exposure_D': float(d[0, j]),
            'model_OLS_gamma_net_0p3_to_1p3D': float(np.polyfit(x[mask], d[mask, j], 1)[0]),
            'model_secant_gamma_net_0p3_to_1p3D': float(1 / (np.interp(1.3, n[:, j], x) - np.interp(.3, n[:, j], x))),
        }
    record = {
        'source': 'User inline chart, 1586 x 1568 display; no local chart raster available. Mounted JPEG was inspected and is an older corridor photograph, not this chart.',
        'reading_method': 'Coarse manual crossing brackets, not digitized point estimates. Bounds are not statistical confidence intervals or a measured uncertainty budget.',
        'axis_scale_discrepancy_percent': float((5 / (16 * np.log10(2)) - 1) * 100),
        'definition': 'One-D secant between visible plateau+.3D and plateau+1.3D. Model OLS in same net-density interval is also reported. Operational plateau is not established true base+fog.',
        'instrument_limit': 'Source Cs41 densitometer Status undisclosed; model uses Gaussian approximate Status M. Numerical envelopes do not cover this unknown systematic error.',
        'conclusion': 'Model secants fall inside all coarse manual source envelopes. This bounded comparison cannot establish or exclude smaller material-response errors; no rejection of current model or fitted correction is warranted.',
        'channels': results,
    }
    (args.output / 'comparison.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=.18, right=.97, top=.88, bottom=.27)
    for j, c in enumerate('RGB'):
        lo, hi = results[c]['chart_secant_manual_envelope_D_per_log10H']
        ax.plot([lo, hi], [j, j], color='0.65', lw=13, solid_capstyle='butt')
        ax.scatter(results[c]['model_secant_gamma_net_0p3_to_1p3D'], j, c=c.lower(), s=85, zorder=3)
    ax.set(yticks=range(3), yticklabels=['Red density', 'Green density', 'Blue density'],
           xlabel='Secant slope: density / log10 exposure', ylim=(-.65, 2.65),
           title='800T / Cs41: bounded neutral-curve check, no fitting')
    ax.grid(axis='x', alpha=.2)
    fig.text(.05, .035, 'Grey: coarse manual chart envelope (not a CI). Dots: frozen FilmLab, approximate Status M.\nDensity-instrument differences remain unknown; this is not an absolute sensitometric validation.', fontsize=9)
    fig.savefig(args.output / 'bounded_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
