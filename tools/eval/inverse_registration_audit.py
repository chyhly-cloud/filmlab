"""Fixed-source inverse predictions under predeclared registration perturbations.

No fitting, RAW decoding, source edits, geometry selection or production changes.
The existing 25um linear-RGB averaging operator and stored sample positions stay
fixed. All shifts and paths use a single common finite aperture support.
Translations are sensitivity controls, not an estimated registration posterior.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/eval')]
from brightness_response_audit import lab


def predeclared_shifts():
    """Target sampled at (y+dy,x+dx); retain all axis/diagonal controls."""
    shifts = [(0, 0)]
    for size in (4, 8):
        shifts.extend([(0, size), (0, -size), (size, 0), (-size, 0),
                       (size, size), (size, -size), (-size, size), (-size, -size)])
    return shifts


def average_with_aperture_support(rgb, sigma):
    """Finite 4sigma Gaussian, averaging in linear RGB before any Lab change."""
    a = np.asarray(rgb, float)
    if a.ndim != 3 or a.shape[-1] != 3 or sigma <= 0:
        raise ValueError('Expected RGB image and positive sigma')
    radius = int(4*sigma + .5)
    finite = np.isfinite(a).all(-1)
    support = ndimage.minimum_filter(finite.astype(np.uint8), size=2*radius+1,
                                     mode='constant', cval=0).astype(bool)
    smooth = ndimage.gaussian_filter(np.nan_to_num(a), (sigma, sigma, 0), truncate=4.)
    return smooth, support


def common_samples(real, predictions, points, sigma, shifts):
    """One shared aperture support, including every target displacement/path."""
    points = np.asarray(points, int)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Points must be N by 2 y/x coordinates')
    real_mean, real_support = average_with_aperture_support(real, sigma)
    height, width = real_support.shape
    keep = np.ones(len(points), bool)
    target_values = []
    for dy, dx in shifts:
        shifted = points + [dy, dx]
        inside = ((shifted[:, 0] >= 0) & (shifted[:, 0] < height)
                  & (shifted[:, 1] >= 0) & (shifted[:, 1] < width))
        valid = np.zeros(len(points), bool)
        valid[inside] = real_support[tuple(shifted[inside].T)]
        keep &= valid
        values = np.full((len(points), 3), np.nan)
        values[inside] = real_mean[tuple(shifted[inside].T)]
        target_values.append(values)
    predictions_mean = {}
    for name, image in predictions.items():
        mean, support = average_with_aperture_support(image, sigma)
        if support.shape != (height, width):
            raise ValueError('All image grids must agree')
        inside = ((points[:, 0] >= 0) & (points[:, 0] < height)
                  & (points[:, 1] >= 0) & (points[:, 1] < width))
        valid = np.zeros(len(points), bool)
        valid[inside] = support[tuple(points[inside].T)]
        keep &= valid
        predictions_mean[name] = mean[tuple(np.clip(points, [0, 0], [height-1, width-1]).T)]
    return keep, np.asarray(target_values), predictions_mean


def run(input_dir, output):
    import colour
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path('/workspace/filmlab-audit/brightness-local/R1-02-3.json').read_text())
    sigma = 25/meta['scan_um_per_pixel']
    shifts = predeclared_shifts()
    report = {'scope': __doc__, 'shifts_dy_dx_scan_pixels': shifts,
              'nominal_scan_um_per_pixel': meta['scan_um_per_pixel'],
              'sigma_scan_pixels': sigma, 'gaussian_truncate': 4.,
              'regions': [], 'limits': [
                  'All perturbations retained; no geometry or source coefficient selected by this audit.',
                  'Integer translations bracket earlier 2-7px seam residuals, not rotations, distortion or parallax.',
                  'The 17 controls are not independent observations or a registration probability distribution.',
                  'Targets move against fixed source/model coordinates and fixed original samples.',
                  'Micron labels retain the nominal36mm scan-width assumption; they are not calibrated film measurements.',
                  'Stable prediction improvement does not certify fitted source radiance or material parameters.',
              ]}
    for name in ('ceiling_lamp', 'middle_ceiling'):
        fit = np.load(input_dir/(name+'_fit.npz'), allow_pickle=False)
        joint = np.load(input_dir/(name+'_joint.npz'), allow_pickle=False)
        metadata = np.load(input_dir/(name+'_metadata_prior.npz'), allow_pickle=False)
        fitmeta = json.loads((input_dir/(name+'_fit.json')).read_text())
        images = {'baseline': joint['baseline'], 'independent_fit': fit['fitted'],
                  'joint_fit': joint['joint'], 'metadata_completed': metadata['image']}
        points = fit['points']
        keep, targets, values = common_samples(joint['real'], images, points, sigma, shifts)
        target_lab = lab(targets)
        prediction_lab = {key: lab(value) for key, value in values.items()}
        replay = {
            'target_Lab_max_abs': float(np.max(abs(target_lab[0][keep]-fit['target_lab'][keep]))),
            'baseline_Lab_max_abs': float(np.max(abs(prediction_lab['baseline'][keep]-fit['baseline_lab'][keep]))),
            'independent_fit_Lab_max_abs': float(np.max(abs(prediction_lab['independent_fit'][keep]-fit['fitted_lab'][keep]))),
            'joint_Lab_max_abs': float(np.max(abs(prediction_lab['joint_fit'][keep]-joint['joint_lab'][keep]))),
            'metadata_Lab_max_abs': float(np.max(abs(prediction_lab['metadata_completed'][keep]-metadata['Lab'][keep]))),
        }
        if max(replay.values()) > 3e-4:
            raise ValueError('Cached measurement replay did not agree: '+str(replay))
        nt = fitmeta['baseline']['train']['samples']
        nv = fitmeta['baseline']['heldout']['samples']
        zones = {'train': np.arange(len(points)) < nt,
                 'heldout': (np.arange(len(points)) >= nt) & (np.arange(len(points)) < nt+nv),
                 'outer': np.arange(len(points)) >= nt+nv}
        region = {'name': name, 'original_samples': len(points),
                  'common_samples': int(keep.sum()), 'measurement_replay': replay,
                  'zones': {}}
        for zone, selected in zones.items():
            mask = keep & selected
            if not mask.any():
                raise ValueError('No common samples in '+zone)
            rows = []
            for shift_index, shift in enumerate(shifts):
                target = target_lab[shift_index][mask]
                paths = {}
                for key, prediction in prediction_lab.items():
                    delta = prediction[mask]-target
                    de = colour.delta_E(prediction[mask], target, method='CIE 2000')
                    paths[key] = {'median_deltaE00': float(np.median(de)),
                                  'mean_deltaE00': float(np.mean(de)),
                                  'Lab_RMSE': float(np.sqrt(np.mean(delta**2))),
                                  'mean_dLab': delta.mean(0).tolist()}
                before = paths['baseline']['median_deltaE00']
                for key in paths:
                    if key != 'baseline':
                        paths[key]['median_error_reduction'] = before-paths[key]['median_deltaE00']
                rows.append({'shift_dy_dx': shift, 'paths': paths})
            summary = {}
            for key in images:
                errors = np.array([row['paths'][key]['median_deltaE00'] for row in rows])
                entry = {'median_deltaE00_range': [float(errors.min()), float(errors.max())],
                         'zero_shift_deltaE00': float(errors[0])}
                if key != 'baseline':
                    gain = np.array([row['paths'][key]['median_error_reduction'] for row in rows])
                    entry.update({'error_reduction_range': [float(gain.min()), float(gain.max())],
                                  'improves_all_shifts': bool(np.all(gain > 0)),
                                  'worsens_all_shifts': bool(np.all(gain < 0)),
                                  'improving_shift_count': int((gain > 0).sum()),
                                  'worsening_shift_count': int((gain < 0).sum())})
                summary[key] = entry
            region['zones'][zone] = {'samples': int(mask.sum()), 'summary': summary, 'shifts': rows}
        report['regions'].append(region)
        np.savez_compressed(output/(name+'_fixed_samples.npz'), points=points, common_support=keep,
                            shifts=np.array(shifts), target_lab=target_lab,
                            **{key+'_lab': value for key, value in prediction_lab.items()})
    (output/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    plot(report, output/'registration_sensitivity.png')
    return report


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for row, region in enumerate(report['regions']):
        for col, zone in enumerate(('heldout', 'outer')):
            ax = axes[row, col]
            for key, label in [('baseline', 'Baseline'), ('joint_fit', 'Joint inverse'),
                               ('metadata_completed', 'Completed metadata')]:
                values = [item['paths'][key]['median_deltaE00']
                          for item in region['zones'][zone]['shifts']]
                ax.plot(values, '.-', label=label)
            ax.set(title=region['name']+' / '+zone, xlabel='Predeclared shift index (0 = unchanged)',
                   ylabel='Median deltaE00')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed inverse estimates: all registration controls retained, no refitting')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('/workspace/filmlab-audit/inverse-source-radiance'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/inverse-registration-audit'))
    args = parser.parse_args()
    report = run(args.input_dir, args.out)
    print(json.dumps({r['name']: {key: value['summary'] for key, value in r['zones'].items()}
                      for r in report['regions']}, indent=2))


if __name__ == '__main__':
    main()
