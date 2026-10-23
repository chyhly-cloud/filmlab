"""Read-only local metric of the fixed RAW-to-scan registration homography.

No optical kernel, source spectrum, source geometry or scan response is changed.
The reported micrometres use the pipeline's two independent nominal36mm width
conventions. A scene-registration homography is NOT film-plane calibration.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def project(matrix, xy):
    xy = np.asarray(xy, float)
    homogeneous = np.concatenate([xy, np.ones(xy.shape[:-1]+(1,))], axis=-1)
    mapped = homogeneous @ np.asarray(matrix, float).T
    return mapped[..., :2]/mapped[..., 2:]


def homography_jacobian(matrix, xy):
    matrix = np.asarray(matrix, float)
    point = np.asarray(xy, float)
    if matrix.shape != (3, 3) or point.shape != (2,):
        raise ValueError('Expected3x3 homography and2-vector coordinates')
    homogeneous = matrix @ np.r_[point, 1.]
    if abs(homogeneous[2]) < 1e-12:
        raise ValueError('Point is at the projective horizon')
    value = homogeneous[:2]/homogeneous[2]
    return (matrix[:2, :2]-value[:, None]*matrix[2, :2][None, :])/homogeneous[2]


def metric_summary(matrix, xy, model_um_per_pixel, scan_um_per_pixel):
    jacobian = homography_jacobian(matrix, xy)
    physical = jacobian*scan_um_per_pixel/model_um_per_pixel
    singular = np.linalg.svd(jacobian, compute_uv=False)
    physical_singular = np.linalg.svd(physical, compute_uv=False)
    return {'native_xy': np.asarray(xy).tolist(), 'scan_xy': project(matrix, xy).tolist(),
            'Jacobian_scan_pixels_per_native_pixel': jacobian.tolist(),
            'singular_scan_pixels_per_native_pixel': singular.tolist(),
            'nominal_physical_Jacobian': physical.tolist(),
            'nominal_physical_singular_values': physical_singular.tolist(),
            'nominal_area_scale': float(abs(np.linalg.det(physical))),
            'nominal_isotropic_area_equivalent_scale': float(np.sqrt(abs(np.linalg.det(physical)))),
            'anisotropy_ratio': float(singular[0]/singular[-1]),
            'mapped_nominal_circle_radii_um': {
                str(radius): (radius*physical_singular).tolist() for radius in (100, 235, 300, 800)}}


def run(cache_dir, output):
    cache_dir, output = Path(cache_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT/'tools/eval/epfl_800t_manifest.csv').open() as stream:
        manifest = next(row for row in csv.DictReader(stream) if row['scene_id'] == 'R1-02-3')
    width, height = map(int, manifest['film_dimensions'].split('x'))
    # These dimensions are the frozen original native half-size decode shape;
    # do not reread RAW or change its resolution during the audit.
    raw_height, raw_width = 1732, 2304
    model_um = 36000/max(raw_height, raw_width)
    scan_um = 36000/width
    report = {'scope': __doc__, 'scene': 'R1-02-3',
              'frozen_native_RAW_shape': [raw_height, raw_width],
              'manifest_scan_dimensions': [width, height],
              'model_um_per_pixel_nominal': model_um, 'scan_um_per_pixel_nominal': scan_um,
              'manifest_scan_processing': manifest['scan_processing'],
              'film_width_calibration_verified': False,
              'scan_has_possible_crop': '可能裁切' in manifest['scan_processing'],
              'regions': [], 'limits': [
                  'Registration H describes corresponding scene appearance, not independently measured film-plane coordinates.',
                  'The digital camera sensor width/FOV is not assumed equal to36mm film width.',
                  'The pipeline assigns36mm to both full RAW model width and saved scan width; the saved scan may be cropped.',
                  'Scan dimensions and ICC/EXIF do not establish untrimmed film-gate dimensions or film camera intrinsics.',
                  'A homography cannot exactly register a depth-varying3D scene with different camera viewpoints/lenses.',
                  'No fitted material coefficient, kernel replacement or geometric correction is authorized by these numbers.',
                  'Earlier translation-only controls do not test systematic radial scale changes.',
              ]}
    for name in ('ceiling_lamp', 'middle_ceiling'):
        a = np.load(cache_dir/(name+'_source.npz'), allow_pickle=False)
        matrix = a['matrix']
        center = project(np.linalg.inv(matrix), np.array([256., 256.]))
        yy, xx = np.where(a['support'])
        source_center = np.array([xx.mean(), yy.mean()])
        center_info = metric_summary(matrix, center, model_um, scan_um)
        source_info = metric_summary(matrix, source_center, model_um, scan_um)
        # Declared800um neighbourhood, independent of RGB or residual values.
        offsets = np.linspace(-800/model_um, 800/model_um, 5)
        samples = [metric_summary(matrix, center+[dx, dy], model_um, scan_um)
                   for dy in offsets for dx in offsets]
        singulars = np.array([item['nominal_physical_singular_values'] for item in samples])
        anisotropy = np.array([item['anisotropy_ratio'] for item in samples])
        # Independently verify the analytic derivative in native-pixel units.
        h = 1e-3
        finite = np.stack([(project(matrix, center+h*unit)-project(matrix, center-h*unit))/(2*h)
                           for unit in np.eye(2)], axis=-1)
        error = float(np.max(abs(finite-homography_jacobian(matrix, center))))
        if error > 1e-8:
            raise ValueError('Analytic homography Jacobian failed finite differences')
        report['regions'].append({'name': name, 'fixed_local_H': matrix.tolist(),
            'at_scan_crop_center': center_info, 'at_strict_source_centroid': source_info,
            'grid_native_points': [item['native_xy'] for item in samples],
            '800um_grid_nominal_singular_min': singulars.min(0).tolist(),
            '800um_grid_nominal_singular_max': singulars.max(0).tolist(),
            '800um_grid_anisotropy_minmax': [float(anisotropy.min()), float(anisotropy.max())],
            'analytic_vs_finite_difference_max': error})
    (output/'geometry-scale.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path('/workspace/filmlab-audit/inverse-source-radiance'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/inverse-registration-audit'))
    args = parser.parse_args()
    report = run(args.cache_dir, args.out)
    print(json.dumps({row['name']: row['at_scan_crop_center'] for row in report['regions']}, indent=2))


if __name__ == '__main__':
    main()
