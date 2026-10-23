"""Geometry-only audit of finite Gaussian/bilinear measurement partitions.

No RGB values, fit residuals, physical parameters or optimized geometry enter.
Train/heldout isolation means zero intersection of *nonzero native columns* of
the actual sparse measurement operator, not merely separated output centres.
Radial labels describe centres unless every finite footprint is in that zone.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]


def _operator(operator):
    p = sparse.csr_matrix(operator, dtype=float, copy=True)
    p.eliminate_zeros()
    if not np.isfinite(p.data).all() or (p.data < 0).any():
        raise ValueError('Measurement operator must have finite nonnegative weights')
    if np.max(abs(np.asarray(p.sum(1)).ravel()-1.), initial=0.) > 1e-8:
        raise ValueError('Each measurement row must have complete unit weight')
    return p


def _rows(rows, count):
    rows = np.asarray(rows)
    if rows.dtype.kind == 'b':
        if rows.shape != (count,):
            raise ValueError('Boolean selection does not match measurement rows')
        rows = np.flatnonzero(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('Rows must be a one-dimensional integer index or boolean selection')
    if ((rows < 0) | (rows >= count)).any() or len(np.unique(rows)) != len(rows):
        raise ValueError('Measurement row indices must be distinct and in bounds')
    return rows


def native_support_overlap(operator, train_rows, heldout_rows):
    """Directly inspect actual P nonzero columns, retaining arbitrarily small weights."""
    p = _operator(operator)
    train, heldout = _rows(train_rows, p.shape[0]), _rows(heldout_rows, p.shape[0])
    left, right = p[train], p[heldout]
    train_columns, heldout_columns = np.unique(left.indices), np.unique(right.indices)
    common = np.intersect1d(train_columns, heldout_columns)
    left_mass = np.asarray(left[:, common].sum(1)).ravel()
    right_mass = np.asarray(right[:, common].sum(1)).ravel()
    return {'train_rows': len(train), 'heldout_rows': len(heldout),
            'train_native_columns': len(train_columns), 'heldout_native_columns': len(heldout_columns),
            'shared_native_columns': len(common), 'strictly_disjoint': len(common) == 0,
            'train_rows_touching_shared_columns': int((left_mass > 0).sum()),
            'heldout_rows_touching_shared_columns': int((right_mass > 0).sum()),
            'maximum_train_row_weight_on_shared_columns': float(left_mass.max(initial=0.)),
            'maximum_heldout_row_weight_on_shared_columns': float(right_mass.max(initial=0.))}, common


def project_native_centres(shape, matrix):
    yy, xx = np.indices(shape)
    homogeneous = np.asarray(matrix, float) @ np.vstack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    return (homogeneous[:2]/homogeneous[2]).T


def geometric_axis_guard(source_shape, matrix, sigma, output_shape=(512, 512)):
    """Conservative centre guard: Gaussian square + projected native neighbour.

    The maximum is measured on all valid bilinear rows at the fixed geometry.
    This contains mapped nonzero native pixel centres on their own side of an
    output axis. It is not a camera uncertainty estimate or a material scale.
    Always verify the actual chosen partitions by native_support_overlap too.
    """
    from corridor_cache_geometry import bilinear_operator
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    a, output_rows = bilinear_operator(source_shape, matrix, output_shape)
    coo = a.tocoo()
    native_xy = project_native_centres(source_shape, matrix)
    y, x = np.divmod(output_rows, output_shape[1])
    target_xy = np.column_stack([x, y])
    displacement = native_xy[coo.col]-target_xy[coo.row]
    extent = np.max(abs(displacement), axis=0)
    radius = int(4*sigma+.5)
    return {'gaussian_square_half_width_output_pixels': radius,
            'maximum_bilinear_native_centre_displacement_xy_output_pixels': extent.tolist(),
            'conservative_axis_guard_output_pixels': int(radius+np.ceil(extent.max())+1),
            'formula': 'round(4*sigma) + ceil(max absolute projected bilinear-neighbour displacement) + 1'}


def native_distance_field(source_shape, matrix, output_distance):
    """Map the frozen scan-distance label field to native sample centres.

    This diagnoses coordinate/label overlap; it is not a measured physical
    photon-origin radius. The original field is distance outside a fixed core.
    """
    xy = project_native_centres(source_shape, matrix)
    return map_coordinates(np.asarray(output_distance, float), xy[:, ::-1].T,
                           order=1, mode='constant', cval=np.nan, prefilter=False)


def radial_footprint_summary(operator, rows, points, output_distance, native_distance,
                             boundary=300., centre_side='near'):
    """Count finite native supports crossing the label boundary and their mass."""
    p = _operator(operator)
    rows = _rows(rows, p.shape[0])
    if centre_side not in ('near', 'outer'):
        raise ValueError('centre_side must be near or outer')
    points = np.asarray(points, int)
    if points.shape != (p.shape[0], 2):
        raise ValueError('Points must give y/x coordinates for every measurement row')
    distance = np.asarray(native_distance, float)
    if distance.shape != (p.shape[1],):
        raise ValueError('Native distance must match operator columns')
    selected = p[rows]
    centres = np.asarray(output_distance)[tuple(points[rows].T)]
    ranges, opposite = [], []
    for i in range(len(rows)):
        row = selected.getrow(i)
        d = distance[row.indices]
        if not np.isfinite(d).all():
            raise ValueError('A measurement footprint exits the fixed distance-label field')
        ranges.append([d.min(), d.max()])
        beyond = d > boundary if centre_side == 'near' else d <= boundary
        opposite.append(row.data[beyond].sum())
    ranges = np.asarray(ranges).reshape(-1, 2)
    opposite = np.asarray(opposite)

    def quantiles(values):
        return np.quantile(values, [0., .5, .95, 1.]).tolist() if len(values) else None

    return {'rows': len(rows), 'centre_side': centre_side, 'boundary_nominal_um': boundary,
            'centre_distance_min_median_max': np.quantile(centres, [0., .5, 1.]).tolist() if len(rows) else None,
            'centre_distance_histogram': {
                'edges_nominal_um': [0, 50, 100, 200, 300, 400, 500, 600, 800, 'inf'],
                'counts': np.histogram(centres, [0, 50, 100, 200, 300, 400, 500, 600, 800, np.inf])[0].tolist()},
            'native_footprint_distance_global_min_max': [float(ranges[:, 0].min()), float(ranges[:, 1].max())] if len(rows) else None,
            'rows_with_nonzero_weight_across_boundary': int((opposite > 0).sum()),
            'opposite_side_row_weight_min_median_p95_max': quantiles(opposite),
            'mean_opposite_side_row_weight': float(opposite.mean()) if len(rows) else None,
            'footprint_distance_ranges': ranges.tolist(),
            'opposite_side_row_weights': opposite.tolist()}


def audit_sample_file(source_file, samples_file, out_file, scan_um_per_pixel,
                      operator_file=None):
    """Read points and named group index arrays; no rendered RGB is consulted.

    Sample NPZ keys: points, train_near, train_outer, heldout_near, heldout_outer
    row indices (or boolean masks); alternatively groups encoded as
    0=inner_train, 1=inner_heldout, 2=outer_train, 3=outer_heldout. Source NPZ
    supplies matrix, q_base shape and distance_um. Optional saved sparse P must
    agree with the reconstructed fixed operator, then is used for the audit.
    Geometry and all sample groups remain fixed.
    """
    from inverse_source_radiance import observation_operator
    source, samples = np.load(source_file), np.load(samples_file)
    points = samples['points']
    shape = source['q_base'].shape[:2]
    sigma = 25/scan_um_per_pixel
    p = observation_operator(shape, source['matrix'], points, sigma)
    operator_replay_error = None
    if operator_file is not None:
        saved = _operator(sparse.load_npz(operator_file))
        if saved.shape != p.shape:
            raise ValueError('Saved sparse operator shape differs from fixed reconstruction')
        delta = saved-p
        operator_replay_error = float(np.max(abs(delta.data), initial=0.))
        if operator_replay_error > 1e-14:
            raise ValueError('Saved sparse operator differs from fixed reconstruction')
        p = saved
    keys = ('train_near', 'train_outer', 'heldout_near', 'heldout_outer')
    if all(key in samples for key in keys):
        groups = {key: _rows(samples[key], len(points)) for key in keys}
    else:
        encoding = np.asarray(samples['groups'])
        if encoding.shape != (len(points),) or not np.isin(encoding, [0, 1, 2, 3]).all():
            raise ValueError('Sample groups must encode 0/1/2/3 for every point')
        groups = {key: np.flatnonzero(encoding == code) for key, code in
                  [('train_near', 0), ('heldout_near', 1), ('train_outer', 2), ('heldout_outer', 3)]}
    train = np.concatenate([groups['train_near'], groups['train_outer']])
    heldout = np.concatenate([groups['heldout_near'], groups['heldout_outer']])
    report = {'scope': __doc__, 'source_file': str(source_file), 'sample_file': str(samples_file),
              'scan_um_per_pixel_nominal': scan_um_per_pixel, 'sigma_output_pixels': sigma,
              'saved_operator_file': str(operator_file) if operator_file is not None else None,
              'saved_operator_replay_max_absolute_error': operator_replay_error,
              'predeclared_axis_guard_output_pixels': int(samples['guard']) if 'guard' in samples else None,
              'operator_nonzero_weights': p.nnz, 'geometry_guard': geometric_axis_guard(shape, source['matrix'], sigma),
              'native_support': {}, 'radial_labels': {}}
    pairs = {'all_train_vs_all_heldout': (train, heldout),
             'train_near_vs_heldout_near': (groups['train_near'], groups['heldout_near']),
             'train_outer_vs_heldout_outer': (groups['train_outer'], groups['heldout_outer']),
             'train_near_vs_heldout_outer': (groups['train_near'], groups['heldout_outer']),
             'train_outer_vs_heldout_near': (groups['train_outer'], groups['heldout_near']),
             'near_train_vs_outer_train_descriptive': (groups['train_near'], groups['train_outer'])}
    for name, (left, right) in pairs.items():
        report['native_support'][name], _ = native_support_overlap(p, left, right)
    native_radius = native_distance_field(shape, source['matrix'], source['distance_um'])
    for name, rows in groups.items():
        report['radial_labels'][name] = radial_footprint_summary(
            p, rows, points, source['distance_um'], native_radius,
            centre_side='near' if name.endswith('near') else 'outer')
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('samples', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--scan-um-per-pixel', type=float, default=36000/5456)
    parser.add_argument('--operator', type=Path)
    args = parser.parse_args()
    result = audit_sample_file(args.source, args.samples, args.out, args.scan_um_per_pixel, args.operator)
    print(json.dumps(result['native_support'], indent=2))
