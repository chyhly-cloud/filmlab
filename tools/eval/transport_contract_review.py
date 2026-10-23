"""Independent numerical audit of the existing conditional halation kernel.

No real images, inferred material coefficients, development, or scanner outputs.
The reference integrates the *same declared angular model*, with no 800T claim.
All powers are dimensionless conditional input-energy fractions, not layer H.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import film

THETA_MIN = 1e-4
THETA_MAX = np.radians(89.5)


def reflected_fraction(mu, refractive_index=1.5):
    """Independent unpolarized internal Fresnel expression in direction cosine."""
    mu = np.asarray(mu, dtype=float)
    sin2_transmitted = refractive_index ** 2 * (1 - mu ** 2)
    cos_transmitted = np.sqrt(np.maximum(1 - sin2_transmitted, 0))
    rs = ((refractive_index * mu - cos_transmitted)
          / (refractive_index * mu + cos_transmitted)) ** 2
    rp = ((mu - refractive_index * cos_transmitted)
          / (mu + refractive_index * cos_transmitted)) ** 2
    return np.where(sin2_transmitted >= 1, 1., (rs + rp) / 2)


def angular_normalization(forward=1.5):
    return (forward + 1) / (np.cos(THETA_MIN) ** (forward + 1)
                            - np.cos(THETA_MAX) ** (forward + 1))


def density_xy(radius, base_um=130., forward=1.5, refractive_index=1.5):
    """Conditional reflected energy / um² from the exact polar Jacobian.

    p(theta)=C sin(theta) cos(theta)^forward;
    r=2t tan(theta), so q(r)=C R(theta) cos(theta)^(forward+3)/(8 pi t²).
    """
    radius = np.asarray(radius, dtype=float)
    mu = 1 / np.sqrt(1 + (radius / (2 * base_um)) ** 2)
    density = (angular_normalization(forward) * reflected_fraction(mu, refractive_index)
               * mu ** (forward + 3) / (8 * np.pi * base_um ** 2))
    support = (radius >= 2 * base_um * np.tan(THETA_MIN)) & (radius <= 2 * base_um * np.tan(THETA_MAX))
    return np.where(support, density, 0.)


def angular_disk_mass(radius, base_um=130., forward=1.5, refractive_index=1.5):
    upper = min(float(np.arctan(radius / (2 * base_um))), THETA_MAX)
    critical = np.arcsin(1 / refractive_index)
    return quad(lambda theta: float(angular_normalization(forward)
                * np.sin(theta) * np.cos(theta) ** forward
                * reflected_fraction(np.cos(theta), refractive_index)),
                THETA_MIN, upper, epsabs=1e-11, points=[critical] if THETA_MIN < critical < upper else None)[0]


def radial_disk_mass(radius, base_um=130., forward=1.5, refractive_index=1.5):
    critical_radius = 2 * base_um * np.tan(np.arcsin(1 / refractive_index))
    minimum = 2 * base_um * np.tan(THETA_MIN)
    return quad(lambda r: float(2 * np.pi * r * density_xy(r, base_um, forward, refractive_index)),
                minimum, radius, epsabs=1e-11,
                points=[critical_radius] if minimum < critical_radius < radius else None)[0]


def pixel_integral(px_um, radius_limit=1500., order=16):
    """Integrate q over pixel squares intersected with the declared circular domain.

    Gauss-Legendre quadrature per pixel; no renormalization. The discarded tail is
    intentionally retained as an overflow ledger, not spread over the raster.
    The source is a point at its pixel centre. Finite-area source pixels require
    an additional source-footprint integral; this is not the unique exact raster
    kernel under every production-image sampling convention.
    """
    n = int(np.ceil(radius_limit / px_um))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros_like(xx, dtype=float)
    cdfs = {r: np.zeros_like(xx, dtype=float) for r in (100., 300., 800., radius_limit)}
    for i, node_y in enumerate(nodes):
        for j, node_x in enumerate(nodes):
            radius = np.hypot(yy + node_y * px_um / 2, xx + node_x * px_um / 2)
            contribution = density_xy(radius) * (weights[i] * weights[j] * px_um ** 2 / 4)
            result += np.where(radius <= radius_limit, contribution, 0)
            for threshold in cdfs:
                cdfs[threshold] += np.where(radius <= threshold, contribution, 0)
    return result, {r: float(a.sum()) for r, a in cdfs.items()}


def area_fraction_inside(px_um, radius, shape, order=24):
    """Uniform raster-pixel measurement aperture, separate from physical transport."""
    n = shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    # Midpoints avoid quadrature weighting of the discontinuous aperture boundary.
    offsets = ((np.arange(order) + .5) / order - .5) * px_um
    result = np.zeros(shape)
    for oy in offsets:
        for ox in offsets:
            result += (xx + ox) ** 2 + (yy + oy) ** 2 <= radius ** 2
    return result / order ** 2


def evaluate(px_um, reference_orders=(8, 16, 32)):
    old = film.halation_kernel(px_um)
    n = old.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px_um
    radius = np.hypot(xx, yy)
    entirely_outside_disk = radius > 1500 + np.sqrt(2) * px_um / 2
    exact_mass = angular_disk_mass(1500)
    references = []
    previous = None
    for order in reference_orders:
        k, cdfs = pixel_integral(px_um, order=order)
        references.append({'order': order, 'mass': float(k.sum()),
                           'mass_minus_angular_reference': float(k.sum() - exact_mass),
                           'L1_change_from_previous_over_exact_mass': None if previous is None else float(np.abs(k - previous).sum() / exact_mass),
                           'cdf_absolute_input_energy': cdfs})
        previous = k
    reference = previous
    old_cdfs = {}
    for r in (100., 300., 800., 1500.):
        mask = area_fraction_inside(px_um, r, old.shape)
        old_cdfs[r] = {'pixel_area_measured_mass': float((old * mask).sum()),
                       'center_mask_mass': float(old[radius <= r].sum()),
                       'independent_continuous_mass': angular_disk_mass(r)}
    return {'px_um': px_um, 'shape': list(old.shape), 'old_mass': float(old.sum()),
            'angular_disk_reference_mass': exact_mass,
            'old_outside_center_mask_fraction': float(old[radius > 1500].sum() / old.sum()),
            'old_wholly_outside_disk_fraction': float(old[entirely_outside_disk].sum() / old.sum()),
            'old_L1_difference_from_integrated_reference_over_exact_mass': float(np.abs(old - reference).sum() / exact_mass),
            'old_cdf': old_cdfs, 'reference_convergence': references}, old, reference


def convolution_ledger(kernel):
    """Zero padding is an open boundary, not a closed-frame energy error."""
    image = np.zeros((41, 41)); image[20, 20] = 1.
    full = fftconvolve(image, kernel, mode='full')
    same = fftconvolve(image, kernel, mode='same')
    return {'kernel_sum': float(kernel.sum()), 'full_convolution_sum': float(full.sum()),
            'retained_frame_sum': float(same.sum()),
            'outside_frame_sum': float(full.sum() - same.sum()),
            'closure_error': float(full.sum() - kernel.sum())}


def multibounce_aperture_probe(px_um, rho):
    """Same analytic reference for the existing four-return red-layer proxy.

    This is a weighted-kernel numerical check, not a closed photon budget.
    """
    kernel = film.multi_bounce_kernel(px_um, 130., 1.5, rho)
    n = kernel.shape[0] // 2
    measurements = []
    for radius in (100., 300., 800.):
        m = min(int(np.ceil(radius / px_um)) + 1, n)
        crop = kernel[n-m:n+m+1, n-m:n+m+1]
        measured = float((crop * area_fraction_inside(px_um, radius, crop.shape)).sum())
        exact = sum(rho ** (bounce - 1) * angular_disk_mass(radius, base_um=130 * bounce)
                    for bounce in range(1, 5))
        measurements.append({'radius_um': radius, 'old_pixel_area_aperture_mass': measured,
                             'analytic_weighted_mass': exact,
                             'relative_difference': measured / exact - 1})
    return {'px_um': px_um, 'rho': rho, 'old_mass': float(kernel.sum()),
            'apertures': measurements}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/autonomous-round/transport'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for px in (15.625, 31.25, 62.5, 70.3125):
        row, old, reference = evaluate(px)
        np.savez_compressed(args.out / f'kernels_{px:g}um.npz', old=old, integrated_reference=reference)
        cases.append(row)
    exact = angular_disk_mass(1500)
    all_reflected = angular_disk_mass(2 * 130 * np.tan(THETA_MAX))
    from filmlab import variation
    from filmlab.materials import emulsion
    rho = film.bounce_retention(emulsion('kodak_vision3_500t').stock, variation.NOMINAL)
    report = {'scope': __doc__,
              'pixel_semantics': 'Reference: point source at pixel centre to receiver pixel area. Finite-area source pixels require a further source-footprint integral; no universal replacement-kernel claim.',
              'independent_jacobian_integral_difference': radial_disk_mass(1500) - exact,
              'conditional_energy_ledger': {'reflected_inside_1500um': exact,
                  'reflected_beyond_1500um': all_reflected - exact,
                  'back_surface_escaped': 1 - all_reflected},
              'production_sha256': hashlib.sha256(Path(film.__file__).read_bytes()).hexdigest(),
              'cases': cases,
              'finite_frame_ledger': convolution_ledger(film.halation_kernel(15.625)),
              'nominal_5219_proxy_rho': rho,
              'active_red_multibounce_apertures': [multibounce_aperture_probe(px, rho[0]) for px in (15.625, 70.3125)],
              'decision': 'Numerical reference only. No production change or 800T attribution. Existing square support and circular normalization mismatch is a small identified implementation inconsistency; material and real-scene attribution remain frozen.'}
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'jacobian_error': report['independent_jacobian_integral_difference'],
        'cases': [{k:v for k,v in row.items() if k not in ('old_cdf','reference_convergence')} for row in cases],
        'ledger': report['finite_frame_ledger']},indent=2))


if __name__ == '__main__':
    main()
