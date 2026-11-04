"""Validate the scanner sampling of the unfitted silver-negative prototype."""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from scipy.signal import fftconvolve

import silver_process as film

OUT = film.ROOT / 'silver-validation'
PITCHES = (2.5, 1.25, .625, .3125, .15625)
APERTURES = (5, 10, 20, 25, 30, 50)
EXPOSURES = (0., .35, .7, 1.4, 2.8, 5.6, 11.2, 45.)


@lru_cache(maxsize=None)
def deficit(population, pitch):
    """Average the isolated grain's transmission loss over each numerical cell."""
    sigma = population.footprint_sigma_um
    radius = int(np.ceil(7 * sigma / pitch))
    centers = np.arange(-radius, radius + 1) * pitch
    nodes, weights = np.polynomial.legendre.leggauss(12)
    sample = centers[:, None] + nodes[None, :] * pitch / 2
    axis = np.exp(-.5 * (sample / sigma) ** 2)
    optical_depth = population.optical_area_um2 / (2 * np.pi * sigma**2)
    samples = -np.expm1(-optical_depth * axis[:, :, None, None] * axis[None, None, :, :])
    return np.einsum('ixjy,x,y->ij', samples, weights, weights) / 4


@lru_cache(maxsize=None)
def factors(pitch, coating=film.COATING):
    """Separable implementation of the two short symmetric convolutions."""
    kernels = []
    for population in coating:
        loss = deficit(population, pitch)
        pair = []
        for k in (-np.log1p(-loss), -loss):
            u, s, v = np.linalg.svd(k)
            keep = s > s[0] * 1e-6
            left = (u[:, keep] * s[keep]).T
            right = v[keep]
            assert np.max(np.abs(k - left.T @ right)) < 1e-6
            pair.append((left.astype('f4'), right.astype('f4')))
        kernels.append(pair)
    return kernels


def convolve_add(output, data, components):
    for y, x in zip(*components):
        horizontal = convolve1d(data, x, axis=1, mode='constant')
        output += convolve1d(horizontal, y, axis=0, mode='constant')


def rates(exposure, coating=film.COATING):
    for population in coating:
        _, eligible = film.probabilities(exposure, population)
        q_dev, q_fog = film.development(population, film.DEV_TIME)
        yield population.per_um2 * (eligible * q_dev + (1 - eligible) * q_fog)


def corrected_film(exposure, pitch, rng):
    depth = np.zeros(exposure.shape, dtype='f4')
    expected_log = np.zeros_like(depth)
    for rate, (footprint, loss) in zip(rates(exposure), factors(pitch)):
        counts = (rate * pitch**2).astype('f4')
        silver = rng.poisson(counts).astype('f4')
        convolve_add(depth, silver, footprint)
        convolve_add(expected_log, counts, loss)
    return film.BASE_T * np.exp(-depth), film.BASE_T * np.exp(expected_log), None


def mean_transmission(exposure, pitch, corrected=True, *, coating=film.COATING):
    if not corrected:
        return film.homogeneous(exposure, pitch)
    log_mean = sum(-rate * pitch**2 * deficit(p, pitch).sum()
                   for p, rate in zip(coating, rates(exposure, coating)))
    return film.BASE_T * np.exp(log_mean)


def shown_positive(transmission, pitch):
    # Keep the prior display curve exactly fixed for the full-frame A/B.
    return film.positive(transmission, pitch)


def stats(exposure, pitch, corrected=True, *, optical_sigma=film.SCAN_SIGMA,
          adjacent=False, apertures=APERTURES, coating=film.COATING):
    """Compute exact T covariance on each discrete lattice, then linearize D."""
    log_factor = None
    for p, rate in zip(coating, rates(exposure, coating)):
        loss = deficit(p, pitch) if corrected else -np.expm1(-film.kernel(p, pitch))
        ac = fftconvolve(loss, loss[::-1, ::-1], mode='full')
        if log_factor is None:
            # Both footprints share the larger population's support.
            radius = max(deficit(q, pitch).shape[0] - 1 for q in coating)
            log_factor = np.zeros((2 * radius + 1,) * 2, dtype='f8')
        offset = (log_factor.shape[0] - ac.shape[0]) // 2
        region = slice(offset, offset + ac.shape[0])
        log_factor[region, region] += rate * pitch**2 * ac
    relative_cov = np.expm1(log_factor)
    center = relative_cov.shape[0] // 2
    rms = []
    for size in apertures:
        b = round(size / pitch)
        if not np.isclose(b * pitch, size):
            raise ValueError('Aperture must be an integer number of cells')
        impulse = np.pad(np.ones(b) / b, int(np.ceil(5 * optical_sigma / pitch)))
        impulse = gaussian_filter(impulse, optical_sigma / pitch, mode='constant')
        ac = np.correlate(impulse, impulse, 'full')
        lag = np.arange(-center, center + 1)
        offsets = np.arange(1 - impulse.size, impulse.size)
        marginal = np.interp(lag, offsets, ac, left=0, right=0)
        if adjacent:
            other = np.interp(lag - b, offsets, ac, left=0, right=0)
            along_x = marginal - other
        else:
            along_x = marginal
        var_rel = np.einsum('i,ij,j->', marginal, relative_cov, along_x)
        rms.append(float(np.sqrt(max(var_rel, 0.)) / np.log(10)))
    return {'D_from_mean_T': float(-np.log10(mean_transmission(exposure, pitch, corrected, coating=coating))),
            'rms_D_linearized': rms}


def monte_carlo(exposure, pitch, aperture_um, seeds):
    size = round(450 / pitch)
    pad = round(25 / pitch)
    b = round(aperture_um / pitch)
    region = np.full((size + 2 * pad,) * 2, exposure, dtype='f4')
    values = []
    for seed in seeds:
        t, _, _ = corrected_film(region, pitch, np.random.default_rng(seed))
        light = gaussian_filter(t, film.SCAN_SIGMA / pitch)[pad:-pad, pad:-pad]
        scanned = film.aperture(light, b)
        values.append((-np.log10(scanned)).ravel())
    d = np.concatenate(values)
    return float(np.std(d)), int(d.size)


def experiment():
    rows = []
    for pitch in PITCHES:
        for exposure in EXPOSURES:
            rows.append({'pitch_um': pitch, 'exposure': exposure,
                         'original': stats(exposure, pitch, False),
                         'corrected': stats(exposure, pitch)})
    high = {r['pitch_um']: r for r in rows if r['exposure'] == 45}
    fine = high[PITCHES[-1]]['corrected']
    coarse = high[PITCHES[0]]['corrected']
    errors = {}
    for pitch, row in high.items():
        errors[str(pitch)] = {
            'density_absolute': abs(row['corrected']['D_from_mean_T'] - fine['D_from_mean_T']),
            'rms_relative': (np.array(row['corrected']['rms_D_linearized'])
                             / fine['rms_D_linearized'] - 1).tolist()}
    assert abs(coarse['D_from_mean_T'] - fine['D_from_mean_T']) < .001
    assert abs(errors['2.5']['rms_relative'][4]) < .05
    # Stochastic rerun checks the covariance calculation at the full-frame
    # output aperture. Each pitch uses independent realizations and 3,600 pixels.
    mc = []
    for pitch in (2.5, 1.25, .625):
        predicted = stats(11.2, pitch)['rms_D_linearized'][4]
        observed, samples = monte_carlo(11.2, pitch, 30, range(120, 136))
        assert abs(observed / predicted - 1) < .08
        mc.append(dict(pitch_um=pitch, predicted=predicted, observed=observed, samples=samples))
    print('Grid convergence passed at 30 µm. High exposure:', errors['2.5'], flush=True)
    return {'aperture_um': APERTURES, 'pitches_um': PITCHES, 'rows': rows, 'high_exposure_errors': errors,
            'monte_carlo_at_30um': mc, 'base_density': float(-np.log10(film.BASE_T)),
            'stock_parameters_changed': False}


def write_external():
    from silver_validation_reference import compare
    film.write_json(compare(stats, mean_transmission, corrected_film), OUT / 'reference.json')


def main():
    OUT.mkdir(exist_ok=True)
    data = experiment()
    write_external()
    film.write_json(data, OUT / 'measurements.json')
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['photos'] = {}
    for i, photo in enumerate(film.PHOTOS):
        data['photos'][photo] = film.render_photo(photo, 421 + i, destination=OUT,
                                                 emulsion=corrected_film, display=shown_positive, model_key=key)
        film.write_json(data, OUT / 'measurements.json')


if __name__ == '__main__':
    main()
