"""Validate continuous Poisson disk transmission and paired scanner sampling."""
from ctypes import CDLL, c_double, c_int
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

import silver_process as film

OUT = film.ROOT / 'silver-emulsion'
PITCH, SIDE, FOV = .125, 416., 406.4
APERTURES = (5., 10., 20., 25., 203.2)
SCAN = (.25, 1., 2., 6.35, 8.69)
DENSITIES = (.682, 1.576)
TAUS = (.1, 1., np.inf)
GRAIN = {'single': (1.,), 'mixture': (.65, 1.25)}
LAGS = (0., 1., 2., 4.)
SEEDS = tuple(range(6401, 6417))
OPTICAL_SIGMA = 1.5
D_LIMIT, RMS_LIMIT, Z_LIMIT = .01, .05, 2.


@lru_cache(maxsize=None)
def weights(input_pitch, input_count, output_pitch, output_count):
    rows, cols, values = [], [], []
    for j in range(output_count):
        lo, hi = j * output_pitch, (j + 1) * output_pitch
        for i in range(int(lo/input_pitch), min(int(np.ceil(hi/input_pitch)), input_count)):
            width = min(hi, (i+1)*input_pitch)-max(lo, i*input_pitch)
            if width > 0:
                rows.append(j)
                cols.append(i)
                values.append(width/output_pitch)
    w = csr_matrix((values, (rows, cols)), shape=(output_count, input_count), dtype='f4')
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1, atol=1e-6)
    return w


def boxes(image, pitch, aperture, count):
    w = weights(pitch, image.shape[0], aperture, count)
    return (w @ (w @ image).T).T


def overlap(r, distance):
    q = np.clip(distance/(2*r), 0, 1)
    return 2*r*r*(np.arccos(q)-q*np.sqrt(1-q*q))


def expectation(density, tau, radii):
    mean = 10**-density
    attenuation = -np.expm1(-tau)
    rate = -np.log(mean)/(attenuation*np.mean(np.pi*np.square(radii)))
    def covariance(distance):
        common_area = sum(overlap(r, distance) for r in radii)/len(radii)
        return mean**2 * np.expm1(rate*attenuation**2*common_area)

    # Integrate the exact covariance against the separable square overlap.
    support = 2*max(radii)
    grid = np.linspace(0, support, 1001)
    cov = covariance(np.hypot(grid[:, None], grid))
    variances = []
    for a in APERTURES:
        axis = (a-grid)/a**2
        axis[[0, -1]] *= .5
        variances.append(float(4*(support/1000)**2 * (axis @ cov @ axis)))
    return rate, dict(mean=mean, point_covariance=covariance(np.array(LAGS)).tolist(),
                      aperture_T_variance=variances)


def d_stats(image):
    positive = image > 0
    zero_fraction = 1 - float(positive.mean())
    if zero_fraction:
        return dict(d_rms=None, adjacent_d_rms=None, zeros=zero_fraction)
    d = -np.log10(image)
    return dict(d_rms=float(np.var(d, dtype='f8')),
                adjacent_d_rms=float(np.mean(np.diff(d, axis=1)**2, dtype='f8')/2),
                zeros=0.)


def field(seed, kind, tau, stamp):
    rng = np.random.default_rng(seed)
    radii = GRAIN[kind]
    rate, _ = expectation(DENSITIES[-1], tau, radii)
    n = round(SIDE/PITCH)
    k = rng.poisson(rate*SIDE**2)
    particles = np.column_stack((rng.uniform(0, SIDE, (k,2)),
                                 rng.choice(radii, k), rng.random(k)))
    low, high = (np.zeros((n,n), dtype='u2') for _ in DENSITIES)
    stamp(low, high, n, PITCH, particles, k)
    output = []
    for density, counts in zip(DENSITIES, (low, high)):
        true_mean = 10**-density
        t = (counts == 0).astype('f4') if np.isinf(tau) else np.exp(-np.float32(tau)*counts)
        centered = t - true_mean
        point = [float(np.mean(centered*np.roll(centered, round(lag/PITCH), axis=1),
                               dtype='f8')) for lag in LAGS]
        raw = [boxes(t, PITCH, a, int(FOV/a + 1e-9)) for a in APERTURES]
        var = [float(np.mean((block-true_mean)**2, dtype='f8')) for block in raw]
        # Optics applied once to the common realization before simulated scans.
        blurred = gaussian_filter(t, OPTICAL_SIGMA/PITCH, mode='wrap')
        del t, centered
        scans = {}
        for pitch in (PITCH,) + SCAN:
            scan = boxes(blurred, PITCH, pitch, int(np.ceil(FOV/pitch)))
            macro = float(-np.log10(boxes(scan, pitch, FOV, 1)[0,0]))
            apertures = []
            for a in APERTURES:
                if a < pitch:
                    apertures.append(None)
                    continue
                block = boxes(scan, pitch, a, int(FOV/a + 1e-9))
                apertures.append(dict(cells=int(block.size), **d_stats(block)))
            scans[str(pitch)] = dict(macro=macro, apertures=apertures)
        output.append(dict(mean=float(np.mean(counts == 0, dtype='f8')) if np.isinf(tau)
                           else float(np.mean(np.exp(-np.float32(tau)*counts), dtype='f8')),
                           point_covariance=point, aperture_T_variance=var, scans=scans))
    return output


def estimates(values, theoretical):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/np.sqrt(values.shape[0])
    z = (mean-theoretical)/se
    return dict(expected=np.asarray(theoretical).tolist(), measured=mean.tolist(),
                seed_se=se.tolist(), z=z.tolist())


def aggregate(tiles, radii, tau):
    output = {}
    for j, density in enumerate(DENSITIES):
        items = [item[j] for item in tiles]
        _, theory = expectation(density, tau, radii)
        checks = {name: estimates([item[name] for item in items], v) for name,v in theory.items()}
        comparison = {}
        for pitch in SCAN:
            ref = [x['scans'][str(PITCH)] for x in items]
            scan = [x['scans'][str(pitch)] for x in items]
            dd = np.array([y['macro']-x['macro'] for x,y in zip(ref,scan)])
            entry = dict(max_abs_macro_delta=float(np.max(np.abs(dd))), apertures={})
            for i, a in enumerate(APERTURES):
                pairs = [(x['apertures'][i],y['apertures'][i]) for x,y in zip(ref,scan)]
                if pairs[0][1] is None:
                    entry['apertures'][str(a)] = {'status': 'below_scan_pixel'}
                    continue
                row = dict(cells_per_seed=pairs[0][1]['cells'],
                           adjacent_pairs_per_seed=pairs[0][1]['cells']-round(np.sqrt(pairs[0][1]['cells'])))
                for metric in ('d_rms', 'adjacent_d_rms'):
                    base, scanned = (np.array([p[k][metric] for p in pairs]) for k in (0,1))
                    assert np.all(np.isfinite(base)) and np.all(np.isfinite(scanned))
                    value = np.sqrt(scanned.mean()/base.mean())-1
                    loo = np.sqrt((scanned.sum()-scanned)/(base.sum()-base))-1
                    se = np.sqrt((len(loo)-1)*np.var(loo))
                    row[metric] = dict(relative=float(value), paired_jackknife_se=float(se),
                                       passed=bool(abs(value) <= RMS_LIMIT))
                entry['apertures'][str(a)] = row
            comparison[str(pitch)] = entry
        output[str(density)] = dict(exact=checks, sampling=comparison)
    return output


def main():
    source = Path(__file__).with_name('silver_field_counts.c')
    key = hashlib.sha256(source.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    cache = OUT / 'field-checkpoint.json'
    old = json.loads(cache.read_text()) if cache.exists() else {}
    work = old.get('cases', {}) if old.get('key') == key else {}
    with TemporaryDirectory(dir=OUT) as tmp:
        library = Path(tmp)/'silver.dylib'
        subprocess.run(['/usr/bin/clang','-O3','-dynamiclib',str(source),'-o',str(library)],
                       check=True, timeout=30)
        handle = CDLL(str(library))
        stamp = handle.silver_counts
        stamp.argtypes = [np.ctypeslib.ndpointer('u2',flags='C_CONTIGUOUS')]*2 + [
            c_int,c_double,np.ctypeslib.ndpointer('f8',flags='C_CONTIGUOUS'),c_int]
        stamp.restype = None
        for kind, radii in GRAIN.items():
            for tau in TAUS:
                name = f'{kind}/{tau:g}'
                if name not in work:
                    tiles = [field(seed, kind, tau, stamp) for seed in SEEDS]
                    work[name] = aggregate(tiles, radii, tau)
                    film.write_json(dict(key=key, cases=work), cache)
                    print(name, 'done', flush=True)
    film.write_json(dict(cases=work, seeds=SEEDS, reference_pitch_um=PITCH, fov_um=FOV,
                         apertures_um=APERTURES, optical_sigma_um=OPTICAL_SIGMA,
                         gates=dict(d_abs=D_LIMIT, rms_relative=RMS_LIMIT, pointwise_z=Z_LIMIT)),
                    OUT/'field-check.json')


if __name__ == '__main__':
    main()
