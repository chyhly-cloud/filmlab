"""Unfitted factorial experiment on the moments of a developed emulsion."""
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import silver_process as film
import silver_sampling as sampling
from silver_validation_reference import read_points, sampled_apertures

OUT = film.ROOT / 'silver-emulsion'
PITCH = .15625
APERTURES = (5, 10, 20, 25)
SPLIT = 16
FINE = tuple(replace(p, per_um2=p.per_um2*SPLIT, optical_area_um2=p.optical_area_um2/SPLIT,
                     footprint_sigma_um=p.footprint_sigma_um/4) for p in film.COATING)


def add_slow(coating):
    return coating + tuple(replace(p, per_um2=p.per_um2*.5, capture=p.capture/(2**stops))
                           for stops in (2, 4) for p in coating)


CASES = {'baseline': film.COATING, 'fine': FINE, 'slow': add_slow(film.COATING),
         'both': add_slow(FINE)}


def render(coating):
    representatives = coating[:2]
    groups = [tuple(p for p in coating if p.footprint_sigma_um == q.footprint_sigma_um)
              for q in representatives]

    def emulsion(exposure, pitch, rng):
        depth = np.zeros(exposure.shape, dtype='f4')
        expected_log = np.zeros_like(depth)
        for group, (footprint, loss) in zip(groups, sampling.factors(pitch, representatives)):
            counts = np.zeros_like(depth)
            for rate in sampling.rates(exposure, group):
                counts += rate
            counts *= pitch**2
            silver = rng.poisson(counts).astype('f4')
            sampling.convolve_add(depth, silver, footprint)
            sampling.convolve_add(expected_log, counts, loss)
        return film.BASE_T * np.exp(-depth), film.BASE_T * np.exp(expected_log), None

    return emulsion


def model_row(coating, stops):
    def d(h, pitch=PITCH):
        return float(-np.log10(sampling.mean_transmission(h, pitch, coating=coating)))

    floor = d(0.)
    anchor = brentq(lambda h: d(h)-floor-.1, 0, 1e4)
    # Compare net densities at the measured D=1 gross patch, then explicitly
    # repeat at gross D=1 to expose the effect of different base+fog densities.
    patch = brentq(lambda h: d(h)-floor-(1-.318), 0, 1e4)
    gross_patch = brentq(lambda h: d(h)-1, 0, 1e4)
    def rms(h, pitch=PITCH, sigma=1.5, apertures=APERTURES, adjacent=True):
        return sampling.stats(h, pitch, optical_sigma=sigma, adjacent=adjacent,
                              apertures=apertures, coating=coating)['rms_D_linearized']

    levels = [d(anchor * 2**s)-floor for s in stops]
    output = dict(floor=floor, anchor=anchor, patch_exposure=patch, net_D=levels,
                  ceiling=d(1e4)-floor, rms=rms(patch), rms_gross_1=rms(gross_patch),
                  rms_optics=[rms(patch,sigma=s) for s in (0.,1.5,3.)])
    if coating == CASES['both']:
        output['rms_sampled'] = sampled_apertures(patch, render(coating), APERTURES)
        output['rms_30um_grid'] = [
            dict(pitch_um=p, gross_D=d(patch,p), rms=rms(patch,p,film.SCAN_SIGMA,(30,),False)[0])
            for p in (2.5,.625,PITCH)]
    return output


def analyze():
    points, apertures, hashes = read_points()
    floor = .318
    origin = float(np.interp(floor+.1, points[:,1], points[:,0]))
    stops = (points[:,0]-origin)/np.log10(2)
    observed_D = points[:,1]-floor
    observed_rms = [row['published_rms_D'] for row in apertures]
    models = {}
    for name, coating in CASES.items():
        m = model_row(coating, stops)
        m['density_rmse'] = float(np.sqrt(np.mean((np.array(m['net_D'])-observed_D)**2)))
        m['rms_error_relative'] = (np.array(m.get('rms_sampled',m['rms'])) / observed_rms-1).tolist()
        models[name] = m
        print(name, 'net D max', m['net_D'][-1], 'RMS', m.get('rms_sampled',m['rms']), flush=True)
    assert np.allclose(models['baseline']['net_D'],models['fine']['net_D'],atol=1e-5)
    assert np.allclose(models['slow']['net_D'],models['both']['net_D'],atol=1e-5)
    joint = models['both']
    grid = joint['rms_30um_grid']
    assert abs(grid[0]['gross_D']-grid[-1]['gross_D']) < .001
    assert abs(grid[0]['rms']/grid[-1]['rms']-1) < .08
    assert np.max(np.abs(np.array(joint['rms_sampled'])/joint['rms']-1)) < .12
    return dict(reference=dict(floor=floor, stops=stops.tolist(), net_D=observed_D.tolist(),
                               rms=observed_rms, aperture_um=APERTURES, hashes=hashes),
                models=models, fit_started=False,
                screen=dict(max_density_error_limit=.15, per_aperture_rms_relative_limit=.25,
                            density_pass=bool(np.max(np.abs(np.array(joint['net_D'])-observed_D))<=.15),
                            rms_pass=bool(np.max(np.abs(joint['rms_error_relative']))<=.25),
                            independent_raw_available=False), photos={}, errors={})


def key():
    sha = hashlib.sha256()
    for name in (__file__, sampling.__file__, film.__file__):
        sha.update(Path(name).read_bytes())
    return sha.hexdigest()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'photo':
        name = sys.argv[2]
        film.render_photo(name, 421+film.PHOTOS.index(name), destination=OUT,
                          emulsion=render(CASES['both']), model_key=key())
        return
    OUT.mkdir(exist_ok=True)
    report = OUT / 'measurements.json'
    data = analyze()
    film.write_json(data, report)
    for name in film.PHOTOS:
        for attempt in (1, 2):
            try:
                subprocess.run([sys.executable, '-u', __file__, 'photo', name], timeout=480, check=True)
                data['photos'][name] = json.loads((OUT / f'{name}.json').read_text())
                data['errors'].pop(name, None)
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                print(name, 'attempt', attempt, str(exc), flush=True)
                data['errors'][name] = str(exc)
        film.write_json(data, report)
    if data['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
