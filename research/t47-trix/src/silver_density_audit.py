"""Audit sensitivity curves and optical capacity before any stock adjustment."""
from dataclasses import replace
import json

import numpy as np
from scipy.optimize import brentq

import silver_emulsion as experiment
import silver_sampling as sampling
from silver_validation_reference import read_points

D_GATE = .10
PITCH = experiment.PITCH


def curve(name, coating, stops, real):
    def density(h):
        return float(-np.log10(sampling.mean_transmission(h, PITCH, coating=coating)))
    base = density(0.)
    anchor = brentq(lambda h: density(h)-base-.1, 0, 100)
    net = np.array([density(anchor*2**s)-base for s in stops])
    error = net-real
    peak = int(np.argmax(np.abs(error)))
    return dict(name=name,
                optical_depth_area_budget=sum(p.optical_area_um2*p.per_um2 for p in coating),
                net_capacity=density(1e4)-base, errors=error.tolist(), net=net.tolist(),
                last_error=float(error[-1]), max_abs_error=float(abs(error[peak])),
                worst_at_stops=float(stops[peak]), passed=bool(abs(error[peak])<=D_GATE))


def main():
    path = experiment.OUT / 'measurements.json'
    data = json.loads(path.read_text())
    stops = np.array(data['reference']['stops'])
    real = np.array(data['reference']['net_D'])
    # Keep total potential optical depth fixed while moving half its
    # site density to the two slower populations. This is not a silver assay.
    original = experiment.CASES['baseline']
    redistribute = tuple(replace(p, per_um2=p.per_um2*.5)
                         for p in experiment.CASES['slow'])
    cases = {**experiment.CASES, 'redistribute': redistribute}
    rows = [curve(name, coating, stops, real) for name, coating in cases.items()]
    assert np.isclose(rows[-1]['net_capacity'], rows[0]['net_capacity'])
    assert np.isclose(rows[-1]['optical_depth_area_budget'], rows[0]['optical_depth_area_budget'])
    assert np.isclose(rows[3]['optical_depth_area_budget'], 2*rows[0]['optical_depth_area_budget'])
    assert abs(rows[3]['last_error']-(data['models']['both']['net_D'][-1]-real[-1])) < .002
    _, apertures, _ = read_points()
    for row in apertures:
        row['sigma2_area'] = (row['published_rms_D']*row['aperture_um'])**2
    kodak = json.loads((experiment.OUT/'kodak.json').read_text())
    result = dict(screen_abs_D=D_GATE, cases=rows, stops=stops.tolist(), real=real.tolist(),
                  required_capacity_at_least=float(real[-1]-D_GATE),
                  grain_apertures=apertures, kodak=kodak,
                  kodak_residuals_status='not evaluated: model has no absolute lux-second or D-76 time calibration')
    experiment.film.write_json(result, experiment.OUT/'density-audit.json')


if __name__ == '__main__':
    main()
