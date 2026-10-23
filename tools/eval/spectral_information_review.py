"""One controlled metamer construction, without fitting film or a target image.

Four fixed nonnegative source SPDs provide one null direction for each of three
specified, linear, three-channel observers. Splitting that direction into its
positive and negative parts gives additive mixtures with identical observations.
One COMMON gain scales both mixtures; neither is independently white-balanced,
normalised or exposed. The film response is the frozen 5219-derived model and
the return response is its uncalibrated optical proxy, not measured 800T.

This is an input identifiability diagnostic, not a correction or real-scene fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import colour
from filmlab import generation, lights, spectral
from filmlab.engine import Engine, Params
from filmlab.incident import spectral_responses
from filmlab.recipes import recipe
from filmlab.stocks import WAVELENGTHS


def construct_pair(sources, observer):
    """Solve only the observation equality, never an exposure/colour target."""
    A = (sources @ observer).T
    _, singular, vh = np.linalg.svd(A, full_matrices=True)
    if singular[-1] <= singular[0] * 1e-10:
        raise ValueError('Fixed source set does not independently span 3 observations')
    z = vh[-1]
    if z[np.flatnonzero(np.abs(z) > 1e-12)[0]] < 0:
        z = -z
    weights = np.stack([np.maximum(z, 0), np.maximum(-z, 0)])
    pair = weights @ sources
    observation = pair @ observer
    # A shared gain chooses an arbitrary common signal unit. It does not hide
    # unequal brightness: equality holds before this gain as well as after it.
    common_gain = 1 / observation[:, 1].mean()
    raw_observation = observation.copy()
    pair *= common_gain
    weights *= common_gain
    observation = pair @ observer
    if not (np.min(pair) >= 0 and np.min(weights) >= 0):
        raise AssertionError('Nonnegative sources must generate nonnegative mixtures')
    np.testing.assert_allclose(observation[0], observation[1], rtol=2e-12, atol=2e-12)
    return pair, weights, {
        'observation_matrix_singular_values': singular.tolist(),
        'observation_matrix_condition_number': float(singular[0] / singular[-1]),
        'unscaled_observations': raw_observation.tolist(),
        'common_gain_applied_to_both': float(common_gain),
        'scaled_observations': observation.tolist(),
        'max_absolute_equality_residual': float(np.max(np.abs(observation[1] - observation[0]))),
        'relative_l2_equality_residual': float(np.linalg.norm(observation[1] - observation[0]) / np.linalg.norm(observation[0])),
        'null_weights_before_common_gain': z.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generation.use('next_sensor')
    engine = Engine(Params(**{**recipe('cinestill_800t'), 'grain': 0, 'roll': 1, 'frame': 0}))
    sens, return_proxy = spectral_responses(engine)
    names = ['2700 K theoretical blackbody', 'CIE LED-B3', 'CIE FL2', 'HPS library source']
    sources = np.stack([lights._blackbody(2700), lights._sd('LED-B3'), lights._sd('FL2'), lights._sd('HPS')])
    observers = [('CIE 1931 2 degree XYZ', spectral.cmfs(), 'Y')]
    for name in ['Nikon 5100 (NPL)', 'Sigma SDMerill (NPL)']:
        sd = colour.MSDS_CAMERA_SENSITIVITIES[name].copy().align(
            colour.SpectralShape(380, 780, 5),
            extrapolator_kwargs={'method': 'Constant', 'left': 0, 'right': 0})
        observers.append((name + ' raw spectral sensitivity', np.clip(sd.values, 0, None), 'G'))
    rows, spectra, weights_out, operators = [], [], [], []
    for name, observer, middle_channel in observers:
        pair, weights, row = construct_pair(sources, observer)
        direct = pair @ sens
        returned = pair @ return_proxy
        row.update(observer=name, common_signal_unit=middle_channel + '=1 for both mixtures',
                   mixture_coefficients=weights.tolist(), direct_layer_H=direct.tolist(),
                   direct_mixture_B_minus_A_stops=np.log2(direct[1] / direct[0]).tolist(),
                   proxy_return_layer_H=returned.tolist(),
                   proxy_return_mixture_B_minus_A_stops=np.log2(returned[1] / returned[0]).tolist(),
                   spectrum_l1_relative_difference=float(np.abs(pair[1] - pair[0]).sum() / pair.mean(0).sum()))
        if name.startswith('CIE'):
            # No cctf, clipping or rendering: the current ideal camera is a
            # fixed linear map of these identical XYZ values, hence identical.
            rgb = lights.camera_rgb(pair)
            row['ideal_linear_camera_rgb'] = rgb.tolist()
            row['ideal_linear_camera_rgb_equality_residual'] = float(np.max(np.abs(rgb[1] - rgb[0])))
        rows.append(row)
        spectra.append(pair)
        weights_out.append(weights)
        operators.append(observer)
    profile = Path(__file__).resolve().parents[2] / 'filmlab/data/profiles/kodak_vision3_500t.json'
    report = {
        'scope': __doc__,
        'fixed_sources': names,
        'wavelength_nm': WAVELENGTHS.tolist(),
        'signal_convention': '81 samples, 380..780nm at5nm; existing common engine summation convention, not SI radiance. One common gain per pair, no independent exposure or colour normalisation.',
        'observer_scope': 'CIE observer is the current ideal camera approximation. Nikon/Sigma are library measurements of other cameras; none is Panasonic G81. Actual sensor filters, lenses and calibration can differ.',
        'film_scope': 'Frozen next_sensor 5219-derived sensitivity and fixed-reference return proxy. No development, scanner, final RGB, actual photo, fitted material coefficient or candidate change.',
        'profile_sha256': hashlib.sha256(profile.read_bytes()).hexdigest(),
        'stock': engine.e.stock.key,
        'cases': rows,
        'decision': 'Identical three-channel measurements do not uniquely determine model layer/return exposure even within a small nonnegative source-mixture family. Additional unsaturated bracket exposures at the same channel sensitivities scale the same three observations and do not remove this ambiguity. This does not quantify the real corridor error or identify its cause; no physical correction justified.',
    }
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    np.savez(args.out / 'controlled_spectra.npz', wavelength_nm=WAVELENGTHS, basis_spectra=sources,
             paired_spectra=np.asarray(spectra), mixture_weights=np.asarray(weights_out),
             observer_operators=np.asarray(operators), frozen_film_response=sens,
             frozen_return_proxy=return_proxy)
    print(json.dumps([{k:r[k] for k in ['observer', 'relative_l2_equality_residual',
                     'direct_mixture_B_minus_A_stops', 'proxy_return_mixture_B_minus_A_stops']}
                      for r in rows], indent=2))


if __name__ == '__main__':
    main()
