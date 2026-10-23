"""Conditional ray-transport checks; no measured-800T claim or parameter fitting.

Uses the existing estimated stack. The alternative accounts for slanted return
paths and both interface reflections on later trips. It remains a ballistic
approximation: scattering is extinction, indices are equal inside the stack,
and the first downward path is normal before redistribution at the base.
Only upward returns are counted, as in the existing model; exposures on later
downward trips are omitted. This is not a complete radiative-transfer solver.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import film, lights, variation
from filmlab.materials import emulsion
from filmlab.stocks import WAVELENGTHS


def return_stack(stock, v, mu):
    """Beer-Lambert upward absorption and full-stack transmission at cos(theta)."""
    s = 10 ** stock.log_sensitivity
    absorb = np.asarray(v.layer_absorbance)[None, :] * s / s.max(0)
    turb = .15 * (550 / WAVELENGTHS) ** 3
    total = absorb + turb[:, None]
    mask = np.maximum(stock.base_density - stock.base_density.min(), 0)
    yellow = v.yellow_filter / (1 + np.exp((WAVELENGTHS - 505) / 9))
    path = 1 / np.asarray(mu).reshape(-1, 1)
    trans = lambda d: 10 ** (-path * d[None, :])
    absorption = (1 - 10 ** (-path[..., None] * total[None, ...])) * (absorb / total)[None, ...]
    back = absorption.copy()
    back[..., 1] *= trans(total[:, 0] + mask / 2)
    back[..., 2] *= trans(total[:, 0] + total[:, 1] + yellow + mask)
    transmission = trans(total.sum(-1) + yellow + mask)
    return back, transmission


def radial_transport(stock, v, spectrum, corrected, count=6000):
    theta = (np.arange(count) + .5) * (np.pi / 2) / count
    mu = np.cos(theta)
    probability = np.sin(theta) * mu ** v.forward
    probability /= probability.sum()
    reflect = film._fresnel_unpolarized(theta)
    down, direct, back_normal = film.emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)
    unit = (spectrum @ (10 ** stock.log_sensitivity)) / (spectrum @ direct)
    back, trans = return_stack(stock, v, mu)
    rho = np.asarray(film.bounce_retention(stock, v))
    edges = np.arange(0, 6005, 5.)
    energy = np.zeros((len(edges) - 1, 3))
    first = np.zeros_like(energy)
    for bounce in range(1, 5):
        if corrected:
            spectral_weight = down[None, :, None] * back * trans[..., None] ** (2 * (bounce - 1))
            amplitude = np.einsum('l,tlc->tc', spectrum, spectral_weight) * unit
            amplitude *= (probability * reflect ** (2 * bounce - 1))[:, None]
        else:
            amplitude = (spectrum @ (down[:, None] * back_normal)) * unit * rho ** (bounce - 1)
            if bounce > 1:
                amplitude *= rho > .01  # Same multiple-return gate as film.halation.
            amplitude = (probability * reflect)[:, None] * amplitude[None, :]
        amplitude *= v.diffuse_fraction
        radius = 2 * v.base_um * bounce * np.tan(theta)
        for c in range(3):
            weight = amplitude[:, c]
            if not corrected and rho[c] <= .01:
                weight = weight * (radius <= 1500)
            binned = np.histogram(radius, edges, weights=weight)[0]
            energy[:, c] += binned
            if bounce == 1: first[:, c] = binned
    return edges, energy, first


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    stock = emulsion('kodak_vision3_500t').stock; v = variation.NOMINAL
    entries = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    selected = [('2700K', lights._blackbody(2700)), ('HPS', lights._sd('HPS')),
                ('LED-B3', lights._sd('LED-B3'))]
    for col, (name, spectrum) in enumerate(selected):
        # Known incident spectrum, equal photopic Y; no RGB reconstruction.
        spectrum = spectrum / (spectrum @ lights.spectral.cmfs())[1]
        for corrected, label in ((False, 'Existing approximation'), (True, 'Slanted-path hypothesis')):
            edges, energy, first = radial_transport(stock, v, spectrum, corrected)
            centers = (edges[1:] + edges[:-1]) / 2
            area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            total = energy.sum(0)
            direct = spectrum @ (10 ** stock.log_sensitivity)
            entry = {'light': name, 'variant': label, 'halo_to_direct': (total / direct).tolist(),
                     'later_return_fraction': (1 - first.sum(0) / total).tolist(),
                     'energy_fraction_beyond_300um': (energy[centers > 300].sum(0) / total).tolist(),
                     'energy_fraction_beyond_800um': (energy[centers > 800].sum(0) / total).tolist()}
            entries.append(entry)
            axes[0, col].plot(centers, energy[:, 0] / area / direct[0], label=label)
            axes[1, col].plot(centers, np.cumsum(energy[:, 0]) / total[0], label=label)
        axes[0, col].set(title=name, xlim=(0, 1000), ylabel='Red-layer halo / direct / um²', xlabel='Radius (um)')
        axes[1, col].set(xlim=(0, 1500), ylabel='Cumulative red-layer halo fraction', xlabel='Radius (um)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Conditional optics audit: estimated stack, no fit; not an 800T calibration')
    fig.savefig(args.out / 'transport.png', dpi=180); plt.close(fig)
    back, transmission = return_stack(stock, v, np.linspace(.01, 1, 101))
    checks = {'max_absorption_plus_transmission': float((back.sum(-1) + transmission).max()),
              'normal_incidence_back_error': float(np.abs(back[-1] - film.emulsion_stack(stock, v.layer_absorbance, v.yellow_filter)[2]).max())}
    report = {'scope': __doc__, 'checks': checks, 'cases': entries}
    (args.out / 'transport.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
