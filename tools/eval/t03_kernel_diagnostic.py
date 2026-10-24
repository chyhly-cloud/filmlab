"""Synthetic kernel/gray diagnostics only. Never loads scene data or acceptance."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filmlab import film, generation, lights, spectral
from filmlab.engine import Engine, Params
from filmlab.materials import emulsion
from filmlab.recipes import recipe


def gray_chroma(size=64):
    import colour
    ref = emulsion('kodak_vision3_500t').stock.reference_illuminant
    rgb = np.tile(.184 * lights.light_rgb(spectral.illuminant(ref)), (size, size, 1))
    eng = Engine(Params(**{**recipe('cinestill_800t_reference'), 'roll': None,
        'grain': 0, 'scene_lights': None, 'scene_illuminant': ref, 'scan_color_strength': 1.0}))
    out = eng.render(rgb)[32, 32]
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(out, 1e-6, 1), apply_cctf_decoding=False))
    return float(np.hypot(lab[1], lab[2]))


def angular_fractions(v, rho, density, samples=200000):
    """Independent midpoint quadrature in theta, including original bounce rule.

    Same A per ring; no rasterization or production kernel/attenuation helper.
    Denominator includes only each ring's historical radial support.
    """
    th = (np.arange(samples) + .5) * (np.radians(89.5) - 1e-4) / samples + 1e-4
    dth = (np.radians(89.5) - 1e-4) / samples
    ang = np.sin(th) * np.cos(th) ** v.forward
    ang /= (ang * dth).sum()
    w = film._fresnel_unpolarized(th) * ang * dth
    if density != 0:
        x = density * (1 / np.cos(th) - 1) * np.log(10)
        w *= -np.expm1(-x) / x
    bounces = 4 if rho > .01 else 1
    max_r = 1500 * bounces
    energies = np.zeros(4)
    for n in range(1, bounces + 1):
        r = 2 * v.base_um * n * np.tan(th)
        active = r <= max_r
        masks = (r <= 235, (r > 235) & (r <= 400), (r > 400) & (r <= 700), r > 700)
        for j, mask in enumerate(masks):
            energies[j] += (w[active & mask] * rho ** (n - 1)).sum()
    return energies / energies.sum()


def diagnostics():
    previous = generation.active()
    try:
        generation.use('next_oblique')
        eng = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0}))
        densities = film.oblique_densities(eng.e.stock, eng.v)
        rho = film.bounce_retention(eng.e.stock, eng.v)
        result = {'status': 'synthetic_checks_pending', 'px_um': 10.0,
                  'density': densities.tolist(), 'rho': list(rho), 'layers': {}}
        for c, name in enumerate('RGB'):
            rows = []
            for density in (0.0, densities[c]):
                k = film.layer_halation_kernel(10.0, eng.v, rho[c], density)
                yy, xx = np.indices(k.shape)
                r = np.hypot(yy-k.shape[0]//2, xx-k.shape[1]//2)*10.0
                masks = (r <= 235, (r > 235)&(r <= 400), (r > 400)&(r <= 700), r > 700)
                rows.append({'total': float(k.sum()), 'fractions': [float(k[m].sum()/k.sum()) for m in masks]})
            result['layers'][name] = {'before': rows[0], 'after': rows[1],
                                      'total_ratio': rows[1]['total']/rows[0]['total'],
                                      'angular_before': angular_fractions(eng.v, rho[c], 0).tolist(),
                                      'angular_after': angular_fractions(eng.v, rho[c], densities[c]).tolist()}
        th = np.linspace(1e-4, np.radians(89.5), 4000)
        r = 2*eng.v.base_um*np.tan(th)
        w = film._fresnel_unpolarized(th)*np.sin(th)*np.cos(th)**eng.v.forward*np.gradient(th)
        w *= film.oblique_attenuation(th, densities[0])
        result['red_single_bounce_angular_fraction_over_700'] = float(w[(r>700)&(r<=1500)].sum()/w[r<=1500].sum())
        result['gray_chroma'] = {}
        for g in ('next', 'next_oblique'):
            generation.use(g)
            result['gray_chroma'][g] = {str(size): gray_chroma(size) for size in (64, 256, 1024)}
        result['status'] = 'synthetic_checks_passed' if (
            all(abs(a-b) <= .005 for row in result['layers'].values()
                for key, angular in (('before', 'angular_before'), ('after', 'angular_after'))
                for a,b in zip(row[key]['fractions'],row[angular]))
            and all(c < .5 for values in result['gray_chroma'].values() for c in values.values())
        ) else 'synthetic_checks_failed'
        return result
    finally:
        generation.use(previous)


if __name__ == '__main__':
    out = json.dumps(diagnostics(), indent=2) + '\n'
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(out)
    print(out)
