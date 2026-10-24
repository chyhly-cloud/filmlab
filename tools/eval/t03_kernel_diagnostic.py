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


def gray_chroma():
    import colour
    ref = emulsion('kodak_vision3_500t').stock.reference_illuminant
    rgb = np.tile(.184 * lights.light_rgb(spectral.illuminant(ref)), (64, 64, 1))
    eng = Engine(Params(**{**recipe('cinestill_800t_reference'), 'roll': None,
        'grain': 0, 'scene_lights': None, 'scene_illuminant': ref, 'scan_color_strength': 1.0}))
    out = eng.render(rgb)[32, 32]
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(out, 1e-6, 1), apply_cctf_decoding=False))
    return float(np.hypot(lab[1], lab[2]))


def diagnostics():
    previous = generation.active()
    try:
        generation.use('next_oblique')
        eng = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0}))
        densities = film.oblique_densities(eng.e.stock, eng.v)
        rho = film.bounce_retention(eng.e.stock, eng.v)
        result = {'status': 'stopped_spec_contradiction', 'px_um': 10.0,
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
                                      'total_ratio': rows[1]['total']/rows[0]['total']}
        th = np.linspace(1e-4, np.radians(89.5), 4000)
        r = 2*eng.v.base_um*np.tan(th)
        w = film._fresnel_unpolarized(th)*np.sin(th)*np.cos(th)**eng.v.forward*np.gradient(th)
        w *= film.oblique_attenuation(th, densities[0])
        result['red_single_bounce_angular_fraction_over_700'] = float(w[(r>700)&(r<=1500)].sum()/w[r<=1500].sum())
        result['gray_chroma'] = {}
        for g in ('next', 'next_oblique'):
            generation.use(g)
            result['gray_chroma'][g] = gray_chroma()
        return result
    finally:
        generation.use(previous)


if __name__ == '__main__':
    out = json.dumps(diagnostics(), indent=2) + '\n'
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(out)
    print(out)
