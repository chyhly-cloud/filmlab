"""Run synthetic/analytic transport experiments; never calibrates real 800T."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.eval.photon_transport import Layer, angle_return, simulate


def synthetic_stack(wavelength):
    """Explicit illustrative coefficients: not Kodak/800T data or estimates."""
    i = [450, 550, 650].index(wavelength)
    mus = [.18, .12, .08][i]
    return [Layer('upper_active', 3, [.06, .015, .004][i], mus, .75),
            Layer('passive_filter', 1, [.35, .02, .002][i], 0),
            Layer('middle_active', 3, [.008, .06, .008][i], mus, .75),
            Layer('lower_active', 3, [.004, .012, .05][i], mus, .75),
            Layer('base', 130, .0001, 0)]


def summary(result):
    e = result['radial_edges']
    ret = result['radial_returned']
    # Thresholds are exact histogram edges, not centre-based approximations.
    return {'ledger': result['ledger'], 'absorbed_by_layer': result['absorbed'].tolist(),
            'rear_return_absorbed_by_layer': result['returned_absorbed'].tolist(),
            'return_energy_outside_300um': (ret[:, e[:-1] >= 300].sum(1)+result['returned_overflow']).tolist(),
            'return_energy_outside_800um': (ret[:, e[:-1] >= 800].sum(1)+result['returned_overflow']).tolist(),
            'returned_overflow': result['returned_overflow'].tolist(),
            'photons': result['photons'], 'events': result['events'], 'seed': result['seed']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--photons', type=int, default=100_000)
    ap.add_argument('--mcml-sample', type=Path)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = {'scope': __doc__, 'material_evidence': 'all material coefficients are hypothetical',
              'angle_cases': [], 'synthetic_wavelengths': [], 'limits': [
                  'steady monochromatic incoherent transport; planar infinite layers',
                  'HG is a stipulated scattering phase function, not measured film scattering',
                  'energy absorption is not latent-image exposure or developed dye density',
                  'no fluorescence, polarization, wave interference, lens or scanner',
                  'no material fit and no adoption into current/next']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, passive in zip(axes, (0., .4)):
        for angle, label in [(False, 'Normal-depth separable control'), (True, 'Oblique absorption/capture')]:
            a = angle_return(passive, .1, angle_dependent=angle)
            order = np.argsort(a['radius'])
            cumulative = np.cumsum(a['energy'][order])/a['energy'].sum()
            ax.plot(a['radius'][order], cumulative, label=label)
            report['angle_cases'].append({'passive_optical_depth': passive, 'active_optical_depth': .1,
                'angle_dependent': angle, 'ledger': a['ledger'],
                'tail_fraction_over_800um': float(a['energy'][a['radius'] > 800].sum()/a['energy'].sum()),
                'tail_absolute_over_800um': float(a['energy'][a['radius'] > 800].sum())})
        ax.set(title=f'Passive optical depth = {passive}', xlabel='Base-front return radius (um)',
               ylabel='Fraction of active-layer return absorption', xlim=(0, 1600), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle('Hypothetical one-return experiment; same angular source and Fresnel boundary')
    fig.savefig(args.out/'angle_absorption.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for wl, color in [(450, 'blue'), (550, 'green'), (650, 'red')]:
        layers = synthetic_stack(wl)
        r = simulate(layers, photons=args.photons, seed=wl)
        row = {'wavelength_nm': wl, 'layers': [asdict(l) for l in layers], **summary(r)}
        report['synthetic_wavelengths'].append(row)
        np.savez_compressed(args.out/f'synthetic_{wl}nm.npz', **{k:v for k,v in r.items() if isinstance(v, np.ndarray)})
        edges = r['radial_edges']; center=(edges[1:]+edges[:-1])/2
        area = np.pi*(edges[1:]**2-edges[:-1]**2)
        # Only active absorbers, excluding filter/base; no photochemical yield applied.
        ret = r['radial_returned'][[0, 2, 3]].sum(0)
        total = r['returned_absorbed'][[0, 2, 3]].sum()
        axes[0].plot(center, ret/area, color=color, label=f'{wl} nm')
        axes[1].plot(edges[1:], np.cumsum(ret)/max(total, 1e-20), color=color, label=f'{wl} nm')
        print(f'synthetic {wl} nm done', flush=True)
    axes[0].set(xlim=(0, 1500), yscale='log', xlabel='Actual absorption radius (um)', ylabel='Returned energy / input / um²')
    axes[1].set(xlim=(0, 1500), ylim=(0, 1), xlabel='Actual absorption radius (um)', ylabel='Cumulative active-layer return fraction')
    axes[0].legend(); fig.suptitle('Synthetic scattering stack: measured radial packet deposition, no kernel fitting')
    fig.savefig(args.out/'synthetic_profiles.png', dpi=170); plt.close(fig)
    base = synthetic_stack(550)
    no_scatter = [Layer(l.name, l.thickness_um, l.mu_a, 0, l.g, l.n) for l in base]
    r = simulate(no_scatter, photons=args.photons, seed=100)
    report['zero_scatter_normal_pencil'] = summary(r)
    r = simulate(base, photons=args.photons, seed=101, n_bottom=base[-1].n)
    report['index_matched_rear'] = summary(r)
    # Repeated seeds and 4x photon-count comparison, no parameter search.
    sizes = [max(1000, args.photons//4), args.photons]
    convergence = []
    for size in sizes:
        rows = []
        for seed in range(4):
            r = simulate(base, photons=size, seed=1000+seed)
            rows.append([r['ledger']['absorbed'], r['ledger']['top_escape'],
                         r['ledger']['bottom_escape'], r['returned_absorbed'][[0, 2, 3]].sum()])
        convergence.append({'photons_per_seed': size, 'seeds': 4,
                            'columns': ['absorbed', 'top_escape', 'bottom_escape', 'active_return_absorbed'],
                            'replicates': rows, 'mean': np.mean(rows, axis=0).tolist(),
                            'standard_error': (np.std(rows, axis=0, ddof=1)/2).tolist()})
    report['convergence'] = convergence
    print('limits and convergence done', flush=True)
    if args.mcml_sample:
        # Reference is the published MCML sample.mco output, not re-executed C.
        sha = hashlib.sha256(args.mcml_sample.read_bytes()).hexdigest()
        if sha != '93b5b5750b36fce506210d6a62dcda119dcc144880cd89e7f17cd8a609cbf0b9':
            raise ValueError('Expected the pinned MCML sample.mco for the hardcoded reference layers')
        text = args.mcml_sample.read_text()
        rat = text.split('RAT #Reflectance, absorption, transmission.')[1].split('A_l')[0]
        target = np.array([float(l.split('#')[0]) for l in rat.splitlines() if l.split('#')[0].strip()])
        ref_layers = [Layer('1', 1000, .0001, .01, .9, 1.37),
                      Layer('2', 1000, .0001, .001, 0, 1.37),
                      Layer('3', 2000, .0002, .001, .7, 1.37)]
        rows = []; size=max(1000, args.photons//4)
        for seed in range(4):
            r = simulate(ref_layers, photons=size, seed=2000+seed, radial_edges=np.arange(0, 20010, 10))
            rows.append([r['ledger'][k] for k in ['specular_reflection', 'top_escape', 'absorbed', 'bottom_escape']])
        mean = np.mean(rows, axis=0); se=np.std(rows, axis=0, ddof=1)/2
        # Reference has 1M packets, with unknown replicate variance; bounded
        # packet tallies give a conservative 0.5/sqrt(N) standard error bound.
        limit=6*np.sqrt(se**2+.25/1_000_000)+5e-7
        report['mcml_published_sample'] = {'source': str(args.mcml_sample),
            'source_sha256': sha,
            'columns': ['specular_reflection', 'diffuse_reflection', 'absorption', 'transmission'],
            'reference_photons': 1_000_000, 'reference': target.tolist(),
            'photons_per_seed': size, 'seeds': 4, 'replicates': rows,
            'our_mean': mean.tolist(), 'our_standard_error': se.tolist(),
            'difference': (mean-target).tolist(), 'conservative_6sigma_limit': limit.tolist(),
            'consistent_with_sample': bool(np.all(np.abs(mean-target) <= limit)),
            'scope': 'published stochastic reference output, not a fresh execution of MCML; cm converted to um'}
    (args.out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
