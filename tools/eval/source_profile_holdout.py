"""No-fit spatial-shape check of actual full-frame legacy reconstruction.

Only two transmitted-light apertures have independent short-exposure truth.
Camera/RGB signal shape is not a measured scene-radiance profile or lamp SPD.
Full-frame load_raw executes before cropping; no local reconstruction replay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zlib
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]


def shape_metrics(estimate, reference, mask):
    """Compare nonnegative spatial distributions on one explicitly common mask.

    Unit-sum normalization removes scalar amplitude. No shift, registration,
    smoothing or shape parameter is estimated using the reference.
    """
    estimate, reference, mask = np.asarray(estimate), np.asarray(reference), np.asarray(mask, bool)
    if estimate.shape != reference.shape or estimate.shape != mask.shape or estimate.ndim != 2:
        raise ValueError('Matching 2D fields and mask required')
    if not mask.any():
        raise ValueError('Empty common support')
    a, b = estimate[mask].astype(float), reference[mask].astype(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0) or min(a.sum(), b.sum()) <= 0:
        raise ValueError('Finite nonnegative nonzero signal required on common support')
    pa, pb = a/a.sum(), b/b.sum()
    coordinates = np.argwhere(mask)
    ca, cb = pa @ coordinates, pb @ coordinates
    return {'common_cells': int(mask.sum()),
            'unit_sum_total_variation': float(.5*np.sum(abs(pa-pb))),
            'centroid_estimate_yx_pixels': ca.tolist(),
            'centroid_reference_yx_pixels': cb.tolist(),
            'centroid_distance_pixels': float(np.linalg.norm(ca-cb)),
            'centroid_estimate_minus_reference_yx_pixels': (ca-cb).tolist(),
            'peak_to_mean_estimate': float(a.max()/a.mean()),
            'peak_to_mean_reference': float(b.max()/b.mean()),
            'peak_to_mean_ratio_estimate_over_reference': float((a.max()/a.mean())/(b.max()/b.mean())),
            'rms_radius_estimate_pixels': float(np.sqrt(pa @ np.sum((coordinates-ca)**2, axis=1))),
            'rms_radius_reference_pixels': float(np.sqrt(pb @ np.sum((coordinates-cb)**2, axis=1)))}


def reference_pair_fields(lamp, data):
    """Reproduce the existing acceptance/assignment order, retaining both views."""
    from radiance_censor_holdout import reference_fields
    truth, support, _, pairs = reference_fields(lamp, data)
    aout, bout, assigned = truth.copy(), truth.copy(), np.zeros(support.shape, bool)
    views = [0]+[i for i, alt in enumerate(lamp['alternatives'], 1) if alt['local_geometry']['pass']]
    order = sorted([(i,j) for at,i in enumerate(views) for j in views[at+1:]], key=lambda x: (0 in x,x))
    for i,j in order:
        a, b = data[f'{i}_camera'].astype(float), data[f'{j}_camera'].astype(float)
        keep = data[f'{i}_safe'] & data[f'{j}_safe'] & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        keep &= (a.min(-1)>.005) & (b.min(-1)>.005)
        error = abs(np.log2(np.maximum(a,1e-9)/np.maximum(b,1e-9)))
        keep &= (error<.15).all(-1) & ~assigned
        aout[keep], bout[keep] = a[keep], b[keep]
        assigned |= keep
    np.testing.assert_array_equal(assigned, support)
    np.testing.assert_array_equal(aout[support], truth[support])
    return aout, bout, support, pairs


def run(out, seeds=32):
    from filmlab import generation, rawio
    root = Path('/workspace/filmlab-audit')
    inputs = root/'measured-sources'
    metadata = json.loads((inputs/'lamp-measurements.json').read_text())
    lamps = [x for x in metadata['lamps'] if x['region']['name'] in ('roof_opening_1','roof_opening_2')]
    assert len(lamps)==2 and len({x['primary_file'] for x in lamps})==1
    src = Path('/workspace/filmlab-data/epfl/data/digital')/lamps[0]['primary_file']
    generation.use('next_sensor')
    capture = {}
    original = rawio.reconstruct_highlights
    def tap(rgb, clip, rng, l_clip=None):
        capture['before'] = rgb.copy()
        capture['clip'] = clip.copy()
        capture['l_clip'] = l_clip
        return original(rgb, clip, rng, l_clip)
    with patch.object(rawio, 'reconstruct_highlights', tap):
        full = rawio.load_raw(src, max_size=3000, seed=0)
    import rawpy
    from inverse_source_radiance import strict_full_cfa
    with rawpy.imread(str(src)) as raw:
        strict_full_frame, _ = strict_full_cfa(raw)
    assert strict_full_frame.shape==full.shape[:2]
    decoder = json.loads((inputs/'decoder.json').read_text())
    matrix = np.asarray(decoder['matrix']) @ rawio.XYZ_TO_SRGB.T
    inverse = np.linalg.inv(matrix)
    # Fixed paired seed arguments only describe algorithm randomness. Every
    # reconstruction still runs in the original full-frame component domain.
    algorithm_profiles = {lamp['region']['name']: [] for lamp in lamps}
    crc = zlib.crc32(str(src).encode())
    for seed in range(seeds):
        result = original(capture['before'], capture['clip'], np.random.default_rng(crc^seed), capture['l_clip'])
        if seed==0:
            np.testing.assert_array_equal(result, full)
        for lamp in lamps:
            x,y=lamp['region']['center']; half=lamp['region']['half']
            algorithm_profiles[lamp['region']['name']].append(
                (result[y-half:y+half,x-half:x+half]@rawio.LUMA).copy())
    del result
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': __doc__, 'source': str(src), 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
              'shape': list(full.shape), 'generation': 'next_sensor', 'load_raw_seed_argument': 0,
              'clip_luminance_seen': capture['l_clip'],
              'paired_seed_arguments': list(range(seeds)),
              'actual_api_seed0_replay_bitwise': True,
              'shape_normalization': 'Each estimate/reference separately has unit sum on exactly the same accepted cells; no refit, shift or rescaling chosen from truth.',
              'interval_interpretation': 'Between-reference differences are accepted-view discrepancies, not confidence intervals or radiance calibration.',
              'regions': [], 'limits': [
                  'Only transmitted-light apertures, not the unresolved corridor lamps.',
                  'The domain is actual primary-any-RGB-saturated cells with two safe accepted views, not the corridor strict-four-CFA support.',
                  'Camera proxy from decoded RGB includes digital transform, XYZ clipping, denoising and highlight reconstruction effects.',
                  'The linear-RGB Y proxy is the actual legacy source-profile definition up to global metering scale, but is not calibrated photopic scene radiance.',
                  'Unit-sum normalization removes total signal error and cannot certify source strength.',
                  'Missing coverage is excluded, never assigned a fabricated reference; reported centroids concern this conditional common mask.',
                  'Seed0 is the exact actual API realization; fixed-seed ranges describe algorithm randomness, never physical uncertainty.',
                  'Reference pairs were accepted using an existing 0.15EV agreement gate; their small remaining shape difference is descriptive and selection-conditioned, not an unbiased error bar.',
                  'No film, scanner, material, source parameter or empirical constant is fitted.']}
    for lamp in lamps:
        name = lamp['region']['scene']+'_'+lamp['region']['name']
        data = np.load(inputs/(name+'_measurements.npz'))
        ref1, ref2, support, pairs = reference_pair_fields(lamp, data)
        core, sat = data['mask'], data['0_sat'].astype(bool)
        domain = core & support & sat.any(-1)
        x,y = lamp['region']['center']; half=lamp['region']['half']
        sl = (slice(y-half,y+half), slice(x-half,x+half))
        np.testing.assert_array_equal(capture['clip'][sl]>.5, sat.any(-1))
        actual_rgb, before_rgb = full[sl], capture['before'][sl]
        strict_domain = domain & strict_full_frame[sl]
        actual_cam, before_cam = actual_rgb@inverse, before_rgb@inverse
        # Independent first/second accepted views define a check of observation
        # variation under the existing geometry and fixed EXIF normalization.
        variants = {'linear_RGB_Y_proxy': (actual_rgb@rawio.LUMA, before_rgb@rawio.LUMA,
                                          (ref1@matrix)@rawio.LUMA, (ref2@matrix)@rawio.LUMA)}
        variants.update({f'camera_proxy_{c}': (actual_cam[...,c], before_cam[...,c], ref1[...,c],ref2[...,c]) for c in range(3)})
        row = {'name': name, 'reference_pairs': pairs,
               'coverage': {'actual_saturated_supported_cells': int(domain.sum()),
                            'all_RGB_flagged_supported_cells': int((domain&sat.all(-1)).sum()),
                            'primary_missing_channels_with_reference': int((domain[...,None]&sat).sum()),
                            'primary_missing_channels_without_reference': int((core[...,None]&sat&~support[...,None]).sum())},
               'metrics': {}}
        for label, (estimate, before, truth, alternate) in variants.items():
            row['metrics'][label] = {'legacy_actual_seed0': shape_metrics(estimate, truth, domain),
                                     'before_prior': shape_metrics(before, truth, domain),
                                     'accepted_reference_pair_difference': shape_metrics(alternate, truth, domain)}
        est_y, before_y, truth_y, alt_y = variants['linear_RGB_Y_proxy']
        row['strict_four_CFA_seed0_control'] = {
            'cells': int(strict_domain.sum()),
            'scope': 'Smaller genuinely four-CFA-saturated subset of the aperture control; still not a corridor source.',
            'legacy_actual_seed0': shape_metrics(est_y,truth_y,strict_domain),
            'before_prior': shape_metrics(before_y,truth_y,strict_domain),
            'accepted_reference_pair_difference': shape_metrics(alt_y,truth_y,strict_domain)}
        seed_metrics = [shape_metrics(v, variants['linear_RGB_Y_proxy'][2], domain)
                        for v in algorithm_profiles[lamp['region']['name']]]
        keys = ['unit_sum_total_variation', 'centroid_distance_pixels',
                'peak_to_mean_ratio_estimate_over_reference', 'rms_radius_estimate_pixels']
        row['legacy_Y_shape_algorithm_seed_distribution'] = {
            'interpretation': 'Algorithm-only fixed-seed range; no truth-based seed choice and no physical coverage claim.',
            'quantiles': {key: dict(zip(['min','p05','median','p95','max'],
                                      np.quantile([m[key] for m in seed_metrics],[0,.05,.5,.95,1]).tolist())) for key in keys},
            'per_seed_metrics': seed_metrics}
        report['regions'].append(row)
        np.savez_compressed(out/(name+'.npz'), reference_camera1=ref1, reference_camera2=ref2,
                            independent_support=support, score_domain=domain, core=core,
                            strict_four_CFA_score_domain=strict_domain,
                            primary_saturation=sat, actual_seed0_rgb=actual_rgb, before_prior_rgb=before_rgb)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    plot_fields(out, report, matrix)
    print(json.dumps(report,ensure_ascii=False), flush=True)
    return report


def plot_fields(out, report, matrix):
    """Scientific signal-map comparison; no source image or output tint edits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from filmlab.rawio import LUMA
    fig, axes = plt.subplots(2,4,figsize=(12,7.2),layout='constrained')
    for row, region in enumerate(report['regions']):
        a=np.load(out/(region['name']+'.npz')); domain=a['score_domain']
        fields=[(a['reference_camera1']@matrix)@LUMA,
                (a['reference_camera2']@matrix)@LUMA,
                a['before_prior_rgb']@LUMA,a['actual_seed0_rgb']@LUMA]
        maps=[np.where(domain,field/field[domain].sum(),np.nan) for field in fields]
        vmax=max(float(np.nanmax(field)) for field in maps)
        for column,(title,field) in enumerate(zip(['Short reference 1','Short reference 2','Before prior','Actual legacy seed0'],maps)):
            ax=axes[row,column]
            image=ax.imshow(field*100,vmin=0,vmax=vmax*100,cmap='magma',interpolation='nearest')
            ax.set(xlim=(64,116),ylim=(116,64),title=title)
            ax.set_title(title,fontsize=11)
            if row==1:ax.set_xlabel('Native half-size pixel x',fontsize=9)
            if column==0:ax.set_ylabel(region['name']+'\nNative half-size pixel y')
        fig.colorbar(image,ax=axes[row].tolist(),label='% of supported-domain signal per pixel')
    fig.suptitle('RAW spatial input check: unit-sum linear RGB-Y proxy\nOnly common, independently corroborated primary-saturated cells; blanks are unscored',fontsize=13)
    fig.savefig(out/'source_profile_comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/source-profile-holdout'))
    p.add_argument('--seeds',type=int,default=32)
    args=p.parse_args(); run(args.out,args.seeds)
