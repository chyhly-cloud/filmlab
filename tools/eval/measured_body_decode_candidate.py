"""Restore float RAW colour on previously declared, fully measured lamp bodies.

Only body_mask & ~saturation.any(-1) changes. Strict-core inverse and accepted
minimum-input-Y partial-edge witnesses remain fixed. The historical local light
weights, chosen member spectra, global class cutoff and all downstream stages
are replayed, not re-estimated after the replacement. This isolates RAW colour
decoding; it neither identifies an emitter SPD nor repairs the RGB-to-spectrum
model. No film/scan samples select this mask or a parameter.
"""

from pathlib import Path
from unittest.mock import patch
import argparse
import gc
import json
import os
import sys
import time
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]
AUDIT = Path('/workspace/filmlab-audit')
NAMES = ('ceiling_lamp', 'middle_ceiling')


def fixed_local_exposure(engine, rgb, weights, shares, members):
    """Replay original _expose_mixed arithmetic with frozen illumination inputs."""
    from filmlab import film, lights
    direct = np.zeros_like(rgb, dtype=float)
    returning = np.zeros_like(rgb, dtype=float)
    for k, category in enumerate(lights.classes()):
        if shares[k] < .003:
            continue
        spectrum = members[category.key][1]
        method = ('spectral' if category.key == 'daylight' else
                  'linear_wp' if 'linear_white' in engine.p.corrections else 'linear')
        wk = weights[..., k, None]
        direct += wk * film.expose(rgb, film.exposure_response(
            engine.e.stock, spectrum, engine.p.lens_filter, method))
        returning += wk * film.halation_response(
            engine.e.stock, spectrum, engine.p.lens_filter, engine.v,
            engine._back_reflect, method)(rgb)
    return np.clip(direct, 1e-9, None), np.clip(returning, 0., None)


def capture_fixed_illumination(engine, inputs):
    """One original RAW decode and one original full-grid illumination estimate."""
    from scipy import ndimage
    from filmlab import lights, rawio
    import oracle_timing as ot

    expected_data = '/workspace/filmlab-data'
    if os.environ.get('FILMLAB_DATA') != expected_data:
        raise ValueError('Require FILMLAB_DATA='+expected_data+' to preserve historical path CRC')
    src, _, bias = ot.sources('R1-02-3')
    prior_capture = json.loads((AUDIT/'lamp-edge-input-audit/capture.json').read_text())
    if str(src) != prior_capture['raw']:
        raise ValueError('Original RAW path/CRC differs from the captured input')
    rgb = rawio.load_raw(src, max_size=3000)
    seed = zlib.crc32(b'R1-02-3') % 1000
    metered, note = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    scale = float(np.sum(metered*rgb)/np.sum(rgb*rgb))
    local = {}
    for name, item in inputs.items():
        xl, yl, xh, yh = item['source_bounds']
        sl = (slice(yl, yh), slice(xl, xh))
        local[name] = {'slice': sl, 'decode_replay': float(np.max(abs(rgb[sl]-item['postprior_rgb'])))}
        if local[name]['decode_replay'] > 2e-6 or abs(scale-float(item['scale'])) > 2e-10:
            raise ValueError('Legacy decoder or metering no longer replays cached inputs')
    del rgb
    captured = {}
    original_weights, original_pick = lights.class_weights, lights.pick_members
    original_nearest, original_zoom = lights.nearest_members, ndimage.zoom

    def weights(*args, **kwargs):
        value = original_weights(*args, **kwargs)
        captured['small_weights'] = value
        return value

    def members(function):
        def wrapped(*args, **kwargs):
            value = function(*args, **kwargs)
            captured['members'] = value
            return value
        return wrapped

    def zoom(value, factors, *args, **kwargs):
        result = original_zoom(value, factors, *args, **kwargs)
        if value is captured.get('small_weights'):
            h, w = metered.shape[:2]
            full = result[:h, :w]
            captured['zoom_factors'] = list(factors)
            captured['shares'] = np.array([float(full[..., k].mean()) for k in range(full.shape[-1])])
            for name, item in local.items():
                item['weights'] = full[item['slice']].copy()
        return result

    with patch.object(lights, 'class_weights', weights), \
            patch.object(lights, 'pick_members', members(original_pick)), \
            patch.object(lights, 'nearest_members', members(original_nearest)), \
            patch.object(ndimage, 'zoom', zoom):
        direct, returning = engine._expose_mixed(metered*2**engine.p.exposure_ev)
    for name, item in local.items():
        a = inputs[name]
        item['full_direct_replay'] = float(np.max(abs(direct[item['slice']]-a['old_direct'])))
        item['full_return_replay'] = float(np.max(abs(returning[item['slice']]-a['old_halo_source'])))
        if max(item['full_direct_replay'], item['full_return_replay']) > 2e-6:
            raise ValueError('Original full-grid exposure no longer matches cached input')
    frame_shape = list(metered.shape)
    del metered, direct, returning
    gc.collect()
    report = {'raw_path': str(src), 'FILMLAB_DATA': os.environ['FILMLAB_DATA'],
              'frame_shape': frame_shape, 'metering': note, 'metering_scale': scale,
              'exposure_ev': engine.p.exposure_ev,
              'small_weight_shape': list(captured['small_weights'].shape),
              'zoom_factors': captured['zoom_factors'],
              'global_class_share': captured['shares'].tolist(),
              'light_member_policy': engine.p.light_member,
              'members': {key: {'name': value[0], 'spectrum': np.asarray(value[1]).tolist()}
                          for key, value in captured['members'].items()}}
    return local, captured, report


def run(out):
    import colour
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import ProjectiveTransform
    from inverse_source_radiance import engine_for, observation_operator
    from lamp_edge_spectral_candidate import transport
    from brightness_response_audit import lab
    from light_acceptance import crop_warp
    from filmlab.spectral import srgb_encode

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_root = AUDIT/'lamp-edge-input-audit'
    edge_root = AUDIT/'lamp-edge-spectral-candidate'
    source_root = AUDIT/'inverse-source-radiance'
    inputs = {name: dict(np.load(input_root/(name+'.npz'))) for name in NAMES}
    engine, meta = engine_for()
    fixed, captured, capture_report = capture_fixed_illumination(engine, inputs)
    report = {'scope': __doc__, 'capture': capture_report, 'regions': [],
              'timing': meta['timing'], 'limits': [
                  'Only existing explicitly reviewed body polygons with all channels measured change.',
                  'Float camera-to-RGB matrix uses the existing decoder proxy, not a measured lamp SPD.',
                  'RAW averages still include lens glare, CFA mixing and noise; no denoise or PSF is inferred.',
                  'Historical illumination weights/member spectra remain a conditional RGB-to-layer mapping.',
                  'Prior same-scan strict-core inverse is inherited and remains an estimate.',
                  'Same-frame heldout checks do not identify true material or scanner parameters.',
                  'No new source geometry, kernel, material, DIR, exposure calibration or scan timing is selected.']}
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)

    def encode(image):
        return Image.fromarray(np.rint(srgb_encode(np.nan_to_num(image))*255).astype('uint8'))

    for name in NAMES:
        a = inputs[name]
        source = np.load(source_root/(name+'_source.npz'))
        previous = np.load(source_root/(name+'_fit.npz'))
        edge = np.load(edge_root/(name+'.npz'))
        fitmeta = json.loads((source_root/(name+'_fit.json')).read_text())
        mask = a['body_mask'].astype(bool) & ~a['saturation'].any(-1)
        if (mask & source['support']).any() or (mask & edge['accepted']).any():
            raise ValueError('Fully measured body overlaps an existing core/edge replacement')
        if int(mask.sum()) != {'ceiling_lamp': 68, 'middle_ceiling': 47}[name]:
            raise ValueError('Previously reviewed fully measured body mask changed')
        factor = float(a['scale'])*2**engine.p.exposure_ev
        old_rgb = a['postprior_rgb']*factor
        float_rgb = (a['camera'] @ a['cam_to_rgb'])*factor
        old_direct, old_return = fixed_local_exposure(
            engine, old_rgb, fixed[name]['weights'], captured['shares'], captured['members'])
        replay = {key: value for key, value in fixed[name].items()
                  if key not in ('slice', 'weights')}
        replay['local_direct_replay'] = float(np.max(abs(old_direct-a['old_direct'])))
        replay['local_return_replay'] = float(np.max(abs(old_return-a['old_halo_source'])))
        if max(replay.values()) > 2e-6:
            raise ValueError('Frozen local mixed exposure failed replay: '+str(replay))
        new_direct, new_return = fixed_local_exposure(
            engine, float_rgb, fixed[name]['weights'], captured['shares'], captured['members'])
        old_d, old_r = old_direct*mask[..., None], old_return*mask[..., None]
        new_d, new_r = new_direct*mask[..., None], new_return*mask[..., None]
        removed, added = transport(engine, old_d, old_r), transport(engine, new_d, new_r)
        # Invert the engine's log exposure including its unchanged latent fog.
        calibration = 10**(engine.log_k-engine.e.speed_loss+engine.layer_cal)
        base_q = edge['minimum_input_Y_q']
        base_H = (10**base_q-10**engine.e.fog_exposure)/calibration
        q_inverse_error = float(np.max(abs(engine._log_exposure(base_H)-base_q)))
        changed_H = base_H-removed+added
        if changed_H[source['safe']].min() < -1e-7:
            raise ValueError('Source subtraction produced negative safe-region exposure')
        changed_q = engine._log_exposure(np.maximum(changed_H, 1e-12))
        P = observation_operator(base_q.shape[:2], source['matrix'], previous['points'],
                                 25/meta['scan_um_per_pixel'])
        warp = crop_warp(ProjectiveTransform(source['matrix']))

        def render(q):
            state = engine._develop(q, blur_px=12/engine.px_um)
            state.dye *= source['factor'][..., None]
            density = engine.scanner.normalized_density(engine._transmittance(state))
            rgb = engine.scanner.to_rgb(density)
            return warp(rgb), lab(P @ rgb.reshape(-1, 3)), rgb, state.dye, density

        before, before_lab, _, _, _ = render(base_q)
        image, prediction, native_rgb, dye, density = render(changed_q)
        replay['candidate_base_Lab'] = float(np.max(abs(before_lab-edge['minimum_input_Y_lab'])))
        replay['candidate_base_RGB'] = float(np.max(abs(before[source['supported']]-edge['minimum_input_Y'][source['supported']])))
        replay['candidate_q_inverse'] = q_inverse_error
        if max(replay['candidate_base_Lab'], replay['candidate_base_RGB']) > 2e-6:
            raise ValueError('Previous edge candidate did not replay with fixed backend')
        nt, nv = fitmeta['baseline']['train']['samples'], fitmeta['baseline']['heldout']['samples']
        target = previous['target_lab']
        selections = {'train_diagnostic_only': slice(0, nt),
                      'heldout': slice(nt, nt+nv), 'outer_300_800_nominal_um': slice(nt+nv, None)}

        def stats(values):
            return {label: {'samples': len(values[s]),
                            'Lab_RMSE': float(np.sqrt(np.mean((values[s]-target[s])**2))),
                            'median_deltaE00': float(np.median(colour.delta_E(values[s], target[s], method='CIE 2000'))),
                            'mean_model_minus_real_Lab': (values[s]-target[s]).mean(0).tolist()}
                    for label, s in selections.items()}

        row = {'name': name, 'fully_measured_body_cells': int(mask.sum()),
               'body_measured_channels': int(mask.sum()*3), 'replay': replay,
               'body_input_unmetered_RGB_max': {'old_preprior': a['preprior_rgb'][mask].max(0).tolist(),
                                              'old_postprior': a['postprior_rgb'][mask].max(0).tolist(),
                                              'float_camera_matrix': (a['camera'] @ a['cam_to_rgb'])[mask].max(0).tolist()},
               'old_direct_sum_layers': old_d.sum((0, 1)).tolist(),
               'new_direct_sum_layers': new_d.sum((0, 1)).tolist(),
               'old_return_source_sum_layers': old_r.sum((0, 1)).tolist(),
               'new_return_source_sum_layers': new_r.sum((0, 1)).tolist(),
               'old_baseline': stats(previous['baseline_lab']),
               'core_inverse': stats(previous['fitted_lab']),
               'prior_minimum_input_Y_edge': stats(before_lab),
               'measured_body_float_decode': stats(prediction)}
        for zone in selections:
            row['measured_body_float_decode'][zone]['median_deltaE00_reduction_vs_edge'] = (
                row['prior_minimum_input_Y_edge'][zone]['median_deltaE00']
                - row['measured_body_float_decode'][zone]['median_deltaE00'])
        report['regions'].append(row)
        np.savez_compressed(out/(name+'.npz'), body_fully_measured_mask=mask,
                            inherited_core_mask=source['support'], inherited_edge_mask=edge['accepted'],
                            old_rgb=old_rgb, float_rgb=float_rgb, old_direct=old_d, new_direct=new_d,
                            old_return_source=old_r, new_return_source=new_r,
                            removed_exposure=removed, added_exposure=added,
                            old_q=base_q, q=changed_q, dye=dye, scan_density=density,
                            native_rgb=native_rgb, image=image, real=source['real'],
                            before=before, old_lab=before_lab, lab=prediction,
                            target_lab=target, points=previous['points'],
                            fixed_local_weights=fixed[name]['weights'], fixed_global_shares=captured['shares'],
                            matrix=source['matrix'])
        encode(image).save(out/(name+'_candidate.png'))
        images = [('真实800T扫描', source['real']), ('旧研究基准', previous['baseline']),
                  ('严格核心逆解', previous['fitted']), ('此前：测量约束边缘', before),
                  ('本轮：全测灯体浮点颜色', image)]
        sheet = Image.new('RGB', (512*len(images), 580), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for index, (title, picture) in enumerate(images):
            sheet.paste(encode(picture), (index*512, 60))
            draw.text((index*512+10, 12), title, font=font, fill='white')
        sheet.save(out/(name+'_comparison.jpg'), quality=98)
        (out/(name+'.json')).write_text(json.dumps(row, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report['elapsed_seconds'] = time.time()-started
    (out/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=AUDIT/'measured-body-decode-candidate')
    args = parser.parse_args()
    run(args.out)
