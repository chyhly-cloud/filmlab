"""Replay the independently validated HDR decoder through fixed film/scan stages.

Only previously declared, fully measured body cells change. Camera-calibrated
numeric range preservation is not a lamp-spectrum or radiance measurement.
Inherited core/partial-edge estimates and all physical/output parameters remain.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]


def run(hdr_crops, out):
    import colour
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import ProjectiveTransform
    from inverse_source_radiance import engine_for, observation_operator
    from measured_body_decode_candidate import fixed_local_exposure
    from lamp_edge_spectral_candidate import transport
    from brightness_response_audit import lab
    from light_acceptance import crop_warp
    from filmlab.spectral import srgb_encode

    audit = Path('/workspace/filmlab-audit')
    prior_root = audit/'measured-body-decode-candidate'
    capture = json.loads((prior_root/'results.json').read_text())['capture']
    members = {key: (value['name'], np.asarray(value['spectrum']))
               for key, value in capture['members'].items()}
    e, meta = engine_for()
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': __doc__, 'coefficient_or_timing_fit': False, 'regions': []}
    encode = lambda a: Image.fromarray(np.rint(srgb_encode(np.nan_to_num(a))*255).astype('uint8'))
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
    for name in ['ceiling_lamp', 'middle_ceiling']:
        prior = np.load(prior_root/(name+'.npz'))
        hdr = np.load(hdr_crops/(name+'.npz'))
        source = np.load(audit/'inverse-source-radiance'/(name+'_source.npz'))
        edge = np.load(audit/'lamp-edge-input-audit'/(name+'.npz'))
        oldfit = np.load(audit/'inverse-source-radiance'/(name+'_fit.npz'))
        ref = json.loads((audit/'inverse-source-radiance'/(name+'_fit.json')).read_text())
        if not np.array_equal(hdr['source_bounds'], edge['source_bounds']):
            raise ValueError('HDR crop geometry differs from original input')
        mask = prior['body_fully_measured_mask']
        if (mask & edge['saturation'].any(-1)).any():
            raise ValueError('This contract cannot replace a genuinely censored camera measurement')
        rgb = hdr['hdr_rgb'] * capture['metering_scale'] * 2**e.p.exposure_ev
        direct, returning = fixed_local_exposure(e, rgb, prior['fixed_local_weights'],
                                                prior['fixed_global_shares'], members)
        direct *= mask[..., None]; returning *= mask[..., None]
        calibration = 10**(e.log_k-e.e.speed_loss+e.layer_cal)
        old_H = (10**prior['old_q']-10**e.e.fog_exposure)/calibration
        H = (old_H - transport(e, prior['old_direct'], prior['old_return_source'])
             + transport(e, direct, returning))
        if H[source['safe']].min() < -1e-7:
            raise ValueError('Nonnegative exposure subtraction failed')
        q = e._log_exposure(np.maximum(H, 1e-12))
        state = e._develop(q, blur_px=12/e.px_um); state.dye *= source['factor'][..., None]
        density = e.scanner.normalized_density(e._transmittance(state))
        native = e.scanner.to_rgb(density)
        P = observation_operator(q.shape[:2], source['matrix'], prior['points'], 25/meta['scan_um_per_pixel'])
        predicted = lab(P @ native.reshape(-1, 3)); target = prior['target_lab']
        image = crop_warp(ProjectiveTransform(source['matrix']))(native)
        nt, nv = ref['baseline']['train']['samples'], ref['baseline']['heldout']['samples']
        metrics = {label: {'Lab_RMSE': float(np.sqrt(np.mean((predicted[s]-target[s])**2))),
                          'median_deltaE00': float(np.median(colour.delta_E(predicted[s], target[s], method='CIE 2000')))}
                   for label, s in [('train_diagnostic', slice(0, nt)), ('heldout', slice(nt, nt+nv)), ('outer', slice(nt+nv, None))]}
        row = {'name': name, 'fully_measured_body_cells': int(mask.sum()), 'metrics': metrics,
               'final_HDR_vs_Bayer_float_body_RGB_max': float(np.max(abs(image[source['supported']]-prior['image'][source['supported']]))),
               'final_HDR_vs_Bayer_float_body_sample_Lab_max': float(np.max(abs(predicted-prior['lab']))),
               'HDR_vs_Bayer_float_body_input_RGB_max': float(np.max(abs(rgb[mask]-prior['float_rgb'][mask]))),
               'unknown_core_or_partial_source_values_newly_recovered': 0,
               'scope_limit': 'Only these full-measurement body cells were changed, not whole-frame HDR adoption or source radiance calibration.'}
        report['regions'].append(row)
        np.savez_compressed(out/(name+'.npz'), image=image, q=q, dye=state.dye, scan_density=density,
                            native_rgb=native, real=source['real'], old_baseline=source['baseline_rgb'],
                            prior_body_float=prior['image'], lab=predicted, target_lab=target,
                            body_fully_measured_mask=mask, hdr_input_rgb=rgb)
        sheet = Image.new('RGB', (1536, 580), (18, 18, 18)); draw = ImageDraw.Draw(sheet)
        for i, (title, values) in enumerate([('真实800T扫描', source['real']),
                                            ('旧研究基准', source['baseline_rgb']),
                                            ('本轮·测量约束与HDR灯体', image)]):
            sheet.paste(encode(values), (i*512, 60)); draw.text((i*512+10, 12), title, font=font, fill='white')
        sheet.save(out/(name+'_comparison.jpg'), quality=98)
    (out/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdr-crops', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/raw-hdr-film-replay'))
    args = parser.parse_args()
    run(args.hdr_crops, args.out)
