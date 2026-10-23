"""Full-frame no-refit validation of the predeclared minimum-input-Y edge witness.

Known camera measurements, not scan error, selected the edge witness. The prior
same-scan strict-core inverse remains an estimate. Neither is a measured SPD.
The historical background illumination inference and fixed print timing remain.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import gc
import json
import sys
import time
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]


def main():
    import colour
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import ProjectiveTransform, warp
    from filmlab import rawio
    from filmlab.incident import IncidentSpectrum
    from filmlab.spectral import srgb_encode
    from inverse_source_radiance import engine_for, observation_operator
    from brightness_response_audit import lab
    from light_acceptance import crop_warp, pixel_scaling
    import oracle_timing as ot

    root = Path('/workspace/filmlab-audit')
    oldroot = root / 'inverse-source-radiance'
    edge_root = root / 'lamp-edge-input-audit'
    out = root / 'lamp-edge-spectral-candidate'
    prep = json.loads((oldroot / 'prepared.json').read_text())
    engine, meta = engine_for()
    src, scan, bias = ot.sources('R1-02-3')
    seed = zlib.crc32(b'R1-02-3') % 1000
    started = time.time()
    rgb = rawio.load_raw(src, max_size=3000)
    metered, note = rawio.film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
    del rgb
    sources, edges = [], []
    for r in prep['regions']:
        name = r['name']
        a = np.load(oldroot / (name + '_source.npz'))
        old = np.load(oldroot / (name + '_fit.npz'))
        candidate = np.load(out / (name + '.npz'))
        captured = np.load(edge_root / (name + '.npz'))
        xl, yl, xh, yh = r['source_bounds']; sl = (slice(yl, yh), slice(xl, xh))
        mask = np.zeros(metered.shape[:2], bool); amplitude = np.zeros(metered.shape[:2])
        mask[sl] = a['support']; amplitude[sl] = a['profile']
        sources.append(IncidentSpectrum(
            spectrum=old['coefficients'] @ a['spectra'], amplitude=amplitude, mask=mask,
            provenance='Inherited strict-core same-scan inverse, NOT measured spectral radiance'))
        c = candidate['minimum_input_Y_coefficients']
        assert not (candidate['accepted'] & a['support']).any()
        edges.append({'name': name, 'slice': sl, 'accepted': candidate['accepted'],
                      'direct': c @ a['direct_response'], 'return': c @ a['return_response'],
                      'old_direct': captured['old_direct'], 'old_return': captured['old_halo_source']})
    old_expose = engine._expose_mixed
    replay = {}

    def expose(rgb):
        direct, returning = old_expose(rgb)
        for item in edges:
            sl, mask = item['slice'], item['accepted']
            difference = max(float(np.max(abs(direct[sl][mask] - item['old_direct'][mask]))),
                             float(np.max(abs(returning[sl][mask] - item['old_return'][mask]))))
            replay[item['name']] = difference
            if difference > 2e-6:
                raise ValueError('Full-frame original source replay mismatch: ' + str(replay))
            direct[sl][mask] = item['direct'][mask]
            returning[sl][mask] = item['return'][mask]
        return direct, returning

    # Only skip fresh auto timing. Keep the captured PrintTiming unchanged.
    engine.p = replace(engine.p, timing='standard')
    with patch.object(engine, '_expose_mixed', expose):
        negative = engine.develop_negative(metered, incident_sources=sources)
    joint = engine.scan(negative)
    del metered
    gc.collect()
    encode = lambda a: Image.fromarray(np.rint(srgb_encode(np.nan_to_num(a)) * 255).astype('uint8'))
    encode(joint).save(out / 'full_edge_minimum_Y_native.png')
    report = {'scope': __doc__, 'mode': 'minimum_input_Y, predeclared without scan selection',
              'shape': list(joint.shape), 'old_source_replay_max': replay,
              'metering': note, 'timing': meta['timing'], 'regions': []}
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
    for r in prep['regions']:
        name = r['name']; a = np.load(oldroot / (name + '_source.npz'))
        local = np.load(out / (name + '.npz')); oldfit = np.load(oldroot / (name + '_fit.npz'))
        previous_joint = np.load(oldroot / (name + '_joint.npz'))
        fd = json.loads((oldroot / (name + '_fit.json')).read_text())
        xl, yl, xh, yh = r['source_bounds']; sl = (slice(yl, yh), slice(xl, xh))
        P = observation_operator(a['q_base'].shape[:2], a['matrix'], local['points'], 25/meta['scan_um_per_pixel'])
        prediction = lab(P @ joint[sl].reshape(-1, 3))
        target = oldfit['target_lab']; nt, nv = fd['baseline']['train']['samples'], fd['baseline']['heldout']['samples']
        metrics = {label: {'Lab_RMSE': float(np.sqrt(np.mean((prediction[s]-target[s])**2))),
                           'median_deltaE00': float(np.median(colour.delta_E(prediction[s], target[s], method='CIE 2000')))}
                   for label, s in [('train', slice(0, nt)), ('heldout', slice(nt, nt+nv)), ('outer', slice(nt+nv, None))]}
        wp = crop_warp(ProjectiveTransform(a['matrix'])); image = wp(joint[sl])
        safe = a['supported']
        metrics['joint_vs_local_candidate_RGB_max'] = float(np.max(abs(image[safe] - local['minimum_input_Y'][safe])))
        report['regions'].append({'name': name, **metrics})
        np.savez_compressed(out / (name + '_joint.npz'), joint=image, joint_lab=prediction,
                            joint_q=negative.log_exposure[sl], real=a['real'], prior_joint=previous_joint['joint'])
        sheet = Image.new('RGB', (1536, 580), (18, 18, 18)); draw = ImageDraw.Draw(sheet)
        for i, (title, picture) in enumerate([('真实800T扫描', a['real']),
                                            ('此前：两灯核心反演', previous_joint['joint']),
                                            ('本轮：实测约束边缘', image)]):
            sheet.paste(encode(picture), (i*512, 60)); draw.text((i*512+10, 12), title, font=font, fill='white')
        sheet.save(out / (name + '_joint_comparison.jpg'), quality=96)
    # Existing alignment, no new geometry fitting.
    oldmeta = json.loads((root / 'epfl-highlights/R1-02-3.json').read_text())
    real = np.load(Path(ot.GEO) / 'geo_R1-02-3.npz')['real']
    step = round(max(joint.shape[:2])/1400)
    H = np.asarray(oldmeta['geometry']) @ pixel_scaling(1/step, 1/step)
    aligned = warp(joint, ProjectiveTransform(H).inverse, output_shape=real.shape[:2], order=1, cval=np.nan)
    height = round(real.shape[0]/real.shape[1]*720)
    sheet = Image.new('RGB', (1440, height+65), (18, 18, 18)); draw = ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(np.rint(real*255).astype('uint8')).resize((720, height), Image.Resampling.LANCZOS), (0, 60))
    sheet.paste(encode(aligned).resize((720, height), Image.Resampling.LANCZOS), (720, 60))
    draw.text((10, 15), '真实800T扫描', font=font, fill='white')
    draw.text((730, 15), '本轮全幅候选 · 非正式管线', font=font, fill='white')
    sheet.save(out / 'full_comparison.jpg', quality=96)
    report['elapsed_seconds'] = time.time() - started
    (out / 'joint_render.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
