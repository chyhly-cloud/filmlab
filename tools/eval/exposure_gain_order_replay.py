"""Eval-only exposure-gain order candidate on two fixed native RAW scenes.

Freeze inferred light weights, selected spectra, negative calibration and scan.
Move ONLY the known recipe exposure multiplier after RGB spectral recovery for
both direct and return-source responses. No new material or output fitting.
This is not a calibrated replacement for unknown scene illumination.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import sys,json,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
ROOT=Path('/workspace/filmlab-audit');OUT=ROOT/'colour-foundation-audit/gain-order-replay'


def job(sid,input_variant='gain_order'):
    import colour
    from filmlab import generation,rawio
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from base_colour_audit import cell_rows
    from skimage.transform import warp,ProjectiveTransform
    from PIL import Image,ImageDraw,ImageFont
    import oracle_timing as ot
    generation.use('next_sensor')
    out=OUT if input_variant=='gain_order' else ROOT/'illumination-inference-trial/real'
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'base-colour-audit'/(sid+'.json')).read_text());td=meta['timing']
    if sid=='cafe':
        saved=np.load(ROOT/'restored-cafe/aligned.npz');mat=json.loads((ROOT/'restored-cafe/settings.json').read_text())['geometry']
    else:
        saved=np.load(ROOT/'epfl-highlights'/(sid+'_aligned.npz'));mat=json.loads((ROOT/'epfl-highlights'/(sid+'.json')).read_text())['geometry']
    seed=zlib.crc32(sid.encode())%1000
    rgb=rawio.load_raw(ot.sources(sid)[0],max_size=1400);metered,_=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=.53)
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    gain=2**e.p.exposure_ev
    oldH,oldS=e._expose_mixed(metered*gain);weights=e.light_W.copy()
    candidate_checks={}
    if input_variant=='gain_order':
        with patch('filmlab.lights.class_weights',return_value=weights):newH,newS=e._expose_mixed(metered)
        newH=newH*gain;newS=newS*gain
    elif input_variant=='no_global_pruning':
        from illumination_inference_trial import unpruned_exposure
        newH,newS,candidate_checks=unpruned_exposure(e,metered*gain)
    else:raise ValueError(input_variant)
    if not np.array_equal(e.light_W,weights):raise ValueError('Illumination weights changed')
    report={'scene':sid,'input_variant':input_variant,'candidate_checks':candidate_checks,
            'scope':__doc__ if input_variant=='gain_order' else 'Only remove global illumination-class pruning, same local weights, member spectra, material, processing and frozen scan.', 'gain':gain,'gain_EV':e.p.exposure_ev,'illumination_weights_exactly_frozen':True,
            'direct_log2_ratio_after_over_before_p50_p95_absmax':np.concatenate([np.percentile(np.log2(newH/oldH),[50,95],axis=(0,1)),np.max(abs(np.log2(newH/oldH)),axis=(0,1))[None]],axis=0).tolist(),
            'limits':['The original engine layer/gray calibration is intentionally frozen; this is an isolated eval candidate.',
                      'Reused scenes; no independent material or scan-response validation.']}
    e.p=replace(e.p,timing='standard')
    e.scanner.timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(a) for a in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    tf=ProjectiveTransform(np.array(mat));shape=saved['real_srgb'].shape[:2];images={'real':colour.models.eotf_sRGB(saved['real_srgb'])}
    for name,h,s in [('before',oldH,oldS),('after',newH,newS)]:
        with patch.object(e,'_expose_mixed',return_value=(h,s)):
            neg=e.develop_negative(metered)
        native=e.scan(neg);images[name]=warp(native,tf.inverse,output_shape=shape,order=1,cval=np.nan)
        del neg;e.state=None
    replay=float(np.nanmax(abs(images['before']-saved['exp_plain'])))
    if replay>2e-5:raise ValueError('Baseline mismatch '+str(replay))
    report['baseline_linear_reproduction_max']=replay
    mask=np.load(ROOT/'base-colour-audit'/(sid+'.npz'))['mask_far_geometric_1500'];rows=cell_rows(images,mask)
    real=np.array([r['Lab']['real'] for r in rows]);stats={}
    for key in ['before','after']:
        a=np.array([r['Lab'][key] for r in rows]);d=a-real
        stats[key]={'median_deltaE00':float(np.median(colour.delta_E(a,real,method='CIE 2000'))),
                    'median_ab_distance':float(np.median(np.linalg.norm(d[:,1:],axis=1))),
                    'median_gap_Lab':np.median(d,axis=0).tolist()}
    report['cells']=len(rows);report['metrics']=stats
    (out/(sid+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(2040,540),(18,18,18));d=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    for i,(key,title) in enumerate([('real','真实800T扫描'),('before','原曝光补偿位置' if input_variant=='gain_order' else '原研究基准'),('after','曝光补偿后置 · 独立实验' if input_variant=='gain_order' else '取消全局光源裁剪 · 实验')]):
        im=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(images[key]))*255)));im.thumbnail((680,470));sheet.paste(im,(i*680,70));d.text((i*680+8,8),title,font=font,fill='white');d.text((i*680+8,37),sid,font=font,fill='white')
    sheet.save(out/(sid+'.jpg'),quality=97)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':job(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'gain_order')
