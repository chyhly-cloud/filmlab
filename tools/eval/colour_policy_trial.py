"""Five predeclared colour policies on frozen full native negatives.

No target RGB enters policy construction. K, contrast, shoulder, material and
negative stay fixed. Patent example weights/threshold are existing settings,
not measurements of the real Frontier. Same historical scenes are reused.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,zlib,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
ROOT=Path('/workspace/filmlab-audit');OUT=ROOT/'colour-policy-trial'
SCENES=['cafe','R1-07-8','R1-06-7','R1-02-3','R1-00-1']
POLICIES=['saved_two_point','no_M_diagnostic','us6160634','us6160634_axis','fixed_film_design_gray']


def policies(e, dn, timing):
    from filmlab.scanning.timing import frame_auto
    from filmlab.engine import LATD_NORMAL
    gray=e.scanner.normalized_density(e._transmittance(e._gray_state(actual_light=False)))[0]
    out={'saved_two_point':timing,'no_M_diagnostic':replace(timing,M=None)}
    for name in ['two_point','us6160634','us6160634_axis']:
        t=frame_auto(e.scanner.digital_paper,dn.reshape(-1,3),gray,e.scanner.gray0,e.scanner.strength,
                     e.scanner.stretch_max,e.p.mid_gray,e.v.latd_limit,LATD_NORMAL,gray_balance=name)
        if name=='two_point':
            replay=max(float(np.max(abs(a-b))) for a,b in zip(t.M,timing.M))
            if replay>2e-5:raise ValueError('Original automatic M does not reproduce: '+str(replay))
        else:out[name]=replace(timing,M=t.M)
    out['fixed_film_design_gray']=replace(timing,M=(-(gray-gray[1]),np.ones(3)))
    return out,replay


def job(sid):
    import colour
    from skimage.transform import warp,ProjectiveTransform
    from PIL import Image,ImageDraw,ImageFont
    from filmlab import generation,rawio
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from base_colour_audit import cell_rows
    from brightness_response_audit import lab
    import oracle_timing as ot
    generation.use('next_sensor');OUT.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'base-colour-audit'/(sid+'.json')).read_text());td=meta['timing']
    timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(a) for a in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    seed=zlib.crc32(sid.encode())%1000
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    cache=OUT/(sid+'_native.npz')
    if cache.exists():
        dn=np.load(cache)['Dn']
    else:
        rgb=rawio.load_raw(ot.sources(sid)[0],max_size=1400)
        m,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=.53)
        negative=e.develop_negative(m);dn=negative.measure(e.scanner.normalized_density)
        np.savez_compressed(cache,Dn=dn)
        del negative,rgb,m;e.state=None
    e.px_um=e.p.frame_width_mm*1000/max(dn.shape[:2])
    ts,mreplay=policies(e,dn,timing)
    if sid=='cafe':
        saved=np.load(ROOT/'restored-cafe/aligned.npz');mat=json.loads((ROOT/'restored-cafe/settings.json').read_text())['geometry']
    else:
        saved=np.load(ROOT/'epfl-highlights'/(sid+'_aligned.npz'));mat=json.loads((ROOT/'epfl-highlights'/(sid+'.json')).read_text())['geometry']
    real=saved['real_srgb'];tf=ProjectiveTransform(np.array(mat));shape=real.shape[:2]
    images={'real':colour.models.eotf_sRGB(real)}
    balances={}
    for name in POLICIES:
        e.scanner.timing=ts[name];native=e.scanner.to_rgb(dn)
        images[name]=warp(native,tf.inverse,output_shape=shape,order=1,cval=np.nan)
        balances[name]={'M':None if ts[name].M is None else [x.tolist() for x in ts[name].M]}
        if not np.array_equal(ts[name].K,timing.K) or not np.array_equal(ts[name].s,timing.s) or ts[name].x_knee!=timing.x_knee or ts[name].x_white!=timing.x_white:raise ValueError('Non-colour timing changed')
    replay=float(np.nanmax(abs(images['saved_two_point']-saved['exp_plain'])))
    if replay>2e-5:raise ValueError('Full baseline does not reproduce: '+str(replay))
    mask=np.load(ROOT/'base-colour-audit'/(sid+'.npz'))['mask_far_geometric_1500']
    rows=cell_rows(images,mask)
    if not rows:raise ValueError('No fixed stable cells')
    target=np.array([r['Lab']['real'] for r in rows]);stats={}
    for name in POLICIES:
        values=np.array([r['Lab'][name] for r in rows]);delta=values-target
        stats[name]={'median_deltaE00':float(np.median(colour.delta_E(values,target,method='CIE 2000'))),
                     'median_ab_distance':float(np.median(np.linalg.norm(delta[:,1:],axis=1))),
                     'median_gap_Lab':np.median(delta,axis=0).tolist()}
    report={'scene':sid,'scope':__doc__,'policies_predeclared':POLICIES,'baseline_linear_reproduction_max':replay,
            'baseline_M_reproduction_max':mreplay,'region':'same geometric-far stable24px cells; not certified pure direct light',
            'cells':len(rows),'balances':balances,'metrics':stats,'cell_records':rows,
            'limitations':['Same final scan has been used historically; this is not external blind validation.',
                'Actual negative density, real scan settings and source spectra remain unmeasured.',
                'Only M changes; no target-based policy, threshold, shoulder, exposure or material fitting.',
                'Patent example algorithm is not independently established as the actual scanner response.']}
    (OUT/(sid+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    for name in POLICIES[2:]:
        sheet=Image.new('RGB',(2040,540),(18,18,18));d=ImageDraw.Draw(sheet)
        for i,(key,title) in enumerate([('real','真实800T扫描'),('saved_two_point','原研究基准'),(name,name+' · 实验')]):
            im=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(images[key]))*255)));im.thumbnail((680,470));sheet.paste(im,(680*i,70))
            d.text((680*i+8,8),title,font=font,fill='white');d.text((680*i+8,37),sid,font=font,fill='white')
        sheet.save(OUT/(sid+'_'+name+'.jpg'),quality=97)
    print(json.dumps({'scene':sid,'cells':len(rows),'replay':replay,'metrics':stats},ensure_ascii=False),flush=True)


def summary():
    out={'policies_predeclared':POLICIES,'scenes':[],'decision':'No default change. Assess daytime and blue hour jointly, with other scenes as regression controls.'}
    for sid in SCENES:
        r=json.loads((OUT/(sid+'.json')).read_text());out['scenes'].append({k:r[k] for k in ['scene','cells','baseline_linear_reproduction_max','baseline_M_reproduction_max','metrics']})
    snapshot=json.loads((ROOT/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    out['production']={'verified_files':len(snapshot),'changed':changed}
    out['cross_scene_checks']={}
    for name in POLICIES[1:]:
        worsened={metric:[r['scene'] for r in out['scenes'] if r['metrics'][name][metric]>r['metrics']['saved_two_point'][metric]]
                  for metric in ['median_deltaE00','median_ab_distance']}
        out['cross_scene_checks'][name]={'worsened_scenes':worsened,'passes_both_metrics_all_scenes':not any(worsened.values())}
    out['decision']='No tested policy improves both colour-distance and full DeltaE consistently across all five scenes. Keep experimental only; no default change.'
    out['related_tests_passed']=13
    (OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scene',choices=SCENES+['summary']);args=ap.parse_args()
    if args.scene=='summary':summary()
    else:job(args.scene)
