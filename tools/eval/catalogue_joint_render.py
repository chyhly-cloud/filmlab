"""Full-frame validation of the two preselected physical 112-spectrum witnesses.

No fitting or candidate selection occurs here. Core spectra remain same-scan
inverse estimates. Partial-edge minY and actual-HDR measured body inputs, the
legacy illumination field, material, transport and PrintTiming remain fixed.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import gc,json,os,sys,time,zlib
import numpy as np
from scipy import sparse

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
ROOT=Path('/workspace/filmlab-audit')
NAMES=('ceiling_lamp','middle_ceiling')


def run(out=ROOT/'catalogue-joint-render'):
    import colour
    from PIL import Image,ImageDraw,ImageFont
    from skimage.transform import ProjectiveTransform,warp
    from filmlab import rawio,film
    from filmlab.incident import IncidentSpectrum,spectral_responses
    from filmlab.spectral import srgb_encode
    from inverse_source_radiance import engine_for
    from measured_body_decode_candidate import fixed_local_exposure
    from light_acceptance import crop_warp,pixel_scaling
    from brightness_response_audit import lab
    import oracle_timing as ot
    if os.environ.get('FILMLAB_DATA')!='/workspace/filmlab-data':raise ValueError('Preserve original FILMLAB_DATA and path CRC')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.time();engine,meta=engine_for()
    capture=json.loads((ROOT/'measured-body-decode-candidate/results.json').read_text())['capture']
    members={k:(v['name'],np.asarray(v['spectrum'])) for k,v in capture['members'].items()}
    src,_,bias=ot.sources('R1-02-3')
    if str(src)!=capture['raw_path']:raise ValueError('RAW path/CRC changed')
    rgb=rawio.load_raw(src,max_size=3000)
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(zlib.crc32(b'R1-02-3')%1000),bias_ev=bias)
    scale=float(np.sum(metered*rgb)/np.sum(rgb*rgb));del rgb
    if abs(scale-capture['metering_scale'])>2e-10:raise ValueError('Legacy frame metering differs')
    shape=metered.shape[:2];sources={'validated_inputs_old_core':[],'catalogue112':[]};items=[];witness_checks={}
    direct_response,return_response=spectral_responses(engine);filt=film._filt(engine.p.lens_filter)
    for name in NAMES:
        with np.load(ROOT/'inverse-source-radiance'/(name+'_source.npz')) as a:
            support=a['support'];profile=a['profile'];dr=a['direct_response'];rr=a['return_response'];spectra10=a['spectra']
        with np.load(ROOT/'inverse-source-radiance'/(name+'_fit.npz')) as a:old_spectrum=a['coefficients']@spectra10
        with np.load(ROOT/'lamp-edge-input-audit'/(name+'.npz')) as a:
            bounds=a['source_bounds'];old_direct=a['old_direct'];old_return=a['old_halo_source'];sat=a['saturation']
        with np.load(ROOT/'catalogue-source-inverse'/(name+'_scene_specific_library.npz')) as a:
            actual_spectra=a['spectra'];S=a['response6'];camera=a['camera_response']
        fit=json.loads((ROOT/'catalogue-source-inverse'/(name+'_catalogue.json')).read_text())
        c=np.array(fit['selected_by_train_only']['witness_coefficients']);raw_negative=float(c.min());c=np.maximum(c,0)
        chosen=c@actual_spectra
        moment=np.r_[(chosen*filt)@direct_response,(chosen*filt)@return_response]*2**engine.p.exposure_ev
        expected=np.array(fit['selected_by_train_only']['moments']);error=float(np.max(abs(moment-expected)/np.maximum(abs(expected),1e-12)))
        if error>1e-7:raise ValueError('A single actual spectrum fails selected direct/return moment replay')
        xl,yl,xh,yh=bounds;sl=(slice(yl,yh),slice(xl,xh));fullmask=np.zeros(shape,bool);amplitude=np.zeros(shape)
        fullmask[sl]=support;amplitude[sl]=profile
        for key,spectrum in [('validated_inputs_old_core',old_spectrum),('catalogue112',chosen)]:
            sources[key].append(IncidentSpectrum(spectrum=spectrum,amplitude=amplitude,mask=fullmask,
                provenance='Fixed same-scan inverse spectral witness, NOT measured source radiance; full-frame verification only'))
        with np.load(ROOT/'lamp-edge-spectral-candidate'/(name+'.npz')) as a:
            accepted=a['accepted'];edge_coeff=a['minimum_input_Y_coefficients']
        with np.load(ROOT/'measured-body-decode-candidate'/(name+'.npz')) as a:
            body=a['body_fully_measured_mask'];weights=a['fixed_local_weights'];shares=a['fixed_global_shares']
        with np.load(ROOT/'raw-hdr-decode-audit/actual'/(name+'.npz')) as a:
            if not np.array_equal(a['source_bounds'],bounds):raise ValueError('Actual HDR crop geometry differs')
            hdr_rgb=a['hdr_rgb']*scale*2**engine.p.exposure_ev
        if (body&sat.any(-1)).any() or (body&accepted).any() or ((body|accepted)&support).any():raise ValueError('Source replacement masks overlap or replace censored body data')
        body_direct,body_return=fixed_local_exposure(engine,hdr_rgb,weights,shares,members)
        items.append({'name':name,'slice':sl,'edge_mask':accepted,'edge_direct':edge_coeff@dr,'edge_return':edge_coeff@rr,
                      'body_mask':body,'body_direct':body_direct,'body_return':body_return,'old_direct':old_direct,'old_return':old_return})
        witness_checks[name]={'negative_coefficient_before_roundoff_cleanup':raw_negative,'source_input_Y':float(c.sum()),'actual_spectrum_replayed_moment_relative_max':error,
                              'active_members':fit['nonzero_witness_members'],'strict_cells':int(support.sum()),'partial_edge_cells':int(accepted.sum()),'fully_measured_HDR_body_cells':int(body.sum())}
        del actual_spectra,edge_coeff,weights,hdr_rgb
    gc.collect()
    # Estimate the historical full-frame illumination exactly once, before any
    # known source replacement. Both full-frame renders reuse that same field.
    direct,returning=engine._expose_mixed(metered*2**engine.p.exposure_ev);old_replay={}
    for item in items:
        sl=item['slice'];mask=item['edge_mask']|item['body_mask']
        replay=max(float(abs(direct[sl][mask]-item['old_direct'][mask]).max()),float(abs(returning[sl][mask]-item['old_return'][mask]).max()))
        if replay>2e-6:raise ValueError('Full original exposure fails cached source replay')
        old_replay[item['name']]=replay
        for label in ['edge','body']:
            mask=item[label+'_mask'];direct[sl][mask]=item[label+'_direct'][mask];returning[sl][mask]=item[label+'_return'][mask]
    items.clear();gc.collect()
    engine.p=replace(engine.p,timing='standard')
    def fixed_expose(_):return direct,returning
    encode=lambda values:Image.fromarray(np.uint8(np.rint(np.clip(srgb_encode(np.nan_to_num(values)),0,1)*255)))
    full={};local_q={};render_seconds={}
    for key in ['validated_inputs_old_core','catalogue112']:
        before=time.time()
        with patch.object(engine,'_expose_mixed',fixed_expose):negative=engine.develop_negative(metered,incident_sources=sources[key])
        full[key]=engine.scan(negative)
        local_q[key]={name:negative.log_exposure[(slice(int(bounds[1]),int(bounds[3])),slice(int(bounds[0]),int(bounds[2])))].copy()
                      for name in NAMES for bounds in [np.load(ROOT/'lamp-edge-input-audit'/(name+'.npz'))['source_bounds']]}
        encode(full[key]).save(out/(key+'_full_native.png'))
        del negative;engine.state=None;gc.collect();render_seconds[key]=time.time()-before
        print('full render complete',key,render_seconds[key],flush=True)
    del metered,direct,returning,sources;gc.collect()
    report={'scope':__doc__,'candidate_selection_or_refit':False,'FILMLAB_DATA':os.environ['FILMLAB_DATA'],'raw_path':str(src),'shape':list(full['catalogue112'].shape),
            'metering':note,'metering_scale':scale,'timing':meta['timing'],'witness_checks':witness_checks,'old_full_exposure_replay_max':old_replay,'render_seconds':render_seconds,'regions':[]}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21);sheets=[]
    for name in NAMES:
        with np.load(ROOT/'inverse-source-radiance'/(name+'_source.npz')) as a:
            matrix=a['matrix'];safe=a['supported'];real=a['real'];native_safe=a['safe']
        split=np.load(ROOT/'joint-radius-source-inverse'/(name+'_split.npz'));P=sparse.load_npz(ROOT/'joint-radius-source-inverse'/(name+'_operator.npz'))
        target=split['target_lab'];groups=split['groups'];bounds=np.load(ROOT/'lamp-edge-input-audit'/(name+'.npz'))['source_bounds'];xl,yl,xh,yh=bounds;sl=(slice(yl,yh),slice(xl,xh))
        images={key:crop_warp(ProjectiveTransform(matrix))(values[sl]) for key,values in full.items()}
        labs={key:lab(P@values[sl].reshape(-1,3)) for key,values in full.items()}
        with np.load(ROOT/'catalogue-source-inverse'/(name+'_catalogue.npz')) as a:
            if not np.array_equal(a['points'],split['points']) or not np.array_equal(a['groups'],groups) or not np.array_equal(a['target_lab'],target):
                raise ValueError('Local catalogue result does not use the exact independently audited joint-radius split')
            local_rgb=a['image'];local_lab=a['lab'];catalogue_q=a['q']
        with np.load(ROOT/'raw-hdr-film-replay'/(name+'.npz')) as a:previous_local=a['image'];previous_q=a['q']
        row={'name':name,'joint_vs_local_catalogue_RGB_max':float(abs(images['catalogue112'][safe]-local_rgb[safe]).max()),
             'joint_vs_local_catalogue_sample_Lab_max':float(abs(labs['catalogue112']-local_lab).max()),
             'joint_vs_local_catalogue_q_safe_max':float(abs(local_q['catalogue112'][name][native_safe]-catalogue_q[native_safe]).max()),
             'joint_baseline_vs_local_HDR_RGB_max':float(abs(images['validated_inputs_old_core'][safe]-previous_local[safe]).max()),
             'joint_baseline_vs_local_HDR_q_safe_max':float(abs(local_q['validated_inputs_old_core'][name][native_safe]-previous_q[native_safe]).max()),'metrics':{}}
        for key,values in labs.items():
            row['metrics'][key]={}
            for i,label in enumerate(['inner_train','inner_heldout','outer_train','outer_heldout']):
                m=groups==i;delta=values[m]-target[m]
                row['metrics'][key][label]={'samples':int(m.sum()),'Lab_RMSE':float(np.sqrt(np.mean(delta**2))),'median_deltaE00':float(np.median(colour.delta_E(values[m],target[m],method='CIE 2000'))),
                    'real_mean_Lab':target[m].mean(0).tolist(),'model_mean_Lab':values[m].mean(0).tolist(),'mean_model_minus_real_Lab':delta.mean(0).tolist()}
        report['regions'].append(row)
        np.savez_compressed(out/(name+'.npz'),real=real,baseline=images['validated_inputs_old_core'],catalogue=images['catalogue112'],baseline_lab=labs['validated_inputs_old_core'],catalogue_lab=labs['catalogue112'],
            target_lab=target,groups=groups,points=split['points'],catalogue_joint_q=local_q['catalogue112'][name],baseline_joint_q=local_q['validated_inputs_old_core'][name])
        sheet=Image.new('RGB',(1536,580),(18,18,18));draw=ImageDraw.Draw(sheet)
        for i,(title,picture) in enumerate([('真实800T扫描',real),('已验证输入 · 原核心逆解',images['validated_inputs_old_core']),('112谱 · 两灯联合验证候选',images['catalogue112'])]):
            sheet.paste(encode(picture),(512*i,60));draw.text((512*i+8,12),title,font=font,fill='white')
        sheet.save(out/(name+'_comparison.jpg'),quality=98);sheets.append(sheet)
    pair=Image.new('RGB',(1536,1160));pair.paste(sheets[0],(0,0));pair.paste(sheets[1],(0,580));pair.save(out/'two_lamp_comparison.jpg',quality=98)
    oldmeta=json.loads((ROOT/'epfl-highlights/R1-02-3.json').read_text());real=np.load(Path(ot.GEO)/'geo_R1-02-3.npz')['real']
    step=round(max(full['catalogue112'].shape[:2])/1400);H=np.asarray(oldmeta['geometry'])@pixel_scaling(1/step,1/step)
    height=round(real.shape[0]/real.shape[1]*720);sheet=Image.new('RGB',(2160,height+65),(18,18,18));draw=ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(np.uint8(np.rint(np.clip(real,0,1)*255))).resize((720,height),Image.Resampling.LANCZOS),(0,60));draw.text((8,15),'真实800T扫描',font=font,fill='white')
    for i,(key,title) in enumerate([('validated_inputs_old_core','已验证输入 · 原核心逆解'),('catalogue112','112谱 · 联合验证候选')],start=1):
        aligned=warp(full[key],ProjectiveTransform(H).inverse,output_shape=real.shape[:2],order=1,cval=np.nan)
        sheet.paste(encode(aligned).resize((720,height),Image.Resampling.LANCZOS),(720*i,60));draw.text((720*i+8,15),title,font=font,fill='white');del aligned
    sheet.save(out/'full_comparison.jpg',quality=98)
    report['elapsed_seconds']=time.time()-started
    report['limits']=['Both strict-core spectra are same-scan inverse witnesses, not independently measured lamp SPDs or calibrated radiance.',
        'This is validation of already selected minY common-spectrum witnesses; no full-frame or heldout result chooses coefficients.',
        'The 19px split is the exact same-frame joint-radius operator; repeated use of these RGB targets is not independent external evidence.',
        'Two-source full-frame transport/development can differ slightly from separately cropped fits; local-vs-joint discrepancies are reported, not silently refitted.',
        'Partial-edge missing channels and intrinsic lamp identities remain uncertain; full-measurement body masking still includes optical mixing.',
        'Near-source appearance can regress despite outer improvements; this candidate does not supersede current or next.']
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True);return report


if __name__=='__main__':run()
