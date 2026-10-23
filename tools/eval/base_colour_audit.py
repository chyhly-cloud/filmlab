"""Cross-scene no-halo normal-area audit; no colour/timing/material fitting.

Fixed cached frame timings and geometry. Primary normal cells are far from
conservative highlight exclusions AND have small modeled halation. Inferred
light classes and D65-removal controls are not measured illumination truth.
"""
from pathlib import Path
from dataclasses import replace
import sys,os,json,zlib,hashlib,argparse
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab,LUMA
ROOT=Path('/workspace/filmlab-audit');OUT=ROOT/'base-colour-audit'


def exclusion_distance(mask,um):
    from scipy.ndimage import distance_transform_edt
    # scipy's implicit exterior is not an observed highlight in a blank mask.
    return distance_transform_edt(~mask)*um if mask.any() else np.full(mask.shape,np.inf)


def cell_rows(images,mask,classes=None,cell=24):
    """Same real-selected stable cells for every endpoint; no error selection."""
    rows=[];height,width=mask.shape
    for y in range(0,height-cell+1,cell):
        for x in range(0,width-cell+1,cell):
            m=mask[y:y+cell,x:x+cell]
            if m.mean()<.9:continue
            values={k:v[y:y+cell,x:x+cell][m] for k,v in images.items()}
            if not all(np.isfinite(v).all() for v in values.values()):continue
            real=lab(values['real']);med=np.median(real,axis=0)
            spread=np.median(abs(real-med),axis=0)
            if spread[0]>4 or np.linalg.norm(spread[1:])>6:continue
            states={k:lab(np.median(v,axis=0)).tolist() for k,v in values.items()}
            item={'yx':[y,x],'pixels':int(m.sum()),'Lab':states,'real_MAD_Lab':spread.tolist()}
            if classes is not None:item['light_weights']=np.mean(classes[y:y+cell,x:x+cell][m],axis=0).tolist()
            rows.append(item)
    return rows


def summarize(rows):
    if not rows:return {'cells':0,'status':'no_support'}
    states={k:np.array([r['Lab'][k] for r in rows]) for k in rows[0]['Lab']};real=states['real']
    result={'cells':len(rows),'status':'descriptive','states':{}}
    for k,values in states.items():
        d=values-real
        result['states'][k]={'median_Lab':np.median(values,0).tolist(),'median_gap_Lab':np.median(d,0).tolist(),
            'mean_gap_Lab':d.mean(0).tolist(),'gap_Lab_p10':np.percentile(d,10,axis=0).tolist(),
            'gap_Lab_p90':np.percentile(d,90,axis=0).tolist(),
            'cyan_blue_gt2_fraction':float(((d[:,1]<-2)&(d[:,2]<-2)).mean()),
            'blue_gt2_fraction':float((d[:,2]<-2).mean()),
            'yellow_gt2_fraction':float((d[:,2]>2).mean())}
    return result


def job(sid):
    import colour,rawpy
    from scipy.ndimage import zoom,binary_dilation,distance_transform_edt
    from skimage.transform import warp,ProjectiveTransform
    from filmlab import generation,rawio,film,lights
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from epfl_highlights import digest
    import oracle_timing as ot
    generation.use('next_sensor');OUT.mkdir(exist_ok=True)
    if sid=='cafe':
        meta=json.loads((ROOT/'restored-cafe/settings.json').read_text());td=meta['timing_values'];mat=meta['geometry']
        old=np.load(ROOT/'restored-cafe/aligned.npz')
    else:
        meta=json.loads((ROOT/'epfl-highlights'/(sid+'.json')).read_text());td=meta['timing'];mat=meta['geometry']
        old=np.load(ROOT/'epfl-highlights'/(sid+'_aligned.npz'))
    rawpath,scanpath,_=ot.sources(sid);hashes={k:digest(p) for k,p in [('raw',rawpath),('scan',scanpath)]}
    if sid!='cafe':
        if hashes['raw']!=meta['raw_sha256'] or hashes['scan']!=meta['film_sha256']:raise ValueError('source hash mismatch')
    rgb=rawio.load_raw(rawpath,max_size=1400);seed=zlib.crc32(sid.encode())%1000
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=.53)
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    tr={};negative=e.develop_negative(metered,trace=tr)
    timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    e.scanner.timing=timing
    shape=old['real_srgb'].shape[:2];tf=ProjectiveTransform(np.array(mat))
    wp=lambda a,fill=np.nan:warp(a,tf.inverse,output_shape=shape,order=1,cval=fill)
    full=wp(e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(negative.state))))
    reproduce=float(np.nanmax(abs(full-old['exp_plain'])))
    if reproduce>2e-5:raise ValueError('cached full RGB not reproduced: '+sid+' '+str(reproduce))
    W=negative.light_weights;W=zoom(W,(rgb.shape[0]/W.shape[0],rgb.shape[1]/W.shape[1],1),order=1)[:rgb.shape[0],:rgb.shape[1]]
    input_q=e._log_exposure(tr['scattered_exposure']);H=tr['scattered_exposure'];total=tr['total_exposure']
    frac=np.maximum(total-H,0)/np.maximum(total,1e-20)
    def render(H):
        q=e._log_exposure(H);st=e._develop(q,blur_px=12/e.px_um);e._agitation(st)
        dn=e.scanner.normalized_density(e._transmittance(st));return e.scanner.to_rgb(dn),dn,st,q
    none,dn,st,q0=render(H);none=wp(none)
    e.scanner.timing=replace(timing,x_knee=None,x_white=None);paper=wp(e.scanner.to_rgb(dn));e.scanner.timing=timing
    # Sole input intervention: replace spatial illumination inference by the
    # pre-existing D65 reference response. Not a proposed correction or truth.
    refH=film.emulsion_scatter(e._expose_lin(metered,e.M),e.px_um)
    ref,refdn,refst,qref=render(refH);ref=wp(ref)
    with rawpy.imread(rawpath) as raw:saturated=rawio.sensor_saturation(raw,per_channel=True)
    step=max(1,int(round(max(saturated.shape[:2])/1400)))
    sat=rawio._downsample(saturated.any(-1).astype(float),step)>0
    if sat.shape!=rgb.shape[:2]:raise ValueError('saturation geometry mismatch')
    measured=wp((~binary_dilation(sat,iterations=2)).astype(float),0)>.999
    real=colour.models.eotf_sRGB(old['real_srgb']);Y=real@LUMA
    # Exclusion masks are deliberately broader than light-source detection;
    # bright reflective surfaces may be excluded, not classified as emitters.
    highlights=(metered@LUMA>=1.)|sat
    um=36000/shape[1];projected=(wp(highlights.astype(float),0)>.01)|(Y>=.85)
    distance=exclusion_distance(projected,um)
    images={'real':real,'nohalo':none,'full':full,'nohalo_D65_input':ref,'nohalo_paper_control':paper}
    valid=measured&np.logical_and.reduce([np.isfinite(v).all(-1) for v in images.values()])
    normal=valid&(Y>=.05)&(Y<.65)
    # Select by geometry and the intervention itself, never target residual.
    negligible=(np.max(abs(lab(full)-lab(none)),axis=-1)<1)&(np.max(wp(frac),axis=-1)<.01)
    weights=wp(W);dq=wp((qref-q0)*np.log2(10))
    classkeys=[c.key for c in lights.classes()]
    masks={'far_1500':normal&(distance>1500)&negligible,
           'far_3000':normal&(distance>3000)&negligible,
           'far_geometric_1500':normal&(distance>1500),
           'near_highlight_0_300':valid&(Y>=.05)&(Y<.85)&(distance>0)&(distance<=300)}
    report={'scene':sid,'source_hashes':hashes,'generation':'next_sensor','timing':td,'input_full_reproduction_max':reproduce,
        'normal_Y_range':[.05,.65],'pixel_um':um,'native_um':e.px_um,'light_class_keys':classkeys,
        'inferred_members':e.light_mix,'regions':{},'groups':{},'exposure_control':{},
        'limitations':['real per-layer exposures and source spectra unmeasured; layer-composition error cannot be measured directly',
            'per-frame stored automatic timing remains fixed; frame offsets are not independent device calibration',
            'normal masks select stable surfaces, not known neutral reflectances',
            'inferred light classes are model outputs, not measured lamp spectra',
            'cafe has distinct camera/scan history and parallax; kept separate']}
    saved={'real':real,'nohalo':none,'full':full,'reference_input':ref,'nohalo_paper':paper,'distance_um':distance,
           'nohalo_q':wp(q0),'reference_q':wp(qref),'nohalo_dye':wp(st.dye),'nohalo_Dn':wp(dn),'light_weights':weights,
           'return_fraction':wp(frac)}
    for name,mask in masks.items():
        rows=cell_rows(images,mask,weights);report['regions'][name]={'pixels':int(mask.sum()),**summarize(rows),'cell_records':rows}
        saved['mask_'+name]=mask
        if name=='far_1500':
            for ci,key in enumerate(classkeys):
                group=[r for r in rows if r['light_weights'][ci]>=.5]
                report['groups'][key]=summarize(group)
            for ci,label in enumerate(['R','G','B']):report['exposure_control'][label+'_median_D65_minus_auto_EV']=float(np.median(dq[mask,ci])) if mask.any() else None
            report['exposure_control']['median_D65_minus_auto_BG_EV']=float(np.median((dq[...,2]-dq[...,1])[mask])) if mask.any() else None
    # A computational neutral witness only, not a measured gray card in scene.
    grey=e._gray_state(actual_light=False)
    neutral=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(grey)))
    report['nominal_reference_gray_under_frame_timing_Lab']=lab(neutral).reshape(-1,3).tolist()
    np.savez_compressed(OUT/(sid+'.npz'),**saved)
    (OUT/(sid+'.json')).write_text(json.dumps(report,indent=2))
    print(sid,'far cells',report['regions']['far_1500']['cells'],'done',flush=True)
    return sid


def main():
    import multiprocessing as mp
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scenes',nargs='+',default=[f'R1-{i:02d}-{i+1}' for i in range(8)]+['cafe']);args=ap.parse_args()
    with mp.get_context('spawn').Pool(2) as pool:
        for sid in pool.imap_unordered(job,args.scenes):print('saved',sid,flush=True)


if __name__=='__main__':main()
