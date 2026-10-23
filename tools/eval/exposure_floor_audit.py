"""Exposure-floor domain audit, without colour or material fitting.

A negative linear sRGB component can be a valid wide-gamut camera-proxy signal.
A floor-limited branch weight is NOT a measured fraction of lost film exposure.
Known-spectrum controls use a CIE camera and the stored material sensitivity;
real RAWs supply no independent per-layer ground truth.
"""
from pathlib import Path
import argparse,json,sys,zlib,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
ROOT=Path('/workspace/filmlab-audit');OUT=ROOT/'exposure-floor-audit'


def coverage(rgb,branch_floor_weights,mask):
    valid=mask&np.isfinite(rgb).all(-1)&np.isfinite(branch_floor_weights).all(-1)
    if not valid.any():return {'pixels':0}
    w=branch_floor_weights[valid]
    return {'pixels':int(valid.sum()),'negative_RGB_pixel_fraction':float(np.any(rgb[valid]<0,axis=-1).mean()),
        'floor_limited_branch_weight_per_layer_mean':w.mean(0).tolist(),
        'floor_limited_branch_weight_per_layer_p95':np.percentile(w,95,axis=0).tolist(),
        'floor_limited_branch_weight_over_half_per_layer_fraction':(w>.5).mean(0).tolist(),
        'any_layer_floor_branch_weight_over_half_fraction':float(np.any(w>.5,axis=-1).mean())}


def controlled():
    import colour
    from filmlab import generation,lights,spectral,film
    from filmlab.materials import emulsion
    generation.use('next_sensor');OUT.mkdir(exist_ok=True)
    cc=colour.SDS_COLOURCHECKERS['ColorChecker N Ohta'];names=list(cc)
    R=np.array([s.copy().align(colour.SpectralShape(380,780,5)).values for s in cc.values()])
    S=10**emulsion('kodak_vision3_500t').stock.log_sensitivity;rows=[]
    for name in ['D65','BB3200','LED-B3','FL11']:
        I=spectral.illuminant(name) if name in ['D65','BB3200'] else lights._sd(name)
        I/=float(lights.camera_rgb(I)@np.array([.2126,.7152,.0722]));rgb=lights.camera_rgb(R*I);truth=(R*I)@S
        response=film.exposure_response(emulsion('kodak_vision3_500t').stock,I,method='spectral' if name=='D65' else 'linear_wp')
        raw=response(rgb);bad=raw<=1e-9
        rows.append({'light':name,'floor_limited_patch_layer_count':int(bad.sum()),'patches':[
            {'name':n,'input_linear_RGB':rgb[j].tolist(),'unclipped_model_H':raw[j].tolist(),'conditional_spectral_H':truth[j].tolist(),
             'floor_limited_layers':np.flatnonzero(bad[j]).tolist()} for j,n in enumerate(names) if bad[j].any()]})
    report={'scope':__doc__,'source':'ColorChecker N Ohta measured reflectances x four known illuminants','lights':rows}
    (OUT/'controlled.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def job(sid):
    from scipy.ndimage import zoom
    from skimage.transform import warp,ProjectiveTransform
    from filmlab import generation,rawio,film,lights
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    import oracle_timing as ot
    generation.use('next_sensor');OUT.mkdir(exist_ok=True)
    rgb=rawio.load_raw(ot.sources(sid)[0],max_size=1400);seed=zlib.crc32(sid.encode())%1000
    metered,_=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=.53)
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    signal=metered*2**e.p.exposure_ev;baseline,_=e._expose_mixed(signal)
    w=e.light_W;h,width=signal.shape[:2];W=zoom(w,(h/w.shape[0],width/w.shape[1],1),order=1)[:h,:width]
    members=lights.pick_members(e.v.frame_seed);floor_weights=np.zeros_like(signal);H=np.zeros_like(signal);branches=[]
    for k,c in enumerate(lights.classes()):
        wk=W[...,k];share=float(wk.mean())
        if share<.003:continue
        spec=members[c.key][1];method='spectral' if c.key=='daylight' else 'linear_wp'
        response=film.exposure_response(e.e.stock,spec,method=method);raw=response(signal);bad=raw<=1e-9
        H+=wk[...,None]*np.clip(raw,1e-9,None);floor_weights+=wk[...,None]*bad
        branches.append({'class':c.key,'member':members[c.key][0],'mean_weight':share,
            'pixels_with_nonpositive_response_per_layer':(raw<=0).sum((0,1)).tolist(),
            'max_weight_on_any_floor_limited_pixel':float(wk[bad.any(-1)].max()) if bad.any() else 0.})
    replay=float(abs(np.clip(H,1e-9,None)-baseline).max())
    if replay>1e-9:raise ValueError('Exposure replay mismatch '+str(replay))
    base=np.load(ROOT/'base-colour-audit'/(sid+'.npz'));shape=base['real'].shape[:2]
    meta=json.loads((ROOT/'restored-cafe/settings.json' if sid=='cafe' else ROOT/'epfl-highlights'/(sid+'.json')).read_text())
    tf=ProjectiveTransform(np.array(meta['geometry']))
    align=lambda a:warp(a,tf.inverse,output_shape=shape,order=1,cval=np.nan)
    ar=align(signal);af=align(floor_weights)
    mask=base['mask_far_geometric_1500']
    report={'scene':sid,'scope':__doc__,'branch_details':branches,'baseline_exposure_reproduction_max':replay,
            'native':coverage(signal,floor_weights,np.ones(signal.shape[:2],bool)),
            'native_positive_Y_gt_1e-4':coverage(signal,floor_weights,signal@rawio.LUMA>1e-4),
            'fixed_geometric_far_region':coverage(ar,af,mask),
            'limits':['Weight of invalid branches is a model diagnostic, not lost true exposure, signal confidence or actual sensor saturation.',
                      'Real RAW linear sRGB is a camera-profile conversion, not measured SSF; physical spectral admissibility remains to be checked.',
                      'Same existing geometric masks, not new independent validation; no processing or output parameter changed.']}
    if sid=='R1-06-7':
        m=np.zeros(shape,bool);m[880:1030,400:780]=True
        report['previous_foreground_grass_rectangle']=coverage(ar,af,m)
    snapshot=json.loads((ROOT/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(changed)
    report['production']={'verified_files':len(snapshot),'changed':changed}
    np.savez_compressed(OUT/(sid+'.npz'),floor_branch_weights=af,input_linear_RGB=ar,region_mask=mask)
    (OUT/(sid+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('scene',choices=['controlled','cafe','R1-07-8','R1-06-7']);a=p.parse_args()
    if a.scene=='controlled':controlled()
    else:job(a.scene)
