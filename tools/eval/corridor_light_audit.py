"""Three manually selected corridor lights: fixed-input optical/development ablation.
No fits. Ring distances assume a 36 mm full scan width; not calibrated film geometry.
"""
from dataclasses import replace
from pathlib import Path
import json, sys, zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]

def main():
    import colour, rawpy
    from PIL import Image, ImageDraw
    from skimage.transform import ProjectiveTransform
    from filmlab import generation, rawio
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from light_acceptance import crop_warp,pixel_scaling
    import oracle_timing as ot
    out=Path('/workspace/filmlab-audit/corridor-lights');out.mkdir(exist_ok=True)
    sid='R1-02-3'; generation.use('next_sensor')
    rawpath,scanpath,bias=ot.sources(sid)
    rgb=rawio.load_raw(rawpath,max_size=3000)
    with rawpy.imread(rawpath) as raw: sat=rawio.sensor_saturation(raw,per_channel=True)
    seed=zlib.crc32(sid.encode())%1000
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    eng=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    tr={}; neg=eng.develop_negative(metered,trace=tr)
    print('negative complete',flush=True)
    prior=json.loads(Path('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json').read_text());t=prior['timing']
    eng.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    dn=neg.measure(eng.scanner.normalized_density)
    exp=eng.scanner.to_rgb(dn)
    eng.scanner.timing=replace(eng.scanner.timing,x_knee=None,x_white=None)
    paper=eng.scanner.to_rgb(dn)
    eng.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    logH=eng._log_exposure(np.clip(tr['scattered_exposure'],1e-9,None))
    state=eng._develop(logH,blur_px=12/eng.px_um);eng._agitation(state);eng._grain(state)
    no=replace(neg,state=state,log_exposure=logH)
    none=eng.scanner.to_rgb(no.measure(eng.scanner.normalized_density))
    with Image.open(scanpath) as im:real=colour.models.eotf_sRGB(np.asarray(im.convert('RGB'),float)/255)
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real']
    h,w=real.shape[:2];step=max(1,round(max(sat.shape[:2])/1400))
    matrix=pixel_scaling(w/oldreal.shape[1],h/oldreal.shape[0])@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    report={'scene':sid,'metering':note,'model_um_per_pixel':eng.px_um,'scan_um_per_pixel':36000/w,'regions':[],'limits':__doc__}
    for name,c in [('near_ceiling',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        x,y=np.array(c)*[w/oldreal.shape[1],h/oldreal.shape[0]];x0,y0=int(round(x))-256,int(round(y))-256
        wp=crop_warp(ProjectiveTransform(np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@matrix))
        images={'real':real[y0:y0+512,x0:x0+512],'exp':wp(exp),'paper':wp(paper),'none':wp(none)}
        yy,xx=np.indices((512,512));dist=np.hypot(xx-256,yy-256)*36000/w
        halo_ev=wp(np.log2(np.maximum(tr['total_exposure'],1e-12)/np.maximum(tr['scattered_exposure'],1e-12)))
        dye=wp(neg.dye-no.dye);sf=wp(sat.astype(float),order=0)
        rows={}
        for lo,hi in [(0,100),(100,200),(200,300)]:
            mask=(dist>=lo)&(dist<hi)
            rows[f'{lo}_{hi}']={'pixels':int(mask.sum()),'mean_added_exposure_EV_RGB':np.mean(halo_ev[mask],0).tolist(),'mean_dye_change_RGB':np.mean(dye[mask],0).tolist(),'any_sensor_saturated_fraction':float((sf[mask].max(-1)>.5).mean()),'all_sensor_saturated_fraction':float((sf[mask].min(-1)>.5).mean()),'output_mean_abs_halo_difference_linear_RGB':np.mean(np.abs(images['exp'][mask]-images['none'][mask]),0).tolist()}
        sheet=Image.new('RGB',(2048,550),(20,20,20));draw=ImageDraw.Draw(sheet)
        for i,(key,label) in enumerate([('real','Real CineStill scan'),('exp','exp_plain (existing halo)'),('paper','paper_plain (existing halo)'),('none','Diagnostic: no optical halo')]):
            arr=np.floor(srgb_encode(np.nan_to_num(images[key]))*255+.5).astype('uint8')
            im=Image.fromarray(arr);im.save(out/(name+'_'+key+'.png'));sheet.paste(im,(512*i,38));draw.text((512*i+8,10),label,fill='white')
        sheet.save(out/(name+'_comparison.png'))
        np.savez_compressed(out/(name+'.npz'),**images,added_exposure_EV=halo_ev,dye_change=dye,saturated_channels=sf)
        report['regions'].append({'name':name,'center_old_scan':c,'rings_um':rows})
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
