"""Validated RAW radiance through unchanged film, chemistry and frozen scanning.

Random priors are replaced only where independent RAW support passes. Unknown
cores retain the baseline prior and remain labelled unresolved. Illumination
weights and library members are frozen for this causal input ablation; real
lamp spectra are not claimed to be measured. No final colour or curve fitting.
"""
import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from filmlab.brackets import MeasuredHighlightPatch,replace_measured_highlights
from linear_raw_measurements import bayer_measurements,decoder_matrix
from light_acceptance import crop_warp,pixel_scaling,region_statistics


def select_patches(lamps,directory):
    patches=[];evidence=[]
    for lamp in lamps:
        region=lamp['region'];name=region['scene']+'_'+region['name']
        data=np.load(directory/(name+'_measurements.npz'))
        core=data['mask'];base=data['0_camera'];base_sat=data['0_sat']
        choices=[]
        for i,alternative in enumerate(lamp['alternatives'],1):
            if not alternative['local_geometry']['pass']:continue
            value=data[f'{i}_camera'];safe=data[f'{i}_safe'];accepted=np.zeros(core.shape,bool)
            # Strong support: two independent alternative exposures, each with
            # all channels safe, agree at every selected pixel within .15 EV.
            for j,other in enumerate(lamp['alternatives'],1):
                if i==j or not other['local_geometry']['pass']:continue
                v2=data[f'{j}_camera'];s2=data[f'{j}_safe']
                ratio=np.log2(np.maximum(value,1e-9)/np.maximum(v2,1e-9))
                accepted|=safe&s2&np.isfinite(ratio).all(-1)&(np.abs(ratio)<.15).all(-1)&(value.min(-1)>.03)&(v2.min(-1)>.03)
            strong=accepted.copy()
            # Single alternate: require all three channels stable over the
            # source's measurable collar, plus agreement in every available
            # primary channel at each replaced pixel. Never admits a fully
            # saturated primary pixel without a second alternative exposure.
            comparisons=[r for r in lamp['comparisons'] if r['first']=='primary' and r['second']==alternative['file']]
            collar_ok=bool(comparisons) and all(c['median_stops'] is not None and abs(c['median_stops'])<.15 for c in comparisons[0]['channels'])
            if collar_ok:
                available=(~base_sat)&(base>.03)
                ratio=np.log2(np.maximum(value,1e-9)/np.maximum(base,1e-9))
                agree=np.all((np.abs(ratio)<.15)|~available,axis=-1)&available.any(-1)
                accepted|=safe&agree&(value.min(-1)>.01)
            accepted&=core&base_sat.any(-1)&np.isfinite(value).all(-1)
            strong&=accepted
            choices.append((alternative['scale'],i,accepted,strong))
        chosen=np.zeros(core.shape,bool);counts=[]
        # Prefer the longest non-clipped validated measurement, for photon SNR.
        for scale,i,accepted,strong in sorted(choices):
            use=accepted&~chosen;chosen|=use
            if not use.any():continue
            x,y=region['center'];half=region['half']
            patches.append(MeasuredHighlightPatch((x-half,y-half),data[f'{i}_camera'],use,
                           name+' | '+lamp['views'][i]+' | fixed metadata normalization and independent RAW checks'))
            counts.append({'file':lamp['views'][i],'pixels':int(use.sum()),'cross_alternate_pixels':int((strong&use).sum())})
        evidence.append({'name':name,'kind':region['kind'],'primary_any_saturated':lamp['primary_any_saturated'],
                         'primary_all_saturated':lamp['primary_all_saturated'],'replaced_pixels':int(chosen.sum()),
                         'replaced_all_saturated_pixels':int((chosen&base_sat.all(-1)).sum()),'chosen':counts})
    return patches,evidence


def freeze_illumination(eng):
    from scipy.ndimage import zoom
    from filmlab import lights,film
    Wsmall=eng.light_W.copy();mix=dict(eng.light_mix)
    def expose(rgb):
        h,w=rgb.shape[:2];W=zoom(Wsmall,(h/Wsmall.shape[0],w/Wsmall.shape[1],1),order=1)[:h,:w]
        H=np.zeros(rgb.shape);source=np.zeros(rgb.shape)
        eng.light_W=Wsmall;eng.light_mix=mix
        for k,c in enumerate(lights.classes()):
            if float(W[...,k].mean())<.003:continue
            name=mix[c.name][0];spec=next(spec for n,spec in c.members if n==name)
            method='spectral' if c.key=='daylight' else ('linear_wp' if 'linear_white' in eng.p.corrections else 'linear')
            H+=W[...,k,None]*film.expose(rgb,film.exposure_response(eng.e.stock,spec,eng.p.lens_filter,method))
            source+=W[...,k,None]*film.halation_response(eng.e.stock,spec,eng.p.lens_filter,eng.v,eng._back_reflect,method)(rgb)
        return np.clip(H,1e-9,None),np.clip(source,0,None)
    return expose


def panel(images,path,name,scan_um,model_um):
    from PIL import Image,ImageDraw,ImageFont
    from filmlab.spectral import srgb_encode
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    output=Image.new('RGB',(3072,1120),(18,18,18))
    for i,(key,title) in enumerate([('real','Real scan'),('prior','Existing prior'),('measured','Validated RAW replacement')]):
        a=np.uint8(np.clip(srgb_encode(np.nan_to_num(images[key]))*255+.5,0,255))
        Image.fromarray(a).save(path.with_name(path.name+'_'+key+'.png'))
        output.paste(Image.fromarray(a).resize((1024,1024),Image.Resampling.NEAREST),(1024*i,96))
        d=ImageDraw.Draw(output);d.text((1024*i+12,8),title,font=font,fill='white')
        d.text((1024*i+12,36),name+' | same film/scan/timing',font=font,fill='white')
        d.text((1024*i+12,65),f'TIFF {scan_um:.2f} um/px; model {model_um:.2f} um/px; display 2x',font=small,fill='white')
        d.rectangle((1024*i+20,1050,1024*i+240,1104),fill=(18,18,18))
        d.text((1024*i+30,1053),'200 um (nominal)',font=small,fill='white')
        d.line((1024*i+30,1085,1024*i+30+round(400/scan_um),1085),fill='white',width=5)
    output.save(path.with_name(path.name+'_compare.png'))


def run_scene(sid,lamps,args):
    import colour
    from PIL import Image
    from scipy.ndimage import binary_dilation
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,rawio,film
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from epfl_highlights import digest
    import oracle_timing as ot
    generation.use('next_sensor')
    rawpath,scanpath,bias=ot.sources(sid)
    for path,field in ((rawpath,'raw_sha256'),(scanpath,'film_sha256')):
        if digest(path)!=ot.MAN[sid][field]:raise ValueError('Primary checksum mismatch')
    camera,sat,meta=bayer_measurements(rawpath)
    decoder,note=decoder_matrix(rawpath)
    old=json.loads((args.cache/(sid+'.json')).read_text())
    prior=rawio.load_raw(rawpath,max_size=3000)
    patches,evidence=select_patches(lamps,args.measurements)
    candidate,replaced=replace_measured_highlights(prior,sat,patches,decoder)
    seed=zlib.crc32(sid.encode())%1000
    metered,meter_note=rawio.film_metering(prior,rng=np.random.default_rng(seed),bias_ev=bias)
    positive=np.flatnonzero(np.abs(prior.ravel())>.01)[0]
    gain=metered.ravel()[positive]/prior.ravel()[positive]
    eng=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    t=old['timing'];eng.scanner.timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(a) for a in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    eng.scanner.conditioning='lab'
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real']
    with Image.open(scanpath) as im: real=colour.models.eotf_sRGB(np.asarray(im.convert('RGB'),float)/255)
    h,w=real.shape[:2];step=max(1,int(round(max(prior.shape[:2])/1400)))
    matrix=pixel_scaling(w/oldreal.shape[1],h/oldreal.shape[0])@np.array(old['geometry'])@pixel_scaling(1/step,1/step)
    whole=ProjectiveTransform(matrix);items=[]
    for lamp in lamps:
        region=lamp['region'];x,y=region['center'];cx,cy=whole([[x,y]])[0]
        x0=int(np.clip(round(cx)-256,0,w-512));y0=int(np.clip(round(cy)-256,0,h-512))
        wp=crop_warp(ProjectiveTransform(np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@matrix))
        yy,xx=np.mgrid[:sat.shape[0],:sat.shape[1]]
        local=np.hypot(xx-x,yy-y)<=region['radius']
        core=wp((sat.any(-1)&local).astype(float),0)>.5
        images={'real':real[y0:y0+512,x0:x0+512]}
        items.append({'name':sid+'_'+region['name'],'region':region,'warp':wp,'core':core,
                      'measured':wp((~binary_dilation(sat.any(-1),iterations=1)).astype(float))>.999,
                      'images':images,'stages':{},'replaced_mask':wp(replaced.astype(float),0)>.5,
                      'unresolved_core':wp((sat.any(-1)&~replaced&local).astype(float),0)>.5})
    del real,camera
    def summarize(item,values):
        from scipy.ndimage import distance_transform_edt
        v=item['warp'](values);distance=distance_transform_edt(~item['core'])*36000/w
        masks={'core':item['core'],'ring_0_150':(distance>0)&(distance<=150),
               'ring_150_300':(distance>150)&(distance<=300),'ring_300_800':(distance>300)&(distance<=800)}
        if not item['core'].any():return {}
        return {name:np.mean(v[mask&np.isfinite(v).all(-1)],0).tolist() for name,mask in masks.items()
                if (mask&np.isfinite(v).all(-1)).sum()>=8}
    for variant,input_rgb in (('prior',metered),('measured',candidate*gain)):
        trace={};neg=eng.develop_negative(input_rgb,trace=trace)
        if variant=='prior':
            frozen=freeze_illumination(eng)
            # Check the frozen implementation against original inference before
            # comparing inputs: exact arithmetic/order and no hidden colour fit.
            checked,checked_src=frozen(input_rgb*2**eng.p.exposure_ev)
            if np.max(np.abs(checked-trace['direct_exposure']))>1e-10:raise ValueError('Frozen exposure mismatch')
            if np.max(np.abs(checked_src-trace['halo_source']))>1e-10:raise ValueError('Frozen halo source mismatch')
            eng._expose_mixed=frozen
            del checked,checked_src
        dn=neg.measure(eng.scanner.normalized_density);status=neg.status_m();rendered=eng.scanner.to_rgb(dn)
        halo=trace['total_exposure']-trace['scattered_exposure']
        for item in items:
            item['images'][variant]=item['warp'](rendered)
            item['stages'][variant]={key:summarize(item,a) for key,a in [('direct_exposure',trace['direct_exposure']),
                        ('halo_source',trace['halo_source']),('halo_exposure',halo),('logH',neg.log_exposure),
                        ('dye',neg.dye),('status_M_density',status),('scan_density',dn)]}
        if variant=='measured':
            unverified=film.halation(np.zeros_like(halo),trace['halo_source']*(sat.any(-1)&~replaced)[...,None],eng.px_um,eng.v,film.bounce_retention(eng.e.stock,eng.v))
            for item in items:
                item['unresolved_halo']=summarize(item,unverified)
        del trace,neg,dn,status,rendered,halo
        print(sid,variant,'chain done',flush=True)
    result=[]
    for item in items:
        images=item['images'];valid=np.isfinite(images['prior']).all(-1)
        stats=region_statistics(images,item['core'],item['measured'],valid,36000/w)
        panel(images,args.out/item['name'],item['name'],36000/w,eng.px_um)
        np.savez_compressed(args.out/(item['name']+'_chain.npz'),**{k:v.astype('float32') for k,v in images.items()},
                            core=item['core'],replaced=item['replaced_mask'],unresolved_core=item['unresolved_core'])
        result.append({'name':item['name'],'region':item['region'],'statistics':stats,'stages':item['stages'],
                       'unresolved_prior_halo':item['unresolved_halo'],'core_pixels_scan':int(item['core'].sum()),
                       'unresolved_core_pixels_scan':int(item['unresolved_core'].sum())})
    report={'scene':sid,'replacement_evidence':evidence,'replaced_raw_pixels':int(replaced.sum()),'meter_gain':float(gain),
            'meter_note':meter_note,'frozen_timing':t,'light_mix':eng.light_mix,'decoder_matrix':decoder.tolist(),
            'decoder_probe':note,'scan_um_per_pixel':36000/w,'model_um_per_pixel':eng.px_um,'scope':__doc__,'regions':result}
    (args.out/(sid+'_chain.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--measurements',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--scenes',nargs='+',default=['R1-00-1','R1-04-5','R1-05-6','R1-06-7'])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    lamps=json.loads((args.measurements/'lamp-measurements.json').read_text())['lamps'];results=[]
    for sid in args.scenes:
        results.append(run_scene(sid,[r for r in lamps if r['region']['scene']==sid],args))
    (args.out/'chain-results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
