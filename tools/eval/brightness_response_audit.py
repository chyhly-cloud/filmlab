"""Brightness-resolved red-response audit; existing probes and fixed EPFL ROIs.

No fitting or new optical/colour candidates. RGB measurements always use linear
sRGB; encoded values are for display only. Probe bins use the unperturbed output;
real-image bins use measured-scan luminance, plus a cross-tab to model luminance.
"""
from dataclasses import replace
from pathlib import Path
import argparse,json,sys,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
LUMA=np.array([.2126,.7152,.0722])
EDGES=np.array([0.,.005,.05,.4,.85,np.inf])
LABELS=['black','dark','transition','bright_transition','near_white']


def brightness_bin(rgb):
    """Input must already be linear sRGB; never decodes or assumes encoded RGB."""
    y=np.asarray(rgb)@LUMA
    out=np.searchsorted(EDGES[1:],np.maximum(y,0),side='right')
    return np.where(np.isfinite(y),out,-1)


def lab(rgb):
    import colour
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb,0,1),apply_cctf_decoding=False))


def chroma_step(a,b):return np.linalg.norm((lab(b)-lab(a))[...,1:],axis=-1)


def med(x):
    a=np.asarray(x);return float(np.nanmedian(a)) if a.size else None


def conversion_info(scanner):
    c=scanner.conversion;matrix=c.P*c.gain
    return {'P':c.P.tolist(),'gain':c.gain.tolist(),'combined_matrix':matrix.tolist(),
            'singular_values':np.linalg.svd(matrix,compute_uv=False).tolist(),
            'condition_number':float(np.linalg.cond(matrix)),
            'scope':'fixed linear density conversion; invertibility is not evidence of correct device colour'}


def probes(root,out):
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    a=np.load(root/'stages.npz');old=json.loads((root/'results.json').read_text())
    generation.use('next');e=Engine(Params(**{**recipe('cinestill_800t'),'grain':0,'roll':None,'timing':'standard','output_conditioning':None}))
    groups={}
    for own in ['exponential','paper_only']:
        bins=brightness_bin(a['base_'+own+'_linear_rgb']);rows=[]
        for i,label in enumerate(LABELS):
            mask=bins==i;row={'bin':label,'samples':int(mask.sum()),'own_binning_output':own}
            if mask.any():
                dc=a['red_0.5_dye']-a['base_dye'];ds=a['red_0.5_scan_density']-a['base_scan_density'];dn=a['red_0.5_normalized_density']-a['base_normalized_density']
                row.update({'baseline_Y':med((a['base_'+own+'_linear_rgb']@LUMA)[mask]),
                            'cyan_increment':med(dc[mask,0]),
                            'density_650nm_increment':med((a['red_0.5_spectral_density']-a['base_spectral_density'])[mask,(650-380)//5]),
                            'scan_density_R_increment':med(ds[mask,0]),'normalized_density_R_increment':med(dn[mask,0]),
                            'density_conversion_norm_ratio':med(np.linalg.norm(dn[mask],axis=-1)/np.maximum(np.linalg.norm(ds[mask],axis=-1),1e-12))})
                for key in ['exponential','paper_only']:
                    b=a['base_'+key+'_linear_rgb'];c=a['red_0.5_'+key+'_linear_rgb'];d=lab(c)-lab(b)
                    row[key]={'baseline_Y':med((b@LUMA)[mask]),'delta_a':med(d[mask,1]),'delta_b':med(d[mask,2]),
                        'chroma_delta':med(np.linalg.norm(d[mask,1:],axis=-1)),
                        'chroma_per_cyan_increment':med((np.linalg.norm(d[:,1:],axis=-1)/np.maximum(dc[:,0],1e-12))[mask]),
                        '8bit_identical_fraction':float(np.all(a['red_0.5_'+key+'_8bit'][mask]==a['base_'+key+'_8bit'][mask],axis=-1).mean()),
                        'display_clipped_fraction':float((a['base_'+key+'_clipped'][mask]|a['red_0.5_'+key+'_clipped'][mask]).any(-1).mean())}
            rows.append(row)
        groups[own]=rows
    report={'scope':'rebin the existing 297-probe experiment; primary summaries use existing +.5 EV finite increments, no new exposure grid',
            'linear_Y_edges':[0,.005,.05,.4,.85,'infinity'],'bins':LABELS,'groups':groups,
            'conversion':conversion_info(e.scanner),'source_timing':old['timing']}
    # Reuse all three existing increments, always assigning bins before perturbation.
    details=[]
    bins=brightness_bin(a['base_exponential_linear_rgb'])
    for ev in [.1,.5,1.]:
        prefix='red_'+str(ev)
        ds=a[prefix+'_scan_density']-a['base_scan_density']
        dn=a[prefix+'_normalized_density']-a['base_normalized_density']
        closure=np.max(abs(dn-ds@np.array(report['conversion']['combined_matrix'])))
        for bi,label in enumerate(LABELS):
            mask=bins==bi
            row={'red_EV':ev,'bin':label,'samples':int(mask.sum()),'conversion_closure_max':float(closure),'outputs':{}}
            for key in ['exponential','paper_only']:
                dx=a[prefix+'_'+key+'_paper_exposure']-a['base_'+key+'_paper_exposure']
                dp=a[prefix+'_'+key+'_paper_density']-a['base_'+key+'_paper_density']
                before=a['base_'+key+'_linear_rgb'];after=a[prefix+'_'+key+'_linear_rgb']
                dc=a[prefix+'_dye'][:,0]-a['base_dye'][:,0]
                row['outputs'][key]={
                    'delta_paper_exposure_R':med(dx[mask,0]),'delta_paper_density_R':med(dp[mask,0]),
                    'paper_density_per_exposure_R':med(abs(dp[mask,0])/np.maximum(abs(dx[mask,0]),1e-15)),
                    'chroma_delta':med(chroma_step(before,after)[mask]),
                    'chroma_per_cyan':med((chroma_step(before,after)/np.maximum(dc,1e-15))[mask]),
                    'max_linear_RGB_delta_median':med(np.max(abs(after-before),axis=-1)[mask]),
                    '8bit_identical_fraction':float(np.all(a[prefix+'_'+key+'_8bit'][mask]==a['base_'+key+'_8bit'][mask],axis=-1).mean())}
            details.append(row)
    report['all_297_rebinned']=details
    (out/'probe_brightness.json').write_text(json.dumps(report,indent=2))
    return report


def scan_local(engine,state,timing):
    from filmlab.scanning.optics import measure,led_response
    from filmlab.scanning.paper import OUTPUT_BLACK
    T=engine._transmittance(state);D=-np.log10(np.maximum(T,1e-30))
    scan=measure(T,engine.scanner.m);dn=engine.scanner.conversion(scan)
    result={'dye':state.dye,'spectral_450_550_650':D[...,[14,34,54]],'scan_density':scan,'normalized_density':dn,
            'scan_floor':(T@led_response(engine.scanner.m)<=1e-6).any(-1).astype(float)}
    for key,t in [('exp',timing),('paper',replace(timing,x_knee=None,x_white=None))]:
        xe=t.paper_exposure(dn);dp=engine.scanner.digital_paper.density(xe)
        raw=engine.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(dn.shape)
        result.update({key:np.clip(raw,0,None),key+'_xe':xe,key+'_dp':dp,
                       key+'_clip':((raw<0)|(raw>1)).any(-1).astype(float)})
    return result,D


def image_scene(sid,regions,out):
    import colour,rawpy
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt,gaussian_filter
    from skimage.transform import ProjectiveTransform
    from filmlab import generation,rawio
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.spectral import srgb_encode
    from light_acceptance import crop_warp,pixel_scaling
    from epfl_highlights import digest
    import oracle_timing as ot
    generation.use('next_sensor')
    rawpath,scanpath,bias=ot.sources(sid)
    for p,key in [(rawpath,'raw_sha256'),(scanpath,'film_sha256')]:
        if digest(p)!=ot.MAN[sid][key]:raise ValueError('Source hash mismatch')
    rgb=rawio.load_raw(rawpath,max_size=3000)
    with rawpy.imread(rawpath) as raw:sat=rawio.sensor_saturation(raw,per_channel=True)
    seed=zlib.crc32(sid.encode())%1000
    metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    eng=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    tr={};negative=eng.develop_negative(metered,trace=tr)
    # Remove only the modeled red return, keeping green/blue return and calibration.
    removed=tr['total_exposure'].copy();removed[...,0]=tr['scattered_exposure'][...,0]
    q=negative.log_exposure;qr=eng._log_exposure(removed)
    st=eng._develop(qr,blur_px=12/eng.px_um);eng._agitation(st);eng._grain(st)
    del rgb,metered,removed,tr
    priorpath=out.parent/'epfl-highlights'/(sid+'.json');prior=json.loads(priorpath.read_text());t=prior['timing']
    timing=PrintTiming(K=np.array(t['K']),s=np.array(t['s']),M=tuple(np.array(v) for v in t['M']),x_knee=t['x_knee'],x_white=t['x_white'])
    with Image.open(scanpath) as im:real=colour.models.eotf_sRGB(np.asarray(im.convert('RGB'),float)/255)
    h,w=real.shape[:2];um=36000/w;step=max(1,round(max(q.shape[:2])/1400))
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real']
    H=pixel_scaling(w/oldreal.shape[1],h/oldreal.shape[0])@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    rows=[]
    for region in regions:
        name=sid+'_'+region['name'];center=np.array(region['center'])*[w/oldreal.shape[1],h/oldreal.shape[0]]
        x0,y0=np.round(center).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0)
        xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[q.shape[1],q.shape[0]])
        transform=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        wp=crop_warp(ProjectiveTransform(transform))
        def sub(state):return replace(state,**{k:v[yl:yh,xl:xh] for k,v in vars(state).items() if isinstance(v,np.ndarray) and v.shape[:2]==q.shape[:2]})
        base,D=scan_local(eng,sub(negative.state),timing);noR,Dr=scan_local(eng,sub(st),timing)
        images={'real':real[y0:y0+512,x0:x0+512]}
        images.update({'base_'+k:wp(v) for k,v in base.items()});images.update({'minus_red_'+k:wp(v) for k,v in noR.items()})
        images['red_EV']=wp((q[yl:yh,xl:xh,0]-qr[yl:yh,xl:xh,0])*np.log2(10))
        images['spectral_delta_rms']=wp(np.sqrt(np.mean((D-Dr)**2,axis=-1)))
        satlocal=sat[yl:yh,xl:xh].any(-1).astype(float)
        measured=wp(satlocal,order=0)<.5
        cache=out.parent/'light-acceptance'/(name+'.npz')
        if cache.exists():
            with np.load(cache) as old:
                core=old['core'];measured&=old['measured'];prior_exp=old['exp_plain']
            error=float(np.nanmax(abs(images['base_exp']-prior_exp)))
            if error>2e-6:raise ValueError('Earlier plain render not reproduced: '+name+str(error))
            dist=distance_transform_edt(~core)*um
            if not core.any():continue
            origin='distance outside pre-existing sensor-core mask'
        else:
            yy,xx=np.indices((512,512));dist=np.hypot(xx-256,yy-256)*um
            # Existing manual centres: 60um central exclusion is only a geometric mask.
            core=dist<=60;dist=np.maximum(dist-60,0);error=None
            origin='distance outside 60um manual central disk, not measured lamp edge'
        valid=np.isfinite(images['base_exp']).all(-1)&np.isfinite(images['base_paper']).all(-1)
        smooth={key:gaussian_filter(np.nan_to_num(images[key]@LUMA),25/um) for key in ['real','base_exp','base_paper']}
        rb=np.searchsorted(EDGES[1:],smooth['real'],side='right');eb=np.searchsorted(EDGES[1:],smooth['base_exp'],side='right')
        stats=[]
        for zone,spatial in [('near_0_300',(dist>0)&(dist<=300)),('outer_300_800',(dist>300)&(dist<=800))]:
            for bi,label in enumerate(LABELS):
                m=spatial&measured&valid&(rb==bi)
                row={'zone':zone,'real_brightness_bin':label,'pixels':int(m.sum())}
                if m.sum()>=30:
                    drgb=images['base_dye']-images['minus_red_dye'];ds=images['base_scan_density']-images['minus_red_scan_density'];dn=images['base_normalized_density']-images['minus_red_normalized_density']
                    qev=images['red_EV'];dose=m&(qev>.001)&(drgb[...,0]>1e-8)
                    row.update({'real_Y':med(smooth['real'][m]),'model_Y':med(smooth['base_exp'][m]),
                        'model_bins':{label2:int((m&(eb==j)).sum()) for j,label2 in enumerate(LABELS)},
                        'real_logY_p90_p10':float(np.ptp(np.percentile(np.log2(np.maximum(smooth['real'][m],1e-6)),[10,90]))),
                        'model_logY_p90_p10':float(np.ptp(np.percentile(np.log2(np.maximum(smooth['base_exp'][m],1e-6)),[10,90]))),
                        'red_added_EV':med(qev[m]),'cyan_increment':med(drgb[m,0]),
                        'spectral_density_delta_rms':med(images['spectral_delta_rms'][m]),
                        'spectral_density_delta_650':med((images['base_spectral_450_550_650']-images['minus_red_spectral_450_550_650'])[m,2]),
                        'scan_density_R_increment':med(ds[m,0]),'normalized_density_R_increment':med(dn[m,0]),
                        'scan_floor_fraction':float((images['base_scan_floor'][m]>.01).mean()),
                        'outputs':{}})
                    real_lab=lab(images['real'])
                    for key in ['exp','paper']:
                        b=images['base_'+key];n=images['minus_red_'+key];dl=lab(b)-lab(n);gap=lab(b)-real_lab
                        row['outputs'][key]={'Y':med(smooth['base_'+key][m]),'dLab_vs_real':np.median(gap[m],axis=0).tolist(),
                            'red_delta_a':med(dl[m,1]),'red_delta_b':med(dl[m,2]),'red_chroma_delta':med(np.linalg.norm(dl[m,1:],axis=-1)),
                            'red_chroma_per_EV':med((np.linalg.norm(dl[...,1:],axis=-1)/np.maximum(qev,1e-8))[dose]),
                            'red_chroma_per_cyan':med((np.linalg.norm(dl[...,1:],axis=-1)/np.maximum(drgb[...,0],1e-8))[dose]),
                            'paper_exposure_red_increment':med((images['base_'+key+'_xe']-images['minus_red_'+key+'_xe'])[m,0]),
                            'paper_density_red_increment':med((images['base_'+key+'_dp']-images['minus_red_'+key+'_dp'])[m,0]),
                            'display_clipped_fraction':float((images['base_'+key+'_clip'][m]>.01).mean())}
                stats.append(row)
        images['measured']=measured;images['distance_um']=dist;images['real_bin']=rb;images['model_bin']=eb
        np.savez_compressed(out/(name+'.npz'),**{k:np.asarray(v,dtype=np.float32) for k,v in images.items()},
                            native_base_spectral_density=D.astype('float32'),native_minus_red_spectral_density=Dr.astype('float32'))
        sheet=Image.new('RGB',(2048,560),(18,18,18));d=ImageDraw.Draw(sheet)
        for j,(key,label) in enumerate([('real','Real scan'),('base_exp','Fixed exp_plain'),('base_paper','Fixed paper_only'),('minus_red_exp','Diagnostic: red return removed')]):
            im=Image.fromarray(np.uint8(np.clip(srgb_encode(np.nan_to_num(images[key]))*255+.5,0,255)))
            sheet.paste(im,(j*512,48));d.text((j*512+8,10),label,fill='white');d.text((j*512+8,26),name,fill='white')
        sheet.save(out/(name+'_comparison.png'))
        rows.append({'name':name,'geometry_origin':origin,'baseline_reproduction_error':error,'statistics':stats})
        print(name,'done',flush=True)
    result={'scene':sid,'metering':note,'source_hashes':{k:ot.MAN[sid][k] for k in ['raw_sha256','film_sha256']},
            'timing':t,'timing_policy':'fixed same saved operator timing for every contrast',
            'generation':'next_sensor','conditioning':None,'grain':0,'red_intervention':'remove modeled red return only, keep G/B return and exposure calibration',
            'bins':LABELS,'linear_Y_edges':[0,.005,.05,.4,.85,'infinity'],'bin_assignment':'25um-smoothed real linear luminance; model cross-tab retained',
            'conversion':conversion_info(eng.scanner),'scan_um_per_pixel':um,'model_um_per_pixel':eng.px_um,
            'limits':['source cores not recovered; red-return amount is a model intervention, not measured lamp radiance',
                      'only unflagged surrounding model pixels used; source prior still influences their returned light',
                      '36mm full scan width assumed; crop/registration and effective resolution limit distances',
                      'scan-only comparison cannot identify physical red-return delta or scanner response'],
            'regions':rows}
    (out/(sid+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--probes',type=Path,required=True);ap.add_argument('--scenes',nargs='*',default=['R1-00-1','R1-02-3','R1-04-5','R1-07-8'])
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    probes(args.probes,args.out)
    regions=json.loads((REPO/'tools/eval/light_acceptance_regions.json').read_text())['regions']
    regions=[r for r in regions if r['name']!='red_beacon']
    regions += [{'scene':'R1-02-3','name':'middle_ceiling','center':[706,235]},
                {'scene':'R1-02-3','name':'glass_reflection','center':[1437,405]}]
    for sid in args.scenes:image_scene(sid,[r for r in regions if r['scene']==sid],args.out)


if __name__=='__main__':main()
