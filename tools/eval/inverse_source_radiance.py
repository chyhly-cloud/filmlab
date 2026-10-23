"""Conditional source-radiance inversion with fixed film/scan, never a material measurement.

Authorized inverse experiment: source spectrum/strength may be fitted to selected
real-scan samples. Direct and returning exposure share one nonnegative spectrum.
Film, transport, source support/profile, timing, geometry and development stay fixed.
Held-out sectors/radii are never used to choose a fit. Missing radiance stays estimated.
"""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import argparse,json,sys,time,zlib
import numpy as np
from scipy import ndimage,sparse
from scipy.optimize import least_squares,minimize_scalar
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab
from corridor_cache_geometry import bilinear_operator
from inverse_radiance_checks import sensor_consistency,near_equivalent_fit_report

LUMA=np.array([.2126,.7152,.0722])
REGIONS=('ceiling_lamp','middle_ceiling')


def shared_spectral_response(coefficients, spectra, direct, returning):
    """One nonnegative spectrum drives BOTH paths in the same linear units."""
    c=np.asarray(coefficients,float);b=np.asarray(spectra,float)
    if c.ndim!=1 or b.ndim!=2 or len(c)!=len(b) or np.any(c<0) or np.any(b<0):
        raise ValueError('nonnegative matching source coefficients and spectra required')
    spectrum=c@b
    return spectrum@direct,spectrum@returning


def replace_source(background,basis,coefficients):
    c=np.asarray(coefficients,float)
    if np.any(c<0):raise ValueError('source coefficients must be nonnegative')
    return np.asarray(background)+np.einsum('k,khwc->hwc',c,basis,optimize=True)


def sector_split(distance, measured, center, *, gap_degrees=12, aperture_radius_pixels=0):
    """Predeclared disjoint sectors, with gaps plus an independent outer radius."""
    yy,xx=np.indices(distance.shape);angle=np.mod(np.arctan2(yy-center[0],xx-center[1]),2*np.pi)
    sector=np.floor(angle/(np.pi/2)).astype(int)
    phase=np.mod(angle,np.pi/2)*180/np.pi
    angular=(phase>gap_degrees)&(phase<90-gap_degrees)
    aperture_guard=(np.abs(yy-center[0])>aperture_radius_pixels)&(np.abs(xx-center[1])>aperture_radius_pixels)
    inner=measured&(distance>=50)&(distance<=300)&angular&aperture_guard
    return inner&(sector%2==0),inner&(sector%2==1),measured&(distance>300)&(distance<=800)


def strict_full_cfa(raw):
    """Existing pile-up criteria, retaining the two green samples separately."""
    from linear_raw_measurements import orient
    data=raw.raw_image_visible;col=raw.raw_colors_visible;black=np.asarray(raw.black_level_per_channel,float)
    threshold=np.full(4,np.inf)
    for c in range(4):
        v=data[col==c]
        if not v.size:continue
        spread=max(float(v.max())-black[c],1.);vals,cnt=np.unique(v[v>=v.max()-.03*spread],return_counts=True)
        peak=float(vals[np.argmax(cnt)]);tol=max(1.,.002*spread)
        top=int((v>=peak-tol).sum());below=int(((v>=peak-21*tol)&(v<peak-tol)).sum())/20
        if top>=50 and top>10*max(below,1.):threshold[c]=peak-tol
    h,w=(np.array(data.shape)//2)*2
    saturated=data[:h,:w]>=threshold[col[:h,:w]]
    full=saturated.reshape(h//2,2,w//2,2).all((1,3))
    return orient(full,raw.sizes.flip),threshold


def engine_for():
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    generation.use('next_sensor');seed=zlib.crc32(b'R1-02-3')%1000
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    meta=json.loads(Path('/workspace/filmlab-audit/brightness-local/R1-02-3.json').read_text());td=meta['timing']
    e.scanner.timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    e.px_um=15.625
    return e,meta


def spectral_library():
    from filmlab import lights
    members={n:s for c in lights.classes() for n,s in c.members}
    names=['D65','2700K 白炽 / 卤素','3300K 白炽 / 卤素','FL2','FL11','LED-B1','LED-B3','LED-B5','HP1','Metal Halide']
    spectra=np.array([members[n] for n in names]);y=lights.camera_rgb(spectra)@LUMA
    return names,spectra/y[:,None]


def prepare(out):
    import rawpy
    from filmlab import rawio,film,lights
    from filmlab.incident import spectral_responses
    from linear_raw_measurements import bayer_measurements,decoder_matrix
    import oracle_timing as ot
    out.mkdir(parents=True,exist_ok=True);e,meta=engine_for();src,_,bias=ot.sources('R1-02-3');seed=zlib.crc32(b'R1-02-3')%1000
    rgb=rawio.load_raw(src,max_size=3000);metered,note=rawio.film_metering(rgb,rng=np.random.default_rng(seed),bias_ev=bias)
    scale=float(np.sum(metered*rgb)/np.sum(rgb*rgb));captured={};old=e._expose_mixed
    def expose(a):
        direct,halo=old(a);captured.update(direct=direct.copy(),halo=halo.copy());return direct,halo
    class Captured(Exception):pass
    def stop(H,source,px,v,rho):captured.update(scattered=H,source=source,rho=rho);raise Captured
    with patch.object(e,'_expose_mixed',expose),patch.object(film,'halation',stop):
        try:e.develop_negative(metered)
        except Captured:pass
    total=film.halation(captured['scattered'],captured['source'],e.px_um,e.v,captured['rho'])
    expected_q=e._log_exposure(np.maximum(total,1e-9))
    with rawpy.imread(src) as raw:
        strict,threshold=strict_full_cfa(raw);sat=rawio.sensor_saturation(raw,per_channel=True)
    cam,_,cammeta=bayer_measurements(src);matrix,probe=decoder_matrix(src)
    cam_to_rgb=matrix@rawio.XYZ_TO_SRGB.T
    labels,_=ndimage.label(sat.any(-1));names,spectra=spectral_library();sens,ret=spectral_responses(e)
    direct_response=spectra@sens*2**e.p.exposure_ev;return_response=spectra@ret*2**e.p.exposure_ev
    camera_response=lights.camera_rgb(spectra)@np.linalg.inv(cam_to_rgb)/scale
    y=metered@LUMA
    root=Path('/workspace/filmlab-audit');reports=[]
    for name in REGIONS:
        cache=np.load(root/'corridor-dye-response/cache'/(name+'.npz'));xl,yl,xh,yh=cache['source_bounds'];sl=(slice(yl,yh),slice(xl,xh));q=cache['q_base'];valid=cache['native_trusted']
        nearest=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True);q=q[tuple(nearest)]
        safe=ndimage.binary_erosion(valid,iterations=12)
        reproduction=float(np.max(abs(q[safe]-expected_q[sl][safe])))
        if reproduction>2e-6:raise ValueError('Raw source/cache q mismatch '+str(reproduction))
        inverse=np.linalg.inv(cache['matrix'])@np.array([256.,256.,1.]);cx,cy=inverse[:2]/inverse[2]
        local_labels=labels[sl];possible=np.argwhere(strict[sl]& (local_labels>0))
        if not len(possible):raise ValueError('No strictly all-four-CFA source cells')
        chosen=possible[np.argmin(np.sum((possible-[cy,cx])**2,axis=1))];component=int(local_labels[tuple(chosen)])
        support=strict[sl]&(local_labels==component)
        # Include all strict source cells of the chosen connected luminaires, no detector-wide merge.
        amplitude=y[sl]*support;reference=float(amplitude[support].mean());profile=amplitude/max(reference,1e-12)
        removed=film.emulsion_scatter(captured['direct'][sl]*support[...,None],e.px_um)
        removed+=film.halation(np.zeros_like(removed),captured['source'][sl]*support[...,None],e.px_um,e.v,captured['rho'])
        total_local=10**q/10**(e.log_k-e.e.speed_loss+e.layer_cal)
        background=total_local-removed
        background_negative=float(background[safe].min())
        if background_negative < -1e-7:raise ValueError('Negative source-subtracted background '+str(background_negative))
        background=np.maximum(background,1e-12)
        shape3=np.repeat(profile[...,None],3,axis=-1)
        gshape=film.emulsion_scatter(shape3,e.px_um)
        kshape=film.halation(np.zeros_like(shape3),shape3,e.px_um,e.v,captured['rho'])
        basis=gshape[None]*direct_response[:,None,None,:]+kshape[None]*return_response[:,None,None,:]
        raw_dye=e._develop(q,blur_px=12/e.px_um).dye
        factors=cache['native_base_dye']/np.maximum(raw_dye,1e-12);factor=np.mean(factors,axis=-1);factor=factor[tuple(nearest)]
        spread=float(np.max(np.ptp(factors[safe],axis=-1)))
        if spread>2e-6:raise ValueError('Agitation is not a reproduced scalar')
        baseline=e._develop(q,blur_px=12/e.px_um);baseline.dye*=factor[...,None]
        local_rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(baseline)))
        from light_acceptance import crop_warp
        from skimage.transform import ProjectiveTransform
        wp=crop_warp(ProjectiveTransform(cache['matrix']));prediction=wp(local_rgb)
        oldcache=np.load(root/'brightness-local'/('R1-02-3_'+name+'.npz'));supported=wp(safe.astype(float))>.999999
        output_error=float(np.nanmax(abs(prediction[supported]-oldcache['base_exp'][supported])))
        if output_error>2e-6:raise ValueError('Baseline output failed replay '+str(output_error))
        np.savez_compressed(out/(name+'_source.npz'),background=background,basis=basis,q_base=q,profile=profile,support=support,removed=removed,total_local=total_local,
          factor=factor,safe=safe,matrix=cache['matrix'],real=cache['real'],baseline_rgb=prediction,measured=cache['measured'],distance_um=cache['distance_um'],supported=supported,
          source_camera=cam[sl][support],source_profile=profile[support],camera_response=camera_response,spectra=spectra,direct_response=direct_response,return_response=return_response)
        report={'name':name,'source_cells':int(support.sum()),'all_RGB_flagged_cells_in_component':int(((local_labels==component)&sat[sl].all(-1)).sum()),'source_bounds':[int(v) for v in cache['source_bounds']],
          'mean_legacy_source_Y':reference,'q_reproduction_max':reproduction,'RGB_reproduction_max':output_error,'agitation_channel_spread':spread,'subtract_add_closure_max':float(np.max(abs(background[safe]+removed[safe]-total_local[safe]))),'background_min_before_floor':background_negative}
        reports.append(report);print('prepared',report,flush=True)
    (out/'prepared.json').write_text(json.dumps({'scene':'R1-02-3','names':names,'spectra':spectra.tolist(),'regions':reports,'metering_scale':scale,'timing':meta['timing'],'sensor_thresholds_4CFA':threshold.tolist(),'probe':probe,'camera':cammeta,'limits':['Source support/profile fixed from censored RAW and legacy shape; not recovered emitting geometry','Camera spectral response is CIE plus inverse digital decoder proxy, NOT measured G81 spectral sensitivity','Coefficients are existing engine input-Y units, not absolute SI radiance','Material absorption, direct sensitivity, transport and scanner remain unvalidated proxies']},indent=2)+'\n')


def observation_operator(shape,matrix,points,sigma):
    """Combine output Gaussian aperture and fixed bilinear warp, without inverse resampling."""
    A,rows=bilinear_operator(shape,matrix);lookup=np.full(512*512,-1,int);lookup[rows]=np.arange(len(rows));radius=int(4*sigma+.5)
    rs=[];cs=[];vs=[]
    for i,(y,x) in enumerate(points):
        yy,xx=np.mgrid[max(0,y-radius):min(512,y+radius+1),max(0,x-radius):min(512,x+radius+1)]
        weight=np.exp(-((yy-y)**2+(xx-x)**2)/(2*sigma*sigma));weight/=weight.sum();indices=lookup[(yy*512+xx).ravel()];ok=indices>=0
        rs.extend([i]*int(ok.sum()));cs.extend(indices[ok]);vs.extend(weight.ravel()[ok])
    G=sparse.coo_matrix((vs,(rs,cs)),shape=(len(points),len(rows))).tocsr();P=G@A
    if np.max(abs(np.asarray(P.sum(1)).ravel()-1))>1e-8:raise ValueError('Incomplete observation aperture')
    return P


def fit(out,name,starts=8,grid=32,max_nfev=55):
    import colour
    from filmlab.spectral import srgb_encode
    from PIL import Image,ImageDraw,ImageFont
    a=np.load(out/(name+'_source.npz'));prep=json.loads((out/'prepared.json').read_text());e,meta=engine_for();counter=[0];start=time.time()
    background=a['background'];basis=a['basis'];qbase=a['q_base'];safe=a['safe'];factor=a['factor'];matrix=a['matrix'];real=a['real'];dist=a['distance_um'];supported=a['supported'];measured=(a['measured']>.5)&supported
    yy,xx=np.where(dist==0);center=(float(yy.mean()),float(xx.mean()));sigma=25/meta['scan_um_per_pixel'];aperture_radius=int(4*sigma+.5)+1
    train,test,outer=sector_split(dist,measured,center,aperture_radius_pixels=aperture_radius)
    # Fixed lattice, then deterministic subsample; no target/error based selection.
    def points(mask,cap):
        yy,xx=np.indices(mask.shape);p=np.argwhere(mask&(yy%3==0)&(xx%3==0));return p[np.linspace(0,len(p)-1,min(cap,len(p))).astype(int)]
    trainp=points(train,160);testp=points(test,160);outerp=points(outer,160);allpoints=np.vstack([trainp,testp,outerp]);nt=len(trainp);nv=len(testp)
    P=observation_operator(qbase.shape[:2],matrix,allpoints,sigma);indices=np.unique(P.indices);Ps=P[:,indices]
    target=ndimage.gaussian_filter(real.astype(float),(sigma,sigma,0))[tuple(allpoints.T)];target_lab=lab(target)
    selected={k:slice(v0,v1) for k,v0,v1 in [('train',0,nt),('heldout',nt,nt+nv),('outer',nt+nv,len(allpoints))]}
    def render(coefficients=None,full=False):
        counter[0]+=1
        q=qbase if coefficients is None else e._log_exposure(np.maximum(replace_source(background,basis,coefficients),1e-12))
        st=e._develop(q,blur_px=12/e.px_um);st.dye*=factor[...,None]
        if full:
            from light_acceptance import crop_warp
            from skimage.transform import ProjectiveTransform
            rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(st)));return crop_warp(ProjectiveTransform(matrix))(rgb),q,st.dye
        flat=lambda v:v.reshape(-1,3)[indices]
        sub=replace(st,dye=flat(st.dye),silver=flat(st.silver),halide=flat(st.halide))
        rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(sub)))
        return lab(Ps@rgb)
    baseline_lab=render();baseline_error=baseline_lab-target_lab
    spectra=a['spectra'];camera_response=a['camera_response'];observed=a['source_camera'];profile=a['source_profile'];amp=next(r['mean_legacy_source_Y'] for r in prep['regions'] if r['name']==name)
    # Saturated camera inequalities are a proxy, with observed lower values. No finite upper from RAW.
    def sensor_violation(c):
        predicted=profile[:,None]*(c@camera_response)[None,:]
        # Absolute 0.03 camera units is explicitly numerical tolerance, not fitted noise/CI.
        return np.minimum(predicted-observed+.03,0).ravel()/.03
    def loss(c,raw_weight=1):
        residual=(render(c)[:nt]-target_lab[:nt]).ravel()/5
        sr=sensor_violation(c)
        return np.r_[residual,raw_weight*sr/np.sqrt(max(len(sr),1))*np.sqrt(len(residual))]
    K=len(spectra);gridrows=[];best_fixed=None
    for k in range(K):
        for ev in np.linspace(-5,7,grid):
            c=np.zeros(K);c[k]=amp*2**ev;v=loss(c);objective=float(np.mean(v*v));gridrows.append({'member':k,'relative_EV':float(ev),'loss':objective,'coefficient':float(c[k])})
            if best_fixed is None or objective<best_fixed[0]:best_fixed=(objective,c)
    # Refine each fixed member independently; the union is still selected by train loss only.
    fixed=[]
    for k in range(K):
        own=min((r for r in gridrows if r['member']==k),key=lambda r:r['loss']);center_ev=own['relative_EV']
        def ff(ev):
            c=np.zeros(K);c[k]=amp*2**ev;v=loss(c);return float(np.mean(v*v))
        opt=minimize_scalar(ff,bounds=(max(-5,center_ev-1),min(7,center_ev+1)),method='bounded',options={'xatol':1e-3,'maxiter':25})
        c=np.zeros(K);c[k]=amp*2**opt.x;fixed.append({'member':k,'coefficients':c.tolist(),'loss':float(opt.fun),'relative_EV':float(opt.x)})
    best1=min(fixed,key=lambda r:r['loss']);upper=amp*2**12
    rng=np.random.default_rng(20261020);initials=[np.array(best1['coefficients'])]
    for i in range(starts-1):
        shape=rng.dirichlet(np.ones(K)*(.3 if i%2 else 1));initials.append(amp*2**rng.uniform(-2,5)*shape)
    trials=[]
    for i,c0 in enumerate(initials):
        sol=least_squares(loss,c0,bounds=(np.zeros(K),np.full(K,upper)),x_scale=np.full(K,amp),jac='2-point',diff_step=1e-3,
          max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6)
        residual=loss(sol.x);row={'start':i,'coefficients':sol.x.tolist(),'loss':float(np.mean(residual**2)),'nfev':sol.nfev,'success':bool(sol.success),'optimality':float(sol.optimality),'message':sol.message,'upper_active':bool(np.any(sol.x>.999*upper))}
        trials.append(row);(out/(name+'_partial_fit.json')).write_text(json.dumps({'trials':trials,'evals':counter[0]},indent=2));print(name,'start',i,row['loss'],'evals',counter[0],time.time()-start,flush=True)
    champion=min(trials,key=lambda r:r['loss']);c=np.array(champion['coefficients']);fitted_lab=render(c);fixed_lab=render(np.array(best1['coefficients']))
    def statistics(values):
        result={}
        for key,s in selected.items():
            gap=values[s]-target_lab[s];result[key]={'samples':len(gap),'Lab_RMSE':float(np.sqrt(np.mean(gap**2))),'mean_dLab':gap.mean(0).tolist(),'median_deltaE00':float(np.median(colour.delta_E(values[s],target_lab[s],method='CIE 2000')))}
        return result
    predicted_cam=profile[:,None]*(c@camera_response)[None,:]
    spectral_sum=float((c@spectra).sum());threshold=champion['loss']*1.02+1e-4
    ensemble=near_equivalent_fit_report(np.array([t['coefficients'] for t in trials]),np.array([t['loss'] for t in trials]),accepted_loss_ceiling=threshold,basis_spectra=spectra)
    output,fitq,dye=render(c,True);fixed_output,_,_=render(np.array(best1['coefficients']),True)
    valid=np.isfinite(output).all(-1)&np.isfinite(real).all(-1)
    report={'name':name,'scope':__doc__,'basis_names':prep['names'],'spectrum':(c@spectra).tolist(),'coefficients':c.tolist(),'fixed_spectrum_best':best1,'fixed_members':fixed,'fixed_grid':gridrows,'multistart':trials,'champion_train_only':champion,'near_equivalent':ensemble,
      'baseline':statistics(baseline_lab),'fixed_spectrum':statistics(fixed_lab),'fitted':statistics(fitted_lab),'source_mean_input_Y':float(c.sum()),'source_relative_to_legacy_mean_Y':float(c.sum()/amp),'source_discrete_spectral_sum':spectral_sum,
      'camera_proxy_lower_bound_fraction_violated':float((predicted_cam<observed-.03).mean()),'camera_proxy_lower_bound_max_deficit':float(np.maximum(observed-predicted_cam,0).max()),
      'camera_proxy_negative_predicted_fraction':float((predicted_cam<0).mean()),'camera_constraints':'All source cells are strictly 4-CFA flagged by existing pile-up criteria; proxy inequalities use fixed numerical .03 tolerance, not a measured noise CI. CIE/inverse-decoder is not true sensor SPD.',
      'bounds':'coefficients 0..4096*legacy_mean_source_Y; finite numerical envelope, NOT radiometric bound','split':'Train 50..300um alternating quadrants, 12 degree gaps and >4-sigma axis guard; held-out opposite quadrants have disjoint finite target Gaussian apertures. Outer 300..800um is descriptive radial holdout; no holdout selection','aperture_radius_pixels':aperture_radius,'total_forward_evaluations':counter[0],'elapsed_seconds':time.time()-start,
      'limits':prep['limits']+['Finite fixed-source-profile/library/model conditional fit; success does not prove true radiance','Some RAW inequalities use ideal-CIE camera proxy; exact raw spectral measurement unavailable','Repeated optimizations are not independent observations; near-equivalent ranges are not confidence intervals','A fixed local positive spectrum may be inadequate for real lamp optical geometry or mixed lens glare']}
    (out/(name+'_fit.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/(name+'_fit.npz'),real=real,baseline=a['baseline_rgb'],fixed_spectrum=fixed_output,fitted=output,coefficients=c,spectrum=c@spectra,train=train,heldout=test,outer=outer,points=allpoints,target_lab=target_lab,fitted_lab=fitted_lab,baseline_lab=baseline_lab,q=fitq,dye=dye)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    sheet=Image.new('RGB',(512*4,580),(18,18,18));d=ImageDraw.Draw(sheet)
    for i,(label,image) in enumerate([('真实800T扫描',real),('原研究基准',a['baseline_rgb']),('谱库单灯候选 · 幅度拟合',fixed_output),('非负谱组合 · 逆拟合',output)]):
        im=Image.fromarray(np.rint(srgb_encode(np.nan_to_num(image))*255).astype('uint8'));sheet.paste(im,(i*512,60));d.text((i*512+10,7),label,font=font,fill='white');d.text((i*512+10,33),name+' · 显影/传播/扫描固定',font=small,fill='white')
    sheet.save(out/(name+'_comparison.jpg'),quality=96)
    print('complete',name,json.dumps({'baseline':report['baseline'],'fitted':report['fitted'],'coefficients':c.tolist(),'evals':counter[0]},ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/inverse-source-radiance'));ap.add_argument('--prepare',action='store_true');ap.add_argument('--region',choices=REGIONS);ap.add_argument('--starts',type=int,default=8);ap.add_argument('--grid',type=int,default=32);ap.add_argument('--max-nfev',type=int,default=55);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(args.out)
    if args.region:fit(args.out,args.region,args.starts,args.grid,args.max_nfev)

if __name__=='__main__':main()
