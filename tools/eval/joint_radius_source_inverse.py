"""One equal-zone source inverse after fixed measured-edge/HDR corrections.

Same-frame, repeatedly inspected real RGB is an inverse target, not independent
material evidence. Only ten nonnegative source-spectrum coefficients vary; the
strict source support/profile, direct/return linkage and backend stay fixed.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
from scipy import ndimage,sparse
from scipy.optimize import least_squares,nnls

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from inverse_source_radiance import engine_for,observation_operator
from brightness_response_audit import lab
ROOT=Path('/workspace/filmlab-audit')


def joint_split(distance,measured,center,guard):
    """Two radial zones, same alternating quadrant labels and finite axis gap."""
    yy,xx=np.indices(distance.shape);dy=yy-center[0];dx=xx-center[1]
    quadrant=np.floor(np.mod(np.arctan2(dy,dx),2*np.pi)/(np.pi/2)).astype(int)
    aperture=(abs(dy)>guard)&(abs(dx)>guard)
    angular=np.mod(np.arctan2(dy,dx),np.pi/2)*180/np.pi
    valid=measured&aperture&(angular>12)&(angular<78)
    inner=valid&(distance>=50)&(distance<=300);outer=valid&(distance>300)&(distance<=800)
    return {'inner_train':inner&(quadrant%2==0),'inner_heldout':inner&(quadrant%2==1),
            'outer_train':outer&(quadrant%2==0),'outer_heldout':outer&(quadrant%2==1)}


def sample(mask,cap=160):
    yy,xx=np.indices(mask.shape);points=np.argwhere(mask&(yy%3==0)&(xx%3==0))
    if len(points)==0:raise ValueError('Empty predeclared sampling region')
    return points[np.linspace(0,len(points)-1,min(cap,len(points))).astype(int)]


def fit(name,out,starts=8,max_nfev=90,prepare_only=False):
    import colour
    source=np.load(ROOT/'inverse-source-radiance'/(name+'_source.npz'))
    oldfit=np.load(ROOT/'inverse-source-radiance'/(name+'_fit.npz'))
    corrected=np.load(ROOT/'raw-hdr-film-replay'/(name+'.npz'))
    edge=np.load(ROOT/'lamp-edge-input-audit'/(name+'.npz'))
    prep=json.loads((ROOT/'inverse-source-radiance/prepared.json').read_text());engine,meta=engine_for()
    out.mkdir(parents=True,exist_ok=True);start=time.time();counter=0
    calibration=10**(engine.log_k-engine.e.speed_loss+engine.layer_cal)
    H=(10**corrected['q']-10**engine.e.fog_exposure)/calibration
    old_core=np.einsum('k,khwc->hwc',oldfit['coefficients'],source['basis'],optimize=True)
    background=H-old_core;minimum=float(background[source['safe']].min())
    if minimum < -1e-7:raise ValueError('Removing inherited core violates nonnegative background')
    background=np.maximum(background,1e-12)
    replay_q=float(abs(engine._log_exposure(background+old_core)[source['safe']]-corrected['q'][source['safe']]).max())
    if replay_q>1e-8:raise ValueError('Measured-edge/HDR q replay fails')
    distance=source['distance_um'];yy,xx=np.where(distance==0);center=(float(yy.mean()),float(xx.mean()))
    sigma=25/meta['scan_um_per_pixel']
    # A native bilinear cell spans >2 scan pixels here. Four-sigma target
    # support alone is insufficient to guarantee native-column separation.
    from skimage.transform import ProjectiveTransform
    tf=ProjectiveTransform(source['matrix']);h,w=source['profile'].shape
    sy,sx=np.meshgrid(np.linspace(0,h-1,5),np.linspace(0,w-1,5));locations=np.c_[sx.ravel(),sy.ravel()]
    projection_margin=max(float(np.max(abs(tf(locations+[dx,dy])-tf(locations)))) for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)])
    guard=int(4*sigma+.5)+int(np.ceil(projection_margin))+1
    masks=joint_split(distance,(source['measured']>.5)&source['supported'],center,guard)
    order=list(masks);point_parts=[sample(masks[key]) for key in order];points=np.vstack(point_parts)
    groups=np.concatenate([np.full(len(p),i) for i,p in enumerate(point_parts)]);train=np.isin(groups,[0,2]);test=~train
    P=observation_operator(source['q_base'].shape[:2],source['matrix'],points,sigma)
    overlap=np.intersect1d(np.unique(P[train].indices),np.unique(P[test].indices))
    if len(overlap):raise ValueError('Native projection footprints overlap train/heldout')
    target=lab(ndimage.gaussian_filter(source['real'].astype(float),(sigma,sigma,0))[tuple(points.T)])
    sparse.save_npz(out/(name+'_operator.npz'),P)
    np.savez_compressed(out/(name+'_split.npz'),points=points,groups=groups,group_names=np.array(order),target_lab=target,
        center=np.array(center),guard=guard,sigma=sigma,native_shape=np.array(source['q_base'].shape[:2]),matrix=source['matrix'],
        distance_um=distance,**masks)
    np.savez_compressed(out/(name+'_case.npz'),background=background,basis=source['basis'],points=points,target_lab=target,groups=groups,
        inner_train_indices=np.where(groups==0)[0],inner_heldout_indices=np.where(groups==1)[0],outer_train_indices=np.where(groups==2)[0],outer_heldout_indices=np.where(groups==3)[0],
        train_indices=np.where(train)[0],heldout_indices=np.where(test)[0],factor=source['factor'],source_profile=source['profile'],support=source['support'],matrix=source['matrix'],
        source_bounds=edge['source_bounds'],baseline_q=corrected['q'],baseline_core_coefficients=oldfit['coefficients'],camera_response=source['camera_response'],source_camera=source['source_camera'],
        direct_response=source['direct_response'],return_response=source['return_response'],spectra=source['spectra'])
    print(name,'prepared',[(key,len(p)) for key,p in zip(order,point_parts)],'native train/heldout overlap',len(overlap),flush=True)
    if prepare_only:return
    indices=np.unique(P.indices);Ps=P[:,indices];camera=source['camera_response'];observed=source['source_camera'];profile=source['source_profile']
    required=(observed/profile[:,None]).max(0)
    if np.any(camera<=0):raise ValueError('Positive library camera response required for scalar feasibility projection')
    def feasible_coefficients(x):
        x=np.asarray(x);prediction=x@camera
        if np.any(x<0) or not np.any(x):raise ValueError('Positive nonnegative spectrum mixture required')
        return x*max(1.,float(np.max(required/np.maximum(prediction,1e-300))))
    def render(x=None,full=False):
        nonlocal counter
        counter+=1
        q=corrected['q'] if x is None else engine._log_exposure(background+np.einsum('k,khwc->hwc',feasible_coefficients(x),source['basis'],optimize=True))
        state=engine._develop(q,blur_px=12/engine.px_um);state.dye*=source['factor'][...,None]
        if full:
            from skimage.transform import ProjectiveTransform
            from light_acceptance import crop_warp
            rgb=engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(state)))
            return crop_warp(ProjectiveTransform(source['matrix']))(rgb),q,state.dye
        sub=replace(state,**{key:getattr(state,key).reshape(-1,3)[indices] for key in ('dye','silver','halide')})
        rgb=engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(sub)))
        return lab(Ps@rgb)
    def residual(x):
        predicted=render(x);parts=[]
        for group in [0,2]:
            delta=predicted[groups==group]-target[groups==group]
            parts.append(delta.ravel()/np.sqrt(2*delta.size))
        return np.concatenate(parts)
    baseline=render();replay=render(oldfit['coefficients'])
    corrected_replay=float(abs(baseline-replay).max())
    if corrected_replay>2e-5:raise ValueError('Inherited corrected-spectrum replay mismatch')
    amp=next(r['mean_legacy_source_Y'] for r in prep['regions'] if r['name']==name);upper=amp*4096;K=len(camera)
    direct_target=edge['old_direct'][source['support']].mean(0);return_target=edge['old_halo_source'][source['support']].mean(0)
    A=np.c_[source['direct_response'],source['return_response']];normalizer=np.maximum(np.r_[direct_target,return_target],1e-8)
    legacy_approx=nnls((A/normalizer).T,np.ones(6))[0]
    initials=[oldfit['coefficients'],legacy_approx,np.full(K,float(oldfit['coefficients'].sum()/K))]
    rng=np.random.default_rng(20261022)
    for i in range(starts-len(initials)):
        initials.append(float(oldfit['coefficients'].sum())*2**rng.uniform(-1,1)*rng.dirichlet(np.ones(K)*(.3 if i%2 else 1)))
    trials=[]
    def jacobian(x):
        base=residual(x);columns=[]
        for k in range(K):
            h=max(abs(x[k])*1e-3,amp*1e-5)
            if x[k]+h>upper:h=-h
            changed=x.copy();changed[k]+=h;columns.append((residual(changed)-base)/h)
        return np.array(columns).T
    for i,initial in enumerate(initials):
        x0=np.maximum(initial,amp*1e-8)
        sol=least_squares(residual,x0,bounds=(np.zeros(K),np.full(K,upper)),jac=jacobian,x_scale=np.full(K,oldfit['coefficients'].sum()/K),
            max_nfev=max_nfev,ftol=2e-8,xtol=2e-8,gtol=2e-6)
        c=feasible_coefficients(sol.x);r=residual(sol.x)
        row={'start':i,'initial_kind':['old FL2-dominant inverse','old local direct/return nonnegative approximation','uniform ten-spectrum mixture'][i] if i<3 else 'predeclared seeded random mixture',
             'coefficients':c.tolist(),'parameter_coefficients':sol.x.tolist(),'equal_zone_Lab_MSE':float(np.sum(r*r)),'success':bool(sol.success),'message':sol.message,'nfev':sol.nfev,'njev':sol.njev,
             'optimality':float(sol.optimality),'numerical_upper_active':bool(np.any(c>upper*.999))}
        trials.append(row);(out/(name+'_partial.json')).write_text(json.dumps({'trials':trials,'forward_evaluations':counter},indent=2))
        print(name,'start',i,'loss',row['equal_zone_Lab_MSE'],'Y',float(c.sum()),'evals',counter,'seconds',time.time()-start,flush=True)
    best=min(trials,key=lambda r:r['equal_zone_Lab_MSE']);c=np.array(best['coefficients']);prediction=render(c);image,q,dye=render(c,True)
    def stats(values):
        result={}
        for group,label in enumerate(order):
            p=values[groups==group];t=target[groups==group];delta=p-t
            result[label]={'samples':len(p),'real_mean_Lab':t.mean(0).tolist(),'model_mean_Lab':p.mean(0).tolist(),'model_minus_real_mean_Lab':delta.mean(0).tolist(),
                           'Lab_RMSE':float(np.sqrt(np.mean(delta**2))),'median_deltaE00':float(np.median(colour.delta_E(p,t,method='CIE 2000')))}
        return result
    predicted_camera=profile[:,None]*(c@camera);deficit=float(np.maximum(observed-predicted_camera,0).max())
    if deficit>1e-9:raise ValueError('Chosen source violates zero-tolerance camera-proxy lower bound')
    near=[t for t in trials if t['equal_zone_Lab_MSE']<=best['equal_zone_Lab_MSE']*1.02+1e-4]
    report={'name':name,'scope':__doc__,'names':prep['names'],'coefficients':c.tolist(),'spectrum':(c@source['spectra']).tolist(),'source_mean_input_Y':float(c.sum()),
            'source_input_units':'Existing engine normalized inputY, not SI radiance; still estimated.','trials':trials,'selected_by':'Only current equal-weight inner/outer training Lab MSE chooses final coefficients. Starts are predeclared, including the old same-frame inverse; current heldout never selects starts or weighting.',
            'near_equivalent_training_solutions':{'count':len(near),'source_Y_min_max':[min(sum(t['coefficients']) for t in near),max(sum(t['coefficients']) for t in near)]},
            'split':{'group_order':order,'samples_by_group':[len(p) for p in point_parts],'target_gaussian_sigma_pixels':sigma,'axis_guard_pixels':guard,'native_bilinear_projection_margin_pixels':projection_margin,'extra_angular_gap_degrees':12,
                     'inner_um':[50,300],'outer_um':[300,800],'native_train_heldout_index_overlap':len(overlap),'policy':'New alternating-quadrant split in both radial zones. Outer now partly trains; it is NOT the former all-outer holdout. Target same-frame RGB has been reused in earlier research; this is not external independent evidence.'},
            'baseline_measured_edges_HDR_old_core':stats(baseline),'joint_radius_fit':stats(prediction),'background_safe_minimum_before_floor':minimum,'q_replay_max':replay_q,'corrected_Lab_replay_max':corrected_replay,
            'raw_proxy_lower_max_deficit':deficit,'raw_proxy_lower_slack_RGB':(c@camera-required).tolist(),'numerical_coefficient_upper':upper,'forward_evaluations':counter,'elapsed_seconds':time.time()-start,
            'limits':['Only one fixed strict support/profile and finite ten-spectrum cone are searched. A failed finite multistart search is not a proof of physical impossibility.',
                      'Camera response is the inherited CIE/digital-decoder proxy, not measured camera spectral sensitivity. Zero lower-bound deficit is conditional on that proxy.',
                      'Feasible projection scales the same nonnegative spectrum for both direct and returned paths. No independent layer or halo-strength fitting.',
                      'Measured partial-edge minY witnesses and HDR full-measurement body inputs remain fixed. Their source identities and unknown channels are not certified.',
                      'Numerical multistart agreement is not a radiometric confidence interval; missing source spectrum and exposure remain estimates.',
                      'No material, DIR, transport kernel, scanner or timing parameter changes.']}
    (out/(name+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/(name+'.npz'),q=q,dye=dye,image=image,real=source['real'],baseline=corrected['image'],coefficients=c,spectrum=c@source['spectra'],predicted_lab=prediction,target_lab=target,points=points,groups=groups)
    from PIL import Image,ImageDraw,ImageFont
    from filmlab.spectral import srgb_encode
    sheet=Image.new('RGB',(1536,580),(18,18,18));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for i,(title,values) in enumerate([('真实800T扫描',source['real']),('已测边缘/HDR · 原核心逆解',corrected['image']),('新近/外训练 · 单谱驱动核心',image)]):
        picture=Image.fromarray(np.uint8(np.rint(np.clip(srgb_encode(np.nan_to_num(values)),0,1)*255)));sheet.paste(picture,(512*i,60));draw.text((512*i+8,12),title,font=font,fill='white')
    sheet.save(out/(name+'_comparison.jpg'),quality=98)
    print(name,'complete',json.dumps({'Y':float(c.sum()),'before':report['baseline_measured_edges_HDR_old_core'],'after':report['joint_radius_fit'],'evals':counter},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--region',choices=['ceiling_lamp','middle_ceiling'],required=True);p.add_argument('--out',type=Path,default=ROOT/'joint-radius-source-inverse');p.add_argument('--starts',type=int,default=8);p.add_argument('--max-nfev',type=int,default=90);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    fit(a.region,a.out,a.starts,a.max_nfev,a.prepare_only)
