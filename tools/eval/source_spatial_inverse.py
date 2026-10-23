"""Conditional inverse of a lamp's spatial radiance inside strict censored support.

Spectrum (FL2, selected by prior source-only study), material, halation kernel,
chemistry, scanner, source support and spatial data split remain fixed. Only a
positive mixture of five declared intensity profiles changes. A fitted profile
is not a measured source or independently calibrated lens PSF.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares,minimize_scalar
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from inverse_source_radiance import engine_for,observation_operator
from brightness_response_audit import lab

TEMPLATE_NAMES=('legacy','uniform','central_distance','major_positive','major_negative')


def spatial_templates(support,legacy_profile):
    """Five fixed positive mean-one profiles, never chosen from target colours."""
    support=np.asarray(support,bool);legacy=np.asarray(legacy_profile,float)
    if support.ndim!=2 or legacy.shape!=support.shape or not support.any():
        raise ValueError('matching nonempty spatial support required')
    if not np.isfinite(legacy).all() or np.any(legacy[support]<=0):
        raise ValueError('legacy source must be finite and positive on support')
    coordinates=np.argwhere(support).astype(float);center=coordinates.mean(0)
    centered=coordinates-center
    _,_,v=np.linalg.svd(centered,full_matrices=False)
    axis=v[0] if len(v) else np.array([1.,0.])
    # Deterministic PCA sign, independent of image errors or targets.
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    projection=centered@axis;projection/=max(np.max(np.abs(projection)),1.)
    yy,xx=np.indices(support.shape);distance=ndimage.distance_transform_edt(support)
    central=.1+distance/max(float(distance.max()),1.)
    positive=np.zeros(support.shape);negative=np.zeros(support.shape)
    positive[support]=1+.8*projection;negative[support]=1-.8*projection
    arrays=[legacy.copy(),support.astype(float),central,positive,negative]
    profiles=[]
    for a in arrays:
        a=np.where(support,a,0.);a/=a[support].mean();profiles.append(a)
    return np.array(profiles),{'source_center_yx':center.tolist(),'principal_axis_yx':axis.tolist(),
                              'minimum_template_on_support':[float(p[support].min()) for p in profiles]}


def spatial_responses(profiles,direct_response,return_response,engine):
    """Same fixed spectrum drives both paths for every spatial profile."""
    from filmlab import film
    rho=film.bounce_retention(engine.e.stock,engine.v)
    basis=[]
    for profile in profiles:
        shape=np.repeat(profile[...,None],3,axis=-1)
        direct=film.emulsion_scatter(shape,engine.px_um)*direct_response
        returning=film.halation(np.zeros_like(shape),shape,engine.px_um,engine.v,rho)*return_response
        basis.append(direct+returning)
    return np.array(basis)


def physical_profile(coefficients,profiles):
    c=np.asarray(coefficients,float);p=np.asarray(profiles,float)
    if c.ndim!=1 or p.ndim!=3 or len(c)!=len(p) or not np.isfinite(c).all() or np.any(c<0):
        raise ValueError('finite nonnegative matching profile coefficients required')
    return np.einsum('k,khw->hw',c,p,optimize=True)


def fit(name,out,starts=8,max_nfev=90,radial_control=False):
    import colour
    from PIL import Image,ImageDraw,ImageFont
    from skimage.transform import ProjectiveTransform
    from light_acceptance import crop_warp
    from filmlab.spectral import srgb_encode
    root=Path('/workspace/filmlab-audit/inverse-source-radiance');out.mkdir(parents=True,exist_ok=True)
    a=np.load(root/(name+'_source.npz'));old=np.load(root/(name+'_fit.npz'))
    reference=json.loads((root/(name+'_fit.json')).read_text());prep=json.loads((root/'prepared.json').read_text())
    engine,meta=engine_for();profiles,geometry=spatial_templates(a['support'],a['profile'])
    k=prep['names'].index('FL2');direct=a['direct_response'][k];returned=a['return_response'][k]
    basis=spatial_responses(profiles,direct,returned,engine)
    # The legacy spatial template under this fixed spectrum must reproduce the
    # already validated FL2 response, without changing the transport kernel.
    basis_error=float(np.max(abs(basis[0]-a['basis'][k])))
    if basis_error>1e-10:raise ValueError('spatial basis fails reference '+str(basis_error))
    nt=reference['baseline']['train']['samples'];nv=reference['baseline']['heldout']['samples'];points=old['points'];target=old['target_lab']
    selected={'train':slice(0,nt),'heldout':slice(nt,nt+nv),'outer':slice(nt+nv,None)}
    training_indices=np.arange(nt)
    radial_details=None
    if radial_control:
        sigma=25/meta['scan_um_per_pixel'];guard=(int(4*sigma+.5)+1)*meta['scan_um_per_pixel']
        distance=a['distance_um'];measured=(a['measured']>.5)&a['supported']
        radial_train=measured&(distance>300)&(distance<500-guard)
        radial_heldout=measured&(distance>500+guard)&(distance<=800)
        def sample(mask):
            yy,xx=np.indices(mask.shape);p=np.argwhere(mask&(yy%3==0)&(xx%3==0))
            return p[np.linspace(0,len(p)-1,min(160,len(p))).astype(int)]
        rp=sample(radial_train);op=sample(radial_heldout)
        points=np.vstack([old['points'][:nt+nv],op,rp])
        target=lab(ndimage.gaussian_filter(a['real'].astype(float),(sigma,sigma,0))[tuple(points.T)])
        selected={'train':slice(0,nt),'heldout':slice(nt,nt+nv),'outer':slice(nt+nv,nt+nv+len(op)),'added_radial_train':slice(nt+nv+len(op),None)}
        training_indices=np.r_[np.arange(nt),np.arange(nt+nv+len(op),len(points))]
        radial_details={'guard_um':guard,'added_training_distance_um':[300,500-guard],
                        'radial_holdout_distance_um':[500+guard,800],'added_training_samples':len(rp),
                        'radial_holdout_samples':len(op),'policy':'Separate exploratory radial-control experiment. Original inner heldout remains untouched; radial gap keeps finite Gaussian apertures disjoint.'}
    P=observation_operator(a['q_base'].shape[:2],a['matrix'],points,25/meta['scan_um_per_pixel']);indices=np.unique(P.indices);Ps=P[:,indices]
    counter=0;start=time.time();background=a['background'];factor=a['factor'];observed=a['source_camera'];camera_color=a['camera_response'][k]
    support=a['support'];profile_values=profiles[:,support]
    def render(c=None,full=False):
        nonlocal counter
        counter+=1
        H=None if c is None else background+np.einsum('k,khwc->hwc',c,basis,optimize=True)
        q=a['q_base'] if c is None else engine._log_exposure(np.maximum(H,1e-12))
        st=engine._develop(q,blur_px=12/engine.px_um);st.dye*=factor[...,None]
        if full:
            rgb=engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(st)))
            return crop_warp(ProjectiveTransform(a['matrix']))(rgb),q
        sub=replace(st,**{key:getattr(st,key).reshape(-1,3)[indices] for key in ('dye','silver','halide')})
        rgb=engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(sub)))
        return lab(Ps@rgb)
    baseline=render();replay=float(np.max(abs(baseline[:nt+nv]-old['baseline_lab'][:nt+nv])))
    initial=np.zeros(len(profiles));initial[0]=reference['source_mean_input_Y']
    fixed=render(initial);fixed_replay=float(np.max(abs(fixed[:nt+nv]-old['fitted_lab'][:nt+nv])))
    if max(replay,fixed_replay)>2e-4:raise ValueError('cached output replay mismatch '+str([replay,fixed_replay]))
    def raw_residual(c):
        prediction=(c@profile_values)[:,None]*camera_color
        return np.minimum(prediction-observed+.03,0).ravel()/.03
    def residual(c):
        colour=(render(c)[training_indices]-target[training_indices]).ravel()/5;raw=raw_residual(c)
        return np.r_[colour,raw*np.sqrt(len(colour)/max(len(raw),1))]
    rng=np.random.default_rng(20261021);initials=[initial]
    for i in range(starts-1):initials.append(reference['source_mean_input_Y']*2**rng.uniform(-1,1)*rng.dirichlet(np.ones(len(profiles))))
    upper=next(r['mean_legacy_source_Y'] for r in prep['regions'] if r['name']==name)*4096
    trials=[]
    for i,c0 in enumerate(initials):
        sol=least_squares(residual,c0,bounds=(np.zeros(len(profiles)),np.full(len(profiles),upper)),x_scale=np.full(len(profiles),reference['source_mean_input_Y']),
                          jac='2-point',diff_step=1e-3,max_nfev=max_nfev,ftol=2e-8,xtol=2e-8,gtol=2e-6)
        r=residual(sol.x);row={'start':i,'coefficients':sol.x.tolist(),'loss':float(np.mean(r*r)),'success':bool(sol.success),'nfev':sol.nfev,'optimality':float(sol.optimality),'upper_active':bool(np.any(sol.x>.999*upper))};trials.append(row)
        (out/(name+'_partial.json')).write_text(json.dumps({'trials':trials,'evals':counter},indent=2))
        print(name,'start',i,row['loss'],'evals',counter,'seconds',time.time()-start,flush=True)
    best=min(trials,key=lambda r:r['loss']);c=np.array(best['coefficients']);fitted=render(c);image,q=render(c,True);intensity=physical_profile(c,profiles)
    predicted=(c@profile_values)[:,None]*camera_color;raw_deficit=np.maximum(observed-predicted,0)
    def stats(values):
        d={}
        for label,s in selected.items():
            delta=values[s]-target[s];d[label]={'samples':len(delta),'Lab_RMSE':float(np.sqrt(np.mean(delta**2))),'mean_dLab':delta.mean(0).tolist(),'median_deltaE00':float(np.median(colour.delta_E(values[s],target[s],method='CIE 2000')))}
        return d
    ensemble=[t for t in trials if t['loss']<=best['loss']*1.02+1e-4]
    ensemble_profiles=np.array([physical_profile(t['coefficients'],profiles)[support] for t in ensemble])
    source_yx=np.argwhere(support);centroid=(intensity[support,None]*source_yx).sum(0)/intensity[support].sum()
    core=a['distance_um']==0
    # Core is intentionally not a fit target, and still subject to registration.
    from scipy.ndimage import gaussian_filter
    core_images={'real':a['real'],'baseline':a['baseline_rgb'],'fixed_profile':old['fitted'],'fitted_spatial':image}
    actual_core=core.copy()
    for values in core_images.values():actual_core&=np.isfinite(values).all(-1)
    core_summary={key:np.median(lab(gaussian_filter(values,(25/meta['scan_um_per_pixel'],25/meta['scan_um_per_pixel'],0)))[actual_core],axis=0).tolist() for key,values in core_images.items()}
    report={'name':name,'spectrum':'FL2 fixed from previous inverse study, not independent source identification','template_names':list(TEMPLATE_NAMES),'geometry':geometry,'coefficients':c.tolist(),'mean_source_input_Y':float(c.sum()),'source_centroid_yx':centroid.tolist(),
       'baseline':stats(baseline),'fixed_profile':stats(fixed),'fitted_spatial':stats(fitted),'trials':trials,'chosen_using':'original inner train and separately declared radial train' if radial_control else 'original train loss only, not heldout/outer','radial_control':radial_details,'forward_evaluations':counter,'elapsed_seconds':time.time()-start,
       'basis_reference_max_error':basis_error,'baseline_Lab_replay_max':replay,'fixed_profile_Lab_replay_max':fixed_replay,'camera_proxy_deficit_max':float(raw_deficit.max()),'camera_proxy_violation_fraction':float((raw_deficit>.03).mean()),
       'raw_feasible_at_zero_tolerance':bool(raw_deficit.max()<1e-8),'near_equivalent_trials':len(ensemble),'near_equivalent_total_input_Y_range':[float(sum(t['coefficients'])) for t in [min(ensemble,key=lambda t:sum(t['coefficients'])),max(ensemble,key=lambda t:sum(t['coefficients']))]],
       'near_equivalent_source_cell_inputY_min':ensemble_profiles.min(0).tolist(),'near_equivalent_source_cell_inputY_max':ensemble_profiles.max(0).tolist(),
       'core_not_fitted':{'pixels':int(actual_core.sum()),'median_Lab_by_path':core_summary},
       'limits':['Only five predeclared positive spatial templates inside fixed strict-four-CFA support; failure is not global source-shape exclusion','Spectrum selected on the same prior picture, not measured; fixed material and scanner remain proxies','Source profile is censored image-plane input, not recovered emitter geometry or measured camera PSF','Spatially held-out source regions are same-frame predictions, not independent scene/material validation','Camera proxy uses ideal CIE plus digital decoder with no measured spectral sensitivity; no SI radiance/energy certification','Numerical coefficient upper bound is not a measured radiance bound','Estimated source shape/brightness never becomes a measurement and is not applied to default pipeline']}
    singular=np.linalg.svd(profiles[:,support],compute_uv=False)
    report['template_linear_rank']=int(np.sum(singular>1e-10*singular[0]))
    report['template_singular_values']=singular.tolist()
    report['template_coefficient_gauge']='uniform equals half each PCA ramp; coefficient non-uniqueness alone is not source-field non-uniqueness'
    (out/(name+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/(name+'.npz'),profiles=profiles,basis=basis,coefficients=c,intensity=intensity,fitted=image,fitted_lab=fitted,target_lab=target,points=points,training_indices=training_indices,baseline=a['baseline_rgb'],fixed_profile=old['fitted'],real=a['real'],train=old['train'],heldout=old['heldout'],outer=old['outer'],q=q)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15);sheet=Image.new('RGB',(2048,580),(18,18,18));draw=ImageDraw.Draw(sheet)
    for i,(title,picture) in enumerate([('真实800T扫描',a['real']),('旧研究基准',a['baseline_rgb']),('原固定源轮廓 · FL2反演',old['fitted']),('未知源空间分布 · 独立候选',image)]):
        im=Image.fromarray(np.rint(srgb_encode(np.nan_to_num(picture))*255).astype('uint8'));sheet.paste(im,(i*512,60));draw.text((i*512+10,8),title,font=font,fill='white');draw.text((i*512+10,35),name+' · 显影/光晕核/扫描固定',font=small,fill='white')
    sheet.save(out/(name+'_comparison.jpg'),quality=96)
    print('complete',name,json.dumps({'baseline':report['baseline'],'fixed':report['fixed_profile'],'spatial':report['fitted_spatial'],'Y':report['mean_source_input_Y'],'evals':counter},ensure_ascii=False),flush=True)
    return report


def strength_profile_diagnostic(name,out):
    """Radius-dependent scalar inverse; outer-used fit is not heldout validation.

    Predeclared 64 log-spaced values followed by bounded refinement of every
    grid-local minimum. This checks finite numerical convergence, not a global
    physical source-strength identification.
    """
    import colour
    root=Path('/workspace/filmlab-audit/inverse-source-radiance');out.mkdir(parents=True,exist_ok=True)
    a=np.load(root/(name+'_source.npz'));old=np.load(root/(name+'_fit.npz'))
    reference=json.loads((root/(name+'_fit.json')).read_text());prep=json.loads((root/'prepared.json').read_text())
    engine,meta=engine_for();k=prep['names'].index('FL2')
    nt=reference['baseline']['train']['samples'];nv=reference['baseline']['heldout']['samples']
    selections={'inner_train':slice(0,nt),'original_inner_heldout':slice(nt,nt+nv),'outer_reused_for_second_fit':slice(nt+nv,None)}
    P=observation_operator(a['q_base'].shape[:2],a['matrix'],old['points'],25/meta['scan_um_per_pixel'])
    indices=np.unique(P.indices);Ps=P[:,indices];target=old['target_lab'];counter=0;cache={}
    proxy=a['profile'][a['support'],None]*a['camera_response'][k]
    if np.any(proxy<=0):raise ValueError('strictly positive camera proxy required')
    lower=float(np.max(a['source_camera']/proxy))
    upper=next(r['mean_legacy_source_Y'] for r in prep['regions'] if r['name']==name)*4096
    def render(log_y):
        nonlocal counter
        key=float(log_y)
        if key in cache:return cache[key]
        counter+=1;y=float(np.exp(key))
        H=a['background']+y*a['basis'][k]
        q=engine._log_exposure(np.maximum(H,1e-12));st=engine._develop(q,blur_px=12/engine.px_um);st.dye*=a['factor'][...,None]
        sub=replace(st,**{field:getattr(st,field).reshape(-1,3)[indices] for field in ('dye','silver','halide')})
        rgb=engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(sub)))
        cache[key]=lab(Ps@rgb);return cache[key]
    grid=np.linspace(np.log(lower),np.log(upper),64);rows={}
    for label in ['inner_train','outer_reused_for_second_fit']:
        sl=selections[label]
        def objective(x):return float(np.mean((render(x)[sl]-target[sl])**2))
        losses=np.array([objective(x) for x in grid]);candidates=[(losses[i],grid[i]) for i in range(len(grid))]
        local=[i for i in range(1,len(grid)-1) if losses[i]<=losses[i-1] and losses[i]<=losses[i+1]]
        # Refine edge-adjacent intervals too: a coarse-grid endpoint may hide a
        # minimum just inside the hard RAW bound or numerical upper bound.
        brackets=[(0,1),(len(grid)-2,len(grid)-1)]+[(i-1,i+1) for i in local]
        refinements=[]
        for lo,hi in brackets:
            sol=minimize_scalar(objective,bounds=(grid[lo],grid[hi]),method='bounded',options={'xatol':1e-7,'maxiter':70})
            candidates.append((float(sol.fun),float(sol.x)));refinements.append({'success':bool(sol.success),'nfev':sol.nfev,'log_interval':[float(grid[lo]),float(grid[hi])],'source_Y':float(np.exp(sol.x)),'Lab_MSE':float(sol.fun)})
        _,best=min(candidates);y=float(np.exp(best));values=render(best);stats={}
        for target_label,s in selections.items():
            delta=values[s]-target[s];stats[target_label]={'Lab_RMSE':float(np.sqrt(np.mean(delta**2))),'mean_dLab':delta.mean(0).tolist(),'median_deltaE00':float(np.median(colour.delta_E(values[s],target[s],method='CIE 2000')))}
        rows[label]={'source_Y':y,'statistics':stats,'raw_proxy_min_margin':float(np.min(y*proxy-a['source_camera'])),'lower_active':bool(y/lower<1.0001),'upper_active':bool(y/upper>.9999),'grid_min_source_Y':float(np.exp(grid[np.argmin(losses)])),'grid_MSE':losses.tolist(),'refinements':refinements}
    report={'name':name,'fixed_spectrum':'FL2 selected on prior same-frame training, not measured SPD','fixed_spatial_profile':'original legacy profile, not fitted here','raw_proxy_hard_lower_source_Y':lower,'numerical_upper_source_Y':upper,'grid_source_Y':np.exp(grid).tolist(),'fits':rows,'inner_fit_to_outer_fit_sourceY_ratio':rows['inner_train']['source_Y']/rows['outer_reused_for_second_fit']['source_Y'],'forward_evaluations':counter,'policy':'Original outer300-800um is deliberately reused as a NEW diagnostic fit target. It is NOT heldout for this second fit; this does not choose a production candidate. Original inner heldout is reported as a same-frame prediction. Fixed direct and returned FL2 paths share one scalar Y.','limits':['Finite 64-grid and bounded local refinements, not strict global proof','RAW bound is ideal camera-proxy consistency, not SI radiance calibration','Changing best amplitude with fit radius may reflect fixed spatial/spectral/scan/background errors; it does not isolate material truth']}
    (out/(name+'_strength_radius.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'name':name,'fits':rows,'sourceY_ratio':report['inner_fit_to_outer_fit_sourceY_ratio'],'evals':counter},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--region',choices=['ceiling_lamp','middle_ceiling'],required=True);p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/source-spatial-inverse'));p.add_argument('--starts',type=int,default=8);p.add_argument('--max-nfev',type=int,default=90);p.add_argument('--radial-control',action='store_true');p.add_argument('--strength-radius-diagnostic',action='store_true');a=p.parse_args()
    if a.strength_radius_diagnostic:strength_profile_diagnostic(a.region,a.out)
    else:fit(a.region,a.out,a.starts,a.max_nfev,a.radial_control)

if __name__=='__main__':main()
