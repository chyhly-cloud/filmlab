"""Finite local exposure feasibility, never a fitted film or rendering candidate.

Nonnegative effective-layer additions are a permissive relaxation, not sufficient
for a realizable source spectrum. Each witness changes only an input patch;
spatial DIR, cached agitation, dye spectra, scanner and timing stay fixed.
Search failure is NOT a proof of global impossibility.
"""
from pathlib import Path
import json,sys,zlib,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab


def add_exposure(q,addition):
    if np.any(np.asarray(addition)<0):raise ValueError('nonnegative extra exposure required')
    return np.log10(10.**q+addition)


def feasible(value,target,L_tolerance=1.,ab_tolerance=2.):
    delta=np.asarray(value)-target
    return bool(abs(delta[0])<=L_tolerance and np.linalg.norm(delta[1:])<=ab_tolerance)


def plot_tradeoff(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for row,region in enumerate(report['regions']):
        for col,zone in enumerate(['real_orange_transition_0_300','transition_100_300']):
            p=next(p for p in region['patches'] if p['zone']==zone);ax=axes[row,col];target=np.array(p['target_Lab'])
            cs=min((s for s in p['fixed_L_constraint_trials'] if s['L_constraint_satisfied']),key=lambda s:s['ab_distance'])
            states=[('No halo',p['nohalo_Lab'],'s','gray'),('Original full halo',p['full_Lab'],'o','orange'),
                    ('Nonnegative input search',p['witness_Lab'],'x','red'),('Search with L constraint',cs['Lab'],'+','blue'),
                    ('Signed input control ONLY',p['signed_control_NOT_returned_light']['Lab'],'D','purple')]
            for label,value,marker,color in states:
                d=np.array(value)-target;ax.scatter(d[0],np.linalg.norm(d[1:]),marker=marker,color=color,s=55,label=label)
            ax.add_patch(Rectangle((-1,0),2,2,facecolor='green',alpha=.15))
            ax.axvline(0,color='gray',lw=.6);ax.axhline(2,color='gray',lw=.6,ls=':')
            ax.set(title=region['name']+' / '+('orange subset' if col==0 else '100-300 um'),
                   xlabel='Output L* minus real L*',ylabel='a*b* distance to real',ylim=(-.3,None))
            if row==0 and col==0:ax.legend(fontsize=7)
    fig.suptitle('Fixed backend, actual local cache: brightness versus colour feasibility\nGreen box is diagnostic tolerance; signed controls are NOT returned light or a correction')
    fig.savefig(Path(out)/'brightness_colour_tradeoff.png',dpi=180);plt.close(fig)


def main():
    from scipy.ndimage import distance_transform_edt,binary_erosion,gaussian_filter
    from scipy.optimize import least_squares,nnls,minimize
    from filmlab import generation,film,lights,spectral
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from corridor_cache_geometry import bilinear_operator
    from light_acceptance import crop_warp
    from skimage.transform import ProjectiveTransform
    from dataclasses import replace
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/workspace/filmlab-audit');out=root/'halo-reachability';out.mkdir(exist_ok=True)
    generation.use('next_sensor');seed=zlib.crc32(b'R1-02-3')%1000
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    prior=json.loads((root/'brightness-local/R1-02-3.json').read_text());td=prior['timing']
    t=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    e.scanner.timing=t;e.px_um=json.loads((root/'corridor-lights/results.json').read_text())['model_um_per_pixel']
    assert np.isneginf(e.e.fog_exposure).all(),'effective exposure addition assumes zero fog'
    # A frozen MODEL optical cone only: existing normal-depth stack, reference
    # D65 calibration, no claim of measured undeveloped absorption or efficiency.
    down,direct,back=film.emulsion_stack(e.e.stock,e.v.layer_absorbance,e.v.yellow_filter)
    E=spectral.illuminant(lights.WB_REFERENCE);sens=10**e.e.stock.log_sensitivity
    units=(E@sens)/(E@direct)*10**(e.log_k-e.e.speed_loss+e.layer_cal)
    optical_basis=down[:,None]*back*units
    report={'scope':__doc__,'generation':'next_sensor','timing':td,'native_um_per_pixel':e.px_um,
            'tolerances':{'abs_L':1.,'ab_distance':2.},'regions':[],
            'optical_cone':'existing proxy normal-depth stack with D65 calibration; necessary membership only, no source energy bound',
            'limits':['input witnesses are not fitted material constants or physical correction proposals',
                      'nonnegative layer exposure is a relaxation of spectral and transport constraints',
                      'unknown source radiance prevents a measured energy budget for these lamps',
                      'finite multistart failure is not a certified global unreachable result',
                      'small local counterfactual patches do not establish simultaneous whole-image feasibility']}
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,name,oldname in zip(axes,['ceiling_lamp','middle_ceiling'],['near_ceiling','middle_ceiling']):
        path=root/'corridor-dye-response/cache'/(name+'.npz');a=np.load(path)
        valid=a['native_trusted'];nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        q1=a['q_base'][tuple(nearest)];q0=a['q_alloff'][tuple(nearest)]
        d1=e._develop(q1,blur_px=12/e.px_um).dye
        factors=a['native_base_dye']/np.maximum(d1,1e-12)
        factor=np.mean(factors,axis=-1);factor=factor[tuple(nearest)]
        safe=binary_erosion(valid,iterations=12)
        fac_error=float(np.max(np.ptp(factors[safe],axis=-1)))
        if fac_error>2e-6:raise ValueError('cached agitation is not scalar/reproduced')
        wp=crop_warp(ProjectiveTransform(a['matrix']));supported=wp(safe.astype(float))>1-1e-10
        old=np.load(root/'corridor-lights'/(oldname+'.npz'))
        diag=np.load(root/'halo-residual-direction'/(name+'.npz'))
        smooth={k:gaussian_filter(old[v],sigma=(25/prior['scan_um_per_pixel'],25/prior['scan_um_per_pixel'],0)) for k,v in [('real','real'),('nohalo','none'),('halo','exp')]}
        targetlab=lab(smooth['real']);A,rows=bilinear_operator(q0.shape[:2],a['matrix'])
        reg={'name':name,'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'agitation_channel_spread_max':fac_error,'patches':[]}
        for zone in ['near_edge_0_100','transition_100_300','real_orange_transition_0_300','darker_outer_300_800_Ylt_.4']:
            zone_mask=diag['mask_'+zone]&supported;yy,xx=np.where(zone_mask)
            if not len(yy):continue
            # Target-only representative; never selected by any model error.
            target_mean=targetlab[zone_mask].mean(0);cost=np.linalg.norm(targetlab[yy,xx]-target_mean,axis=-1)
            iy=int(np.argmin(cost));y,x=int(yy[iy]),int(xx[iy]);target=targetlab[y,x]
            impulse=np.zeros((512,512));impulse[y,x]=1
            weights=np.asarray(A.T@gaussian_filter(impulse,25/prior['scan_um_per_pixel']).ravel()[rows]).reshape(q0.shape[:2])
            wy,wx=np.where(weights>0);yl,yh=wy.min()-5,wy.max()+6;xl,xh=wx.min()-5,wx.max()+6
            if min(yl,xl)<0 or yh>q0.shape[0] or xh>q0.shape[1]:raise ValueError('patch context unavailable')
            if not valid[yl:yh,xl:xh].all():raise ValueError('untrusted patch context')
            q=q0[yl:yh,xl:xh];fac=factor[yl:yh,xl:xh];w=weights[yl:yh,xl:xh]
            control=np.zeros(w.shape);control[2:-2,2:-2]=1 # target + unchanged 2-pixel chemistry context outside intervention
            href=np.sum(10**q*w[...,None],axis=(0,1))/w.sum()
            def response(u,detail=False):
                addition=href*np.expm1(np.log(2)*np.asarray(u))
                changed=add_exposure(q,control[...,None]*addition)
                st=e._develop(changed,blur_px=12/e.px_um);st.dye*=fac[...,None]
                T=e._transmittance(st);dn=e.scanner.normalized_density(T);rgb=e.scanner.to_rgb(dn)
                value=lab(np.sum(rgb*w[...,None],axis=(0,1)))
                if detail:return value,addition,st.dye,T,dn,rgb,changed
                return value
            nohalo=response([0.,0.,0.]);cached=lab(smooth['nohalo'][y,x]);repro=float(np.max(abs(nohalo-cached)))
            if repro>2e-4:raise ValueError('nohalo local complete backend not reproduced: '+str(repro))
            # Curve saturation bounds, not a chosen light-energy budget. Beyond
            # this all local characteristic curves are at the stored plateau.
            qcap=e.e.log_exposure.max()+3
            upper=np.log2(1+10**qcap/href)
            trials=[]
            for start in [np.array([.5,.05,.0001]),np.ones(3),np.array([1.,.01,1.])]:
                f=least_squares(lambda u:response(u)-target,start,bounds=(np.zeros(3),upper),
                                jac='3-point',max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-8)
                value=response(f.x);trials.append((np.linalg.norm(value-target),f,value))
            _,sol,value=min(trials,key=lambda v:v[0]);detail=response(sol.x,True);addition=detail[1]
            # Explicit luminance constraint, rather than trading L against a/b
            # in a single least-squares score. This is input feasibility, not a
            # chosen correction. Keep unsuccessful attempts in the record.
            def colour_loss(u):
                gap=response(u)[1:]-target[1:]
                return float(gap@gap)
            con={'type':'ineq','fun':lambda u:np.array([1-(response(u)[0]-target[0]),1+(response(u)[0]-target[0])])}
            constrained=[]
            for start in [np.zeros(3),sol.x]:
                cs=minimize(colour_loss,start,method='SLSQP',bounds=list(zip(np.zeros(3),upper)),constraints=con,
                            options={'maxiter':100,'ftol':1e-8,'eps':1e-4})
                cv=response(cs.x);constrained.append({'u':cs.x.tolist(),'Lab':cv.tolist(),'error_Lab':(cv-target).tolist(),
                    'ab_distance':float(np.linalg.norm(cv[1:]-target[1:])),
                    'L_constraint_satisfied':bool(abs(cv[0]-target[0])<=1.0001),'success':bool(cs.success)})
            # Signed-exposure control only: demonstrates whether the same
            # backend can encode the colour if the fixed-base restriction is
            # relaxed. Never call a negative addition physical returned light.
            lower=-.99*np.min(10**q[control>0],axis=0)/href
            def signed_response(z):
                changed=np.log10(10**q+control[...,None]*(href*z))
                st=e._develop(changed,blur_px=12/e.px_um);st.dye*=fac[...,None]
                rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(st)))
                return lab(np.sum(rgb*w[...,None],axis=(0,1)))
            signed_start=addition/href
            signed_trials=[]
            for start in [signed_start,np.array([.5,.1,max(-.1,lower[2]/2)])]:
                trial=least_squares(lambda z:signed_response(z)-target,start,
                    bounds=(lower,np.maximum(2**upper-1,10)),jac='3-point',max_nfev=150,
                    xtol=1e-9,ftol=1e-9,gtol=1e-8)
                signed_trials.append(trial)
            sf=min(signed_trials,key=lambda trial:np.linalg.norm(signed_response(trial.x)-target))
            signed_value=signed_response(sf.x)
            # Cone membership is separately audited; never present arbitrary
            # independent layer controls as a measured lamp spectrum.
            basis=optical_basis/np.maximum(np.linalg.norm(optical_basis,axis=1,keepdims=True),1e-30)
            coef,resid=nnls(basis.T,addition,maxiter=2000)
            cone_relative=float(resid/max(np.linalg.norm(addition),1e-20))
            full=lab(smooth['halo'][y,x]);current_delta=(10**a['q_base']-10**a['q_alloff'])[yl:yh,xl:xh]
            row={'zone':zone,'zone_pixels':int(zone_mask.sum()),'output_yx':[y,x],'target_Lab':target.tolist(),
                 'nohalo_Lab':nohalo.tolist(),'full_Lab':full.tolist(),'witness_Lab':value.tolist(),
                 'nohalo_reproduction_Lab_max':repro,'witness_feasible':feasible(value,target),
                 'witness_error_Lab':(value-target).tolist(),'u_addition_parameter':sol.x.tolist(),
                 'extra_effective_H_RGB':addition.tolist(),'effective_H_reference':href.tolist(),
                 'center_total_exposure_increment_EV':np.log2(1+addition/href).tolist(),
                 'current_mean_extra_H_RGB':np.sum(current_delta*w[...,None],axis=(0,1)).tolist(),
                 'proxy_optical_cone_relative_residual':cone_relative,
                 'proxy_cone_positive_wavelengths_nm':(380+5*np.flatnonzero(coef>1e-8)).tolist(),
                 'fixed_L_constraint_trials':constrained,
                 'signed_control_NOT_returned_light':{'relative_addition_H_over_reference':sf.x.tolist(),
                     'Lab':signed_value.tolist(),'error_Lab':(signed_value-target).tolist(),'feasible':feasible(signed_value,target)},
                 'trials':[{'cost_Lab_norm':float(s[0]),'evaluations':s[1].nfev,'success':bool(s[1].success)} for s in trials]}
            reg['patches'].append(row)
            np.savez_compressed(out/(name+'_'+zone+'.npz'),target_Lab=target,base_q=q,control=control,weights=w,
                                agitation=fac,extra_H=addition,witness_dye=detail[2],witness_T=detail[3],witness_Dn=detail[4],
                                witness_RGB=detail[5],witness_q=detail[6],optical_basis=optical_basis)
            ax.plot([nohalo[1],full[1]],[nohalo[2],full[2]],'-',alpha=.6)
            ax.scatter(*nohalo[1:],marker='s',color='gray');ax.scatter(*full[1:],marker='o',color='orange')
            ax.scatter(*target[1:],marker='*',s=120,color='green');ax.scatter(*value[1:],marker='x',s=70,color='purple')
            ax.annotate(zone,(target[1],target[2]),fontsize=6)
            (out/'results.partial.json').write_text(json.dumps(report|{'active_region':reg},indent=2))
            print(name,zone,'feasible',row['witness_feasible'],'dLab',row['witness_error_Lab'],'extraEV',sol.x,'cone',cone_relative,flush=True)
        report['regions'].append(reg);ax.set(title=name,xlabel='a*',ylabel='b*');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
    fig.suptitle('Fixed-backend INPUT witnesses only: square=nohalo, circle=full, star=real, x=witness')
    fig.savefig(out/'reachability.png',dpi=170)
    plot_tradeoff(report,out)
    (out/'results.json').write_text(json.dumps(report,indent=2));print('saved',out,flush=True)


if __name__=='__main__':main()
