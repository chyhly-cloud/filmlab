"""Hard-censoring source PSF check, distinct from soft fits and output projection.

A_min(shape,b) is the largest amplitude lower bound imposed by censored pixels
inside the declared source support. A=A_min+exp(extra) therefore satisfies all
these bounds by construction. A positive finite-energy shape is a necessary
model property, not proof that an extended aperture follows that shape.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tools/eval')]
from clipped_psf_inverse import profile,integrated_signal,error_metrics


def hard_prediction(kind,p,yy,xx,observed,saturation,source):
    cx,cy,extra,logw,b,beta=p
    g=profile(kind,(xx-cx)**2+(yy-cy)**2,np.exp(logw),beta)
    censored=np.asarray(saturation,bool)&source
    amin=float(np.max(np.maximum(observed[censored]-b,0)/np.maximum(g[censored],1e-250),initial=0))
    amplitude=amin+np.exp(extra)
    return b+amplitude*g,amin,amplitude


def hard_fit(observed,saturation,source,radius,kind,extra_cap=100.,starts=4,max_nfev=220):
    observed=np.asarray(observed,float);sat=np.asarray(saturation,bool);yy,xx=np.indices(observed.shape);cy,cx=np.array(observed.shape)//2
    rr=np.hypot(xx-cx,yy-cy);sector=np.floor((np.arctan2(yy-cy,xx-cx)+np.pi)/(np.pi/4)).astype(int)%8
    domain=(rr<=min(3*radius,60))&np.isfinite(observed);train=domain&(sector%2==0)&~sat;test=domain&(sector%2==1)&~sat
    shift=max(2.,radius/3);bmax=max(.02,float(np.percentile(observed[domain],90)))
    low=[cx-shift,cy-shift,-12.,np.log(.4),0.,1.05];high=[cx+shift,cy+shift,np.log(extra_cap),np.log(2*radius),bmax,12.]
    if kind=='gaussian':low[-1]=1.999999;high[-1]=2.000001
    if kind=='disk':low[-1]=.5;high[-1]=radius;high[3]=np.log(radius)
    bg=float(np.clip(np.percentile(observed[train],10),0,bmax*.9));trials=[]
    def predict(p):return hard_prediction(kind,p,yy,xx,observed,sat,source)
    def residual(p):
        # Evaluate only fitting and inequality samples during optimization.
        # Full-grid prediction is retained for final validation and plotting.
        px,py,extra,logw,b,beta=p;w=np.exp(logw)
        c=source&sat
        gc=profile(kind,(xx[c]-px)**2+(yy[c]-py)**2,w,beta)
        amin=float(np.max(np.maximum(observed[c]-b,0)/np.maximum(gc,1e-250),initial=0))
        gt=profile(kind,(xx[train]-px)**2+(yy[train]-py)**2,w,beta)
        fitted=b+(amin+np.exp(extra))*gt
        return np.arcsinh(fitted/.01)-np.arcsinh(observed[train]/.01)
    for seed in range(starts):
        p0=np.array([cx+(-.5 if seed%2==0 else .5),cy,np.log(.1+seed*.3),np.log(max(.6,radius*(.2+.15*seed))),bg,2. if kind=='gaussian' else radius*(.3+.2*seed) if kind=='disk' else 1.5+seed%3])
        sol=least_squares(residual,p0,bounds=(low,high),loss='soft_l1',f_scale=.15,max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        pred,amin,amp=predict(sol.x);r=residual(sol.x);cost=float(np.sum(2*.15**2*(np.sqrt(1+(r/.15)**2)-1)))
        c=source&sat
        trials.append({'parameters':sol.x.tolist(),'robust_train_objective':cost,'nfev':sol.nfev,'success':bool(sol.success),'A_min':amin,'extra_amplitude':float(np.exp(sol.x[2])),'amplitude':amp,'integrated_signal':integrated_signal(kind,amp,np.exp(sol.x[3]),sol.x[5]),'minimum_censoring_margin':float(np.min(pred[c]-observed[c])) if c.any() else None,'extra_upper_bound_hit':bool(np.exp(sol.x[2])>=extra_cap*.99),'heldout_available_asinh_RMSE':float(np.sqrt(np.mean((np.arcsinh(pred[test]/.01)-np.arcsinh(observed[test]/.01))**2)))})
    selected=min(range(len(trials)),key=lambda i:trials[i]['robust_train_objective']);pred,amin,amp=predict(np.array(trials[selected]['parameters']));c=source&sat
    if c.any() and np.min(pred[c]-observed[c])<-1e-9:raise ValueError('Hard censoring lower bound violated')
    return pred,{'kind':kind,'extra_amplitude_cap':extra_cap,'selected_trial':selected,'trials':trials,'best':trials[selected],'train_available_samples':int(train.sum()),'heldout_available_samples':int(test.sum())}


def run():
    from radiance_censor_holdout import reference_fields
    directory=Path('/workspace/filmlab-audit/measured-sources');out=Path('/workspace/filmlab-audit/clipped-psf-inverse');meta=json.loads((directory/'lamp-measurements.json').read_text())
    report={'scope':__doc__,'parameterization':'A=max_censored((observation-b)/shape)+exp(extra); exact source-core inequality','limits':['Only declared source-core censoring constraints are imposed; neighboring lights outside support are not forced to belong to this PSF.','No core truth enters optimization or start/model selection.','100/1000 bound applies to extra amplitude above required A_min, not the total physically bounded source.','Camera-signal integrals exclude background and are not SI radiance.','Extended apertures are not necessarily point sources; failure is conditional on these shape assumptions.'],'regions':[]}
    for lamp in meta['lamps']:
        if lamp['region']['name'] not in ['roof_opening_1','roof_opening_2']:continue
        name=lamp['region']['scene']+'_'+lamp['region']['name'];a=np.load(directory/(name+'_measurements.npz'));truth,support,unknown,pairs=reference_fields(lamp,a);obs=a['0_camera'].astype(float);sat=a['0_sat'];source=a['mask'];missing=source[...,None]&support[...,None]&sat
        region={'name':name,'hidden_missing_channels':int(missing.sum()),'floor':error_metrics(obs,truth,missing),'models':[]};saved={'observed':obs,'truth':truth,'score_missing':missing}
        for kind in ['gaussian','moffat']:
            for extra_cap in [100.,1000.]:
                pred=np.empty_like(obs);records=[]
                for c in range(3):
                    pred[...,c],record=hard_fit(obs[...,c],sat[...,c],source,lamp['region']['radius'],kind,extra_cap);records.append(record)
                # Reconstruct only unknown core samples; available observations stay measurements.
                estimate=np.where(source[...,None]&sat,pred,obs)
                row={'kind':kind,'extra_cap':extra_cap,'channels':records,'hidden_core_error':error_metrics(estimate,truth,missing),'observed_channel_max_change':float(np.max(abs(estimate[~sat]-obs[~sat]),initial=0))};region['models'].append(row);saved[kind+'_'+str(int(extra_cap))]=estimate.astype('float32');print(name,kind,extra_cap,row['hidden_core_error'],flush=True)
        report['regions'].append(region);np.savez_compressed(out/(name+'_hard.npz'),**saved);(out/'hard-results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'hard-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
