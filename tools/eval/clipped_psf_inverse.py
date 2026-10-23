"""Conditional PSF extrapolation of censored RAW cores, independently scored.

Gaussian/Moffat are explicit source-plus-camera-blur hypotheses, not measured
lens PSFs or 800T transport. Only primary available samples and censoring bounds
train. Independent short-exposure core measurements score predictions afterward.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/eval')]


def profile(kind,r2,width,beta=2.):
    if width<=0 or (kind=='moffat' and beta<=1):raise ValueError('finite-energy profile requires width>0 and Moffat beta>1')
    if kind=='gaussian':return np.exp(-np.asarray(r2)/(2*width**2))
    if kind=='moffat':return (1+np.asarray(r2)/width**2)**(-beta)
    if kind=='disk':
        from scipy.special import chndtr
        if beta<=0:raise ValueError('disk radius must be positive')
        return np.clip(chndtr((beta/width)**2,2,np.asarray(r2)/width**2),0,1)
    raise ValueError('unknown profile')


def integrated_signal(kind,amplitude,width,beta=2.):
    if amplitude<0:raise ValueError('amplitude must be nonnegative')
    profile(kind,0,width,beta)
    if kind=='disk':return float(amplitude*np.pi*beta**2)
    return float(amplitude*2*np.pi*width**2 if kind=='gaussian' else amplitude*np.pi*width**2/(beta-1))


def model(kind,parameters,yy,xx):
    cx,cy,loga,logw,bg,beta=parameters
    return np.exp(loga)*profile(kind,(xx-cx)**2+(yy-cy)**2,np.exp(logw),beta)+bg


def censored_residual(prediction,observation,saturation,scale=.01):
    pred=np.arcsinh(np.asarray(prediction)/scale);obs=np.arcsinh(np.asarray(observation)/scale)
    return np.where(saturation,np.minimum(pred-obs,0),pred-obs)


def fit_channel(observed,saturation,radius,kind,peak_cap=100.,starts=8,max_nfev=180):
    observed=np.asarray(observed,float);sat=np.asarray(saturation,bool);yy,xx=np.indices(observed.shape)
    cy,cx=np.array(observed.shape)//2;r=np.hypot(xx-cx,yy-cy);angle=np.arctan2(yy-cy,xx-cx)
    sector=np.floor((angle+np.pi)/(np.pi/4)).astype(int)%8
    domain=(r<=min(3*radius,60))&np.isfinite(observed)
    train=domain&(sector%2==0);test=domain&~train
    if (train&~sat).sum()<40:raise ValueError('insufficient available wing samples')
    # Fixed geometric bounds, not selected by core/heldout truth.
    shift=max(2.,radius/3);wmax=max(3.,2*radius)
    low=np.array([cx-shift,cy-shift,np.log(.001),np.log(.4),0.,1.05])
    high=np.array([cx+shift,cy+shift,np.log(peak_cap),np.log(wmax),max(.02,float(np.nanpercentile(observed[domain],90))),12.])
    if kind=='gaussian':low[-1]=1.999999;high[-1]=2.000001
    bg=float(np.clip(np.nanpercentile(observed[train&~sat],10),0,high[4]*.9))
    peak=float(np.clip(np.nanmax(observed[domain])-bg,.002,peak_cap*.9))
    trials=[]
    for seed in range(starts):
        width=max(.6,radius*(.15+.12*(seed%4)))
        a=min(peak*(1+seed//4*5),peak_cap*.9)
        initial=np.array([cx+(.5 if seed%2 else -.5),cy,np.log(a),np.log(width),bg,2. if kind=='gaussian' else 1.5+seed%3])
        initial=np.minimum(np.maximum(initial,low+1e-9),high-1e-9)
        fun=lambda p:censored_residual(model(kind,p,yy[train],xx[train]),observed[train],sat[train])
        result=least_squares(fun,initial,bounds=(low,high),loss='soft_l1',f_scale=.15,max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        pred=model(kind,result.x,yy,xx);res=fun(result.x)
        # Selection uses the same declared robust training loss, never heldout score.
        objective=float(np.sum(2*.15**2*(np.sqrt(1+(res/.15)**2)-1)))
        trials.append({'parameters':result.x.tolist(),'objective':objective,'nfev':result.nfev,'success':bool(result.success),'peak':float(np.exp(result.x[2])),'width':float(np.exp(result.x[3])),'beta':float(result.x[5]),'background':float(result.x[4]),'integrated_signal':integrated_signal(kind,np.exp(result.x[2]),np.exp(result.x[3]),result.x[5]),'peak_at_upper_bound':bool(np.exp(result.x[2])>peak_cap*.99)})
    chosen=min(range(len(trials)),key=lambda i:trials[i]['objective']);best=trials[chosen];prediction=model(kind,np.array(best['parameters']),yy,xx)
    available_test=test&~sat
    record={'profile':kind,'peak_cap':peak_cap,'selection':'minimum robust training objective; heldout/core truth never selects','train_available':int((train&~sat).sum()),'train_censored':int((train&sat).sum()),'heldout_available':int(available_test.sum()),'heldout_asinh_RMSE':float(np.sqrt(np.mean(censored_residual(prediction[available_test],observed[available_test],False)**2))),'raw_profile_censored_lower_bound_violation_fraction':float(np.mean(prediction[domain&sat]<observed[domain&sat]-1e-6)) if (domain&sat).any() else None,'selected_trial':chosen,'trials':trials,'best':best,'fit_evaluations':sum(t['nfev'] for t in trials)}
    return prediction,record,train,test


def error_metrics(estimate,truth,mask):
    use=np.asarray(mask,bool)&np.isfinite(truth)&(truth>0)&np.isfinite(estimate)&(estimate>0)
    if not use.any():return {'channels':0}
    e=np.log2(estimate[use]/truth[use])
    return {'channels':int(use.sum()),'median_error_EV':float(np.median(e)),'median_absolute_error_EV':float(np.median(abs(e))),'p90_absolute_error_EV':float(np.percentile(abs(e),90)),'fraction_within_quarter_stop':float((abs(e)<=.25).mean())}


def run(directory,out,starts=8):
    from radiance_censor_holdout import reference_fields
    meta=json.loads((directory/'lamp-measurements.json').read_text());report={'scope':__doc__,'starts':starts,'region_results':[],'limits':['Models are source-plus-camera-blur hypotheses; extended apertures need not follow a point-source PSF.','Primary clipped observations constrain only lower bounds.','Peak caps 100 and 1000 are numerical sensitivity bounds, not measured radiance upper limits.','Independent exposures are EXIF-normalized relative camera RGB, not calibrated spectral radiance.','Angular sectors are deterministic heldout checks, not independent photons or scenes.','No core truth selects profile family, start, peak cap, centre, or shape bounds.','Integrated signal is relative camera signal times pixel area; background is excluded.','No 800T material, scanner, DIR or halation kernel is changed.']}
    names=['roof_opening_1','roof_opening_2','facade_lamp','walkway_1','distant_lamp']
    for lamp in meta['lamps']:
        if lamp['region']['name'] not in names:continue
        name=lamp['region']['scene']+'_'+lamp['region']['name'];data=np.load(directory/(name+'_measurements.npz'))
        truth,support,unknown,pairs=reference_fields(lamp,data);core=data['mask'];radius=lamp['region']['radius']
        actual=lamp['region']['name'].startswith('roof_opening')
        caps=[None] if actual else [.2,.4]
        region={'name':name,'kind':lamp['region']['kind'],'corroborated_core_cells':int((core&support).sum()),'corroborated_original_missing_core_cells':int((core&support&data['0_sat'].any(-1)).sum()),'cases':[]}
        for cap in caps:
            obs=data['0_camera'].astype(float) if cap is None else np.minimum(truth,cap)
            sat=data['0_sat'].astype(bool) if cap is None else (truth>=cap)|unknown
            missing=core[...,None]&support[...,None]&sat
            if missing.sum()<12:continue
            case={'censoring':'actual_primary' if cap is None else 'artificial','cap':cap,'score_channels':int(missing.sum()),'baseline_floor':error_metrics(obs,truth,missing),'models':[]}
            saved={'observed':obs.astype('float32'),'truth':truth.astype('float32'),'saturation':sat,'score_missing':missing}
            for kind in ['gaussian','moffat']:
                for peakcap in [100.,1000.]:
                    prediction=np.empty_like(obs);channels=[]
                    for c in range(3):
                        pred,record,train,test=fit_channel(obs[...,c],sat[...,c],radius,kind,peakcap,starts)
                        prediction[...,c]=pred;channels.append(record)
                    # Predictions only replace unknown channels; measured values remain unchanged.
                    estimate=np.where(sat,np.maximum(prediction,obs),obs)
                    row={'kind':kind,'peak_cap':peakcap,'channels':channels,'raw_profile_hidden_core_error':error_metrics(prediction,truth,missing),'projection_policy':'max(observed censoring lower bound, raw fitted profile); projected values no longer necessarily form one PSF','hidden_core_error':error_metrics(estimate,truth,missing),'measured_channel_max_change':float(np.max(abs(estimate[~sat]-obs[~sat]),initial=0))}
                    case['models'].append(row);saved[kind+'_'+str(int(peakcap))]=estimate.astype('float32')
                    print(name,cap,kind,peakcap,row['hidden_core_error'],flush=True)
            np.savez_compressed(out/(name+'_'+('actual' if cap is None else str(cap))+'.npz'),**saved)
            region['cases'].append(case)
        report['region_results'].append(region);(out/'results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measurements',type=Path,default=Path('/workspace/filmlab-audit/measured-sources'));p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/clipped-psf-inverse'));p.add_argument('--starts',type=int,default=8);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);run(a.measurements,a.out,a.starts)


if __name__=='__main__':main()
