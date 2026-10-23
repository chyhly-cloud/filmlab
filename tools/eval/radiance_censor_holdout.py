"""Censored RAW holdout: mathematical guesses tested on independent exposures.

No film/scan fitting. Artificial censoring uses declared camera-signal caps;
reported 'truth' is a relative camera measurement corroborated by another RAW,
not a measured spectrum or actual source radiance. Known lamp edges/apertures
must not be generalized to unrecovered lamp cores.
The legacy prior is replayed on local camera-to-RGB crops: its area-based class,
preprocessing and RNG sequence differ from full load_raw. Those prior scores
are crop controls, not actual full-pipeline reconstruction benchmarks.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
from scipy.optimize import nnls
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from filmlab import rawio
from tools.eval.sensor_constrained_input import constrain_camera


def censor(camera,cap,unknown=None):
    camera=np.asarray(camera,float)
    sat=camera>=cap
    if unknown is not None:sat|=np.asarray(unknown,bool)
    return np.minimum(camera,cap),sat


def local_regression(observed,saturation,holdout):
    """Fit only uncensored surrounding pixels; source target never trains/calibrates.

    Nonnegative through-origin coefficients predict a missing channel from the
    remaining available channels. Calibration residual percentiles are merely
    local empirical intervals: changed material/illumination can break coverage.
    Full RGB censoring returns the observation with an infinite upper bound.
    """
    obs=np.asarray(observed,float);sat=np.asarray(saturation,bool);core=np.asarray(holdout,bool)
    out=obs.copy();lo=obs.copy();hi=np.full_like(obs,np.inf);yy,xx=np.indices(core.shape)
    records=[]
    for c in range(3):
        other=[i for i in range(3) if i!=c]
        for pattern in [(other[0],),(other[1],),tuple(other)]:
            active=np.zeros(3,bool);active[list(pattern)]=True
            use=core&sat[...,c]&np.all(~sat[...,list(pattern)],axis=-1)
            # Exactly this set of available other channels; disjoint predictions.
            rest=[i for i in other if i not in pattern]
            if rest:use&=np.all(sat[...,rest],axis=-1)
            if not use.any():continue
            clean=~core&~sat[...,c]&np.all(~sat[...,list(pattern)],axis=-1)
            clean&=(obs[...,c]>.002)&(obs[...,list(pattern)].min(-1)>.002)
            train=clean&((xx+yy)%2==0);cal=clean&~train
            if train.sum()<30 or cal.sum()<20:continue
            X=obs[...,list(pattern)];beta,_=nnls(X[train],obs[...,c][train])
            predicted=X[use]@beta
            out[...,c][use]=np.maximum(obs[...,c][use],predicted)
            cp=X[cal]@beta;ok=cp>.001
            if ok.sum()<20:continue
            residual=np.log2(obs[...,c][cal][ok]/cp[ok])
            q=np.quantile(residual,[.05,.95])
            lo[...,c][use]=np.maximum(obs[...,c][use],np.maximum(predicted,1e-12)*2**q[0])
            hi[...,c][use]=np.maximum(lo[...,c][use],np.maximum(predicted,1e-12)*2**q[1])
            records.append({'channel':c,'features':list(pattern),'train':int(train.sum()),'calibration':int(ok.sum()),'coefficients':beta.tolist(),'residual_EV_quantiles_05_95':q.tolist()})
    # No correction can touch an observed channel.
    out[~sat]=obs[~sat];lo[~sat]=obs[~sat];hi[~sat]=obs[~sat]
    return out,lo,hi,records


def reference_fields(lamp,data):
    """Use two geometrically accepted, unsaturated exposures agreeing <0.15EV.

    Longer safe exposures are preferred only after this fixed agreement gate.
    No per-view radiometric gain is estimated. Views already share EXIF scale.
    """
    base=data['0_camera'].astype(float);initial_unknown=data['0_sat'].astype(bool)
    signal=base.copy();support=np.zeros(base.shape[:2],bool);pairs=[]
    views=[0]+[i for i,a in enumerate(lamp['alternatives'],1) if a['local_geometry']['pass']]
    # Alt-alt support first: it can corroborate primary's originally missing core.
    order=sorted([(i,j) for a,i in enumerate(views) for j in views[a+1:]],key=lambda x:(0 in x,x))
    for i,j in order:
        a=data[f'{i}_camera'].astype(float);b=data[f'{j}_camera'].astype(float)
        keep=data[f'{i}_safe']&data[f'{j}_safe']&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
        keep&=(a.min(-1)>.005)&(b.min(-1)>.005)
        error=np.abs(np.log2(np.maximum(a,1e-9)/np.maximum(b,1e-9)))
        keep&=(error<.15).all(-1)&~support
        if not keep.any():continue
        signal[keep]=a[keep];support|=keep
        pairs.append({'views':[lamp['views'][i],lamp['views'][j]],'pixels':int(keep.sum()),'median_max_abs_EV':float(np.median(error[keep].max(-1)))})
    unknown=initial_unknown.copy();unknown[support]=False
    return signal,support,unknown,pairs


def score(estimate,truth,missing,lo=None,hi=None):
    valid=missing&np.isfinite(estimate)&(truth>0)
    positive=valid&(estimate>0)
    err=np.full_like(truth,np.nan,dtype=float);err[positive]=np.log2(estimate[positive]/truth[positive])
    e=err[positive]
    row={'channels':int(missing.sum()),'positive_predictions':int(positive.sum()),'nonpositive_predictions':int((valid&~positive).sum())}
    if e.size:
        row.update({'median_error_EV':float(np.median(e)),'median_absolute_error_EV':float(np.median(abs(e))),'p90_absolute_error_EV':float(np.percentile(abs(e),90)),'within_0.25_EV_fraction':float((abs(e)<=.25).mean())})
    if lo is not None:
        eligible=missing&np.isfinite(lo)&np.isfinite(hi)
        row.update({'finite_interval_channels':int(eligible.sum()),'finite_interval_fraction':float(eligible.sum()/max(missing.sum(),1)),'finite_interval_coverage':float(((truth>=lo)&(truth<=hi))[eligible].mean()) if eligible.any() else None,'median_interval_width_EV':float(np.median(np.log2(hi[eligible]/np.maximum(lo[eligible],1e-12)))) if eligible.any() else None})
    return row


def run(directory,out,thresholds,seeds):
    from filmlab.rawio import reconstruct_highlights,XYZ_TO_SRGB
    metadata=json.loads((directory/'lamp-measurements.json').read_text())
    decoder=json.loads((directory/'decoder.json').read_text())
    matrix=np.array(decoder['matrix'])@XYZ_TO_SRGB.T;inverse=np.linalg.inv(matrix)
    report={'scope':__doc__,'caps_camera_units':thresholds,'random_prior_seeds':seeds,'limits':['The accepted views share metadata exposure normalization, not absolute radiance calibration.','Only two-view safe, stable support is scored; principal lamp cores lacking support remain untested.','Artificial censoring is not a replacement for actual saturated-source recovery.','legacy_prior calls the actual RGB implementation with rawio.clip_luminance on the actual primary file. It runs on local camera-to-RGB crops, so area-based prior class, preprocessing and RNG differ from full load_raw. Scores are crop controls, not actual full-pipeline benchmarks. If ISO is unreadable it uses a fallback-stop branch. Available EXIF clipping estimates scale with artificial cap; neither branch independently calibrates radiance or physical uncertainty.','Local interval percentiles are empirical calibration intervals, not guaranteed confidence intervals under spectrum/background shifts.','No lamp class or material parameters are fitted; no film/scanner image is used.'],'regions':[]}
    for lamp in metadata['lamps']:
        name=lamp['region']['scene']+'_'+lamp['region']['name'];path=directory/(name+'_measurements.npz');data=np.load(path)
        signal,support,unknown,pairs=reference_fields(lamp,data);core=data['mask'];truth=signal.copy();primary_sat=data['0_sat']
        rawpath=Path('/workspace/filmlab-data/epfl/data/digital')/lamp['primary_file']
        l_clip=rawio.clip_luminance(str(rawpath))
        region={'name':name,'EXIF_clipping_luminance_estimate':l_clip,'kind':lamp['region']['kind'],'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'core_cells':int(core.sum()),'corroborated_core_cells':int((support&core).sum()),'corroborated_original_any_saturated_cells':int((support&core&primary_sat.any(-1)).sum()),'corroborated_original_all_RGB_flagged_cells':int((support&core&primary_sat.all(-1)).sum()),'reference_pairs':pairs,'cases':[]}
        for cap in list(thresholds)+[None]:
            if cap is None:
                obs=data['0_camera'].astype(float);sat=data['0_sat'].astype(bool);synth=sat.copy()
            else:
                obs,sat=censor(signal,cap,unknown);synth=signal>=cap
            test=core&support;missing=test[...,None]&synth
            if missing.sum()<12:continue
            available=~sat
            ratio,_=constrain_camera(obs,sat,mode='local_ratio')
            regression,low,high,fit=local_regression(obs,sat,core)
            # Crop camera-to-RGB prior control; NOT the full load_raw preprocessing/area/RNG domain.
            predictions={'floor':obs,'local_ratio':ratio,'local_regression':regression}
            intervals={'local_regression':(low,high)}
            clipping=None if l_clip is None else l_clip*(1. if cap is None else cap)
            samples=[]
            for seed in range(seeds):
                rgb=reconstruct_highlights(obs@matrix,sat.any(-1).astype(float),np.random.default_rng(seed),clipping)
                samples.append((rgb@inverse).astype('float32'))
            stack=np.stack(samples);predictions['legacy_prior']=np.median(stack,0)
            intervals['legacy_prior']=tuple(np.quantile(stack,[.05,.95],axis=0))
            groups={'all':missing,'partial_RGB':missing&(~sat.all(-1))[...,None],'full_RGB':missing&sat.all(-1)[...,None],'originally_primary_missing':missing&primary_sat,'primary_previously_measurable':missing&~primary_sat}
            row={'cap':cap,'censored_core_cells':int((test&synth.any(-1)).sum()),'full_RGB_censored_core_cells':int((test&sat.all(-1)).sum()),'regression_fits':fit,'metrics':{}}
            for method,estimate in predictions.items():
                entry={'max_observed_channel_change':float(np.max(abs(estimate[available]-obs[available]),initial=0)),'groups':{}}
                for label,mask in groups.items():
                    if mask.any():entry['groups'][label]=score(estimate,truth,mask,*(intervals.get(method,(None,None))))
                row['metrics'][method]=entry
            region['cases'].append(row)
        report['regions'].append(region)
        print(name,'corroborated',region['corroborated_core_cells'],'originally missing',region['corroborated_original_any_saturated_cells'],'cases',len(region['cases']),flush=True)
        (out/'results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--measurements',type=Path,default=Path('/workspace/filmlab-audit/measured-sources'));parser.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/radiance-censor-holdout'));parser.add_argument('--seeds',type=int,default=32);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    run(args.measurements,args.out,[.025,.05,.1,.2,.4,.8,1.6],args.seeds)


if __name__=='__main__':main()
