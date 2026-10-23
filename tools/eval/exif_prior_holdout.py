"""Two actual censored-source controls for the opt-in recorded-EXIF candidate.

Replays the exact full-frame reconstruct_highlights input captured from load_raw,
including its preceding LibRaw XYZ ceiling and denoising. Same paired seeds,
prior/MAX_STOPS and measured exposure normalization; no fitting to hidden truth.
Reference camera values are independently corroborated short-exposure samples.
Inverse-decoder camera outputs are only model proxies, not new measurements.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,sys,time,zlib
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from filmlab import generation,rawio
from radiance_censor_holdout import reference_fields


def channel_metrics(estimate,truth,mask,low,high):
    use=mask&np.isfinite(estimate)&(truth>0)
    positive=use&(estimate>0)
    error=np.log2(estimate[positive]/truth[positive])
    finite=mask&np.isfinite(low)&np.isfinite(high)
    widths=finite&(low>0)&(high>0)
    return {'missing_channels_scored':int(mask.sum()),'positive_predictions':int(positive.sum()),
      'nonpositive_predictions':int((use&~positive).sum()),
      'median_signed_error_EV':float(np.median(error)) if error.size else None,
      'median_absolute_error_EV':float(np.median(abs(error))) if error.size else None,
      'p90_absolute_error_EV':float(np.percentile(abs(error),90)) if error.size else None,
      'within_quarter_EV_fraction':float((abs(error)<=.25).mean()) if error.size else None,
      'finite_interval_fraction':float(finite.sum()/max(mask.sum(),1)),
      'algorithm_05_95_interval_coverage':float(((truth>=low)&(truth<=high))[finite].mean()) if finite.any() else None,
      'positive_interval_width_channels':int(widths.sum()),
      'median_algorithm_interval_width_EV':float(np.median(np.log2(high[widths]/low[widths]))) if widths.any() else None}


def run(directory,out,seeds=32):
    metadata=json.loads((directory/'lamp-measurements.json').read_text());decoder=json.loads((directory/'decoder.json').read_text())
    lamps=[x for x in metadata['lamps'] if x['region']['name'] in ('roof_opening_1','roof_opening_2')]
    assert len(lamps)==2 and len({x['primary_file'] for x in lamps})==1
    src=Path('/workspace/filmlab-data/epfl/data/digital')/lamps[0]['primary_file']
    mt={};rawio._exif_complete(src,trace=mt)
    old_lclip=rawio.clip_luminance(src);new_lclip=rawio.clip_luminance(src,complete_exif=True)
    if old_lclip is not None or new_lclip is None or mt['conflicts']:
        raise ValueError('This declared case must have missing legacy ISO and matching completed metadata')
    generation.use('next_sensor');original=rawio.reconstruct_highlights;capture={}
    def tap(rgb,clip,rng,l_clip=None):
        capture['rgb']=rgb.copy();capture['clip']=clip.copy();capture['lclip_seen']=l_clip
        return original(rgb,clip,rng,l_clip)
    with patch.object(rawio,'reconstruct_highlights',tap):old_actual=rawio.load_raw(src,max_size=3000)
    before=capture['rgb'];clip=capture['clip'];assert capture['lclip_seen']==old_lclip
    # Each full decode precedes the prior in exactly the same domain.
    with patch.object(rawio,'reconstruct_highlights',tap):new_actual=rawio.load_raw(src,max_size=3000,complete_exif=True)
    np.testing.assert_array_equal(before,capture['rgb']);np.testing.assert_array_equal(clip,capture['clip'])
    assert capture['lclip_seen']==new_lclip
    matrix=np.asarray(decoder['matrix'])@rawio.XYZ_TO_SRGB.T;inverse=np.linalg.inv(matrix)
    cases=[]
    for lamp in lamps:
        name=lamp['region']['scene']+'_'+lamp['region']['name'];p=directory/(name+'_measurements.npz')
        a=np.load(p);truth,support,_,pairs=reference_fields(lamp,a)
        x,y=lamp['region']['center'];half=lamp['region']['half'];sl=(slice(y-half,y+half),slice(x-half,x+half))
        core=a['mask'];sat=a['0_sat'].astype(bool);valid=core&support;missing=valid[...,None]&sat
        assert truth.shape==before[sl].shape
        cases.append({'name':name,'cache':p,'truth':truth,'pairs':pairs,'support':support,'core':core,'sat':sat,'valid':valid,'missing':missing,'slice':sl,'samples':{'legacy':[],'complete_exif':[]},'before_camera_proxy':before[sl]@inverse})
    crc=zlib.crc32(str(src).encode())
    replay={}
    for mode,lc,actual in [('legacy',old_lclip,old_actual),('complete_exif',new_lclip,new_actual)]:
        for seed in range(seeds):
            result=original(before,clip,np.random.default_rng(crc^seed),lc)
            if seed==0:
                np.testing.assert_array_equal(result,actual);replay[mode]=True
            for c in cases:c['samples'][mode].append(result[c['slice']].copy())
        print(mode,'paired seeds complete',seeds,flush=True)
    report={'scope':__doc__,'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shape':list(before.shape),
      'metadata_trace':mt,'legacy_clip_luminance':old_lclip,'candidate_nominal_clip_luminance':new_lclip,
      'candidate_exposure':rawio.exif_exposure(src,complete_exif=True),'paired_seed_arguments':list(range(seeds)),
      'actual_rng_rule':'crc32(str(source_path)) XOR load_raw seed argument','actual_api_seed0_replay_bitwise':replay,
      'pre_prior_inputs_equal_bitwise':True,'prior_parameters':{'MAX_STOPS':rawio.MAX_STOPS,'HIGHLIGHT_PRIOR':rawio.HIGHLIGHT_PRIOR,'LINE_PRIOR':rawio.LINE_PRIOR},'cases':[],
      'limits':['No value or model was selected using hidden truth.','Truth is accepted relative camera-channel measurement, not lamp spectral radiance.','Inverse decoder maps reconstructed RGB to camera PROXY; actual load_raw before-prior data include XYZ clipping and denoising.','The nominal cd/m2 clipping estimate is neither measured lamp brightness nor an independently calibrated radiometric conversion.','32-seed intervals describe algorithm randomness; they are not credible intervals or physical uncertainty.','These apertures are extended transmitted-light controls, not recovered primary luminaire cores.','All reconstructed missing channels remain unknown estimates, even if a particular seed matches short-exposure truth.']}
    for c in cases:
        truth=c['truth'];missing=c['missing'];groups={'all_missing':missing,'partial_RGB_flags':missing&(~c['sat'].all(-1))[...,None],'all_RGB_flags':missing&c['sat'].all(-1)[...,None]}
        row={'name':c['name'],'input_cache_sha256':hashlib.sha256(c['cache'].read_bytes()).hexdigest(),'independent_reference_pairs':c['pairs'],
          'coverage':{'core_cells':int(c['core'].sum()),'primary_any_saturated_core_cells':int((c['core']&c['sat'].any(-1)).sum()),'two_exposure_supported_missing_cells':int((c['valid']&c['sat'].any(-1)).sum()),'missing_channels_total_core':int((c['core'][...,None]&c['sat']).sum()),'independent_truth_supported_missing_channels':int(missing.sum()),'remaining_without_truth_channels':int((c['core'][...,None]&c['sat']&~c['support'][...,None]).sum()),'all_reconstructed_missing_channels_still_estimated':True},'paths':{}}
        save={'truth_camera':truth,'missing':missing,'core':c['core'],'independent_support':c['support'],'before_prior_camera_proxy':c['before_camera_proxy']}
        for mode in ['legacy','complete_exif']:
            rgb_stack=np.stack(c['samples'][mode]);cam=rgb_stack@inverse
            med=np.median(cam,axis=0);lo,hi=np.quantile(cam,[.05,.95],axis=0)
            actual_camera=cam[0]
            entry={'camera_proxy':{key:channel_metrics(med,truth,mask,lo,hi) for key,mask in groups.items() if mask.any()},'actual_seed0_camera_proxy':{key:channel_metrics(actual_camera,truth,mask,lo,hi) for key,mask in groups.items() if mask.any()}}
            Y=rgb_stack@rawio.LUMA;Ytruth=(truth@matrix)@rawio.LUMA;Ymask=c['valid']&c['sat'].any(-1)
            yp=np.median(Y,0);positive=Ymask&(Ytruth>0)&(yp>0);er=np.log2(yp[positive]/Ytruth[positive]);yl,yh=np.quantile(Y,[.05,.95],axis=0)
            entry['linear_RGB_Y_proxy']={'cells':int(positive.sum()),'median_signed_error_EV':float(np.median(er)),'median_absolute_error_EV':float(np.median(abs(er))),'p90_absolute_error_EV':float(np.percentile(abs(er),90)),'algorithm_05_95_interval_coverage':float(((Ytruth>=yl)&(Ytruth<=yh))[positive].mean())}
            available=~c['sat'];entry['max_available_channel_change_from_preprior_camera_proxy']=float(np.max(abs(med-c['before_camera_proxy'])[available],initial=0))
            row['paths'][mode]=entry
            save[mode+'_camera_proxy_median']=med;save[mode+'_camera_proxy_05']=lo;save[mode+'_camera_proxy_95']=hi
        np.savez_compressed(out/(c['name']+'.npz'),**save);report['cases'].append(row)
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measurements',type=Path,default=Path('/workspace/filmlab-audit/measured-sources'));p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/exif-prior-holdout'));p.add_argument('--seeds',type=int,default=32);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    t=time.time();report=run(a.measurements,a.out,a.seeds)
    for case in report['cases']:
        print(case['name'],json.dumps({m:v['camera_proxy']['all_missing'] for m,v in case['paths'].items()}),flush=True)
    print('elapsed',time.time()-t,flush=True)


if __name__=='__main__':main()
