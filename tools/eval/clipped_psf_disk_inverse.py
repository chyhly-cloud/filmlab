"""Declared extended-aperture control: uniform disk convolved with Gaussian.

CDF is the noncentral chi-square disk integral of a normalized 2D Gaussian.
Finite source flux is pi*radius**2*amplitude, excluding background. It remains
an assumed circular uniform aperture and isotropic camera blur, not a measured
camera PSF, and cannot establish actual 800T properties.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tools/eval')]
from clipped_psf_hard_inverse import hard_fit
from clipped_psf_inverse import error_metrics
from radiance_censor_holdout import reference_fields


def run():
    directory=Path('/workspace/filmlab-audit/measured-sources');out=Path('/workspace/filmlab-audit/clipped-psf-inverse');metadata=json.loads((directory/'lamp-measurements.json').read_text());report={'scope':__doc__,'declared_bounds':{'disk_radius_native_halfsize_px':[.5,18.],'gaussian_sigma_px':[.4,18.],'center_offset_px':[-6.,6.],'starts_per_channel':4},'limits':['Only Gaussian isotropic camera blur and uniform circular source are included.','Short-exposure truth only scores hidden source-core channels, never selects the model or bounds.','Hard source-core sensor censoring bounds imposed through exact A_min; no output projection repairs the shape.','Extra-amplitude numerical cap 100 is not a physical amplitude cap.','Same source-minus-background positive finite integral does not establish correct source geometry or true radiance.'],'regions':[]}
    for lamp in metadata['lamps']:
        if lamp['region']['name'] not in ['roof_opening_1','roof_opening_2']:continue
        name=lamp['region']['scene']+'_'+lamp['region']['name'];a=np.load(directory/(name+'_measurements.npz'));truth,support,unknown,pairs=reference_fields(lamp,a);obs=a['0_camera'].astype(float);sat=a['0_sat'];source=a['mask'];missing=source[...,None]&support[...,None]&sat
        prediction=np.empty_like(obs);records=[]
        for c in range(3):
            prediction[...,c],record=hard_fit(obs[...,c],sat[...,c],source,lamp['region']['radius'],'disk',100.,starts=4,max_nfev=220);records.append(record)
        estimate=np.where(source[...,None]&sat,prediction,obs)
        row={'name':name,'hidden_missing_channels':int(missing.sum()),'floor':error_metrics(obs,truth,missing),'channels':records,'hidden_core_error':error_metrics(estimate,truth,missing),'observed_channel_max_change':float(np.max(abs(estimate[~sat]-obs[~sat]),initial=0))}
        report['regions'].append(row);np.savez_compressed(out/(name+'_disk.npz'),observed=obs,truth=truth,score_missing=missing,estimate=estimate.astype('float32'));print(name,row['hidden_core_error'],flush=True);(out/'disk-results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'disk-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
