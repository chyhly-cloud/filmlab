"""Export per-region measurement evidence and unresolved coverage for handoff.

Reads existing results; no model execution, acceptance tuning or parameter fit.
Evidence levels are descriptive categories, not statistical confidence scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def build(directory, bracket_file):
    measurements=json.loads((directory/'lamp-measurements.json').read_text())
    chains=json.loads((directory/'chain-results.json').read_text())
    brackets=json.loads(bracket_file.read_text())
    acquisitions={row['file']:row for scene in brackets['scenes'] for row in [scene['primary']]+scene['alternates']}
    lamps={r['region']['scene']+'_'+r['region']['name']:r for r in measurements['lamps']}
    records=[]
    for scene in chains:
        for e in scene['replacement_evidence']:
            name=e['name'];lamp=lamps[name]
            with np.load(directory/(name+'_measurements.npz')) as data:
                total=int(data['mask'].sum())
            missing=e['primary_any_saturated'];all_missing=e['primary_all_saturated'];accepted=e['replaced_pixels']
            accepted_all=e['replaced_all_saturated_pixels'];cross=sum(r['cross_alternate_pixels'] for r in e['chosen'])
            if not 0<=accepted_all<=accepted<=missing<=total or not 0<=accepted_all<=all_missing<=missing:
                raise ValueError('Inconsistent coverage counts: '+name)
            if not 0<=cross<=accepted:raise ValueError('Inconsistent evidence counts: '+name)
            level=('no_missing_camera_samples' if missing==0 else 'not_recovered' if accepted==0 else
                   'cross_exposure_rgb' if cross==accepted else 'partial_channel_collar' if cross==0 else 'mixed')
            files=[lamp['primary_file']]+[r['file'] for r in lamp['alternatives']]
            records.append({'name':name,'region':lamp['region'],'evidence_level':level,
                'evidence_counts':{'independent_alternate_agreement':cross,'primary_channel_and_collar_agreement':accepted-cross},
                'camera_coverage':{'roi_units':total,'primary_unsaturated_units':total-missing,
                    'primary_any_saturated':missing,'primary_all_saturated':all_missing,
                    'new_measured_units':accepted,'new_measured_originally_all_saturated':accepted_all,
                    'unresolved_any_saturated':missing-accepted,'unresolved_all_saturated':all_missing-accepted_all,
                    'new_measurement_fraction_of_roi':accepted/total,
                    'recovery_fraction_of_missing':accepted/missing if missing else None,
                    'recovery_fraction_of_all_saturated':accepted_all/all_missing if all_missing else None},
                'spectrum_status':'not_measured','radiometry':'relative camera RGB; no absolute spectral radiance calibration',
                'unresolved_policy':'retain baseline estimates and label unresolved; estimates are not measurements',
                'halation_calibration_eligible':False,
                'calibration_blockers':['actual source spectrum unavailable','complete incident source and lens-glare separation unavailable'],
                'selected_measurements':e['chosen'],
                'source_files':[{'file':f,'sha256':acquisitions[f]['sha256'],'time_s':acquisitions[f]['time_s'],
                                 'f_number':acquisitions[f]['f_number'],'iso':acquisitions[f]['iso'],
                                 'timestamp':acquisitions[f]['timestamp']} for f in files],
                'registration_and_saturation_checks':lamp['alternatives'],'temporal_comparisons':lamp['comparisons']})
    if len(records)!=len(lamps) or len({r['name'] for r in records})!=len(records):
        raise ValueError('Regions missing or duplicated')
    return {'record_date':'2026-10-20','scope':'Fixed per-region acceptance snapshot; no generalization from apertures to lamps',
            'units':'native half-size Bayer cells; not individual full-resolution sensor pixels',
            'evidence_levels':{'cross_exposure_rgb':'All accepted pixels agree between independent alternate RAW exposures; not full spectra',
                'partial_channel_collar':'Weaker evidence from a single alternate plus primary measured channels and source collar',
                'not_recovered':'No new samples accepted; baseline estimates remain unverified',
                'no_missing_camera_samples':'No missing primary camera channels in this ROI; source spectrum still unknown'},
            'study_status':'freeze halation strength, scanner curves and empirical-colour tuning pending independent source/material measurements',
            'controlled_spectral_tests':'Known-spectrum background invariance is a separate controlled test, not real-scene spectral measurement',
            'summary':{'regions':len(records),'new_measured_units':sum(r['camera_coverage']['new_measured_units'] for r in records),
                'new_measured_originally_all_saturated':sum(r['camera_coverage']['new_measured_originally_all_saturated'] for r in records),
                'real_regions_with_measured_spectrum':0},'regions':records}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True);ap.add_argument('--brackets',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    record=build(args.results,args.brackets)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record['summary']))


if __name__=='__main__':main()
