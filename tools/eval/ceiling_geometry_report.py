"""Geometry-controlled halo assessment; aggregates fixed results, no selection."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def main():
    from brightness_response_audit import lab
    from local_background_contrast import rectangle
    from source_profile_holdout import shape_metrics
    from filmlab.spectral import srgb_encode
    from PIL import Image,ImageDraw,ImageFont
    root=Path('/workspace/filmlab-audit');out=root/'ceiling-geometry-replay'
    geometry=json.loads((root/'ceiling-line-registration/results.json').read_text())
    replay=json.loads((out/'results.json').read_text())
    name='ceiling_lamp'
    historic=np.load(root/'halo-residual-direction'/(name+'.npz'))
    config=json.loads((REPO/'tools/eval/local_background_regions.json').read_text())[name]
    target=rectangle(historic['distance_um'].shape,config['target'])&historic['measured']&(historic['distance_um']>0)&(historic['distance_um']<=300)
    inner=target&(historic['distance_um']<=100);outer=target&(historic['distance_um']>200)
    sigma=25/json.loads((root/'brightness-local/R1-02-3.json').read_text())['scan_um_per_pixel']
    images={'real':historic['real'],'historic_halo':historic['halo']}
    for key in ['legacy_full','corrected_inputs_old_core','fixed112','field_original_sectors']:
        a=np.load(out/(name+'_'+key+'.npz'))
        images[key+'_original_geometry']=a['old_image']
        images[key+'_ceiling_geometry']=a['corrected_image']
    refit=root/'censored-source-field/ceiling-geometry'
    for label,path in [('new_geometry_field_forward',refit),('new_geometry_field_reverse',refit/'reverse-sectors')]:
        images[label]=np.load(path/(name+'.npz'))['image']
    contrasts={}
    for key,value in images.items():
        l=lab(gaussian_filter(np.nan_to_num(value),(sigma,sigma,0)))
        contrasts[key]={'outer_label_minus_inner_label_Lab':(np.median(l[outer],0)-np.median(l[inner],0)).tolist()}
    reproduction=float(np.max(abs(np.array(contrasts['historic_halo']['outer_label_minus_inner_label_Lab'])-
                                   contrasts['legacy_full_original_geometry']['outer_label_minus_inner_label_Lab'])))
    if reproduction>1e-4:raise ValueError('Historical local-contrast version does not replay inside measured ceiling regions')
    comparisons={}
    for label,path in [('original_geometry',root/'censored-source-field'),('accepted_ceiling_geometry',refit)]:
        a=np.load(path/(name+'.npz'));b=np.load(path/'reverse-sectors'/(name+'.npz'))
        x=json.loads((path/(name+'.json')).read_text());y=json.loads((path/'reverse-sectors'/(name+'.json')).read_text())
        comparisons[label]={'source_field_comparison':shape_metrics(a['field'],b['field'],a['support']),
                            'forward':x,'reverse':y}
    snapshot=json.loads((root/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError('Unexpected production changes: '+str(changed))
    result={'date_utc':'2026-10-21','scope':__doc__,'goal_complete':False,
            'geometry_measurements':geometry,'fixed_negative_replay':replay,
            'fixed_ceiling_region_contrasts':{'inner_label_pixels':int(inner.sum()),'outer_label_pixels':int(outer.sum()),
                'meaning':'Same previously selected scan-coordinate ceiling regions. Legacy radial labels are retained for controlled comparison; no new film-mm or clipping-status certification.',
                'historical_Lab_contrast_reproduction_max':reproduction,'states':contrasts},
            'source_field_role_reversal':comparisons,
            'historical_constraint_version_note':'The negative outward a* trend belongs to the historical legacy-input state. Source-inverse research candidates already have a positive but insufficient trend. None is adopted as current/next.',
            'production':{'verified_files':len(snapshot),'changed_files':changed},
            'new_targeted_tests_passed':7,'related_targeted_tests_passed':25,
            'decision':'Retain independently validated near-ceiling-plane affine for qualified evaluation. Middle geometry and fixture parallax are unverified. No source/material/scan candidate passes orange-halo acceptance.',
            'limits':['Withheld corner landmarks were not used for line fitting, but structure-filter pixel supports are not statistically independent.',
                      'Some line matches are outliers; soft-L1 fitting plus withheld corners validates the affine, not every line measurement.',
                      'A@H is scene correspondence, not a film-plane mm calibration. No optical kernel scale changed.',
                      'Fixed targets/group names and legacy radial labels are not automatically re-certified RAW-unsaturated or physical annuli after remapping.',
                      'Three-step QPs can reach iteration limits; only feasible full nonlinear decreases are accepted, not certified optima.',
                      'No independent unsaturated fixture landmark set was acquired; all lamp parallax is not excluded.']}
    for path in [out/'round_results.json',REPO/'tools/eval/ceiling_geometry_round_results.json']:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(1536,560),(18,18,18));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
    for col,(label,value) in enumerate([
        ('真实800T扫描',images['real']),
        ('格线校验几何 · 旧输入基准',images['legacy_full_ceiling_geometry']),
        ('格线校验几何 · 源反演候选',images['fixed112_ceiling_geometry'])]):
        image=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(value))*255)))
        sheet.paste(image,(512*col,48));draw.text((512*col+8,12),label,font=font,fill='white')
    sheet.save(out/'real_legacy_candidate_geometry_control.jpg',quality=98)
    print(json.dumps({'historical_contrast_reproduction_max':reproduction,
                       'field_TV_before_after':[comparisons[k]['source_field_comparison']['unit_sum_total_variation'] for k in comparisons],
                       'decision':result['decision']},ensure_ascii=False))


if __name__=='__main__':main()
