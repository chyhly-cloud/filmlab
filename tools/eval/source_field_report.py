"""Aggregate the predeclared field/sector experiments without fitting/selecting."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]
from source_profile_holdout import shape_metrics


def main():
    from PIL import Image, ImageDraw, ImageFont
    from filmlab.spectral import srgb_encode
    audit = Path('/workspace/filmlab-audit')
    root = audit/'censored-source-field'
    rows = []
    sheet = Image.new('RGB',(2048,1160),(18,18,18)); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for row,name in enumerate(['ceiling_lamp','middle_ceiling']):
        forward = json.loads((root/(name+'.json')).read_text())
        reverse = json.loads((root/'reverse-sectors'/(name+'.json')).read_text())
        assert forward['training_groups']==[0,2]
        assert reverse['training_groups']==[1,3]
        a = np.load(root/(name+'.npz')); b = np.load(root/'reverse-sectors'/(name+'.npz'))
        for key in ['points','groups','target_lab','support','initial_field']:
            np.testing.assert_array_equal(a[key],b[key])
        shape = shape_metrics(a['field'],b['field'],a['support'])
        rows.append({'name':name,'same_initialization_and_data_verified':True,
                     'source_field_role_reversal_comparison':shape,
                     'forward':forward,'reverse':reverse,
                     'reverse_prediction_on_original_train_component_MSE':float(np.mean([
                         reverse['fitted_field'][key]['Lab_RMSE']**2 for key in ['inner_train','outer_train']]))})
        real=np.load(audit/'inverse-source-radiance'/(name+'_source.npz'))['real']
        baseline=np.load(audit/'catalogue-source-inverse'/(name+'_catalogue.npz'))['image']
        for col,(label,values) in enumerate([
            ('真实800T扫描',real),('上一轮 · 固定空间轮廓',baseline),
            ('逐单元场 · 原象限训练',a['image']),('逐单元场 · 对调象限训练',b['image'])]):
            image=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(values))*255)))
            sheet.paste(image,(col*512,row*580+60))
            draw.text((col*512+8,row*580+15),label,font=font,fill='white')
    sheet.save(root/'both_directions_comparison.jpg',quality=98)
    controls=json.loads((audit/'source-profile-holdout/results.json').read_text())
    snapshot=json.loads((audit/'lamp-edge-spectral-candidate/production_snapshot.json').read_text())['production_files_sha256']
    changed=[p for p,h in snapshot.items() if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h]
    if changed:
        raise ValueError('Unexpected production changes: '+str(changed))
    result={'date_utc':'2026-10-21','scope':__doc__,'goal_complete':False,
            'production':{'checked_files':len(snapshot),'changed_files':changed},
            'reports':['docs/CENSORED_SOURCE_FIELD.md','docs/CENSORED_SOURCE_FIELD_REVIEW.md','docs/SOURCE_PROFILE_HOLDOUT.md'],
            'new_targeted_tests_passed':7,'regions':rows,
            'independent_profile_controls':[{
                'name':x['name'],'coverage':x['coverage'],
                'Y_proxy_shape':x['metrics']['linear_RGB_Y_proxy'],
                'strict_four_CFA_seed0_control':x['strict_four_CFA_seed0_control'],
                'seed_quantiles':x['legacy_Y_shape_algorithm_seed_distribution']['quantiles']}
                for x in controls['regions']],
            'decision':'Retain independently established aperture-profile failure and diagnostic field mechanism; no field/spatial/material/scanner candidate promoted.',
            'interpretation':'Free fields expose previously untested shape leverage but do not recover the orange transition and are highly unstable under a sector role reversal. Shape concentration alone is not proof of physical impossibility.',
            'limits':['Known synthetic output recovery does not certify recovered source shape.',
                      'Opposite-sector fitting reuses targets and a previously selected SPD; not external cross-validation.',
                      'Independent short-exposure shape controls are apertures, not the unresolved corridor lamps.'],
            'comparison_image':str(root/'both_directions_comparison.jpg')}
    for path in [root/'round_results.json',REPO/'tools/eval/source_field_round_results.json']:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'verified_production_files':len(snapshot),
                      'field_TV_between_sector_roles':[x['source_field_role_reversal_comparison']['unit_sum_total_variation'] for x in rows],
                      'decision':result['decision']},ensure_ascii=False))


if __name__=='__main__':
    main()
