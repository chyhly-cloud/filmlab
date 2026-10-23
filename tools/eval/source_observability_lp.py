"""Actual-scene source ambiguity at fixed model return-R/G responses.

Two catalogue LP extrema, never a fitted material or an observational confidence
interval. Every spectrum drives all six responses together and obeys the same
qualified camera-proxy censoring lower bounds. Full nonlinear images are checked.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linprog

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def extreme_Y(responses, selected_indices, target, camera, required, maximize=False):
    S=np.asarray(responses,float)[:,selected_indices];target=np.asarray(target,float)
    scale=np.maximum(abs(target),1e-12)
    return linprog((-1 if maximize else 1)*np.ones(len(S)),
                   A_eq=S.T/scale[:,None],b_eq=target/scale,
                   A_ub=-np.asarray(camera).T,b_ub=-np.asarray(required),
                   bounds=(0,None),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})


def run():
    from catalogue_source_inverse import MomentCase
    from filmlab.spectral import srgb_encode
    from PIL import Image,ImageDraw,ImageFont
    root=Path('/workspace/filmlab-audit');fitroot=root/'catalogue-source-inverse/ceiling-geometry'
    out=root/'source-observability-lp';out.mkdir(exist_ok=True)
    case=MomentCase('ceiling_lamp',ceiling_geometry=True);a,e=case.a,case.e
    fit=json.loads((fitroot/'ceiling_lamp_catalogue.json').read_text())
    library=np.load(fitroot/'ceiling_lamp_scene_specific_library.npz')
    S,C=library['response6'],library['camera_response']
    m=np.array(fit['selected_by_train_only']['moments'])
    required=np.max(a['source_camera']/a['source_profile'][a['support'],None],axis=0)
    report={'scope':__doc__,'geometry':case.geometry,'held_fixed_response_indices':[3,4],
            'held_fixed_response_values':m[3:5].tolist(),'baseline_train_MSE':fit['selected_by_train_only']['train_equal_zone_MSE'],
            'endpoints':[],'limits':['Returned R/G are frozen MODEL moments, not independently measured optical exposure.',
                'The Y interval only covers the finite112 library and two fixed responses; not a confidence interval or bound on actual lamp radiance.',
                'Unknown source shape, sensor SSF, material and scanner calibration are outside this LP.',
                'Similar final images do not certify the true source; nor does source ambiguity itself explain the remaining orange-halo error.']}
    safe=np.load(root/'inverse-source-radiance/ceiling_lamp_source.npz')['safe']
    states=[]
    for maximize in [False,True]:
        sol=extreme_Y(S,[3,4],m[3:5],C,required,maximize)
        if not sol.success:raise ValueError(sol.message)
        c=np.maximum(sol.x,0.);moments=c@S
        closure=float(np.max(abs(moments[3:5]-m[3:5])/m[3:5]))
        if closure>1e-8:raise ValueError('LP roundoff violates fixed responses')
        prediction=case.render(moments);image,q=case.render(moments,full=True)
        H=a['background']+np.einsum('m,mhwc->hwc',moments,case.B)
        st=e._develop(q,blur_px=12/e.px_um);st.dye*=a['factor'][...,None]
        trans=e._transmittance(st);density=-np.log10(np.maximum(trans,1e-300))
        sd=e.scanner.normalized_density(trans);rgb=e.scanner.to_rgb(sd)
        timing=e.scanner.timing
        balanced=sd if timing.M is None else timing.M[0]+timing.M[1]*sd
        xe_pre=timing.K-balanced*timing.s
        xe_post=timing.paper_exposure(sd)
        dp_pre=e.scanner.digital_paper.density(xe_pre)
        dp_post=e.scanner.digital_paper.density(xe_post)
        paper_only=e.scanner._reference_print(dp_pre,e.scanner.digital_paper.k)
        row={'endpoint':'maximum_Y' if maximize else 'minimum_Y','source_input_Y':float(c.sum()),
             'coefficients':c.tolist(),'six_response_moments':moments.tolist(),'fixed_RG_relative_closure':closure,
             'camera_proxy_max_deficit':float(np.maximum(required-c@C,0).max()),
             'metrics':case.stats(prediction),
             'train_component_MSE':float(np.mean([case.stats(prediction)[k]['Lab_RMSE']**2 for k in ['inner_train','outer_train']])),
             'nonzero_members':[{'name':str(n),'coefficient':float(v)} for n,v in zip(library['names'],c) if v>1e-7]}
        report['endpoints'].append(row)
        states.append({'H':H,'q':q,'dye':st.dye,'spectral_density81':density,'scan_density':sd,'native_linear_RGB':rgb,
                       'image':image,'sample_Lab':prediction,'paper_exposure_before_extra_shoulder':xe_pre,
                       'paper_exposure_after_extra_shoulder':xe_post,'paper_density_default':dp_post,
                       'paper_density_paper_only':dp_pre,'paper_only_linear_RGB':paper_only})
        np.savez_compressed(out/(row['endpoint']+'.npz'),**states[-1],source_spectrum=c@library['spectra'])
    comparisons={}
    core=a['support']
    for key in ['q','dye','spectral_density81','scan_density','paper_exposure_before_extra_shoulder',
                'paper_exposure_after_extra_shoulder','paper_density_default','paper_density_paper_only',
                'native_linear_RGB','paper_only_linear_RGB']:
        difference=abs(states[1][key]-states[0][key])
        comparisons[key]={'core_absolute_difference_p50_p95_max':np.percentile(difference[core],[50,95,100]).tolist(),
                          'safe_native_absolute_difference_max':float(difference[safe].max())}
    delta_ev=np.log2(states[1]['H'][core]/states[0]['H'][core])
    comparisons['core_layer_exposure_EV_maxY_over_minY']={
        'per_layer_median':np.median(delta_ev,axis=0).tolist(),'per_layer_min':delta_ev.min(0).tolist(),'per_layer_max':delta_ev.max(0).tolist()}
    comparisons['sample_Lab_absolute_difference_max']=float(abs(states[1]['sample_Lab']-states[0]['sample_Lab']).max())
    encoded=[np.uint8(np.rint(srgb_encode(x['native_linear_RGB'])*255)) for x in states]
    comparisons['safe_native_sRGB8']={'different_code_pixel_fraction':float(np.any(encoded[0]!=encoded[1],axis=-1)[safe].mean()),
                               'maximum_absolute_code_difference':int(abs(encoded[0].astype(int)-encoded[1].astype(int))[safe].max())}
    report['source_input_Y_ratio']=report['endpoints'][1]['source_input_Y']/report['endpoints'][0]['source_input_Y']
    report['stage_comparison']=comparisons
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(1024,560),(18,18,18));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    for i,state in enumerate(states):
        picture=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(state['image']))*255)))
        sheet.paste(picture,(512*i,48))
        draw.text((512*i+8,12),f"LP源Y极值 {report['endpoints'][i]['source_input_Y']:.2f} · 非实测",font=font,fill='white')
    sheet.save(out/'conditional_input_ambiguity.jpg',quality=98)
    print(json.dumps({'sourceY_ratio':report['source_input_Y_ratio'],'stages':comparisons},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':run()
