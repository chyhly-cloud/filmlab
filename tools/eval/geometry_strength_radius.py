"""Finite near/outer amplitude-consistency diagnostic after accepted ceiling geometry.

The spectrum, source profile, corrected background, film and scan stay fixed.
Each zone reuses existing target RGB for its own scalar fit. This is not a new
outer holdout, material fit, measured lamp brightness or global optimum proof.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def ray_bounds(moments, coordinate_scale, camera_response, required, upper_z=4096.):
    """Hard camera-proxy floor and the unchanged six-coordinate numeric box."""
    m, scale, camera, floor = map(np.asarray,(moments,coordinate_scale,camera_response,required))
    if np.any(m<0) or not np.any(m>0) or np.any(scale<=0) or np.any(camera<=0) or np.any(floor<0):
        raise ValueError('Nonnegative ray, positive scales/camera and nonnegative censor bounds required')
    lower=float(np.max(floor/camera))
    upper=float(np.min(upper_z/(m[m>0]/scale[m>0])))
    if not 0<lower<upper:
        raise ValueError('A positive finite feasible logarithmic interval is required')
    return lower,upper


def main():
    from catalogue_source_inverse import MomentCase
    root=Path('/workspace/filmlab-audit')
    fitted_dir=root/'catalogue-source-inverse/ceiling-geometry'
    out=root/'geometry-source-refit-review';out.mkdir(exist_ok=True)
    case=MomentCase('ceiling_lamp',ceiling_geometry=True)
    a=case.a
    fitted=json.loads((fitted_dir/'ceiling_lamp_catalogue.json').read_text())
    m=np.array(fitted['selected_by_train_only']['moments'])
    c=np.maximum(fitted['selected_by_train_only']['witness_coefficients'],0.)
    data=np.load(fitted_dir/'ceiling_lamp_scene_specific_library.npz')
    S,C=data['response6'],data['camera_response']
    if np.max(abs(c@S-m)/np.maximum(abs(m),1e-10))>1e-8:
        raise ValueError('Common source witness no longer matches the selected response ray')
    camera=c@C
    required=np.max(a['source_camera']/a['source_profile'][a['support'],None],axis=0)
    lower,upper=ray_bounds(m,case.scale,camera,required)
    if not lower<=1<=upper:
        raise ValueError('Joint fitted source is outside inherited hard/numeric bounds')
    grid=np.linspace(np.log(lower),np.log(upper),32)
    cache={}
    def evaluate(log_t):
        key=float(log_t)
        if key not in cache:
            prediction=case.render(m*np.exp(key))
            losses={g:float(np.mean((prediction[a['groups']==g]-a['target_lab'][a['groups']==g])**2)) for g in [0,2]}
            cache[key]=(prediction,losses)
        return cache[key]
    grid_losses=np.array([[evaluate(z)[1][g] for g in [0,2]] for z in grid])
    rows=[]
    for col,(group,label) in enumerate([(0,'inner_training_only'),(2,'outer_training_reused_diagnostic')]):
        values=grid_losses[:,col]
        local=[i for i in range(1,len(grid)-1) if values[i]<=values[i-1] and values[i]<=values[i+1]]
        # Refine every detected interior basin and both boundary-adjacent cells;
        # do not assume a single minimum or use another zone to select a basin.
        candidates=[{'log_multiplier':float(z),'component_MSE':float(v),'kind':'declared_grid'} for z,v in zip(grid,values)]
        intervals=sorted(set(local+[0,len(grid)-1]))
        refinements=[]
        for i in intervals:
            lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
            result=minimize_scalar(lambda z:evaluate(z)[1][group],bounds=(lo,hi),method='bounded',
                                   options={'xatol':2e-6,'maxiter':100})
            row={'log_multiplier':float(result.x),'component_MSE':float(result.fun),
                 'kind':'bounded_log_refinement','success':bool(result.success),'iterations':int(result.nit),
                 'bracket_log_multiplier':[float(lo),float(hi)]}
            refinements.append(row);candidates.append(row)
        best=min(candidates,key=lambda x:x['component_MSE'])
        t=float(np.exp(best['log_multiplier']))
        prediction=evaluate(best['log_multiplier'])[0]
        deficit=float(np.maximum(required-t*camera,0).max())
        if deficit>1e-7:
            raise ValueError('Scalar diagnostic violated the inherited camera-proxy floor')
        rows.append({'fit_target':label,'fit_group':group,'multiplier':t,'source_input_Y':float(t*c.sum()),
                     'source_input_units':'Existing D65-normalized engine input Y, not SI radiance.',
                     'selected_by':'Only the designated single training-zone component MSE; other zones never choose amplitude.',
                     'fit_component_MSE':best['component_MSE'],'selected_candidate':best,'bounded_refinements':refinements,
                     'camera_proxy_max_deficit':deficit,'numerical_boundary_active':bool(t<=lower*(1+1e-4) or t>=upper*(1-1e-4)),
                     'all_original_groups':case.stats(prediction)})
    ratio=rows[0]['source_input_Y']/rows[1]['source_input_Y']
    result={'scope':__doc__,'name':case.name,'geometry':case.geometry,
            'fixed_spectrum_members':fitted['nonzero_witness_members'],
            'joint_fit_source_input_Y':float(c.sum()),'same_SPD_moment_relative_closure':float(np.max(abs(c@S-m)/np.maximum(abs(m),1e-10))),
            'camera_proxy_required':required.tolist(),'camera_proxy_response_at_multiplier1':camera.tolist(),
            'hard_lower_multiplier':lower,'numeric_upper_multiplier':upper,'numeric_upper_rule':'min(4096/(selected_moment/old_coordinate_scale)); not a measured radiance ceiling',
            'log_grid_points_per_zone':32,'grid_multiplier':np.exp(grid).tolist(),'grid_zone_component_MSE':grid_losses.tolist(),
            'fits':rows,'inner_over_outer_source_input_Y_ratio':float(ratio),
            'joint_fit_all_original_groups':case.stats(evaluate(0.)[0]),
            'unique_complete_model_forward_evaluations':case.counter,
            'limits':['Finite log grid plus local bounded refinements do not prove global optimality.',
                      'Both zones and the selected SPD have already used the same final scan; no external blind test or independent source measurement.',
                      'Old radius/group labels remain fixed image regions, not newly calibrated physical annuli.',
                      'This differs from the historical FL2/old geometry split in several inputs; a ratio change cannot be attributed solely to geometry.',
                      'A remaining ratio difference may include source, spatial, background, development or scanner model errors; no unique physical root cause follows.']}
    (out/'strength_radius.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(out/'strength_radius.npz',grid_multiplier=np.exp(grid),grid_zone_MSE=grid_losses,
                        selected_moments=m,required=required,camera_response=camera,
                        points=a['points'],groups=a['groups'],target_lab=a['target_lab'])
    print(json.dumps({'near_sourceY':rows[0]['source_input_Y'],'outer_sourceY':rows[1]['source_input_Y'],
                      'near_over_outer':ratio,'forwards':case.counter},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
