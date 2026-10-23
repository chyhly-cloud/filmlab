"""Spectral admissibility of camera-proxy RGB; never fit a target photograph.

The ideal observer is only a proxy. Real RAW camera-profile RGB need not be
exact CIE colourimetry. Infeasibility is not evidence that actual light is
unphysical, nor permission to alter measured camera channels silently.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import linprog,minimize,Bounds,LinearConstraint
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def nonnegative_spectrum(rgb,camera_matrix,smooth=False):
    rgb=np.asarray(rgb,float);C=np.asarray(camera_matrix,float)
    scale=max(float(np.max(abs(rgb))),1e-12);target=rgb/scale
    # Minimum L-infinity RGB residual, with nonnegative spectral radiance.
    constraints=np.vstack([np.column_stack([C.T,-np.ones(3)]),np.column_stack([-C.T,-np.ones(3)])])
    lp=linprog(np.r_[np.zeros(len(C)),1.],A_ub=constraints,b_ub=np.r_[target,-target],bounds=(0,None),method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not lp.success:raise ValueError(lp.message)
    residual=float(lp.x[-1]);row={'minimum_relative_max_RGB_residual':residual,
        'LP_spectrum_RGB_closure_relative':float(np.max(abs(lp.x[:-1]@C-target)))}
    if not smooth or residual>1e-8:return row,None
    # Explicit mathematical prior: smallest discrete curvature, exact camera RGB.
    # No material/film colour is involved in this objective.
    D=np.diff(np.eye(len(C)),n=2,axis=0);Q=D.T@D
    fun=lambda x:float(x@Q@x);jac=lambda x:2*Q@x
    sol=minimize(fun,lp.x[:-1],jac=jac,method='SLSQP',bounds=Bounds(0,np.inf),
        constraints=[LinearConstraint(C.T,target,target)],options={'ftol':1e-11,'maxiter':1500})
    closure=float(np.max(abs(sol.x@C-target)))
    row.update({'smooth_success':bool(sol.success),'smooth_message':str(sol.message),'iterations':int(sol.nit),
                'smooth_RGB_closure_relative':closure,'minimum_normalized_spectral_radiance':float(sol.x.min())})
    if not sol.success or closure>1e-7 or sol.x.min()<-1e-9:return row,None
    return row,sol.x*scale


def main():
    import colour
    from filmlab import generation,lights,spectral,film
    from filmlab.materials import emulsion
    generation.use('next_sensor');root=Path('/workspace/filmlab-audit');out=root/'exposure-floor-audit';out.mkdir(exist_ok=True)
    C=lights.camera_rgb(np.eye(81));S=10**emulsion('kodak_vision3_500t').stock.log_sensitivity
    cc=colour.SDS_COLOURCHECKERS['ColorChecker N Ohta'];names=list(cc)
    R=np.array([v.copy().align(colour.SpectralShape(380,780,5)).values for v in cc.values()])
    results={'scope':__doc__,'controlled':[],'real_input_samples':[],
        'prior':'Nonnegative spectral radiance; optional minimum discrete second-difference norm subject to exact ideal-camera RGB. This is an unmeasured smoothness prior, not recovered true light.'}
    arrays={}
    for name in ['BB3200','FL11']:
        I=spectral.illuminant(name) if name=='BB3200' else lights._sd(name)
        I/=float(lights.camera_rgb(I)@np.array([.2126,.7152,.0722]));rgb=lights.camera_rgb(R*I);truth=(R*I)@S
        old=film.exposure_response(emulsion('kodak_vision3_500t').stock,I,method='linear_wp')(rgb)
        for j in np.flatnonzero((old<=1e-9).any(-1)):
            checks,L=nonnegative_spectrum(rgb[j],C,smooth=True)
            row={'light':name,'patch':names[j],'input_RGB':rgb[j].tolist(),'old_unclipped_H':old[j].tolist(),
                'conditional_true_H':truth[j].tolist(),'checks':checks,'candidate_H':None if L is None else (L@S).tolist()}
            if L is not None:
                arrays[name+'_'+str(j)]=L
                row['candidate_layer_EV_error']=np.log2((L@S)/truth[j]).tolist()
            # Existing spectral lookup control, to check whether positivity alone helps.
            lookup=lights.response_under(I,np.eye(81))(rgb[j:j+1])[0]
            row['existing_spectral_lookup_relative_RGB_error']=float(np.max(abs(lights.camera_rgb(lookup)-rgb[j]))/np.max(abs(rgb[j])))
            results['controlled'].append(row)
    for sid in ['cafe','R1-07-8','R1-06-7']:
        a=np.load(out/(sid+'.npz'));rgb=a['input_linear_RGB'];weights=a['floor_branch_weights'];mask=a['region_mask']
        valid=mask&np.isfinite(rgb).all(-1)&(np.max(weights,axis=-1)>.5)
        xy=np.argwhere(valid);count=len(xy)
        # Uniform deterministic index spacing in the input-defined set; no target RGB/error selection.
        indices=np.linspace(0,count-1,min(count,16),dtype=int) if count else []
        rows=[]
        for ix in indices:
            y,x=xy[ix];row,_=nonnegative_spectrum(rgb[y,x],C,smooth=False)
            row.update({'yx':[int(y),int(x)],'input_RGB':rgb[y,x].tolist()});rows.append(row)
        results['real_input_samples'].append({'scene':sid,'input_selected_pixels':count,'sample_count':len(rows),'samples':rows,
            'minimum_cone_residual_p50_p95_max':np.percentile([r['minimum_relative_max_RGB_residual'] for r in rows],[50,95,100]).tolist() if rows else None})
    np.savez_compressed(out/'controlled_radiance_candidates.npz',**arrays)
    (out/'spectral_admissibility.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
