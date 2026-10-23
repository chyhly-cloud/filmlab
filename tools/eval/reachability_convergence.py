"""Necessary local convergence checks on two saved actual-input patches.

Repeat solver accuracy/parameterization, not a new material/strength search.
Boundary KKT checks establish local stationarity only, never global exclusion.
"""
from pathlib import Path
import sys,json,zlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab


def finite_jacobian(function,x,epsilon):
    columns=[]
    for c in range(len(x)):
        plus=x.copy();minus=x.copy();plus[c]+=epsilon;minus[c]=max(0,minus[c]-epsilon)
        columns.append((function(plus)-function(minus))/(plus[c]-minus[c]))
    return np.stack(columns,axis=-1)


def main():
    from scipy.optimize import least_squares,minimize,brentq,minimize_scalar
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    root=Path('/workspace/filmlab-audit/halo-reachability');old=json.loads((root/'results.json').read_text())
    generation.use('next_sensor');e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':15,'grain':0,'output_conditioning':None}))
    td=old['timing'];e.scanner.timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white']);e.px_um=old['native_um_per_pixel']
    report={'scope':__doc__,'checks':[],'limits':['no global lower bound or interval certificate','only two representative points, not all pixels or spatially arbitrary returned light',
        'nonnegative effective exposure is necessary but not sufficient for physical lamp radiance/SPD',
        'diagnostic |deltaL|<=1 and ab<=2 tolerances are not observational confidence limits']}
    for name,zone in [('ceiling_lamp','real_orange_transition_0_300'),('middle_ceiling','transition_100_300')]:
        meta=next(p for g in old['regions'] if g['name']==name for p in g['patches'] if p['zone']==zone)
        a=np.load(root/(name+'_'+zone+'.npz'));q=a['base_q'];control=a['control'];w=a['weights'];fac=a['agitation'];target=a['target_Lab'];href=np.array(meta['effective_H_reference'])
        def f(u):
            addition=href*np.expm1(np.log(2)*u)
            changed=np.log10(10**q+control[...,None]*addition)
            st=e._develop(changed,blur_px=12/e.px_um);st.dye*=fac[...,None]
            rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(st)))
            return lab(np.sum(rgb*w[...,None],axis=(0,1)))
        x=np.array(meta['u_addition_parameter']);repro=float(np.max(abs(f(x)-meta['witness_Lab'])))
        if repro>1e-8:raise ValueError('saved search endpoint not reproduced')
        bound=np.log2(1+10**(e.e.log_exposure.max()+3)/href)
        repeat=least_squares(lambda u:f(u)-target,x,bounds=(np.zeros(3),bound),jac='3-point',
            max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-11)
        # Independent coordinates: linear added-H / Href instead of log2(1+H/Href).
        def linear_f(z):return f(np.log2(1+z))
        linear=least_squares(lambda z:linear_f(z)-target,np.expm1(np.log(2)*x),bounds=(np.zeros(3),2**bound-1),
            jac='3-point',max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-11)
        x=repeat.x;j=finite_jacobian(f,x,1e-4);j2=finite_jacobian(f,x,2e-5);res=f(x)-target;gradient=j.T@res
        active=x<1e-6;projected=gradient.copy();projected[active]=np.minimum(projected[active],0)
        constraints={'type':'ineq','fun':lambda u:1-abs(f(u)[0]-target[0])}
        constrained=[]
        for eps in [1e-4,2e-5]:
            start=np.array(min(meta['fixed_L_constraint_trials'],key=lambda p:p['ab_distance'])['u'])
            sol=minimize(lambda u:np.sum((f(u)[1:]-target[1:])**2),start,method='SLSQP',
                bounds=list(zip(np.zeros(3),bound)),constraints=constraints,
                options={'maxiter':150,'ftol':1e-10,'eps':eps})
            constrained.append({'epsilon':eps,'success':bool(sol.success),'message':sol.message,'u':sol.x.tolist(),
                'error_Lab':(f(sol.x)-target).tolist(),'ab_distance':float(np.linalg.norm(f(sol.x)[1:]-target[1:])),
                'L_constraint_violation':float(max(0,abs(f(sol.x)[0]-target[0])-1))})
        # Independent check on the observed B=0 active face: solve the L=target+1
        # boundary exactly by a scalar bracket, then minimise colour along it.
        # This tests that branch; it is NOT a certificate for the whole cube.
        Lcap=target[0]+1
        # R alone can plateau below Lcap, so a root need not exist. Keep this
        # local check around the observed solution rather than claiming a scan
        # of saturated, flat branches of the whole domain.
        rlimit=min(bound[0],x[0]+.5)
        if f(np.array([rlimit,0.,0.]))[0]>Lcap:
            rmax=brentq(lambda r:f(np.array([r,0.,0.]))[0]-Lcap,0,rlimit,xtol=1e-12)
        else:rmax=rlimit
        def contour(r):
            if abs(f(np.array([r,0.,0.]))[0]-Lcap)<1e-10:g=0.
            else:g=brentq(lambda g:f(np.array([r,g,0.]))[0]-Lcap,0,bound[1],xtol=1e-12)
            return np.array([r,g,0.])
        scalar=minimize_scalar(lambda r:np.sum((f(contour(r))[1:]-target[1:])**2),bounds=(0,rmax),method='bounded',options={'xatol':1e-10})
        cont_u=contour(scalar.x);cont_value=f(cont_u)
        report['checks'].append({'scene':name,'zone':zone,'saved_endpoint_reproduction_max':repro,
            'original_error_Lab':meta['witness_error_Lab'],'tight_repeat_error_Lab':res.tolist(),
            'tight_repeat_status':{'success':bool(repeat.success),'message':repeat.message,'optimality':repeat.optimality,'nfev':repeat.nfev},
            'linear_coordinate_error_Lab':(linear_f(linear.x)-target).tolist(),
            'linear_coordinate_status':{'success':bool(linear.success),'message':linear.message,'optimality':linear.optimality,'nfev':linear.nfev},
            'coordinate_output_disagreement_Lab_norm':float(np.linalg.norm(f(x)-linear_f(linear.x))),
            'finite_J_step_disagreement_relative':float(np.linalg.norm(j-j2)/np.linalg.norm(j2)),
            'J_singular_values':np.linalg.svd(j2,compute_uv=False).tolist(),
            'half_squared_error_gradient':gradient.tolist(),'lower_bound_active':active.tolist(),
            'projected_gradient_max_abs':float(np.max(abs(projected))),'fixed_L_repeats':constrained,
            'B_zero_exact_L_contour':{'success':bool(scalar.success),'u':cont_u.tolist(),
                'error_Lab':(cont_value-target).tolist(),'ab_distance':float(np.linalg.norm(cont_value[1:]-target[1:])),
                'scope':'only observed B=0, L=target+1 branch, not global reachability'}})
    (root/'convergence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
