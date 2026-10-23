"""Audit the units and finite-change limits of the existing q0 dye decomposition.

Reads existing derivatives and the 297 finite perturbations; creates no exposure
probe or fitted constant. Finite corner/path examples are mathematical attribution
controls, not real film states, optical measurements or proposed film behaviour.
"""
from pathlib import Path
from itertools import permutations
import hashlib,json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
LOG10_2=float(np.log10(2.))


def ev_to_log10_exposure(ev):return np.asarray(ev)*LOG10_2


def corner_path_contributions(values):
    """Every coordinate-order decomposition of a scalar 3-variable function."""
    values=np.asarray(values,float)
    if values.shape!=(8,):raise ValueError('need eight scalar corner values')
    results=[]
    for order in permutations(range(3)):
        contributions=np.zeros(3);mask=0
        for i in order:
            next_mask=mask|(1<<i);contributions[i]=values[next_mask]-values[mask];mask=next_mask
        results.append({'order':list(order),'contributions':contributions.tolist(),'total':float(contributions.sum())})
    return results


def shapley_corner_allocation(values):
    """Exact permutation average for the specified eight-corner static game."""
    paths=corner_path_contributions(values)
    return np.mean([p['contributions'] for p in paths],axis=0)


def corner_marginal_ranges(values):
    """A finite, non-differentiability-safe attribution bound, not causality."""
    values=np.asarray(values,float);ranges=[]
    for i in range(3):
        marginal=[values[s|(1<<i)]-values[s] for s in range(8) if not s&(1<<i)]
        ranges.append({'coordinate':i,'marginals':marginal,'minimum':float(min(marginal)),'maximum':float(max(marginal))})
    return ranges


def nonlinear_counterexample(start,end):
    """F(C,M,Y)=C*M+Y, monotone in every coordinate for C,M>0."""
    start=np.asarray(start,float);end=np.asarray(end,float)
    points=np.array([[end[i] if mask&(1<<i) else start[i] for i in range(3)] for mask in range(8)])
    values=points[:,0]*points[:,1]+points[:,2]
    return {'function':'F(C,M,Y)=C*M+Y; C,M positive on the box; all first partials positive',
            'start':start.tolist(),'end':end.tolist(),'corners':points.tolist(),'corner_values':values.tolist(),
            'endpoint_change':float(values[-1]-values[0]),'paths':corner_path_contributions(values),
            'shapley':shapley_corner_allocation(values).tolist(),'marginal_ranges':corner_marginal_ranges(values),
            'mixed_second_derivatives':{'CM':1,'CY':0,'MY':0},
            'pair_swap_bound_CM':float(abs((end[0]-start[0])*(end[1]-start[1])))}


def main():
    from brightness_response_audit import lab
    root=Path('/workspace/filmlab-audit');probe_path=root/'nonneutral-review/chain/stages.npz';matrix_path=root/'transfer-matrices/matrices.npz'
    a=np.load(probe_path);m=np.load(matrix_path);meta=json.loads((root/'transfer-matrices/results.json').read_text())
    matches=np.flatnonzero(np.all(a['base_q']==0,axis=-1))
    if len(matches)!=1:raise ValueError('expected one q0')
    i=int(matches[0]);jc=m['eps_1e-05_dye'][i];red_dye=jc[:,0]
    report={'scope':__doc__,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [probe_path,matrix_path]},
            'state':{'index':i,'q':a['base_q'][i].tolist(),'generation':'next','roll':None,'grain':0,'engine_timing_mode':'standard',
                'effective_timing':'fixed saved R1-02-3 PrintTiming after engine construction','timing':meta['source_timing'],'output_conditioning':None,
                'logH_derivative_step':1e-5,'logH_definition':'log10 of normalized effective layer exposure; not log2 and not absolute irradiance'},
            'q_change_per_EV':LOG10_2,'dye_CMY_per_red_log10H':red_dye.tolist(),'outputs':{},
            'finite_attribution_counterexamples':[
                nonlinear_counterexample([1,1,1],[2,2,2]),
                nonlinear_counterexample([1,2,1],[2,1,.5])],
            'limits':['q0 is a homogeneous synthetic layer-exposure state, not measured corridor radiance',
                      'per-EV means perturbing only effective red-layer exposure while holding green/blue logH fixed',
                      'no threshold, physical light-loss fraction or finite real-scan colour difference is implied by a local derivative',
                      'finite allocations depend on order and static-game definition; total endpoint difference is fixed',
                      'Shapley is uniquely defined for specified corners/weights, not uniquely identified physical chemical causation']}
    for out in ['exponential','paper_only']:
        scan=m[out+'_Lab_per_dye'][i];terms=scan*red_dye[None,:]
        saved=m[out+'_red_Lab_terms_CMY'][i];total=terms.sum(axis=-1);direct=m['eps_1e-05_'+out+'_Lab'][i,:,0]
        if np.max(abs(terms-saved))>1e-10:raise ValueError('saved terms fail independent product closure')
        baseline_rgb=a['base_'+out+'_linear_rgb'][i];baseline_lab=lab(baseline_rgb)
        rows=[]
        for ev in [.1,.5,1.]:
            endpoint_lab=lab(a[f'red_{ev}_'+out+'_linear_rgb'][i]);actual=endpoint_lab-baseline_lab
            first_order=total*ev_to_log10_exposure(ev);residual=actual-first_order
            rows.append({'red_EV':ev,'delta_red_log10H':float(ev_to_log10_exposure(ev)),
                'first_order_Lab':first_order.tolist(),'actual_cached_finite_Lab':actual.tolist(),
                'actual_minus_first_order_Lab':residual.tolist(),
                'actual_over_first_order_b':float(actual[2]/first_order[2]) if abs(first_order[2])>1e-12 else None,
                'source_or_endpoint_display_clipping':bool((a['base_'+out+'_clipped'][i]|a[f'red_{ev}_'+out+'_clipped'][i]).any())})
        report['outputs'][out]={'baseline_RGB':baseline_rgb.tolist(),'baseline_Lab':baseline_lab.tolist(),
            'scan_Lab_per_dye_CMY':scan.tolist(),'terms_Lab_per_red_log10H':terms.tolist(),
            'total_Lab_per_red_log10H':total.tolist(),'direct_full_chain_Lab_per_red_log10H':direct.tolist(),
            'local_chain_sum_max_abs_error':float(np.max(abs(total-direct))),
            'terms_Lab_per_red_EV':(terms*LOG10_2).tolist(),'total_Lab_per_red_EV':(total*LOG10_2).tolist(),
            'half_EV_first_order_terms_Lab':(terms*LOG10_2*.5).tolist(),
            'half_EV_first_order_total_Lab':(total*LOG10_2*.5).tolist(),
            'finite_existing_checks':rows}
    outdir=root/'corridor-dye-response';outdir.mkdir(exist_ok=True,parents=True)
    (outdir/'units.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for name,values in report['outputs'].items():
        print(name,'b perlog10H terms:',values['terms_Lab_per_red_log10H'][2],'total',values['total_Lab_per_red_log10H'][2])
        print('b perEV terms:',values['terms_Lab_per_red_EV'][2],'total',values['total_Lab_per_red_EV'][2])
        for row in values['finite_existing_checks']:print(row)


if __name__=='__main__':main()
