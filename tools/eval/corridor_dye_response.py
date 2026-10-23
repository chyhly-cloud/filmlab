"""Actual cached corridor endpoints: finite dye attribution and local validity.

No RAW decode, new synthetic grid, chemistry change or output fit. Eight dye
coalitions are diagnostic combinations, not physically proposed negatives.
Attributions average all six orders and are not unique chemical causal effects.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,zlib,itertools,math
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab,LUMA


def shapley_three(values):
    """Finite output differences assigned to three inputs; values[0..7] are corners."""
    values=np.asarray(values);result=[];order_values=[[] for _ in range(3)]
    for i in range(3):
        term=np.zeros_like(values[0])
        for mask in range(8):
            if mask&(1<<i):continue
            size=mask.bit_count();weight=math.factorial(size)*math.factorial(2-size)/6
            term+=weight*(values[mask|(1<<i)]-values[mask])
        result.append(term)
    for order in itertools.permutations(range(3)):
        mask=0
        for i in order:
            next_mask=mask|(1<<i);order_values[i].append(values[next_mask]-values[mask]);mask=next_mask
    return np.stack(result,axis=-1),np.stack([np.stack(v) for v in order_values],axis=-1)


def error_stats(prediction,actual,mask):
    p=prediction[mask];a=actual[mask];err=p-a
    scale=np.linalg.norm(a)
    return {'relative_L2':float(np.linalg.norm(err)/max(scale,1e-12)),
            'median_vector_error':float(np.median(np.linalg.norm(err,axis=-1))),
            'channel_RMSE':np.sqrt(np.mean(err**2,axis=0)).tolist(),
            'p90_vector_error':float(np.percentile(np.linalg.norm(err,axis=-1),90)),
            'mean_prediction':np.mean(p,axis=0).tolist(),'mean_actual':np.mean(a,axis=0).tolist()}


def scanner(e,dye,t):
    from filmlab.scanning.optics import measure
    from filmlab.scanning.paper import OUTPUT_BLACK
    density=dye@e._dye_spectra('CD-4').T+e.e.stock.base_density
    trans=10**(-density);sd=measure(trans,e.scanner.m);dn=e.scanner.conversion(sd)
    out={'D650':density[...,(650-380)//5],'scan_D':sd,'Dn':dn}
    for key,timing in [('exp',t),('paper',replace(t,x_knee=None,x_white=None))]:
        xe=timing.paper_exposure(dn);dp=e.scanner.digital_paper.density(xe)
        raw=e.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(dye.shape)
        out.update({key:np.maximum(raw,0),key+'_raw':raw,key+'_xe':xe,key+'_Dp':dp})
    return out


def main():
    from skimage.transform import ProjectiveTransform
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from light_acceptance import crop_warp
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path('/workspace/filmlab-audit/corridor-dye-response'));ap.add_argument('--regions',nargs='*',default=['ceiling_lamp','middle_ceiling','glass_reflection'])
    args=ap.parse_args();root=args.root;root.mkdir(exist_ok=True,parents=True)
    sid='R1-02-3';seed=zlib.crc32(sid.encode())%1000;generation.use('next_sensor')
    e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':seed%36,'grain':0,'output_conditioning':None}))
    prior=json.loads(Path('/workspace/filmlab-audit/epfl-highlights/R1-02-3.json').read_text());td=prior['timing']
    t=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    report={'scope':__doc__,'timing':td,'generation':'next_sensor','seed':seed,'regions':[],'no_RAW_read':True,
            'mask_policy':'existing measured surrounds, nominal distance 0..300um, REAL smoothed linear Y .4..85; same mask for every comparison',
            'attribution':'Shapley average over all 6 dye orders; means preserve additive closure, not unique chemistry causation'}
    for name in args.regions:
        path=root/'cache'/(name+'.npz')
        if not path.exists():raise FileNotFoundError('Verified inverse-projection cache required: '+str(path))
        a=np.load(path);old=np.load('/workspace/filmlab-audit/brightness-local/R1-02-3_'+name+'.npz')
        wp=crop_warp(ProjectiveTransform(a['local_transform']))
        d1=a['native_base_dye'];mask=(old['measured']>.5)&(old['distance_um']>0)&(old['distance_um']<=300)&(old['real_bin']==3)
        original_pixels=int(mask.sum());trusted_output=wp(a['native_trusted'].astype(float))>1-1e-10
        mask&=trusted_output
        cached={'trusted_output':trusted_output,'real':old['real'],'full_exp':old['base_exp'],'full_paper':old['base_paper'],'mask':mask,'distance_um':old['distance_um'],'real_bin':old['real_bin'],'model_bin':old['model_bin']}
        region={'name':name,'cases':[],'pixels':int(mask.sum()),'original_pixels':original_pixels,'native_cache_policy':'untrusted native endpoint increments held at zero, primary mask only uses trusted support'}
        for case,field in [('red_return','native_minus_red_dye'),('all_return','native_alloff_dye')]:
            d0=a[field].copy();d0[~a['native_trusted']]=d1[~a['native_trusted']];delta=d1-d0
            base=scanner(e,d0,t);full=scanner(e,d1,t)
            images={key:{'base':wp(base[key]),'full':wp(full[key])} for key in ['exp','paper']}
            for key in images:
                images[key]['base_lab']=lab(images[key]['base']);images[key]['full_lab']=lab(images[key]['full'])
                images[key]['actual']=images[key]['full_lab']-images[key]['base_lab']
                cached[case+'_'+key+'_base']=images[key]['base']
            corners={'exp':[],'paper':[]};rgb_corners={'exp':[],'paper':[]}
            # Eight exact finite combinations; none is promoted as a corrected film.
            for coalition in range(8):
                if coalition==0:state=base
                elif coalition==7:state=full
                else:state=scanner(e,d0+delta*np.array([(coalition>>i)&1 for i in range(3)]),t)
                for key in corners:
                    rgb=wp(state[key]);rgb_corners[key].append(rgb);corners[key].append(lab(rgb))
            finite={};orders={};finite_rgb={}
            for key in corners:
                finite[key],orders[key]=shapley_three(corners[key]);finite_rgb[key],_=shapley_three(rgb_corners[key])
                cached[case+'_'+key+'_finite_Lab_CMY']=finite[key]
                cached[case+'_'+key+'_finite_Y_CMY']=np.einsum('...ci,c->...i',finite_rgb[key],LUMA)
            # Baseline/midpoint/endpoint local Jacobians times the actual native dye increment.
            predictions={key:[] for key in corners};dir_fields={key:[] for key in corners}
            for alpha in [0.,.5,1.]:
                point=d0+alpha*delta;state=base if alpha==0 else full if alpha==1 else scanner(e,point,t)
                derivatives={key:[] for key in corners};eps=1e-5
                for c in range(3):
                    shift=np.eye(3)[c]*eps
                    plus=scanner(e,point+shift,t);minus=scanner(e,point-shift,t)
                    for key in corners:derivatives[key].append(wp((plus[key]-minus[key])/(2*eps)*delta[...,c,None]))
                for key in corners:
                    rgb=wp(state[key]);eps_rgb=1e-6
                    jl=np.stack([(lab(rgb+np.eye(3)[c]*eps_rgb)-lab(rgb-np.eye(3)[c]*eps_rgb))/(2*eps_rgb) for c in range(3)],axis=-1)
                    dr=np.stack(derivatives[key],axis=-1)
                    contributions=np.einsum('...oc,...ci->...oi',jl,dr)
                    predictions[key].append(contributions.sum(-1));dir_fields[key].append(dr.sum(-1))
                    if alpha==0:cached[case+'_'+key+'_linear_Lab_CMY']=contributions
            # Refine path quadrature, not the material/scene parameters.
            quarter={key:[] for key in corners}
            for alpha in [.25,.75]:
                eps_t=1e-4;plus=scanner(e,d0+(alpha+eps_t)*delta,t);minus=scanner(e,d0+(alpha-eps_t)*delta,t);state=scanner(e,d0+alpha*delta,t)
                for key in corners:
                    dr=wp((plus[key]-minus[key])/(2*eps_t));rgb=wp(state[key]);eps_rgb=1e-6
                    jl=np.stack([(lab(rgb+np.eye(3)[c]*eps_rgb)-lab(rgb-np.eye(3)[c]*eps_rgb))/(2*eps_rgb) for c in range(3)],axis=-1)
                    quarter[key].append(np.einsum('...oc,...c->...o',jl,dr))
            gradients={key:{0.:predictions[key][0],.25:quarter[key][0],.5:predictions[key][1],.75:quarter[key][1],1.:predictions[key][2]} for key in corners}
            fine_fields={};path_intervals=4
            def integrate(n,key):
                weights=np.array([1 if j in (0,n) else 4 if j%2 else 2 for j in range(n+1)])
                return sum(weights[j]*gradients[key][j/n] for j in range(n+1))/(3*n)
            while True:
                fine_fields={key:integrate(path_intervals,key) for key in corners}
                relative=[error_stats(fine_fields[key],images[key]['actual'],mask)['relative_L2'] for key in corners]
                if max(relative)<.005 or path_intervals>=32:break
                path_intervals*=2
                for j in range(1,path_intervals,2):
                    alpha=j/path_intervals;eps_t=1e-4
                    plus=scanner(e,d0+(alpha+eps_t)*delta,t);minus=scanner(e,d0+(alpha-eps_t)*delta,t)
                    for key in corners:
                        pr=wp(plus[key]);mi=wp(minus[key]);dr=(pr-mi)/(2*eps_t);rgb=(pr+mi)/2;eps_rgb=1e-6
                        jl=np.stack([(lab(rgb+np.eye(3)[c]*eps_rgb)-lab(rgb-np.eye(3)[c]*eps_rgb))/(2*eps_rgb) for c in range(3)],axis=-1)
                        gradients[key][alpha]=np.einsum('...oc,...c->...o',jl,dr)
            print(name,case,'quadrature intervals',path_intervals,flush=True)
            if case=='red_return':expected_base=old['minus_red_exp']
            else:
                oldname={'ceiling_lamp':'near_ceiling','middle_ceiling':'middle_ceiling','glass_reflection':'glass_reflection'}[name]
                expected_base=np.load('/workspace/filmlab-audit/corridor-lights/'+oldname+'.npz')['none']
            reproduction=float(np.max(abs(images['exp']['base'][mask]-expected_base[mask])))
            if reproduction>2e-6:raise ValueError('Cached endpoint not reproduced in target: '+str(reproduction))
            case_report={'case':case,'outputs':{},'base_endpoint_reproduction_primary_max':reproduction,'max_actual_spectral_density_increment':float(np.max(abs((d1-d0)@e._dye_spectra('CD-4').T))),
                         'mean_effective_exposure_increment_EV_RGB':np.mean(wp((a['q_base']-a['q_minus_red' if case=='red_return' else 'q_alloff'])*np.log2(10))[mask],axis=0).tolist(),
                         'mean_dye_increment':np.mean(wp(delta)[mask],axis=0).tolist(),
                         'median_dye_increment':np.median(wp(delta)[mask],axis=0).tolist(),
                         'mean_D650_increment':float(np.mean(wp(full['D650']-base['D650'])[mask])),
                         'mean_scan_density_increment':np.mean(wp(full['scan_D']-base['scan_D'])[mask],axis=0).tolist(),
                         'mean_Dn_increment':np.mean(wp(full['Dn']-base['Dn'])[mask],axis=0).tolist()}
            for key in corners:
                actual=images[key]['actual'];terms=finite[key];p=predictions[key]
                coarse=(p[0]+4*p[1]+p[2])/6
                fine=fine_fields[key]
                previous=integrate(path_intervals//2,key)
                cached[case+'_'+key+'_integral_intervals']=np.array(path_intervals)
                cached[case+'_'+key+'_actual_Lab']=actual;cached[case+'_'+key+'_baseline_linear_Lab']=p[0];cached[case+'_'+key+'_integral_Lab']=fine
                full_gap=images[key]['full_lab']-lab(old['real']);base_gap=images[key]['base_lab']-lab(old['real'])
                mean_terms=np.mean(terms[mask],axis=0);mean_total=np.mean(actual[mask],axis=0)
                order_mean=np.mean(orders[key][:,mask],axis=1)  # (6,Lab,CMY)
                out={'mean_actual_delta_Lab':mean_total.tolist(),'median_actual_delta_Lab':np.median(actual[mask],axis=0).tolist(),
                     'mean_finite_Lab_by_CMY':mean_terms.tolist(),
                     'mean_path_order_min_Lab_by_CMY':order_mean.min(0).tolist(),'mean_path_order_max_Lab_by_CMY':order_mean.max(0).tolist(),
                     'finite_additive_closure_max':float(np.max(abs(terms.sum(-1)[mask]-actual[mask]))),
                     'baseline_J_error':error_stats(p[0],actual,mask),'endpoint_J_error':error_stats(p[2],actual,mask),
                     'midpoint_J_error':error_stats(p[1],actual,mask),'integrated_J_error':error_stats(fine,actual,mask),
                     'quadrature_refinement_difference':error_stats(fine,previous,mask),
                     'quadrature_intervals':path_intervals,
                     'mean_full_minus_real_Lab':np.mean(full_gap[mask],axis=0).tolist(),'mean_base_minus_real_Lab':np.mean(base_gap[mask],axis=0).tolist(),
                     'mean_Y_contribution_CMY':np.mean(cached[case+'_'+key+'_finite_Y_CMY'][mask],axis=0).tolist(),
                     'mean_xe_increment':np.mean(wp(full[key+'_xe']-base[key+'_xe'])[mask],axis=0).tolist(),
                     'mean_Dp_increment':np.mean(wp(full[key+'_Dp']-base[key+'_Dp'])[mask],axis=0).tolist(),
                     'full_RGB_reproduction_max':float(np.max(abs(images[key]['full'][mask]-old['base_'+key][mask]))),
                     'model_nearwhite_fraction':float((old['model_bin'][mask]==4).mean()),
                     'model_mean_Y':float(np.mean((images[key]['full']@LUMA)[mask])),
                     'actual_chroma_median':float(np.median(np.linalg.norm(actual[mask,1:],axis=-1))),
                     'chroma_per_cyan_median':float(np.median(np.linalg.norm(actual[mask,1:],axis=-1)/np.maximum(abs(wp(delta)[mask,0]),1e-6)))}
                # Stratify by BASE model brightness on the same real-bright mask.
                strata=[]
                for bi,label in [(2,'model_Y_.05_.4'),(3,'model_Y_.4_.85'),(4,'model_Y_ge_.85')]:
                    m=mask&(old['model_bin']==bi)
                    if m.sum()<20:continue
                    strata.append({'label':label,'pixels':int(m.sum()),'mean_actual_delta_Lab':np.mean(actual[m],axis=0).tolist(),
                                   'mean_CMY_Lab':np.mean(terms[m],axis=0).tolist(),
                                   'baseline_J_error':error_stats(p[0],actual,m),'integrated_J_error':error_stats(fine,actual,m)})
                out['model_brightness_strata']=strata;case_report['outputs'][key]=out
            region['cases'].append(case_report)
            print(name,case,'complete',flush=True)
        np.savez_compressed(root/(name+'.npz'),**cached);report['regions'].append(region)
    report['limits']=['all effects are model counterfactual differences, not measured real-film halo',
        'dye combinations and path choices are diagnostic, not physical corrections',
        'finite-difference local matrices are not globally valid across actual return amplitudes',
        'inverse projection must pass separate cache reconstruction tests and coverage checks',
        'real image mismatch cannot uniquely attribute unmeasured source/material/device error']
    (root/('results.json' if len(args.regions)==3 else args.regions[0]+'_results.json')).write_text(json.dumps(report,indent=2));print('saved',root,flush=True)


if __name__=='__main__':main()
