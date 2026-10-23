"""Full local transfer matrices on the existing 99 backgrounds, no new grid/fit.

Three existing difference step sizes are reused for each layer. Status M is the
repository's Gaussian approximation, not ISO calibration. Matrices have stage-
specific units; their norms are not energy efficiencies. Boundary derivatives
are flagged rather than treated as measured material discontinuities.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from nonneutral_chain_audit import evaluate
from brightness_response_audit import lab,brightness_bin,LABELS,LUMA


def jacobian(function,x,epsilon):
    """Output (...,m) by input (...,3) central derivative, independent rows."""
    return np.stack([(function(x+np.eye(3)[c]*epsilon)-function(x-np.eye(3)[c]*epsilon))/(2*epsilon) for c in range(3)],axis=-1)


def sensor_density_jacobian(transmittance,response):
    """Exact derivative of -log10(sum T*R) with respect to spectral density D."""
    t=np.asarray(transmittance);r=np.asarray(response)
    weight=t[...,None]*r
    denominator=weight.sum(axis=-2,keepdims=True)
    if np.any(denominator<=0):raise ValueError('positive detector response required')
    return np.swapaxes(weight/denominator,-2,-1)


def main():
    from filmlab import generation,densitometry
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.scanning.optics import measure,led_response
    from filmlab.scanning.paper import OUTPUT_BLACK
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/transfer-matrices'));args=ap.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    source=Path('/workspace/filmlab-audit/nonneutral-review/chain');a=np.load(source/'stages.npz');meta=json.loads((source/'results.json').read_text());td=meta['timing']
    generation.use('next');e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'standard','roll':None,'grain':0,'output_conditioning':None}))
    t=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    q=a['base_q'];base=evaluate(e,q,t);d=base['dye'];T=10**(-base['spectral_density'])
    st=e._develop(q);dye_spectrum=e._dye_spectra(st.dye_agent)
    keys=['dye','spectral_density','scan_density','normalized_density']
    variants=['exponential','paper_only']
    extra=lambda ev:{**ev,'status_M_approx':densitometry.measure(10**(-ev['spectral_density'])),
                     **{v+'_Lab':lab(ev[v+'_linear_rgb']) for v in variants}}
    base=extra(base);keys+=['status_M_approx']
    for v in variants:keys += [v+'_paper_exposure',v+'_paper_density',v+'_raw_rgb',v+'_linear_rgb',v+'_Lab']
    saved={};steps=[.001,.0001,.00001]
    for eps in steps:
        plus=[extra(evaluate(e,q+np.eye(3)[c]*eps,t)) for c in range(3)]
        minus=[extra(evaluate(e,q-np.eye(3)[c]*eps,t)) for c in range(3)]
        for key in keys:saved[f'eps_{eps}_{key}']=np.stack([(plus[c][key]-minus[c][key])/(2*eps) for c in range(3)],axis=-1)
    primary=steps[-1];prefix=f'eps_{primary}_';jc=saved[prefix+'dye'];jd=saved[prefix+'spectral_density']
    exact_D_q=np.einsum('li,nij->nlj',dye_spectrum,jc)
    exact_scan_q=sensor_density_jacobian(T,led_response(e.scanner.m))@exact_D_q
    exact_status_q=sensor_density_jacobian(T,densitometry.status_response('M'))@exact_D_q
    matrix=e.scanner.conversion.P*e.scanner.conversion.gain
    exact_dn_q=matrix.T@exact_scan_q
    closure={'spectrum_max_abs':float(np.max(abs(jd-exact_D_q))),
             'scan_density_max_abs':float(np.max(abs(saved[prefix+'scan_density']-exact_scan_q))),
             'status_M_approx_max_abs':float(np.max(abs(saved[prefix+'status_M_approx']-exact_status_q))),
             'Dn_max_abs':float(np.max(abs(saved[prefix+'normalized_density']-exact_dn_q)))}
    bins=brightness_bin(base['exponential_linear_rgb']);report={'scope':__doc__,'source_sha256':hashlib.sha256((source/'stages.npz').read_bytes()).hexdigest(),
        'source_timing':td,'probe_backgrounds':len(q),'step_sizes_log10H':steps,'primary_step':primary,
        'baseline_reproduction_max':float(np.max(abs(base['exponential_linear_rgb']-a['base_exponential_linear_rgb']))),
        'analytical_stage_closure':closure,'brightness_bins':LABELS,'linear_Y_edges':[0,.005,.05,.4,.85,'infinity'],'groups':[],'scanner_chain':{}}
    def scan_dye(dyes,v):
        density=dyes@dye_spectrum.T+e.e.stock.base_density
        dn=e.scanner.normalized_density(10**(-density));tim=t if v=='exponential' else replace(t,x_knee=None,x_white=None)
        dp=e.scanner.digital_paper.density(tim.paper_exposure(dn))
        return np.maximum(e.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(dyes.shape),0)
    for v in variants:
        jrgb_d=jacobian(lambda x:scan_dye(x,v),d,primary);jlab_d=jacobian(lambda x:lab(scan_dye(x,v)),d,primary)
        jr=saved[prefix+v+'_linear_rgb'];jl=saved[prefix+v+'_Lab']
        pred_rgb=jrgb_d@jc;pred_lab=jlab_d@jc
        saved[v+'_RGB_per_dye']=jrgb_d;saved[v+'_Lab_per_dye']=jlab_d
        # Each column is one dye's contribution to the red-layer exposure derivative.
        red_terms=jlab_d*jc[:,None,:,0];saved[v+'_red_Lab_terms_CMY']=red_terms
        report['scanner_chain'][v]={'RGB_max_abs_residual':float(np.max(abs(jr-pred_rgb))),
            'Lab_max_abs_residual':float(np.max(abs(jl-pred_lab))),
            'Lab_red_terms_max_abs_residual':float(np.max(abs(jl[:,:,0]-red_terms.sum(-1))))}
    for bi,label in enumerate(LABELS):
        mask=bins==bi;r={'bin':label,'samples':int(mask.sum()),'outputs':{}}
        r['median_dye_J']=np.median(jc[mask],axis=0).tolist()
        r['dye_smallest_singular_median']=float(np.median(np.linalg.svd(jc[mask],compute_uv=False)[:,-1]))
        for v in variants:
            j=saved[prefix+v+'_Lab'];terms=saved[v+'_red_Lab_terms_CMY'];small=saved[prefix+v+'_linear_rgb'];big=saved['eps_0.001_'+v+'_linear_rgb']
            diff=np.linalg.norm(small-big,axis=(1,2))/np.maximum(np.linalg.norm(small,axis=(1,2)),1e-8)
            r['outputs'][v]={'median_Lab_per_red_logH':np.median(j[mask,:,0],axis=0).tolist(),
                'median_a_terms_CMY':np.median(terms[mask,1,:],axis=0).tolist(),
                'median_b_terms_CMY':np.median(terms[mask,2,:],axis=0).tolist(),
                'median_Y_per_red_logH':float(np.median((saved[prefix+v+'_linear_rgb'][:,:,0]@LUMA)[mask])),
                'relative_RGB_step_disagreement_gt1pct':int((diff[mask]>.01).sum()),
                'median_chroma_gain_per_red_logH':float(np.median(np.linalg.norm(j[mask,1:,0],axis=1))),
                'display_clipped_fraction':float(a['base_'+v+'_clipped'][mask].any(-1).mean())}
        report['groups'].append(r)
    report['step_convergence']={}
    for v in variants:
        fine=saved[prefix+v+'_linear_rgb'];middle=saved['eps_0.0001_'+v+'_linear_rgb']
        rel=np.linalg.norm(fine-middle,axis=(1,2))/np.maximum(np.linalg.norm(fine,axis=(1,2)),1e-8)
        report['step_convergence'][v]={'fraction_gt1pct':float((rel>.01).mean()),'max_relative':float(rel.max())}
    qi=int(np.flatnonzero(np.all(q==0,axis=-1))[0]);report['neutral_q0_example']={'index':qi,'q':q[qi].tolist(),'dye_J':jc[qi].tolist(),
        'outputs':{v:{'Lab_red_terms_CMY':saved[v+'_red_Lab_terms_CMY'][qi].tolist(),
                      'Lab_per_red_logH':saved[prefix+v+'_Lab'][qi,:,0].tolist()} for v in variants}}
    saved.update(q=q,base_dye=d,base_exponential_RGB=base['exponential_linear_rgb'],brightness_bin=bins,
                 analytic_scan_J=exact_scan_q,analytic_statusM_J=exact_status_q,analytic_Dn_J=exact_dn_q)
    np.savez_compressed(out/'matrices.npz',**saved)
    report['limits']=['local derivatives of existing effective model, not measured 800T chemistry',
        'same timing and no output conditioning; independent scan audit handles empirical conditioning',
        'approximate Status M uses Gaussian channels, not ISO 5-3 table',
        'piecewise LUT, display clipping and material plateaux can invalidate a unique central derivative',
        'stage Jacobians have different dimensions/units; no energy-efficiency claim']
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key in zip(axes,['dye','exponential_Lab','paper_only_Lab']):
        mat=saved[prefix+key];image=ax.imshow(mat[:,1,:],aspect='auto',cmap='coolwarm');fig.colorbar(image,ax=ax)
        ax.set_xticks(range(3),['R input','G input','B input']);ax.set_xlabel('Input effective log10 H');ax.set_ylabel('Existing background index');ax.set_title('Magenta dye derivatives' if key=='dye' else key+' a* derivatives')
    fig.suptitle('Existing backgrounds: local response matrices, fixed timing; no new fit');fig.savefig(out/'matrices.png',dpi=170)
    plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,v in zip(axes,variants):
        terms=saved[v+'_red_Lab_terms_CMY'][qi,1:,:];x=np.arange(2);pos=np.zeros(2);neg=np.zeros(2)
        for c,(name,color) in enumerate([('Cyan dye','#36a4ae'),('Magenta dye','#c34b9b'),('Yellow dye','#c4aa19')]):
            value=terms[:,c];bottom=np.where(value>=0,pos,neg);ax.bar(x,value,bottom=bottom,label=name,color=color)
            pos+=np.maximum(value,0);neg+=np.minimum(value,0)
        ax.plot(x,terms.sum(-1),'ko',label='Total derivative');ax.axhline(0,color='gray',lw=.7)
        ax.set_xticks(x,['a* (red–green)','b* (yellow–blue)']);ax.set_title(v);ax.set_ylabel('Derivative per red-layer log10 H');ax.legend(fontsize=8)
    fig.suptitle('Same q=(0,0,0), fixed timing: chemistry cross-effects change colour direction')
    fig.savefig(out/'cross_layer_colour.png',dpi=180);plt.close(fig)



if __name__=='__main__':main()
