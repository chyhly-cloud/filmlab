"""Joint radius/wavelength/layer audit of EXISTING hypothetical transport.

No new MC implementation, material parameters, strength sweep or 800T fit.
Energy deposition and stipulated unit-yield photon counts are kept separate
from FilmLab sensitivity-calibrated H. Algebraic products are diagnostics only.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def marginal_product(joint):
    """Per-layer wavelength/radius independence preserving BOTH marginals."""
    j=np.asarray(joint,float) # wavelength, receiving layer, radial bin
    mass=j.sum(axis=2,keepdims=True);shape=j.sum(axis=0,keepdims=True)
    total=j.sum(axis=(0,2),keepdims=True)
    return np.divide(mass*shape,total,out=np.zeros_like(j),where=total>0)


def relative_tv(actual,approx):
    den=np.sum(actual)
    return float(.5*np.sum(abs(actual-approx))/den) if den>0 else None


def main():
    from filmlab import film
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/workspace/filmlab-audit');out=root/'halo-joint-transport';out.mkdir(exist_ok=True)
    report={'scope':__doc__,'wavelengths_nm':[450,550,650],'receiving_order':['lower_R_label','middle_G_label','upper_B_label'],
        'materials':'reuse photon_transport_audit.synthetic_stack; all coefficients hypothetical, not undeveloped 800T measurements',
        'cases':[],'limits':['rank failure demonstrates representational loss, not real-film error magnitude',
            'three-line sources are declared controls, not measured corridor lamp spectra',
            'MC cached run has one seed per wavelength; no significance claims from per-bin noise',
            'ballistic angular integral supplies the deterministic coupling check',
            'no calibration from absorbed energy to actual 800T latent image or exposure',
            'no transport connected to current/next; no inference of real 800T material parameters']}
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    for row,(case,dirname,names,field,active,tail) in enumerate([
        ('full_scattering_MC','photon-transport',['synthetic_450nm','synthetic_550nm','synthetic_650nm'],'radial_returned',[3,2,0],'returned_overflow'),
        ('ballistic_oblique_integral','angular-colour',['450_True','550_True','650_True'],'radial_absorbed',[0,1,3],'overflow')]):
        arrays=[np.load(root/dirname/(n+'.npz')) for n in names];edges=arrays[0]['radial_edges']
        joint=np.stack([np.column_stack([a[field][active],a[tail][active]]) for a in arrays])
        budget_errors=[]
        for a in arrays:
            budget=a['returned_absorbed'][active] if row==0 else a['absorbed'][active]
            budget_errors.append(float(np.max(abs(a[field][active].sum(-1)+a[tail][active]-budget))))
        if max(budget_errors)>2e-12:raise ValueError('radial absorption plus overflow does not conserve cached absorption')
        sep=marginal_product(joint)
        # Per-layer rank-one BEST L2 residual lower bound for any A(lambda,c) Kc(r),
        # not just the current geometric kernel or the marginal-product control.
        ranks=[]
        for c in range(3):
            singular=np.linalg.svd(joint[:,c,:],compute_uv=False)
            ranks.append({'layer':report['receiving_order'][c],'singular_values':singular.tolist(),
                          'best_rank1_relative_L2':float(np.linalg.norm(singular[1:])/np.linalg.norm(singular))})
        item={'case':case,'cache_sha256':{n:hashlib.sha256((root/dirname/(n+'.npz')).read_bytes()).hexdigest() for n in names},
              'active_absorbed_energy_per_equal_energy_input':float(joint.sum()/3),
              'radial_plus_overflow_vs_absorption_max_error':budget_errors,
              'marginal_product_TV':relative_tv(joint,sep),'rank_tests':ranks,'rings':[],'source_controls':[]}
        for lo,hi in [(0,100),(100,300),(300,800),(800,1500),(1500,float(edges[-1]))]:
            bins=(edges[:-1]>=lo)&(edges[1:]<=hi)
            energy=joint[:,:,:-1][:,:,bins].sum(-1)/3
            photon=energy*np.array([450,550,650])[:,None]/550 # stipulated ONE active event per absorbed photon
            def shares(x):return (x/x.sum()).tolist() if x.sum()>0 else None
            r={'radius_um':[lo,hi],'energy_lambda_by_layer':energy.tolist(),
               'energy_layer_share':shares(energy.sum(0)),'energy_wavelength_share':shares(energy.sum(1)),
               'unit_yield_relative_photon_layer_share':shares(photon.sum(0))}
            item['rings'].append(r)
        # Match all wavelength/layer integrated masses and the equal-source
        # per-layer radial shape algebraically. Equal mixture must reproduce;
        # other sources expose lost lambda/r correlation. No material is fitted.
        for label,source in [('equal',[1/3]*3),('blue_rich',[.6,.3,.1]),('red_rich',[.1,.3,.6])]:
            source=np.array(source);actual=np.einsum('w,wcr->cr',source,joint);approx=np.einsum('w,wcr->cr',source,sep)
            item['source_controls'].append({'source':label,'input_energy_weights':source.tolist(),
                'joint_vs_per_layer_separable_TV':relative_tv(actual,approx),
                'layer_TV':[relative_tv(actual[c],approx[c]) for c in range(3)]})
        # Use actual production first-return function with SAME stipulated
        # geometry as the ballistic control, on its stored finite radius range.
        # Only compare conditional shape inside 0..1500um, explicitly report
        # omitted physical tails. This is NOT an energy-conserving replacement.
        if row==1:
            px=10.;k=film.halation_kernel(px,base_um=130,forward=1.5,r_max_um=1500)
            yy,xx=np.indices(k.shape);radius=np.hypot(yy-(k.shape[0]-1)/2,xx-(k.shape[1]-1)/2)*px
            pedges=np.array([0,100,300,800,1500]);kh=np.histogram(radius.ravel(),pedges,weights=k.ravel())[0];kh/=kh.sum()
            item['production_first_return_same_geometry']={'px_um':px,'base_um':130,'forward':1.5,
                'conditioning':'conditional radial shape inside 1500um ONLY; no tail renormalization fed to transport',
                'rings':pedges.tolist(),'shared_kernel_ring_shares':kh.tolist(),'wavelength_layer_comparisons':[]}
            for w in range(3):
                for c in range(3):
                    h=np.array([joint[w,c,:-1][(edges[:-1]>=lo)&(edges[1:]<=hi)].sum() for lo,hi in zip(pedges[:-1],pedges[1:])])
                    h/=h.sum()
                    item['production_first_return_same_geometry']['wavelength_layer_comparisons'].append({
                        'wavelength_nm':[450,550,650][w],'layer':report['receiving_order'][c],'actual_ring_shares':h.tolist(),
                        'TV_vs_shared_kernel':relative_tv(h,kh)})
        np.savez_compressed(out/(case+'.npz'),joint_energy=joint,separable_control=sep,radial_edges=edges)
        report['cases'].append(item)
        centres=(edges[:-1]+edges[1:])/2
        for col,(label,source) in enumerate([('equal',[1/3]*3),('blue-rich',[.6,.3,.1]),('red-rich',[.1,.3,.6])]):
            ax=axes[row,col];actual=np.einsum('w,wcr->cr',source,joint)[:,:-1];approx=np.einsum('w,wcr->cr',source,sep)[:,:-1]
            # Smoothed energy histograms only for display; numeric rings use raw tallies.
            actual=gaussian_filter(actual,(0,1));approx=gaussian_filter(approx,(0,1))
            for c,color in enumerate(['red','green','blue']):
                den=actual.sum(0);d2=approx.sum(0)
                ax.plot(centres,np.divide(actual[c],den,out=np.zeros_like(den),where=den>0),color=color,label=report['receiving_order'][c])
                ax.plot(centres,np.divide(approx[c],d2,out=np.zeros_like(d2),where=d2>0),color=color,ls='--',alpha=.6)
            ax.set(xlim=(0,1500),ylim=(0,1),xlabel='Actual absorption radius (um)',ylabel='Active-layer energy share',title=case+' / '+label)
            if col==0:ax.legend(fontsize=6)
    # Same-spectral-shape separable positive control must have no representation loss.
    wavelengths=np.array([1.,2.,3.]);layers=np.array([2.,1.,.2]);radii=np.array([.1,.4,.3,.2])
    control=wavelengths[:,None,None]*layers[None,:,None]*radii[None,None,:]
    report['exact_separable_control_max_error']=float(np.max(abs(control-marginal_product(control))))
    fig.suptitle('HYPOTHETICAL materials: solid joint transport; dashed per-layer wavelength/radius separation\nAlgebraic diagnostic preserves marginals; not a proposed film or measured exposure')
    fig.savefig(out/'radial_composition.png',dpi=170);plt.close(fig)
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
