"""Equal total absorption, different deposition depth: a sufficiency test.

One declared homogeneous hypothetical absorber, matched interfaces, 650nm.
Depth boxes used by the MC check are numerical bins of the SAME material,
not fast/slow film sublayers. No 800T quantum efficiency or dye law is inferred.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys
import numpy as np
from scipy.integrate import quad
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from photon_transport import Layer,simulate


def absorption_bins(edges,mu_a,*,back=False):
    """Exact absorbed-photon fractions for unit monochromatic incident flux."""
    edges=np.asarray(edges,float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]!=0 or np.any(np.diff(edges)<=0):raise ValueError('depth edges must increase from zero')
    if not np.isfinite(mu_a) or mu_a<0:raise ValueError('nonnegative finite mu_a required')
    thickness=edges[-1]
    if back:return np.exp(-mu_a*(thickness-edges[1:]))-np.exp(-mu_a*(thickness-edges[:-1]))
    return np.exp(-mu_a*edges[:-1])-np.exp(-mu_a*edges[1:])


def integrated_response(thickness,mu_a,q_front,q_back,*,back=False):
    """Exact linear photon-to-latent weighting, q(z) interpolates end efficiencies.

    This q is a stipulated expected event yield per absorbed photon, not a
    sensitometric dye density. Both flat q and gradient q belong to known inputs.
    """
    if thickness<=0 or mu_a<0 or not np.isfinite([thickness,mu_a,q_front,q_back]).all():raise ValueError('finite positive geometry and nonnegative absorption required')
    if not 0<=q_front<=1 or not 0<=q_back<=1:raise ValueError('yield must be in [0,1]')
    tau=mu_a*thickness;absorbed=-np.expm1(-tau)
    if mu_a==0:return {'total_absorbed':0.,'first_depth_moment':0.,'effective_latent':0.}
    # ∫ z mu_a exp(-mu_a z) dz; stable for the finite optical depth used here.
    moment=(-np.expm1(-tau)-tau*np.exp(-tau))/mu_a
    if back:moment=thickness*absorbed-moment
    latent=q_front*absorbed+(q_back-q_front)*moment/thickness
    return {'total_absorbed':float(absorbed),'first_depth_moment':float(moment),'effective_latent':float(latent)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--photons',type=int,default=160000);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    thickness=3.;mu_a=.05;bins=48;edges=np.linspace(0,thickness,bins+1);centres=(edges[1:]+edges[:-1])/2
    # Same geometry/declared coefficient as the earlier hypothetical 650nm
    # lower absorber, but isolated to remove all other confounders.
    layers=[Layer('same_homogeneous_material_depth_bin',thickness/bins,mu_a,0,n=1.5) for _ in range(bins)]
    report={'scope':__doc__,'material':{'wavelength_nm':650,'thickness_um':thickness,'mu_a_per_um':mu_a,'mu_s_per_um':0,'refractive_index':1.5,
                    'evidence':'hypothetical coefficient reused from synthetic_stack(650); not film data'},
            'normalization':'same unit incident monochromatic photon flux both directions; equal total absorption analytically, no direction-dependent scaling',
            'yield_cases':[],'mc_checks':[]}
    for label,q0,q1 in [('constant_yield',.5,.5),('hypothetical_depth_gradient',.2,.8)]:
        pair=[]
        for back in [False,True]:
            result=integrated_response(thickness,mu_a,q0,q1,back=back)
            direction='back' if back else 'front'
            q=lambda z:q0+(q1-q0)*z/thickness
            density=lambda z:mu_a*np.exp(-mu_a*(thickness-z if back else z))
            numerical=quad(lambda z:density(z)*q(z),0,thickness,epsabs=1e-13,epsrel=1e-13)[0]
            result.update(direction=direction,numerical_quad_latent=float(numerical),quad_error=float(abs(numerical-result['effective_latent'])),
                          absorbed_depth_mean=result['first_depth_moment']/result['total_absorbed'])
            pair.append(result)
        report['yield_cases'].append({'label':label,'q_front':q0,'q_back':q1,'cases':pair,
            'latent_difference_back_minus_front':pair[1]['effective_latent']-pair[0]['effective_latent']})
    figdata={};absorption_exact={}
    for back,seed in [(False,6401),(True,6402)]:
        # Reversing depth labels models illumination from the physical back;
        # every segment has identical optical parameters, so tracer coordinates
        # run from its entrance in both independent runs.
        mc=simulate(layers,photons=args.photons,seed=seed,n_top=1.5,n_bottom=1.5,max_events=200)
        observed=mc['absorbed'][::-1] if back else mc['absorbed']
        exact=absorption_bins(edges,mu_a,back=back);direction='back' if back else 'front'
        sigma=np.sqrt(exact*(1-exact)/args.photons)
        row={'direction':direction,'seed':seed,'photons':args.photons,'ledger':mc['ledger'],
             'max_abs_depth_bin_error':float(np.max(abs(observed-exact))),
             'max_depth_bin_standard_errors':float(np.max(abs(observed-exact)/np.maximum(sigma,1/args.photons))),
             'total_absorption_observed':float(observed.sum()),'total_absorption_exact':float(exact.sum()),'responses':[]}
        for q0,q1 in [(.5,.5),(.2,.8)]:
            q=q0+(q1-q0)*centres/thickness
            exact_continuous=integrated_response(thickness,mu_a,q0,q1,back=back)['effective_latent']
            exact_binned=float(exact@q);observed_response=float(observed@q)
            variance=max(0.,float(exact@(q*q))-exact_binned**2)/args.photons
            row['responses'].append({'q_front':q0,'q_back':q1,'observed':observed_response,
                'exact_at_bin_centres':exact_binned,'exact_continuous':exact_continuous,
                'known_depth_discretization_error':exact_binned-exact_continuous,
                'monte_carlo_standard_error':float(np.sqrt(variance))})
        report['mc_checks'].append(row);figdata[direction]=observed;absorption_exact[direction]=exact
    report['sufficiency_verdict']={
        'constant_yield':'same layer-integrated absorption implies same effective exposure; depth redistribution alone does not refute 3 effective layer records',
        'gradient_yield':'equal unweighted absorbed dose can give different latent expectation only because a hypothetical depth-dependent yield was introduced',
        'stronger_boundary':'a static known q(z) is still reducible to H_eff=integral q(z) A(z) dz; this counterexample does not require extra developed-state sublayers',
        'what_would_be_required':'independent evidence of differing chemistry/nonlinear local response such that equal H_eff still gives different developed spectral density under controlled conditions',
        'real_800T':'no such controlled measurement was supplied; the existing 3-record effective model has not been refuted by this experiment'}
    report['response_units']='effective_latent denotes a stipulated linear expected primary photo-event statistic, not measured stable grains, dye or Engine.FilmState.latent'
    report['limits']=['Matched interfaces remove all Fresnel/halation confounders; no scattered return or real lamp is represented',
        'The gradient is a labelled hypothetical counterexample, not a fit or claim about actual 800T',
        'MC depth boxes are identical numerical subdivisions and do not create multiple-speed emulsions',
        'No source curves, scan colour, timing, material constants or current/next defaults changed']
    np.savez_compressed(args.out/'depth_profiles.npz',depth_edges=edges,front_exact=absorption_exact['front'],back_exact=absorption_exact['back'],front_mc=figdata['front'],back_mc=figdata['back'])
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for direction,color in [('front','#267bb9'),('back','#d47822')]:
        axes[0].plot(centres,absorption_exact[direction]/np.diff(edges),color=color,label=direction+' illumination')
        axes[0].scatter(centres,figdata[direction]/np.diff(edges),color=color,s=8,alpha=.5)
    axes[0].set(xlabel='Depth from physical front (um)',ylabel='Absorbed photons / incident photon / um',title='Same total absorption, mirrored deposition');axes[0].legend()
    labels=[];values=[]
    for c in report['yield_cases']:
        for result in c['cases']:
            labels.append(c['label']+'\n'+result['direction']);values.append(result['effective_latent'])
    axes[1].bar(np.arange(4),values,color=['#267bb9','#d47822']*2)
    axes[1].set_xticks(np.arange(4),['Constant q\nfront','Constant q\nback','Gradient q\nfront','Gradient q\nback'])
    axes[1].set(ylabel='Expected primary photo-events / incident photon',title='Difference appears in the stipulated gradient case')
    fig.suptitle('Known hypothetical absorber and yields; not measured CineStill chemistry')
    fig.savefig(args.out/'depth_sufficiency.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
