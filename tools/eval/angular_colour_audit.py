"""Layer-resolved angle/capture audit with declared hypothetical absorbers.

Optical-energy absorption is not photosensitivity, latent image or positive RGB.
No 800T optical constants are inferred from developed density or sensitivity.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from photon_transport import Layer,fresnel,simulate
from photon_transport_audit import synthetic_stack


def deposit_segment(hist,edges,weight,mu_a,thickness,mu,start_radius,*,optical_cosine=None):
    """Exact ballistic absorption histogram for a finite plane-parallel layer.

    Radius grows monotonically along each prescribed ray. Coordinates are at
    actual absorption depth, rather than forcing every event onto one plane.
    All inputs except hist/edges may be vectors of rays.
    """
    mu=np.atleast_1d(mu);weight=np.broadcast_to(weight,mu.shape)
    start=np.broadcast_to(start_radius,mu.shape);tan=np.sqrt(1-mu*mu)/mu
    optical_mu=mu if optical_cosine is None else np.broadcast_to(optical_cosine,mu.shape)
    absorbed=weight*(-np.expm1(-mu_a*thickness/optical_mu))
    if mu_a==0:return weight.copy(),0.
    # Integrate exponential attenuation between intersections with each annulus.
    for bi,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        s0=np.clip(np.divide(lo-start,tan,out=np.full_like(mu,np.inf),where=tan>0),0,thickness)
        s1=np.clip(np.divide(hi-start,tan,out=np.full_like(mu,np.inf),where=tan>0),0,thickness)
        normal=tan==0
        s0[normal]=np.where((start[normal]>=lo)&(start[normal]<hi),0,thickness)
        s1[normal]=thickness
        hist[bi]+=np.sum(weight*(np.exp(-mu_a*s0/optical_mu)-np.exp(-mu_a*s1/optical_mu)))
    tail_start=np.clip(np.divide(edges[-1]-start,tan,out=np.full_like(mu,np.inf),where=tan>0),0,thickness)
    overflow=np.sum(weight*(np.exp(-mu_a*tail_start/optical_mu)-np.exp(-mu_a*thickness/optical_mu)))
    normal=(tan==0)&(start>=edges[-1]);overflow+=absorbed[normal].sum()
    return weight*np.exp(-mu_a*thickness/optical_mu),float(overflow)


def base_return(layers,*,base_um=130.,forward=1.5,count=6000,oblique=True,edges=None):
    """Unit energy arriving at front of transparent base, prescribed angular mix.

    layers are in upward encounter order (near base first). Only one rear
    reflection is included; top escape leaves the system (index matched top).
    Oblique=False keeps the SAME geometric ray/absorption positions but uses
    normal-depth attenuation: a deliberate nonphysical separability control.
    """
    if any(l.mu_s!=0 for l in layers):raise ValueError('ballistic control requires mu_s=0')
    edges=np.asarray(edges if edges is not None else np.arange(0,5001,25),float)
    mu=(np.arange(count)+.5)/count;prob=(forward+1)*mu**forward;prob/=prob.sum()
    reflect,_=fresnel(mu,1.5,1.)
    weight=prob*reflect;bottom=float(np.sum(prob*(1-reflect)))
    tan=np.sqrt(1-mu*mu)/mu;radius=2*base_um*tan
    hist=np.zeros((len(layers),len(edges)-1));overflow=np.zeros(len(layers));absorbed=[]
    for i,l in enumerate(layers):
        prior=weight.copy()
        weight,overflow[i]=deposit_segment(hist[i],edges,weight,l.mu_a,l.thickness_um,mu,radius,
                                           optical_cosine=mu if oblique else 1.)
        absorbed.append(float((prior-weight).sum()));radius+=l.thickness_um*tan
    ledger={'bottom_escape':bottom,'absorbed':float(np.sum(absorbed)),'top_escape':float(weight.sum())}
    ledger['closure_error']=sum(ledger.values())-1
    return {'radial_edges':edges,'radial_absorbed':hist,'overflow':overflow,'absorbed':np.array(absorbed),'ledger':ledger}


def analytic_ballistic(layers,mu,edges,*,n_top=1.5,n_bottom=1.):
    """Exact down+one-rear-return absorption; index-matched top, equal indices.

    This finite-thickness control is independently compared with the MC tracer.
    It includes actual outward and return absorption coordinates and base loss.
    """
    if any(l.mu_s!=0 or l.n!=n_top for l in layers):raise ValueError('matched ballistic layers required')
    depths=np.r_[0,np.cumsum([l.thickness_um for l in layers])]
    tan=np.sqrt(1-mu*mu)/mu;hist=np.zeros((len(layers),len(edges)-1));ret=np.zeros_like(hist)
    overflow=np.zeros(len(layers));rover=np.zeros(len(layers));weight=np.ones(1)
    for i,l in enumerate(layers):
        weight,overflow[i]=deposit_segment(hist[i],edges,weight,l.mu_a,l.thickness_um,np.array([mu]),np.array([depths[i]*tan]))
    reflect=float(fresnel(mu,n_top,n_bottom)[0]);bottom=float(weight[0]*(1-reflect));weight*=reflect
    for i in reversed(range(len(layers))):
        l=layers[i];start=(2*depths[-1]-depths[i+1])*tan
        weight,rover[i]=deposit_segment(ret[i],edges,weight,l.mu_a,l.thickness_um,np.array([mu]),np.array([start]))
    hist+=ret;overflow+=rover
    absorbed=hist.sum(1)+overflow;returned=ret.sum(1)+rover
    ledger={'bottom_escape':bottom,'top_escape':float(weight[0]),'absorbed':float(absorbed.sum())};ledger['closure_error']=sum(ledger.values())-1
    return {'radial_absorbed':hist,'radial_returned':ret,'overflow':overflow,'returned_overflow':rover,'absorbed':absorbed,'returned_absorbed':returned,'ledger':ledger}


def rings(result,indices):
    edges=result['radial_edges'];hist=result['radial_absorbed'][indices];rows=[]
    for lo,hi in [(0,300),(300,800),(800,1500),(1500,5000)]:
        values=hist[:,(edges[:-1]>=lo)&(edges[1:]<=hi)].sum(1)
        rows.append({'radius_um':[lo,hi],'active_absorbed_energy':values.tolist(),
                     'active_layer_share':(values/values.sum()).tolist() if values.sum()>0 else None})
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--photons',type=int,default=120000)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    report={'scope':__doc__,'source':'unit optical energy at base-front; same stipulated angular distribution at each wavelength',
            'material_evidence':'all mu_a coefficients are declared hypothetical; names R/G/B only label layers, not latent-image quantum yields',
            'wavelengths':[],'mc_cross_checks':[]}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,wl in enumerate([450,550,650]):
        original=synthetic_stack(wl)[:-1]
        upward=[Layer(l.name,l.thickness_um,l.mu_a,0,n=l.n) for l in original[::-1]]
        cases=[]
        for oblique,label,style in [(False,'Normal-depth control','--'),(True,'Oblique depth','-')]:
            r=base_return(upward,oblique=oblique);active=[0,1,3]
            row={'label':label,'ledger':r['ledger'],'absorbed_by_layer':r['absorbed'].tolist(),'rings':rings(r,active)};cases.append(row)
            centers=(r['radial_edges'][:-1]+r['radial_edges'][1:])/2
            hist=r['radial_absorbed'][active];den=hist.sum(0)
            for li,(name,color) in enumerate([('lower/R','red'),('middle/G','green'),('upper/B','blue')]):
                share=np.divide(hist[li],den,out=np.full_like(den,np.nan),where=den>1e-10)
                axes[0,col].plot(centers,share,style,color=color,label=f'{name}: {label}')
            area=np.pi*np.diff(r['radial_edges']**2)
            axes[1,col].plot(centers,den/area,style,color='black',label=label)
            np.savez_compressed(args.out/f'{wl}_{oblique}.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        report['wavelengths'].append({'nm':wl,'upward_layers':[asdict(l) for l in upward],'cases':cases})
        axes[0,col].set(title=f'{wl} nm: fraction of active-layer absorption',xlim=(0,1600),ylim=(0,1),xlabel='Actual absorption radius (um)',ylabel='Layer energy share')
        axes[1,col].set(xlim=(0,1600),yscale='log',xlabel='Actual absorption radius (um)',ylabel='Active absorbed energy / incident / um²');axes[1,col].legend(fontsize=8)
    axes[0,0].legend(fontsize=6);fig.suptitle('Known hypothetical stack: layer composition and radius are not separable\nOptical energy shares are not positive-image RGB or measured CineStill exposure')
    fig.savefig(args.out/'layer_radius.png',dpi=180);plt.close(fig)
    # Finite-geometry single-pass crosscheck, not a repeat of the angular-source experiment.
    layers=[Layer(l.name,l.thickness_um,l.mu_a,0,n=1.5) for l in synthetic_stack(550)]
    edges=np.arange(0,2001,25.)
    for mu in [.9,.6,.3]:
        exact=analytic_ballistic(layers,mu,edges)
        mc=simulate(layers,photons=args.photons,seed=int(mu*1000),n_top=1.5,n_bottom=1.,incident_cosine=mu,radial_edges=edges)
        per_layer=mc['returned_absorbed']-exact['returned_absorbed']
        # Per-bin absolute discrepancies are reported, without tuning or rescaling histograms.
        report['mc_cross_checks'].append({'internal_cosine':mu,'photons':args.photons,
            'exact_ledger':exact['ledger'],'mc_ledger':mc['ledger'],
            'exact_returned_absorbed':exact['returned_absorbed'].tolist(),'mc_returned_absorbed':mc['returned_absorbed'].tolist(),
            'max_abs_returned_layer_error':float(np.max(abs(per_layer))),
            'max_abs_returned_radial_bin_error':float(np.max(abs(mc['radial_returned']-exact['radial_returned']))),
            'total_radial_L1_error':float(np.sum(abs(mc['radial_returned']-exact['radial_returned']))),
            'max_layer_standard_errors':float(np.max(abs(per_layer)/np.maximum(np.sqrt(exact['returned_absorbed']*(1-exact['returned_absorbed'])/args.photons),1/args.photons)))})
        np.savez_compressed(args.out/f'mc_check_{mu}.npz',exact=exact['radial_returned'],mc=mc['radial_returned'],edges=edges)
    report['equal_energy_three_line_mixture']=[]
    for ci in [0,1]:
        rr=[]
        for ri in range(4):
            values=np.mean([np.array(w['cases'][ci]['rings'][ri]['active_absorbed_energy']) for w in report['wavelengths']],axis=0)
            rr.append({'radius_um':report['wavelengths'][0]['cases'][ci]['rings'][ri]['radius_um'],
                       'active_absorbed_energy':values.tolist(),'active_layer_share':(values/values.sum()).tolist()})
        report['equal_energy_three_line_mixture'].append({'label':report['wavelengths'][0]['cases'][ci]['label'],'rings':rr})
    report['limits']=['No scattering in the new analytic control; MC support is limited to this ballistic geometry',
        'One rear reflection and index-matched top; no later round trips in this experiment',
        'Initial base-front angular source is stipulated, not generated by real-film scattering',
        'No wavelength redistribution or fluorescence; film-layer energy is not dye/exposure/RGB',
        'These hypothetical coefficients cannot identify actual 800T absorption, g, quantum yield or layer allocation',
        'No kernel resizing, tail deletion, energy renormalization or fit to photographs; current/next unchanged']
    (args.out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
