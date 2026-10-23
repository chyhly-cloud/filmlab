"""Fixed-timing nonneutral layer-exposure probes, not a fit or 800T validation.

logH is the engine's effective-latent proxy, not a microscopic latent-image model.
All probes are homogeneous: no spatial transport, grain or DIR diffusion. The
experiment changes exposure, never chemistry constants or scanner parameters.
"""
import argparse
from dataclasses import replace
import hashlib,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))


def evaluate(engine,q,timing):
    from filmlab import chemistry
    from filmlab.scanning.optics import measure
    from filmlab.scanning.paper import OUTPUT_BLACK
    from filmlab.spectral import srgb_encode
    state=chemistry.run(engine.e,engine.proc,q,blur_px=0,tau_factor=engine.v.tau_factor)
    T=engine._transmittance(state)
    density=-np.log10(np.maximum(T,1e-30))
    dn=engine.scanner.normalized_density(T)
    out={'q':q,'dye':state.dye,'spectral_density':density,
         'scan_density':measure(T,engine.scanner.m),'normalized_density':dn,
         'halide_cap_reached':state.dye>=engine.e.coupler_cap-1e-9}
    for key,t in [('exponential',timing),('paper_only',replace(timing,x_knee=None,x_white=None))]:
        xe=t.paper_exposure(dn);dp=engine.scanner.digital_paper.density(xe)
        raw=engine.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(q.shape)
        lin=np.clip(raw,0,None)
        encoded=srgb_encode(lin)
        out.update({key+'_paper_exposure':xe,key+'_paper_density':dp,
                    key+'_raw_rgb':raw,key+'_linear_rgb':lin,
                    key+'_8bit':np.floor(encoded*255+.5).astype('uint8'),
                    key+'_clipped':(raw<=0)|(raw>=1)})
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--timing',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import colour
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    generation.use('next')
    engine=Engine(Params(**{**recipe('cinestill_800t'),'timing':'standard','roll':None,'grain':0,'output_conditioning':None}))
    td=json.loads(args.timing.read_text())['timing']
    t=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    levels=np.arange(-2.,3.01,.5);backgrounds=[(g,b) for g in [-.6,0.,.6] for b in [-.6,0.,.6]]
    q=np.array([[r,r+g,r+b] for r in levels for g,b in backgrounds])
    baseline=evaluate(engine,q,t)
    arrays={'base_'+k:v for k,v in baseline.items()}
    report={'scope':__doc__,'generation':'next','timing':td,
            'timing_source':str(args.timing),'timing_source_sha256':hashlib.sha256(args.timing.read_bytes()).hexdigest(),
            'timing_policy':'one pre-existing R1-02-3 timing shared by every probe and both outputs; no auto timing or refit',
            'effective_logH_levels':levels.tolist(),'green_blue_logH_offsets':backgrounds,
            'red_increment_EV':[.1,.5,1.], 'finite_differences_log10H':[.001,.0001,.00001],
            'limits':['synthetic layer-exposure interventions, not measured lamp spectra',
                      'dye/latent response is current model behaviour, not proof of physical fidelity',
                      'paper_only is a comparator, not a known scanner ground truth',
                      'fixed-timing response depends on the declared timing, not universal for all scans'],
            'rows':[]}
    def lab(rgb):return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb,0,1),apply_cctf_decoding=False))
    def logrg(rgb):return np.log(np.maximum(rgb[:,0],1e-6)/np.maximum(rgb[:,1],1e-6))
    responses={}
    for ev in [.1,.5,1.]:
        moved=q.copy();moved[:,0]+=np.log10(2)*ev
        changed=evaluate(engine,moved,t);responses[ev]=changed
        arrays.update({f'red_{ev}_{k}':v for k,v in changed.items()})
        dl=changed['spectral_density']-baseline['spectral_density']
        predicted=(changed['dye']-baseline['dye'])@engine._dye_spectra(chemistry_agent(engine)).T
        for i,source in enumerate(q):
            row={'logH':source.tolist(),'red_increment_EV':ev,
                 'dye_change_CMY':(changed['dye'][i]-baseline['dye'][i]).tolist(),
                 'spectral_density_change_rms':float(np.sqrt(np.mean(dl[i]**2))),
                 'spectral_density_change_450_550_650':dl[i,[(450-380)//5,(550-380)//5,(650-380)//5]].tolist(),
                 'dye_to_spectral_density_closure':float(np.max(abs(dl[i]-predicted[i]))),
                 'normalized_scan_density_change':(changed['normalized_density'][i]-baseline['normalized_density'][i]).tolist(),
                 'outputs':{}}
            for key in ['exponential','paper_only']:
                a=baseline[key+'_linear_rgb'];b=changed[key+'_linear_rgb']
                da=lab(b)-lab(a)
                row['outputs'][key]={'linear_RGB_before':a[i].tolist(),'linear_RGB_after':b[i].tolist(),
                    'delta_Lab':da[i].tolist(),'delta_log_R_over_G':float((logrg(b)-logrg(a))[i]),
                    'chroma_delta_norm':float(np.linalg.norm(da[i,1:])),
                    'paper_exposure_change':(changed[key+'_paper_exposure'][i]-baseline[key+'_paper_exposure'][i]).tolist(),
                    'paper_density_change':(changed[key+'_paper_density'][i]-baseline[key+'_paper_density'][i]).tolist(),
                    'paper_input_outside_curve':bool(np.any((changed[key+'_paper_exposure'][i]<engine.scanner.digital_paper.x.min())|(changed[key+'_paper_exposure'][i]>engine.scanner.digital_paper.x.max())|(baseline[key+'_paper_exposure'][i]<engine.scanner.digital_paper.x.min())|(baseline[key+'_paper_exposure'][i]>engine.scanner.digital_paper.x.max()))),
                    '8bit_before':baseline[key+'_8bit'][i].tolist(),'8bit_after':changed[key+'_8bit'][i].tolist(),
                    '8bit_identical':bool(np.array_equal(baseline[key+'_8bit'][i],changed[key+'_8bit'][i])),
                    'display_clipping':bool(np.any(baseline[key+'_clipped'][i]|changed[key+'_clipped'][i]))}
            exp=row['outputs']['exponential'];paper=row['outputs']['paper_only']
            row['exponential_over_paper_chroma_delta']=exp['chroma_delta_norm']/paper['chroma_delta_norm'] if paper['chroma_delta_norm']>1e-8 else None
            report['rows'].append(row)
    jac=[]
    for step in [.001,.0001,.00001]:
        plus=q.copy();minus=q.copy();plus[:,0]+=step;minus[:,0]-=step
        p,m=evaluate(engine,plus,t),evaluate(engine,minus,t)
        deriv={k:(p[k]-m[k])/(2*step) for k in ['dye','spectral_density','normalized_density','exponential_linear_rgb','paper_only_linear_rgb']}
        arrays.update({f'jac_{step}_{k}':v for k,v in deriv.items()});jac.append(deriv)
    stability={}
    for key in jac[0]:
        difference=np.max(abs(jac[1][key]-jac[2][key]),axis=-1)
        scale=np.maximum(np.max(abs(jac[2][key]),axis=-1),1e-6)
        stability[key]={'max_absolute_difference_h1e4_vs_h1e5':float(difference.max()),
                        'fraction_relative_difference_over_1percent':float((difference/scale>.01).mean())}
    report['finite_difference_stability']=stability
    report['own_cyan_gain_min']=float(jac[-1]['dye'][:,0].min())
    report['zero_cyan_gain_count']=int((abs(jac[-1]['dye'][:,0])<1e-6).sum())
    # Heatmaps use +0.5 EV finite increments, not a local-derivative approximation.
    plus=responses[.5]
    fig,ax=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    dye=(plus['dye'][:,0]-baseline['dye'][:,0]).reshape(len(levels),len(backgrounds))
    de=lab(plus['exponential_linear_rgb'])-lab(baseline['exponential_linear_rgb'])
    dp=lab(plus['paper_only_linear_rgb'])-lab(baseline['paper_only_linear_rgb'])
    chroma_exp=np.linalg.norm(de[:,1:],axis=1).reshape(dye.shape)
    chroma_paper=np.linalg.norm(dp[:,1:],axis=1).reshape(dye.shape)
    for j,(a,data,title) in enumerate(zip(ax,[dye,chroma_exp,chroma_paper],['Cyan dye increment','Exponential: |delta(a*,b*)|','Paper only: |delta(a*,b*)|'])):
        im=a.imshow(data,origin='lower',aspect='auto',vmin=0,vmax=None if j==0 else max(chroma_exp.max(),chroma_paper.max()));fig.colorbar(im,ax=a);a.set(title=title,xlabel='(G,B) logH offsets',ylabel='Baseline red logH')
        a.set_xticks(range(len(backgrounds)),[str(p) for p in backgrounds],rotation=65,fontsize=7)
        a.set_yticks(range(len(levels)),[f'{v:g}' for v in levels])
    fig.suptitle('Known +0.5 EV red-layer exposure; fixed chemistry/timing; homogeneous synthetic probes')
    fig.savefig(args.out/'red_response.png',dpi=170);plt.close(fig)
    # A finite-increment colour chart; encoded display only, never an input to measurements.
    from PIL import Image,ImageDraw
    chart=Image.new('RGB',(900,len(levels)*76+60),(24,24,24));d=ImageDraw.Draw(chart)
    for column,(label,data) in enumerate([('Baseline',baseline['exponential_8bit']),('+0.5 EV red: exp',plus['exponential_8bit']),('+0.5 EV red: paper_only',plus['paper_only_8bit'])]):
        d.text((column*300+8,10),label,fill='white')
        for j,r in enumerate(levels):
            # Neutral green/blue offset (index 4) keeps this chart readable.
            i=j*len(backgrounds)+4;d.rectangle((column*300+4,j*76+55,column*300+296,j*76+125),fill=tuple(data[i]))
            if column==0:d.text((8,j*76+58),f'logH {r:g}',fill='white')
    chart.save(args.out/'red_increment_swatches.png')
    np.savez_compressed(args.out/'stages.npz',**arrays)
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('nonneutral probes:',len(report['rows']),'rows; fixed timing; no fit',flush=True)


def chemistry_agent(engine):
    from filmlab.chemistry import native_dye_agent
    # Final colour developer determines the actual dye agent for spectral conversion.
    agents=[s.developer for s in engine.proc.steps if hasattr(s,'developer')]
    from filmlab.chemistry import DEVELOPERS
    return next((DEVELOPERS[a].agent for a in reversed(agents) if DEVELOPERS[a].agent),native_dye_agent(engine.e))


if __name__=='__main__':main()
