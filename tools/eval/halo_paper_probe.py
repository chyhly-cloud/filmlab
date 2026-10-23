"""Reuse all 297 red probes to isolate paper spectral colour from its tone curve."""
from pathlib import Path
import json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from halo_mechanism_ablation import separable_paper_rgb
from brightness_response_audit import brightness_bin,lab,LABELS


def main():
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.paper import OUTPUT_BLACK
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('/workspace/filmlab-audit/halo-mechanisms');out.mkdir(exist_ok=True)
    a=np.load('/workspace/filmlab-audit/nonneutral-review/chain/stages.npz')
    generation.use('next');e=Engine(Params(**{**recipe('cinestill_800t'),'grain':0,'roll':None,'output_conditioning':None}))
    k=e.scanner.digital_paper.k;rows=[]
    bins=brightness_bin(a['base_exponential_linear_rgb'])
    for shoulder in ['exponential','paper_only']:
        b=a['base_'+shoulder+'_linear_rgb'];ib=np.maximum(separable_paper_rgb(e.scanner.digital_paper,a['base_'+shoulder+'_paper_density'],OUTPUT_BLACK),0)
        for ev in [.1,.5,1.]:
            prefix='red_'+str(ev);c=a[prefix+'_'+shoulder+'_linear_rgb'];ic=np.maximum(separable_paper_rgb(e.scanner.digital_paper,a[prefix+'_'+shoulder+'_paper_density'],OUTPUT_BLACK),0)
            dc=a[prefix+'_dye'][:,0]-a['base_dye'][:,0]
            for bi,label in enumerate(LABELS):
                mask=bins==bi;r={'shoulder':shoulder,'red_EV':ev,'bin':label,'samples':int(mask.sum()),'outputs':{}}
                for key,before,after in [('spectral_paper',b,c),('independent_channels',ib,ic)]:
                    d=lab(after)-lab(before)
                    r['outputs'][key]={'median_delta_a':float(np.median(d[mask,1])),
                        'median_chroma_delta':float(np.median(np.linalg.norm(d[mask,1:],axis=-1))),
                        'median_chroma_per_cyan':float(np.median(np.linalg.norm(d[mask,1:],axis=-1)/np.maximum(dc[mask],1e-12))),
                        'clipped_fraction':float((((before[mask]<0)|(before[mask]>1))|((after[mask]<0)|(after[mask]>1))).any(-1).mean())}
                actual_flags=(a['base_'+shoulder+'_clipped'][mask]|a[prefix+'_'+shoulder+'_clipped'][mask]).any(-1)
                r['outputs']['spectral_paper']['clipped_fraction']=float(actual_flags.mean())
                rows.append(r)
    dp=np.repeat(np.linspace(0,float(e.scanner.digital_paper.curve.max()),30)[:,None],3,axis=-1)
    actual=e.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK);ideal=separable_paper_rgb(e.scanner.digital_paper,dp,OUTPUT_BLACK)
    report={'scope':__doc__,'rows':rows,'neutral_linear_RGB_max_difference':float(np.max(abs(actual-ideal))),
            'limits':'same Dp; neutral-matched independent channels are an isolation control, not a measured Frontier LUT; bins fixed by original exponential baseline'}
    (out/'paper_probes.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,shoulder in zip(axes,['exponential','paper_only']):
        selected=[r for r in rows if r['shoulder']==shoulder and r['red_EV']==.5]
        for key in ['spectral_paper','independent_channels']:
            ax.plot(range(5),[r['outputs'][key]['median_chroma_per_cyan'] for r in selected],'o-',label=key)
        ax.set_xticks(range(5),['<.005','.005–.05','.05–.4','.4–.85','≥.85']);ax.set_title(shoulder)
        ax.set_xlabel('Original baseline linear Y');ax.set_ylabel('Chroma increment / cyan dye increment');ax.legend(fontsize=8)
    fig.suptitle('Same negative, timing and paper density; +0.5 EV red-layer probe')
    fig.savefig(out/'paper_colour_isolation.png',dpi=170)
    print('neutral gray max RGB mismatch:',report['neutral_linear_RGB_max_difference'])
    for r in rows:
        if r['red_EV']==.5:print(r)


if __name__=='__main__':main()
