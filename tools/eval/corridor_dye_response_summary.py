"""Combine cache-only local finite responses; no fitting or extra image probes."""
from pathlib import Path
import json,csv,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from filmlab.spectral import srgb_encode
    root=Path('/workspace/filmlab-audit/corridor-dye-response');names=['ceiling_lamp','middle_ceiling','glass_reflection']
    reports=[json.loads((root/(n+'_results.json')).read_text()) for n in names]
    if any(r['timing']!=reports[0]['timing'] for r in reports):raise ValueError('timing changed')
    report={**reports[0],'regions':[r['regions'][0] for r in reports]};rows=[]
    for region in report['regions']:
        region['real_pairing_status']='local glass-light correspondence visibly fails; numerical decomposition only' if region['name']=='glass_reflection' else 'fixed registration retained; source radiance/SPD unknown'
        for case in region['cases']:
            for variant,v in case['outputs'].items():
                a=np.array(v['mean_finite_Lab_by_CMY']);total=np.array(v['mean_actual_delta_Lab']);cross=a[:,1:].sum(-1)
                r={'region':region['name'],'case':case['case'],'variant':variant,'pixels':region['pixels'],
                    'mean_delta_L':total[0],'mean_delta_a':total[1],'mean_delta_b':total[2],
                    'mean_b_C':a[2,0],'mean_b_M':a[2,1],'mean_b_Y':a[2,2],'mean_b_cross_net':cross[2],
                    'b_cancellation_fraction':float(-cross[2]/a[2,0]),
                    'mean_a_C':a[1,0],'mean_a_cross_net':cross[1],
                    'response_hue_rotation_deg':float(np.degrees(np.arctan2(total[2],total[1])-np.arctan2(a[2,0],a[1,0]))),
                    'baseline_J_relative_error':v['baseline_J_error']['relative_L2'],'integrated_J_relative_error':v['integrated_J_error']['relative_L2'],
                    'mean_gap_L':v['mean_full_minus_real_Lab'][0],'mean_gap_a':v['mean_full_minus_real_Lab'][1],'mean_gap_b':v['mean_full_minus_real_Lab'][2],
                    'model_mean_Y':v['model_mean_Y'],'model_nearwhite_fraction':v['model_nearwhite_fraction'],
                    'input_R_EV_mean':case['mean_effective_exposure_increment_EV_RGB'][0],
                    'delta_D650_mean':case['mean_D650_increment'],'delta_DscanR_mean':case['mean_scan_density_increment'][0]}
                rows.append(r)
    report['summary_rows']=rows
    (root/'results.json').write_text(json.dumps(report,indent=2))
    with (root/'summary.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,region in zip(axes,report['regions']):
        case=next(c for c in region['cases'] if c['case']=='all_return')['outputs']['exp']
        terms=np.array(case['mean_finite_Lab_by_CMY'])[1:,:];x=np.arange(2);pos=np.zeros(2);neg=np.zeros(2)
        for c,(label,color) in enumerate([('Cyan pathway','#36a4ae'),('Magenta pathway','#c34b9b'),('Yellow pathway','#c4aa19')]):
            v=terms[:,c];ax.bar(x,v,bottom=np.where(v>=0,pos,neg),label=label,color=color);pos+=np.maximum(v,0);neg+=np.minimum(v,0)
        ax.plot(x,terms.sum(-1),'ko',label='Full finite change');ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x,['a* (red–green)','b* (yellow–blue)'])
        ax.set_title(region['name']+(' (local mismatch)' if region['name']=='glass_reflection' else ''));ax.set_ylabel('Mean finite Lab change (all returns on minus off)')
    axes[0].legend(fontsize=7);fig.suptitle('Actual cached amplitudes / fixed timing / real-bright 0–300 µm mask\nShapley dye contributions are diagnostic allocations, not chemical measurements',fontsize=11)
    fig.savefig(root/'actual_dye_contributions.png',dpi=180);plt.close(fig)
    # Each diagnostic map is shown only on the trusted real-bright target mask.
    fig,axes=plt.subplots(3,5,figsize=(16,10.5),layout='constrained')
    for row,name in enumerate(names):
        a=np.load(root/(name+'.npz'));m=a['mask'];terms=a['all_return_exp_finite_Lab_CMY'];total=a['all_return_exp_actual_Lab'];cross=terms[...,2,1:].sum(-1)
        fields=[('Real scan',None),('Existing exp_plain',None),('Cyan pathway: delta b*',terms[...,2,0]),('Cross pathways M+Y: delta b*',cross),('Full finite delta b*',total[...,2])]
        for col,(title,values) in enumerate(fields):
            ax=axes[row,col]
            if col<2:
                image=a['real'] if col==0 else a['full_exp'];ax.imshow(np.clip(srgb_encode(np.nan_to_num(image)),0,1));ax.contour(m,levels=[.5],colors=['#ffdc35'],linewidths=.4)
            else:
                im=ax.imshow(np.ma.masked_where(~m,values),cmap='coolwarm',vmin=-12,vmax=12);fig.colorbar(im,ax=ax,shrink=.65)
            ax.set_title(title if row==0 else '',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(name+(' (pairing unreliable)' if name=='glass_reflection' else ''),fontsize=9)
    fig.suptitle('Fixed real-bright mask (outline); maps use Lab units, not recoloured photo candidates',fontsize=12)
    fig.savefig(root/'local_maps.png',dpi=180);plt.close(fig)
    print('rows',len(rows),'saved',root)


if __name__=='__main__':main()
