"""Summarize the fixed-timing brightness audit without changing its bins or fitting."""
from pathlib import Path
import argparse,csv,json
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    p=json.loads((root/'probe_brightness.json').read_text());rows=[]
    for path in sorted(root.glob('R*.json')):
        s=json.loads(path.read_text())
        for r in s['regions']:
            for v in r['statistics']:
                if 'outputs' not in v:continue
                e=v['outputs']['exp'];q=v['outputs']['paper']
                rows.append({'region':r['name'],'geometry':r['geometry_origin'],'zone':v['zone'],
                    'real_bin':v['real_brightness_bin'],'pixels':v['pixels'],'real_Y':v['real_Y'],
                    'exp_Y':v['model_Y'],'paper_Y':q['Y'],
                    'exp_nearwhite_fraction':v['model_bins']['near_white']/v['pixels'],
                    'red_EV':v['red_added_EV'],'cyan_delta':v['cyan_increment'],
                    'D650_delta':v['spectral_density_delta_650'],'Dscan_R_delta':v['scan_density_R_increment'],
                    'Dn_R_delta':v['normalized_density_R_increment'],'exp_xe_R_delta':e['paper_exposure_red_increment'],
                    'paper_xe_R_delta':q['paper_exposure_red_increment'],'exp_Dp_R_delta':e['paper_density_red_increment'],
                    'paper_Dp_R_delta':q['paper_density_red_increment'],
                    'exp_red_chroma':e['red_chroma_delta'],'paper_red_chroma':q['red_chroma_delta'],
                    'exp_red_chroma_per_cyan':e['red_chroma_per_cyan'],'paper_red_chroma_per_cyan':q['red_chroma_per_cyan'],
                    'exp_a_gap':e['dLab_vs_real'][1],'paper_a_gap':q['dLab_vs_real'][1],
                    'exp_b_gap':e['dLab_vs_real'][2],'paper_b_gap':q['dLab_vs_real'][2],
                    'real_logY_span':v['real_logY_p90_p10'],'exp_logY_span':v['model_logY_p90_p10'],
                    'exp_clipped':e['display_clipped_fraction'],'paper_clipped':q['display_clipped_fraction'],
                    'scan_floor':v['scan_floor_fraction']})
    with (root/'local_brightness.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(5)
    for key,label,col in [('exponential','Extra exponential + paper','#d17822'),('paper_only','Paper only','#247ca2')]:
        y=[r['outputs'][key]['chroma_per_cyan'] for r in p['all_297_rebinned'] if r['red_EV']==.5]
        ax[0].plot(x,y,'o-',label=label,color=col)
    ax[0].set_xticks(x,['<.005','.005–.05','.05–.4','.4–.85','≥.85'])
    ax[0].set_xlabel('Baseline linear Y (exp output); fixed +0.5 EV red increment')
    ax[0].set_ylabel('Median chroma step / cyan dye increment')
    ax[0].set_title('Existing probes: strongest loss near white');ax[0].legend(fontsize=8)
    colors={'R1-00-1':'#9467bd','R1-02-3':'#2ca02c','R1-04-5':'#d62728','R1-07-8':'#1f77b4'}
    for sid,c in colors.items():
        subset=[r for r in rows if r['zone']=='near_0_300' and r['region'].startswith(sid)]
        ax[1].scatter([r['exp_Y'] for r in subset],[r['exp_red_chroma_per_cyan'] for r in subset],color=c,label=sid)
    ax[1].axvspan(.85,1,color='gray',alpha=.15)
    ax[1].set_xlabel('Model linear Y; groups assigned by REAL scan brightness')
    ax[1].set_ylabel('Median red-return chroma step / cyan increment')
    ax[1].set_title('EPFL 0–300 µm: response remains outside white');ax[1].legend(fontsize=8)
    fig.suptitle('Fixed timing; float linear sRGB → Lab. Conditional model response ≠ measured real-film derivative.',fontsize=10)
    fig.savefig(root/'brightness_response.png',dpi=180);plt.close(fig)
    summary={'regions':len(set(r['region'] for r in rows)),'populated_region_zone_bins':len(rows),
        'probe_conversion_closure_max':max(r['conversion_closure_max'] for r in p['all_297_rebinned']),
        'max_scan_floor_fraction':max(r['scan_floor'] for r in rows),
        'reproduction_error_max':max(r['baseline_reproduction_error'] or 0 for path in root.glob('R*.json') for r in json.loads(path.read_text())['regions']),
        'limits':'pixels are spatially correlated; no significance test, no calibrated real-film red derivative; manual ROI distances are geometric only'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
