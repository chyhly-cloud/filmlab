"""Summarise target-selected far cells, without fitting any correction."""
from pathlib import Path
import json,sys,csv
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from base_colour_audit import ROOT,OUT,cell_rows,summarize


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report={'scope':__doc__,'scenes':[],'selection_note':'strict negligible-return far masks can be empty; geometric far cells are a separately declared descriptive comparison, not pure halo-free truth',
        'limits':['normal surfaces are not calibrated gray cards','one EPFL roll/lab; cafe distinct camera and unverified editing history',
            'real per-layer H not observed, so final colour cannot uniquely identify blue sensitivity error',
            'no proof of global unreachable colours from local numerical search','cells are correlated; no pixel-based significance tests']}
    table=[]
    for path in sorted(OUT.glob('R1-*.json'))+([OUT/'cafe.json'] if (OUT/'cafe.json').exists() else []):
        r=json.loads(path.read_text());name=r['scene'];a=np.load(OUT/(name+'.npz'))
        rows=r['regions']['far_geometric_1500']['cell_records'];normal=summarize(rows)
        # More distant sensitivity subset, with identical cell and texture rules.
        images={'real':a['real'],'nohalo':a['nohalo'],'full':a['full'],
                'nohalo_D65_input':a['reference_input'],'nohalo_paper_control':a['nohalo_paper']}
        distant=cell_rows(images,a['mask_far_geometric_1500']&(a['distance_um']>3000),a['light_weights'])
        strata={}
        for label,predicate in [('real_neutral_Cle12',lambda v:np.linalg.norm(v[1:])<=12),
            ('real_warm_a_b_positive',lambda v:v[1]>0 and v[2]>0 and np.linalg.norm(v[1:])>=10),
            ('real_cool_b_negative',lambda v:v[2]<0),('real_L_25_60',lambda v:25<=v[0]<60),('real_L_60_85',lambda v:60<=v[0]<85)]:
            subset=[p for p in rows if predicate(np.array(p['Lab']['real']))];strata[label]=summarize(subset)
        groups={key:summarize([p for p in rows if p['light_weights'][ci]>=.5]) for ci,key in enumerate(r['light_class_keys'])}
        groups['mixed_no_class_ge_half']=summarize([p for p in rows if max(p['light_weights'])<.5])
        controls=[]
        far_mask=a['mask_far_geometric_1500']
        dq_map=(a['reference_q']-a['nohalo_q'])*np.log2(10)
        for p in rows:
            y,x=p['yx'];m=far_mask[y:y+24,x:x+24]
            dq=dq_map[y:y+24,x:x+24][m]
            controls.append({'yx':[y,x],'D65_minus_auto_BG_EV':float(np.median(dq[:,2]-dq[:,1])),
                'D65_minus_auto_Lab':(np.array(p['Lab']['nohalo_D65_input'])-p['Lab']['nohalo']).tolist(),
                'paper_minus_exp_Lab':(np.array(p['Lab']['nohalo_paper_control'])-p['Lab']['nohalo']).tolist()})
        item={'scene':name,'cached_full_reproduction_max':r['input_full_reproduction_max'],
            'strict_far_cells':r['regions']['far_1500']['cells'],'geometric_far':normal,'geometric_far_3000':summarize(distant),
            'real_selected_strata':strata,'inferred_light_groups':groups,'input_and_output_controls':controls,
            'nominal_reference_gray_Lab':r['nominal_reference_gray_under_frame_timing_Lab']}
        report['scenes'].append(item)
        if not rows:continue
        row={'scene':name,'cells':len(rows),'cells_3000':len(distant),'strict_cells':item['strict_far_cells']}
        for state in ['nohalo','full','nohalo_D65_input','nohalo_paper_control']:
            gap=normal['states'][state]['median_gap_Lab'];row.update({state+'_'+k:float(gap[i]) for i,k in enumerate(['L','a','b'])})
        row['nohalo_cyan_blue_cell_fraction']=normal['states']['nohalo']['cyan_blue_gt2_fraction']
        if controls:
            row['D65_minus_auto_BG_EV_median']=float(np.median([c['D65_minus_auto_BG_EV'] for c in controls]))
        table.append(row)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    if table:
        with (OUT/'summary.tsv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained');x=np.arange(len(table))
    for ax,c in zip(axes,['L','a','b']):
        for state,label,offset,color in [('nohalo','No halation',-.25,'#3b80ab'),('full','Full halation',0,'#d59239'),
                    ('nohalo_D65_input','D65 input control only',.25,'#9d55a0')]:
            ax.bar(x+offset,[v[state+'_'+c] for v in table],width=.23,label=label,color=color)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x,[v['scene'] for v in table],rotation=60,ha='right')
        ax.set_ylabel('Median cell '+c+'* (model - real)');ax.set_title('Geometric far normal regions')
    axes[0].legend(fontsize=7);fig.suptitle('Fixed timing; target-selected stable normal cells >1.5 mm from highlight exclusions\nActual far surfaces, not known gray cards; D65 removal is not a correction')
    fig.savefig(OUT/'normal_regions.png',dpi=180);plt.close(fig)
    print(json.dumps(table,indent=2))


if __name__=='__main__':main()
