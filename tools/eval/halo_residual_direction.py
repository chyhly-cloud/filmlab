"""Fixed-cache E/V test: does complete model halation reduce the real residual?

E=Lab_real-Lab_nohalo is an endpoint target residual, NOT measured real halation.
V=Lab_halo-Lab_nohalo is the finite model intervention. No RAW reads or renders,
operator refits, synthetic probes or material changes. Euclidean Lab geometry
and perceptual deltaE00 are reported separately, not used to choose parameters.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
from brightness_response_audit import lab,LUMA


def vector_metrics(E,V,*,target_floor=1.,movement_floor=.2):
    """Euclidean geometry; a positive dot alone does not imply improvement."""
    E=np.asarray(E);V=np.asarray(V);dot=np.sum(E*V,axis=-1)
    before=np.linalg.norm(E,axis=-1);after=np.linalg.norm(E-V,axis=-1);movement=np.linalg.norm(V,axis=-1)
    valid=(before>target_floor)&(movement>movement_floor)
    cosine=np.divide(dot,before*movement,out=np.full_like(dot,np.nan,dtype=float),where=valid)
    improvement=before**2-after**2
    return dict(before=before,after=after,movement=movement,dot=dot,cosine=cosine,valid=valid,
                squared_improvement=improvement,towards=dot>0,improved=improvement>1e-10)


def distribution(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not x.size:return None
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.percentile(x,10)),'p90':float(np.percentile(x,90))}


def geometry_summary(E,V):
    m=vector_metrics(E,V);valid=m['valid'];dot=float(np.sum(m['dot']));den=np.linalg.norm(E)*np.linalg.norm(V)
    return {'aggregate_cosine':float(dot/den) if den>1e-12 else None,
            'pixel_cosine':distribution(m['cosine']),'direction_defined_fraction':float(valid.mean()),
            'towards_fraction_when_defined':float(m['towards'][valid].mean()) if valid.any() else None,
            'actual_distance_improved_fraction':float(m['improved'].mean()),
            'positive_direction_but_distance_worse_fraction':float((m['towards']&~m['improved']).mean()),
            'distance_before':distribution(m['before']),'distance_after':distribution(m['after']),
            'distance_reduction':distribution(m['before']-m['after']),
            'mean_target_E':np.mean(E,axis=0).tolist(),'mean_added_V':np.mean(V,axis=0).tolist(),
            'vector_identity_max_error':float(np.max(abs(m['squared_improvement']-(2*m['dot']-m['movement']**2))))}


def circular_abs_angle(a,b):
    return np.abs((a-b+180)%360-180)


def state_summary(rgb,Lab,target_rgb,target_Lab):
    c=np.linalg.norm(Lab[:,1:],axis=-1);ct=np.linalg.norm(target_Lab[:,1:],axis=-1)
    h=np.degrees(np.arctan2(Lab[:,2],Lab[:,1]));ht=np.degrees(np.arctan2(target_Lab[:,2],target_Lab[:,1]));valid=(c>=5)&(ct>=5)
    gap=Lab-target_Lab
    white=((rgb@LUMA)>=.85)&(c<10);true_white=((target_rgb@LUMA)>=.85)&(ct<10)
    return {'mean_Lab':np.mean(Lab,axis=0).tolist(),'median_Lab':np.median(Lab,axis=0).tolist(),
            'L':distribution(Lab[:,0]),'a':distribution(Lab[:,1]),'b':distribution(Lab[:,2]),'chroma':distribution(c),
            'mean_gap_Lab':np.mean(gap,axis=0).tolist(),'median_gap_Lab':np.median(gap,axis=0).tolist(),
            'absolute_error_L':distribution(abs(gap[:,0])),'absolute_error_a':distribution(abs(gap[:,1])),
            'absolute_error_b':distribution(abs(gap[:,2])),'absolute_error_chroma':distribution(abs(c-ct)),
            'hue_error_deg_when_both_chromas_ge5':distribution(circular_abs_angle(h[valid],ht[valid])),
            'hue_error_defined_fraction':float(valid.mean()),'model_low_chroma_fraction':float((c<5).mean()),
            'brighter_than_real_by_gt1_L_fraction':float((gap[:,0]>1).mean()),
            'more_yellow_than_real_by_gt1_b_fraction':float((gap[:,2]>1).mean()),
            'both_bright_and_yellow_excess_fraction':float(((gap[:,0]>1)&(gap[:,2]>1)).mean()),
            'nearwhite_fraction':float(white.mean()),'false_nearwhite_fraction':float((white&~true_white).mean())}


def main():
    import colour
    from scipy.ndimage import gaussian_filter
    from filmlab.spectral import srgb_encode
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/workspace/filmlab-audit');out=root/'halo-residual-direction';out.mkdir(exist_ok=True)
    meta=json.loads((root/'brightness-local/R1-02-3.json').read_text());um=meta['scan_um_per_pixel']
    report={'scope':__doc__,'scene':'R1-02-3','timing':meta['timing'],'generation':'next_sensor','conditioning':None,
            'grain':0,'source_hashes':meta['source_hashes'],'scan_um_per_pixel':um,'regions':[],
            'primary_measurement':'same Gaussian sigma=25um on LINEAR RGB of all three states, then Lab; native-pixel result retained',
            'colour_region_definition':'existing measured surrounds at d=0..300um, REAL smoothed Y=.05..85,a>0,b>0,Cab>10; target only, no model/error selection',
            'white_definition':'diagnostic linearY>=.85 and Cab<10; not a material/device saturation threshold',
            'cosine_definition':'pixel cosine only when |E|>1 Lab and |V|>.2 Lab; aggregate uses all vectors'}
    display=[];flat_rows=[]
    for name,oldname in [('ceiling_lamp','near_ceiling'),('middle_ceiling','middle_ceiling')]:
        p=root/'corridor-lights'/(oldname+'.npz');a=np.load(p);b=np.load(root/'brightness-local'/('R1-02-3_'+name+'.npz'))
        images={'real':a['real'],'nohalo':a['none'],'halo':a['exp']}
        # Reuse independently reproduced cached endpoints for verification only.
        latest=np.load(root/'corridor-dye-response'/(name+'.npz'))
        m=latest['mask'];errors={'real':float(np.nanmax(abs(images['real']-b['real']))),
            'full':float(np.nanmax(abs(images['halo']-b['base_exp']))),
            'nohalo_target':float(np.max(abs(images['nohalo'][m]-latest['all_return_exp_base'][m])))}
        if max(errors.values())>2e-6:raise ValueError('cache endpoint mismatch')
        dist=b['distance_um'];measured=b['measured']>.5
        core=np.load(root/'light-acceptance'/('R1-02-3_'+name+'.npz'))['core'].astype(bool) if (root/'light-acceptance'/('R1-02-3_'+name+'.npz')).exists() else dist==0
        # Same region definitions for both scales and all states, fixed before residuals.
        smooth={key:gaussian_filter(np.nan_to_num(rgb),sigma=(25/um,25/um,0)) for key,rgb in images.items()}
        real_lab=lab(smooth['real']);Y=smooth['real']@LUMA;C=np.linalg.norm(real_lab[...,1:],axis=-1)
        valid=np.logical_and.reduce([np.isfinite(rgb).all(-1) for rgb in images.values()])
        zones={'sensor_core':core,'near_edge_0_100':(dist>0)&(dist<=100)&measured,
               'transition_100_300':(dist>100)&(dist<=300)&measured,
               'real_orange_transition_0_300':(dist>0)&(dist<=300)&measured&(Y>=.05)&(Y<.85)&(real_lab[...,1]>0)&(real_lab[...,2]>0)&(C>10),
               'outer_300_800':(dist>300)&(dist<=800)&measured,
               'darker_outer_300_800_Ylt_.4':(dist>300)&(dist<=800)&measured&(Y<.4),
               'dark_outer_300_800_Ylt_.05':(dist>300)&(dist<=800)&measured&(Y<.05),
               'previous_realbright_0_300':(dist>0)&(dist<=300)&measured&(b['real_bin']==3)}
        zones={k:v&valid for k,v in zones.items()}
        region={'name':name,'cache_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'endpoint_reproduction':errors,
                'core_policy':'sensor-core proxy, output comparison allowed but radiance remains prior-dependent',
                'zones':{k:int(v.sum()) for k,v in zones.items()},'scales':{}}
        saved={**images,'distance_um':dist,'measured':measured,**{'mask_'+k:v for k,v in zones.items()}}
        for scale,rgbs in [('native',images),('25um',smooth)]:
            labs={k:lab(rgb) for k,rgb in rgbs.items()};E=labs['real']-labs['nohalo'];V=labs['halo']-labs['nohalo'];Eafter=E-V
            if np.max(abs(Eafter[valid]-(labs['real']-labs['halo'])[valid]))>1e-12:raise ValueError('residual identity broken')
            deN=colour.delta_E(labs['real'],labs['nohalo'],method='CIE 2000');deH=colour.delta_E(labs['real'],labs['halo'],method='CIE 2000')
            gain_L=np.abs(E[...,0])-np.abs(Eafter[...,0]);gain_ab=np.linalg.norm(E[...,1:],axis=-1)-np.linalg.norm(Eafter[...,1:],axis=-1)
            saved[scale+'_E']=E;saved[scale+'_V']=V;saved[scale+'_deltaE00_reduction']=deN-deH
            saved[scale+'_L_error_reduction']=gain_L;saved[scale+'_ab_error_reduction']=gain_ab
            rows=[]
            for zone,mask in zones.items():
                row={'zone':zone,'pixels':int(mask.sum()),'status':'evaluated' if mask.sum()>=30 else 'insufficient_pixels'}
                if mask.sum()<30:rows.append(row);continue
                r=labs['real'][mask];n=labs['nohalo'][mask];h=labs['halo'][mask];e=E[mask];v=V[mask]
                row.update({'states':{key:state_summary(rgb[mask],labs[key][mask],rgbs['real'][mask],r) for key,rgb in rgbs.items()},
                            'Lab_geometry':geometry_summary(e,v),'ab_geometry':geometry_summary(e[:,1:],v[:,1:]),
                            'brightness_L':{'mean_E':float(e[:,0].mean()),'mean_V':float(v[:,0].mean()),
                                'error_before':distribution(abs(e[:,0])),'error_after':distribution(abs((e-v)[:,0])),
                                'error_reduction':distribution(gain_L[mask]),'improved_fraction':float((gain_L[mask]>1e-10).mean())},
                            'deltaE00_before':distribution(deN[mask]),'deltaE00_after':distribution(deH[mask]),
                            'deltaE00_reduction':distribution((deN-deH)[mask]),'deltaE00_improved_fraction':float((deN[mask]>deH[mask]+1e-10).mean()),
                            'sensor_any_saturated_fraction':float((a['saturated_channels'][mask].max(-1)>.5).mean())})
                cn=np.linalg.norm(n[:,1:],axis=-1);ch=np.linalg.norm(h[:,1:],axis=-1);cr=np.linalg.norm(r[:,1:],axis=-1)
                row['chroma_change']=distribution(ch-cn);row['chroma_absolute_error_reduction']=distribution(abs(cn-cr)-abs(ch-cr))
                # Compare hue on the SAME samples: separate state thresholds can
                # otherwise make neutralisation look like a hue improvement.
                hue_valid=(cn>=5)&(ch>=5)&(cr>=5)
                hn=np.degrees(np.arctan2(n[:,2],n[:,1]));hh=np.degrees(np.arctan2(h[:,2],h[:,1]));hr=np.degrees(np.arctan2(r[:,2],r[:,1]))
                en=circular_abs_angle(hn[hue_valid],hr[hue_valid]);eh=circular_abs_angle(hh[hue_valid],hr[hue_valid])
                row['hue_common']={'pixels':int(hue_valid.sum()),'fraction':float(hue_valid.mean()),
                    'before_deg':distribution(en),'after_deg':distribution(eh),
                    'improved_fraction':float((eh<en).mean()) if hue_valid.any() else None}
                rows.append(row)
                flat_rows.append({'region':name,'scale':scale,'zone':zone,'pixels':row['pixels'],
                    **{s+'_'+axis:row['states'][s]['mean_Lab'][j] for s in images for j,axis in enumerate(['L','a','b'])},
                    **{s+'_C':row['states'][s]['chroma']['mean'] for s in images},
                    'E_L':row['brightness_L']['mean_E'],'V_L':row['brightness_L']['mean_V'],
                    'E_a':row['ab_geometry']['mean_target_E'][0],'E_b':row['ab_geometry']['mean_target_E'][1],
                    'V_a':row['ab_geometry']['mean_added_V'][0],'V_b':row['ab_geometry']['mean_added_V'][1],
                    'ab_cosine':row['ab_geometry']['aggregate_cosine'],'ab_improved_fraction':row['ab_geometry']['actual_distance_improved_fraction'],
                    'L_improved_fraction':row['brightness_L']['improved_fraction'],'L_error_reduction_mean':row['brightness_L']['error_reduction']['mean'],
                    'DE00_before_mean':row['deltaE00_before']['mean'],'DE00_after_mean':row['deltaE00_after']['mean'],
                    'baseline_bright_yellow_excess_fraction':row['states']['nohalo']['both_bright_and_yellow_excess_fraction']})
            region['scales'][scale]=rows
            if scale=='25um':display.append(dict(name=name,images=images,masks=zones,E=E,V=V,gainL=gain_L,gainab=gain_ab,gainDE=deN-deH))
        np.savez_compressed(out/(name+'.npz'),**saved);report['regions'].append(region)
    report['limits']=['E is a model-to-target residual, not pure measured real halation',
        'halation-off negative is not an observable real-film control; base error versus missing physical halo is not uniquely identified',
        'three final states alone cannot prove scanner attenuation or distinguish input/development/scanner origins of the nohalo output',
        'two lights in one scene/roll/lab are not independent material/device calibration',
        'native and 25um averaging compare measurement scales, not sharpening or a proposed rendering change',
        'pixels are correlated descriptive samples; no significance claims or fitted strength factors']
    (out/'results.json').write_text(json.dumps(report,indent=2))
    import csv
    with (out/'summary.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat_rows[0]),delimiter='\t');w.writeheader();w.writerows(flat_rows)
    fig,axes=plt.subplots(2,3,figsize=(12,8.3),layout='constrained')
    for row,d in enumerate(display):
        for col,key in enumerate(['real','nohalo','halo']):
            ax=axes[row,col];ax.imshow(np.clip(srgb_encode(d['images'][key]),0,1),interpolation='nearest')
            ax.set_title({'real':'Real CineStill scan','nohalo':'No halation','halo':'Full halation'}[key]);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(d['name'])
    fig.suptitle('R1-02-3 | identical registration and fixed timing | original cached photos')
    fig.savefig(out/'three_state_photos.png',dpi=160);plt.close(fig)
    # Original-resolution photos; smoothing is confined to measurement maps.
    fig,axes=plt.subplots(2,5,figsize=(16,6.8),layout='constrained')
    for row,d in enumerate(display):
        for col,key in enumerate(['real','nohalo','halo']):
            ax=axes[row,col];ax.imshow(np.clip(srgb_encode(np.nan_to_num(d['images'][key])),0,1));ax.set_title({'real':'Real CineStill scan','nohalo':'Same timing: no halation','halo':'Same timing: full halation'}[key],fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(d['name'])
        for col,key,title in [(3,'gainL','L* error reduction (+ improves)'),(4,'gainab','a*b* error reduction (+ improves)')]:
            ax=axes[row,col];mask=(np.load(out/(d['name']+'.npz'))['distance_um']<=800)
            im=ax.imshow(np.ma.masked_where(~mask,d[key]),cmap='RdYlGn',vmin=-10,vmax=10);fig.colorbar(im,ax=ax,shrink=.7);ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Fixed inputs/development/timing; original photos. Residual-change maps at 25 µm measurement scale.',fontsize=11)
    fig.savefig(out/'three_states_and_residuals.png',dpi=190);plt.close(fig)
    # Mean endpoint arrows, with target and intervention separately labelled.
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    selected=['sensor_core','near_edge_0_100','transition_100_300','real_orange_transition_0_300','darker_outer_300_800_Ylt_.4']
    for ax,region in zip(axes,report['regions']):
        for zone,color in zip(selected,['#777777','#c22e2e','#d18a20','#a537a2','#2575a8']):
            r=next(row for row in region['scales']['25um'] if row['zone']==zone)
            if r['status']!='evaluated':continue
            pts={k:np.array(v['mean_Lab'])[1:] for k,v in r['states'].items()};n=pts['nohalo'];h=pts['halo'];target=pts['real']
            ax.scatter(*n,color=color,marker='s');ax.scatter(*h,color=color,marker='o');ax.scatter(*target,color=color,marker='*',s=90,label=zone)
            ax.annotate('',xy=h,xytext=n,arrowprops=dict(arrowstyle='->',color=color,lw=1.8))
            ax.plot([n[0],target[0]],[n[1],target[1]],':',color=color,alpha=.7)
        ax.set_title(region['name']);ax.set_xlabel('a*');ax.set_ylabel('b*');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.legend(fontsize=6)
    fig.suptitle('Squares=nohalo; circles=full; stars=real. Solid V, dotted E. Mean endpoints, not measured halo spectra.',fontsize=10)
    fig.savefig(out/'target_and_added_colour_vectors.png',dpi=190);plt.close(fig)
    print('saved',out)


if __name__=='__main__':main()
