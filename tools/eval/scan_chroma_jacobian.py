"""Cyan-dye scan Jacobian at the existing 99+297 negative states; no fitting.

Perturbation is in physical dye-spectrum coefficients: D(lambda)+/-eps*epsilon_C.
This isolates the model scanner, not a measured scanner or film reaction. Algebraic
Lab outside the RGB gamut is labelled; it is not the displayed colour. Empirical
SATURATION_MATRIX/LAB_TINT is a fixed intervention, never called film physics.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]
LUMA=np.array([.2126,.7152,.0722]);STEPS=[1e-3,1e-4,1e-5]
BINS=['black','dark','transition','bright_transition','near_white']


def linear_lab(rgb):
    """Algebraic linear RGB->Lab; callers explicitly choose any clipping stage."""
    import colour
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.asarray(rgb),apply_cctf_decoding=False))


def condition_original(raw_rgb,scanner_model):
    from filmlab.scanner import SATURATION_MATRIX
    from filmlab.scanning.conditioning import output_tint
    matrix=np.maximum(raw_rgb,0)@SATURATION_MATRIX.T
    output=output_tint(matrix,scanner_model.output_tint)
    return output,matrix


def scan_response(engine,spectral_density,timing):
    from filmlab.scanning.optics import measure,led_response
    from filmlab.scanning.paper import OUTPUT_BLACK
    import colour
    trans=10.**(-np.asarray(spectral_density));sd=measure(trans,engine.scanner.m)
    dn=engine.scanner.conversion(sd)
    result={'scan_D':sd,'Dn':dn,'detector_floor':(trans@led_response(engine.scanner.m))<=1e-6}
    for shoulder,t in [('exp',timing),('paper',replace(timing,x_knee=None,x_white=None))]:
        xe=t.paper_exposure(dn);dp=engine.scanner.digital_paper.density(xe)
        raw=engine.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(dn.shape)
        conditioned,matrix=condition_original(raw,engine.scanner.m)
        result[shoulder+'_xe']=xe;result[shoulder+'_Dp']=dp
        result[shoulder+'_conditioning_matrix_rgb']=matrix
        result[shoulder+'_conditioning_input_L']=linear_lab(np.maximum(matrix,0))[...,0]
        for cond,b in [('off',raw),('on',conditioned)]:
            for domain,rgb in [('raw',b),('nonnegative',np.maximum(b,0)),('display',np.clip(b,0,1))]:
                prefix=shoulder+'_'+cond+'_'+domain
                result[prefix+'_rgb']=rgb
                result[prefix+'_Lab']=linear_lab(rgb)
                result[prefix+'_XYZ']=colour.sRGB_to_XYZ(rgb,apply_cctf_decoding=False)
    return result


def central_difference(before,after,step):return (np.asarray(after)-np.asarray(before))/(2*step)


def crosses_threshold(minus,plus,threshold):
    return ((np.asarray(minus)<=threshold)!=(np.asarray(plus)<=threshold)).any(axis=-1)


def rotation_degrees(a,b,floor=1e-8):
    """Signed a->b angle, undefined for near-zero vectors (NaN)."""
    a=np.asarray(a);b=np.asarray(b)
    angle=np.degrees(np.arctan2(a[...,0]*b[...,1]-a[...,1]*b[...,0],np.sum(a*b,axis=-1)))
    return np.where((np.linalg.norm(a,axis=-1)>floor)&(np.linalg.norm(b,axis=-1)>floor),angle,np.nan)


def med(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def real_gamut_check(engine,timing):
    """Read existing native negative crops only; no RAW decode or redevelopment."""
    from PIL import Image
    from skimage.transform import ProjectiveTransform
    from light_acceptance import crop_warp,pixel_scaling
    from filmlab.scanning.paper import OUTPUT_BLACK
    import oracle_timing as ot
    root=Path('/workspace/filmlab-audit');sid='R1-02-3'
    prior=json.loads((root/'epfl-highlights'/(sid+'.json')).read_text())
    oldreal=np.load(Path(ot.GEO)/('geo_'+sid+'.npz'))['real'];oh,ow=oldreal.shape[:2]
    _,scanpath,_=ot.sources(sid)
    with Image.open(scanpath) as im:w,h=im.size
    raw_shape=(1732,2304);step=round(max(raw_shape)/1400)
    H=pixel_scaling(w/ow,h/oh)@np.array(prior['geometry'])@pixel_scaling(1/step,1/step)
    td=prior['timing']
    from filmlab.scanning.timing import PrintTiming
    fixed=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    rows=[]
    for name,center in [('ceiling_lamp',(676,92)),('middle_ceiling',(706,235)),('glass_reflection',(1437,405))]:
        a=np.load(root/'brightness-local'/(sid+'_'+name+'.npz'));density=a['native_base_spectral_density']
        x0,y0=np.round(np.array(center)*[w/ow,h/oh]).astype(int)-256
        corners=ProjectiveTransform(H).inverse(np.array([[x0,y0],[x0+512,y0],[x0,y0+512],[x0+512,y0+512]]))
        xl,yl=np.maximum(np.floor(corners.min(0)).astype(int)-3,0);xh,yh=np.minimum(np.ceil(corners.max(0)).astype(int)+4,[raw_shape[1],raw_shape[0]])
        if density.shape[:2]!=(yh-yl,xh-xl):raise ValueError('native geometry mismatch')
        matrix=np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])@H@np.array([[1,0,xl],[0,1,yl],[0,0,1]])
        warp=crop_warp(ProjectiveTransform(matrix));dn=engine.scanner.normalized_density(10**(-density.astype(float)))
        dp=engine.scanner.digital_paper.density(fixed.paper_exposure(dn))
        raw=engine.scanner.digital_paper.reference_print(dp,OUTPUT_BLACK).reshape(dn.shape)
        reproduced=warp(np.maximum(raw,0));error=float(np.nanmax(abs(reproduced-a['base_exp'])))
        if error>2e-6:raise ValueError('local gamut baseline not reproduced')
        warped=warp(raw);r_flag=warp((raw[...,0]<0).astype(float))
        for bi in [1,2,3,4]:
            mask=(a['measured']>.5)&(a['distance_um']>0)&(a['distance_um']<=300)&(a['real_bin']==bi)&np.isfinite(warped).all(-1)
            if mask.sum()<30:continue
            rows.append({'region':name,'real_bin':bi,'distance':[0,300],'pixels':int(mask.sum()),'baseline_reproduction_error':error,
                'raw_R_negative_fraction':float((warped[mask,0]<0).mean()),
                'raw_any_negative_fraction':float((warped[mask]<0).any(-1).mean()),
                'raw_any_above_one_fraction':float((warped[mask]>1).any(-1).mean()),
                'negative_native_R_interpolation_weight_median':float(np.median(r_flag[mask])),
                'negative_native_R_contributes_over1percent_fraction':float((r_flag[mask]>.01).mean())})
    return rows


def main():
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    from filmlab.scanning.timing import PrintTiming
    from filmlab.scanner import SATURATION_MATRIX,LAB_TINT
    from nonneutral_chain_audit import chemistry_agent
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/scan-chroma-jacobian'));args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    root=Path('/workspace/filmlab-audit/nonneutral-review/chain');a=np.load(root/'stages.npz');prior=json.loads((root/'results.json').read_text());td=prior['timing']
    generation.use('next');e=Engine(Params(**{**recipe('cinestill_800t'),'timing':'standard','roll':None,'grain':0,'output_conditioning':None}))
    timing=PrintTiming(K=np.array(td['K']),s=np.array(td['s']),M=tuple(np.array(v) for v in td['M']),x_knee=td['x_knee'],x_white=td['x_white'])
    groups=['base','red_0.1','red_0.5','red_1.0'];density=np.concatenate([a[g+'_spectral_density'] for g in groups]);dye=np.concatenate([a[g+'_dye'] for g in groups])
    # Identify full actual dye spectra via the final developer, not a guessed colour axis.
    dye_spectra=e._dye_spectra(chemistry_agent(e));delta=dye_spectra[:,0]
    original=scan_response(e,density,timing);n=len(dye);base_bins=np.searchsorted([.005,.05,.4,.85],a['base_exponential_linear_rgb']@LUMA,side='right')
    fixed_bins=np.tile(base_bins,len(groups));state_groups=np.repeat(np.arange(len(groups)),len(base_bins))
    arrays={'spectral_density':density,'dye':dye,'cyan_spectral_density_per_unit':delta,'original_baseline_bin':fixed_bins,'state_group':state_groups}
    arrays.update({'base_'+k:v for k,v in original.items()});derivatives=[];diagnostics=[]
    # Cached base data must reproduce; otherwise stop before attributing any derivative.
    reproduction={}
    for shoulder,cached in [('exp','exponential'),('paper','paper_only')]:
        expected=np.concatenate([a[g+'_'+cached+'_raw_rgb'] for g in groups])
        error=float(np.max(abs(original[shoulder+'_off_raw_rgb']-expected)));reproduction[shoulder]=error
        if error>2e-10:raise ValueError('cached negative/output mismatch: '+str(error))
    for step in STEPS:
        plus=scan_response(e,density+step*delta,timing);minus=scan_response(e,density-step*delta,timing)
        deriv={k:central_difference(minus[k],plus[k],step) for k in original if original[k].dtype!=bool}
        arrays.update({f'eps_{step}_jac_'+k:v for k,v in deriv.items()});derivatives.append(deriv)
        flags={'physical_cyan_bounds':(dye[:,0]-step>=0)&(dye[:,0]+step<=e.e.coupler_cap[0]),
               'detector_floor_crossing':np.any(minus['detector_floor']!=plus['detector_floor'],axis=-1)}
        for sh in ['exp','paper']:
            paper=e.scanner.digital_paper
            flags[sh+'_paper_curve_crossing']=np.any(np.searchsorted(paper.x,minus[sh+'_xe'])!=np.searchsorted(paper.x,plus[sh+'_xe']),axis=-1)
            flags[sh+'_paper_LUT_crossing']=np.any(np.searchsorted(paper._lut[0],minus[sh+'_Dp'])!=np.searchsorted(paper._lut[0],plus[sh+'_Dp']),axis=-1)
            if sh=='exp':flags[sh+'_shoulder_crossing']=crosses_threshold(minus[sh+'_xe'],plus[sh+'_xe'],timing.x_knee)
            tint_l=np.array(LAB_TINT)[:,0]
            flags[sh+'_tint_anchor_crossing']=np.searchsorted(tint_l,minus[sh+'_conditioning_input_L'])!=np.searchsorted(tint_l,plus[sh+'_conditioning_input_L'])
            flags[sh+'_conditioning_input_clamp_crossing']=crosses_threshold(minus[sh+'_conditioning_matrix_rgb'],plus[sh+'_conditioning_matrix_rgb'],0)
            for cond in ['off','on']:
                prefix=sh+'_'+cond
                rawminus=minus[prefix+'_raw_rgb'];rawplus=plus[prefix+'_raw_rgb']
                flags[prefix+'_black_clamp_crossing']=crosses_threshold(rawminus,rawplus,0)
                flags[prefix+'_white_clamp_crossing']=crosses_threshold(rawminus,rawplus,1)
                # Compare central and one-sided derivatives, not just two step sizes.
                for domain in ['raw','nonnegative','display']:
                    key=prefix+'_'+domain+'_Lab'
                    left=(original[key]-minus[key])/step;right=(plus[key]-original[key])/step
                    scale=np.maximum(np.linalg.norm(deriv[key],axis=-1),1e-6)
                    flags[prefix+'_'+domain+'_one_sided_disagreement']=np.linalg.norm(left-right,axis=-1)/scale>.01
        diagnostics.append(flags);arrays.update({f'eps_{step}_flag_'+k:v for k,v in flags.items()})
    stability={}
    for k in derivatives[-1]:
        jac=derivatives[-1][k];diff=abs(derivatives[-2][k]-jac)
        scale=np.maximum(np.max(abs(jac),axis=-1),1e-6) if jac.ndim>1 else np.maximum(abs(jac),1e-6)
        ratio=np.max(diff,axis=-1)/scale if diff.ndim>1 else diff/scale
        stable=ratio<=.01;arrays['stable_'+k]=stable
        coarse=abs(derivatives[0][k]-derivatives[1][k])
        coarse_scale=np.maximum(np.max(abs(derivatives[1][k]),axis=-1),1e-6) if jac.ndim>1 else np.maximum(abs(derivatives[1][k]),1e-6)
        coarse_ratio=np.max(coarse,axis=-1)/coarse_scale if coarse.ndim>1 else coarse/coarse_scale
        stability[k]={'max_absolute_eps1e4_vs_1e5':float(diff.max()),'over_1percent_count':int((~stable).sum()),
            'max_absolute_eps1e3_vs_1e4':float(coarse.max()),'coarse_over_1percent_count':int((coarse_ratio>.01).sum())}
    rows=[];d=derivatives[-1]
    for scope,selection in [('99_baselines',state_groups==0),('297_existing_perturbations',state_groups>0)]:
        for bi,label in enumerate(BINS):
            selected=selection&(fixed_bins==bi)
            for sh in ['exp','paper']:
                for cond in ['off','on']:
                    for domain in ['raw','nonnegative','display']:
                        prefix=sh+'_'+cond+'_'+domain;dl=d[prefix+'_Lab'];rgb=original[prefix+'_rgb'];drgb=d[prefix+'_rgb'];xyz=original[prefix+'_XYZ']
                        for mode,mask in [('all',selected),('stable',selected&arrays['stable_'+prefix+'_Lab']&(~diagnostics[-1][sh+'_'+cond+'_'+domain+'_one_sided_disagreement'])),('stable_inside_paper_gamut',selected&arrays['stable_'+prefix+'_Lab']&(~diagnostics[-1][sh+'_'+cond+'_'+domain+'_one_sided_disagreement'])&(~((original[sh+'_off_raw_rgb']<0)|(original[sh+'_off_raw_rgb']>1)).any(-1)))]:
                            if not mask.any():continue
                            da=med(dl[mask,1]);db=med(dl[mask,2]);length=np.hypot(da,db)
                            row={'scope':scope,'bin':label,'shoulder':sh,'conditioning':cond,'domain':domain,'selection':mode,'states':int(mask.sum()),
                                 'da_dcyan':da,'db_dcyan':db,'dY_dcyan':med((drgb@LUMA)[mask]),'dab_norm_median':med(np.linalg.norm(dl[mask,1:],axis=-1)),
                                 'direction_of_component_medians_deg':float(np.degrees(np.arctan2(db,da))) if length>1e-8 else None,
                                 'negative_a_fraction':float((dl[mask,1]<-1e-6).mean()),'positive_a_positive_b_fraction':float(((dl[mask,1]>1e-6)&(dl[mask,2]>1e-6)).mean()),
                                 'outside_RGB_display_fraction':float(((rgb[mask]<0)|(rgb[mask]>1)).any(-1).mean()),
                                 'negative_XYZ_fraction':float((xyz[mask]<0).any(-1).mean()),
                                 'Lab_origin_count':int((np.linalg.norm(xyz[mask],axis=-1)<1e-12).sum()),
                                 'source_paper_raw_outside_display_fraction':float(((original[sh+'_off_raw_rgb'][mask]<0)|(original[sh+'_off_raw_rgb'][mask]>1)).any(-1).mean()),
                                 'conditioned_raw_outside_display_fraction':float(((original[sh+'_on_raw_rgb'][mask]<0)|(original[sh+'_on_raw_rgb'][mask]>1)).any(-1).mean()),
                                 'black_clamp_crossing_count':int(diagnostics[-1][sh+'_'+cond+'_black_clamp_crossing'][mask].sum()),
                                 'white_clamp_crossing_count':int(diagnostics[-1][sh+'_'+cond+'_white_clamp_crossing'][mask].sum()),
                                 'rotation_vs_unconditioned_raw_deg_median':med(rotation_degrees(d[sh+'_off_raw_Lab'][mask,1:],dl[mask,1:]))}
                            rows.append(row)
    sign_summary=[]
    for sh in ['exp','paper']:
        source_outside=((original[sh+'_off_raw_rgb']<0)|(original[sh+'_off_raw_rgb']>1)).any(-1)
        for cond in ['off','on']:
            for domain in ['raw','nonnegative','display']:
                k=sh+'_'+cond+'_'+domain+'_Lab';negative=d[k][:,1]<-1e-6
                for scope,selection in [('99_baselines',state_groups==0),('297_existing_perturbations',state_groups>0)]:
                    sign_summary.append({'scope':scope,'shoulder':sh,'conditioning':cond,'domain':domain,
                        'negative_a_count':int((negative&selection).sum()),
                        'negative_a_source_inside_paper_raw_gamut_count':int((negative&selection&(~source_outside)).sum()),
                        'source_outside_paper_raw_gamut_count':int((selection&source_outside).sum())})
    reversal_examples=[]
    base_negative=(state_groups==0)&(d['exp_off_nonnegative_Lab'][:,1]<-1e-6)&(d['exp_off_raw_Lab'][:,1]>0)
    for i in np.flatnonzero(base_negative):
        reversal_examples.append({'baseline_index':int(i),'q':a['base_q'][i].tolist(),
            'baseline_nonnegative_Y':float(original['exp_off_nonnegative_rgb'][i]@LUMA),'original_bin':BINS[fixed_bins[i]],
            'raw_RGB':original['exp_off_raw_rgb'][i].tolist(),'raw_XYZ':original['exp_off_raw_XYZ'][i].tolist(),
            'raw_Lab':original['exp_off_raw_Lab'][i].tolist(),'raw_da_dcyan':float(d['exp_off_raw_Lab'][i,1]),
            'nonnegative_da_dcyan':float(d['exp_off_nonnegative_Lab'][i,1])})
    local_gamut=real_gamut_check(e,timing)
    report={'scope':__doc__,'states':n,'base_states':99,'existing_red_exposure_states':297,'perturbation_axis':'cyan dye coefficient using actual CD-4 full81 spectrum; no dye/development refit',
            'generation':'next','grain':0,'roll':None,'timing':td,'eps':STEPS,'bins':BINS,'bin_policy':'unperturbed original exponential nonnegative linear RGB Y bins, fixed across existing state groups and every intervention',
            'reproduction_max_abs_raw_RGB':reproduction,'conditioning_matrix':SATURATION_MATRIX.tolist(),'conditioning_tint':LAB_TINT,
            'finite_difference_stability':stability,'rows':rows,'negative_a_stage_summary':sign_summary,
            '99_baseline_black_clamp_reversal_examples':reversal_examples,'real_local_gamut_check':local_gamut,
            'physical_cyan_bounds_valid_all_steps':bool(all(f['physical_cyan_bounds'].all() for f in diagnostics)),
            'limits':['only scanner model derivative, not independently measured Frontier response','negative linear RGB means outside this RGB gamut, not nonphysical spectrum or invalid XYZ; raw Lab from nonnegative physical XYZ is valid colourimetry but not the clipped display colour',
                      'on_raw includes original preconditioning nonnegative clamps; on_nonnegative is an extra diagnostic clamp, not a different calibrated pipeline',
                      'near-zero chroma derivative direction undefined; rotation statistics omit norms below1e-8',
                      'stable means eps1e4 vs1e5 agree within1percent at norm floor1e-6 and one-sided derivatives agree; flags retain LUT/curve/clamp crossings',
                      'no quantization enters any derivative; 396 states are correlated existing states, not396 independent measurements']}
    np.savez_compressed(args.out/'stages.npz',**arrays)
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plot(report,args.out)
    print('saved',args.out,'reproduction',reproduction)


def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,sh in enumerate(['exp','paper']):
        for cond,style in [('off','-'),('on','--')]:
            for domain,color in [('raw','#c28423'),('nonnegative','#307db5'),('display','#b02f50')]:
                rows=[r for r in report['rows'] if r['scope']=='99_baselines' and r['shoulder']==sh and r['conditioning']==cond and r['domain']==domain and r['selection']=='stable']
                indexed={r['bin']:r for r in rows};x=np.arange(5)
                for j,key in enumerate(['da_dcyan','db_dcyan']):
                    axes[j,col].plot(x,[indexed.get(b,{}).get(key,np.nan) for b in BINS],style,marker='o',color=color,label=cond+':'+domain)
                    axes[j,col].set_xticks(x,['<.005','.005–.05','.05–.4','.4–.85','≥.85'])
                    axes[j,col].set_title(sh+' / '+('a*' if j==0 else 'b*')+' derivative')
                    axes[j,col].set_ylabel('Lab units / cyan dye coefficient');axes[j,col].axhline(0,color='black',lw=.5)
    axes[0,0].legend(fontsize=7);fig.suptitle('Pure scanning Jacobian: cyan dye → full spectrum → fixed scanner → Lab (stable baselines)')
    fig.savefig(out/'scan_chroma_jacobian.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
