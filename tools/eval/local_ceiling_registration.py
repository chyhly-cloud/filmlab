"""Local ceiling-plane geometry from RAW/scan structure, never lamp-colour fitting.

Only remote dark-grid corners select an affine residual. Geometric validation
precedes any fixed-model colour evaluation. Fixture parallax is not measured.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from skimage.feature import corner_harris, corner_peaks, match_template
from skimage.measure import ransac
from skimage.transform import AffineTransform, ProjectiveTransform

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def structure(rgb):
    """Dark-ridge log-luminance; scalar exposure disappears before matching."""
    y=np.asarray(rgb)@np.array([.2126,.7152,.0722])
    valid=np.isfinite(y)&(y>0)
    if not valid.any():raise ValueError('No positive finite structure input')
    log=np.log(np.maximum(np.where(valid,y,np.median(y[valid])),1e-8))
    ridge=np.maximum(gaussian_filter(log,12)-gaussian_filter(log,1.5),0.)
    return ridge


def patch_match(source,target,point,search=16,radius=12):
    """Integer centres (y,x); source patch vs local target, no source-target warp."""
    y,x=map(int,point);h,w=source.shape
    if min(y,x,h-1-y,w-1-x)<search+radius:return None
    patch=source[y-radius:y+radius+1,x-radius:x+radius+1]
    window=target[y-radius-search:y+radius+search+1,x-radius-search:x+radius+search+1]
    if np.std(patch)<1e-5:return None
    scores=match_template(window,patch,pad_input=False)
    iy,ix=np.unravel_index(np.argmax(scores),scores.shape)
    mask=np.ones(scores.shape,bool);mask[max(0,iy-3):iy+4,max(0,ix-3):ix+4]=False
    runner=float(scores[mask].max())
    score=float(scores[iy,ix]);margin=score-runner
    if score<.6 or margin<.025:return None
    matched=np.array([y+iy-search,x+ix-search])
    return matched,score,margin


def estimate(source_xy,target_xy,train):
    """Train-only robust affine plus predeclared independent landmark checks."""
    p,q,train=np.asarray(source_xy,float),np.asarray(target_xy,float),np.asarray(train,bool)
    record={'training_points':int(train.sum()),'validation_points':int((~train).sum()),
            'adopt_for_diagnostic':False}
    if min(train.sum(),(~train).sum())<6:
        record['status']='insufficient_geometric_correspondences';return None,record
    model,inliers=ransac((p[train],q[train]),AffineTransform,min_samples=3,
                          residual_threshold=2.,max_trials=1000,rng=20261021)
    if model is None:
        record['status']='affine_not_estimated';return None,record
    initial=np.linalg.norm(p-q,axis=1);error=np.linalg.norm(model(p)-q,axis=1)
    sv=np.linalg.svd(model.params[:2,:2],compute_uv=False)
    center_shift=model(np.array([[256.,256.]]))[0]-[256.,256.]
    angle=float(np.degrees(np.arctan2(model.params[1,0],model.params[0,0])))
    spread=np.ptp(p[train],axis=0)
    hull=ConvexHull(p[train])
    inside=bool(np.all(hull.equations[:,:2]@np.array([256.,256.])+hull.equations[:,2]<=1e-8))
    condition=float(np.linalg.cond(np.c_[(p[train]-256)/256,np.ones(train.sum())]))
    held_initial=float(np.median(initial[~train]));held=float(np.median(error[~train]))
    checks={'train_inliers_at_least_six':int(inliers.sum())>=6,
            'spans_both_axes_150px':bool(np.all(spread>=150)),
            'landmarks_not_near_collinear':condition<=20,
            'center_inside_training_hull':inside,
            'validation_median_at_most_1p5px':held<=1.5,
            'validation_p90_at_most_3px':float(np.percentile(error[~train],90))<=3.,
            'validation_improves_20percent':held_initial>.75 and held<=.8*held_initial,
            'small_affine':bool(np.all((sv>=.97)&(sv<=1.03)) and abs(angle)<=3 and np.linalg.norm(center_shift)<=16)}
    record.update(status='validated_geometry' if all(checks.values()) else 'geometry_not_accepted',
                  adopt_for_diagnostic=bool(all(checks.values())),checks=checks,
                  matrix_raw_warp_to_scan=model.params.tolist(),
                  center_displacement_xy_scan_pixels=center_shift.tolist(),singular_values=sv.tolist(),
                  rotation_degrees=angle,inliers=inliers.tolist(),
                  training_design_condition_number=condition,
                  initial_residual_pixels=initial.tolist(),fitted_residual_pixels=error.tolist(),
                  validation_initial_median_px=held_initial,validation_fitted_median_px=held,
                  validation_fitted_p90_px=float(np.percentile(error[~train],90)))
    return model,record


def run():
    from PIL import Image,ImageDraw,ImageFont
    from light_acceptance import crop_warp
    from inverse_source_radiance import engine_for,observation_operator
    from brightness_response_audit import lab
    from measurement_aperture_audit import native_support_overlap
    from catalogue_source_inverse import MomentCase
    from filmlab.spectral import srgb_encode
    root=Path('/workspace/filmlab-audit');out=root/'local-ceiling-registration';out.mkdir(exist_ok=True)
    report={'scope':__doc__,'criteria_fixed_before_colour_evaluation':True,'regions':[],
            'source_selection':'RAW Harris corners only, distance>800 nominal um, border40, no fixture/colour targets.',
            'fit_validation_split':'Fixed 128px spatial checkerboard, point-coordinate parity; no lamp-region RGB or model output selects correspondence/fit.',
            'limits':['Only ceiling-plane alignment is constrained; fixture parallax needs separate unsaturated structural landmarks.',
                      'Local affine scale is scene registration, not physical film-mm calibration or kernel correction.',
                      'Dark-ridge photometric matching and periodic grids can fail; rejected transforms are not used for colour assessment.',
                      'Targets and spectra were already inspected/fitted; this is a geometry control, not external physical validation.']}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
    for name in ['ceiling_lamp','middle_ceiling']:
        a=np.load(root/'inverse-source-radiance'/(name+'_source.npz'))
        edge=np.load(root/'lamp-edge-input-audit'/(name+'.npz'))
        raw=crop_warp(ProjectiveTransform(a['matrix']))(edge['preprior_rgb'])
        raw_structure,real_structure=structure(raw),structure(a['real'])
        yy,xx=np.indices(raw_structure.shape)
        allowed=(a['distance_um']>800)&a['supported']&(yy>=40)&(yy<472)&(xx>=40)&(xx<472)
        response=corner_harris(raw_structure,method='k',sigma=1.5)
        response[~allowed]=0
        points=corner_peaks(response,min_distance=24,threshold_rel=.03,num_peaks=100,exclude_border=40)
        matches=[]
        for yx in points:
            match=patch_match(raw_structure,real_structure,yx)
            if match is None:continue
            p,score,margin=match
            back=patch_match(real_structure,raw_structure,p)
            if back is None or np.linalg.norm(back[0]-yx)>1.5:continue
            matches.append({'raw_yx':yx.tolist(),'real_yx':p.tolist(),'NCC':score,'peak_margin':margin})
        source=np.array([m['raw_yx'][::-1] for m in matches]).reshape(-1,2)
        target=np.array([m['real_yx'][::-1] for m in matches]).reshape(-1,2)
        train=((source[:,0]//128+source[:,1]//128).astype(int)%2)==0
        transform,geometry=estimate(source,target,train)
        row={'name':name,'raw_corners':len(points),'correspondences':matches,'training_flags':train.tolist(),'geometry':geometry,
             'fixture_parallax':'Not independently constrained by these ceiling-grid matches.'}
        # Geometric correspondence overlays are display-only per-image normalized
        # structure fields. No normalized image enters the colour evaluation.
        sheet=Image.new('RGB',(1024,560),(18,18,18));draw=ImageDraw.Draw(sheet)
        for col,(title,values) in enumerate([('RAW天花板格线',raw_structure),('真实扫描格线',real_structure)]):
            value=np.clip(values/max(np.percentile(values,99),1e-8),0,1)
            sheet.paste(Image.fromarray(np.uint8(np.rint(value*255))).convert('RGB'),(512*col,48))
            draw.text((512*col+8,12),title,font=font,fill='white')
        for i,(p,q) in enumerate(zip(source,target)):
            color='lime' if train[i] else 'orange'
            for col,pt in enumerate([p,q]):
                x,y=pt+np.array([col*512,48]);draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                draw.text((x+5,y),str(i),fill=color)
        sheet.save(out/(name+'_structure_matches.jpg'),quality=98)
        np.savez_compressed(out/(name+'_geometry.npz'),source_xy=source,target_xy=target,train=train,
                            raw_structure=raw_structure,real_structure=real_structure,allowed=allowed)
        if geometry['adopt_for_diagnostic']:
            case=MomentCase(name);e=case.e;matrix=transform.params@a['matrix']
            P=observation_operator(case.a['source_profile'].shape,matrix,case.a['points'],25/case.meta['scan_um_per_pixel'])
            overlap=native_support_overlap(P,np.isin(case.a['groups'],[0,2]),np.isin(case.a['groups'],[1,3]))
            row['new_geometry_native_overlap_audit']=overlap[0]
            states={
                'corrected_inputs_old_core':np.load(root/'raw-hdr-film-replay'/(name+'.npz'))['q'],
                'fixed112':np.load(root/'catalogue-source-inverse'/(name+'_catalogue.npz'))['q'],
                'free_source_field':np.load(root/'censored-source-field'/(name+'.npz'))['q']}
            row['fixed_states_without_refit']={};images=[]
            for key,q in states.items():
                st=e._develop(q,blur_px=12/e.px_um);st.dye*=case.a['factor'][...,None]
                rgb=e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(st)))
                prediction=lab(P@rgb.reshape(-1,3))
                previous=lab(case.P@rgb.reshape(-1,3))
                row['fixed_states_without_refit'][key]={'old_geometry':case.stats(previous),'ceiling_geometry':case.stats(prediction)}
                images.append((key,crop_warp(ProjectiveTransform(matrix))(rgb)))
            comparison=Image.new('RGB',(2048,560),(18,18,18));d=ImageDraw.Draw(comparison)
            for col,(label,values) in enumerate([('真实800T扫描',a['real']),*images]):
                picture=Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(values))*255)))
                comparison.paste(picture,(col*512,48));d.text((col*512+8,12),label,font=font,fill='white')
            comparison.save(out/(name+'_fixed_states_comparison.jpg'),quality=98)
            row['new_matrix']=matrix.tolist()
        report['regions'].append(row)
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,json.dumps(geometry,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':run()
