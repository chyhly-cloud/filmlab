"""Ceiling affine from line-normal displacements, held-out corner validation.

Straight lines constrain only their normal coordinate. No fixture core, halo
colour, or film prediction chooses the transform or its acceptance thresholds.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from skimage.transform import ProjectiveTransform

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/eval')]


def normal_design(points_xy,normals_xy):
    p=(np.asarray(points_xy)-256)/256
    n=np.asarray(normals_xy)
    return np.c_[n[:,0]*p[:,0],n[:,0]*p[:,1],n[:,0],
                 n[:,1]*p[:,0],n[:,1]*p[:,1],n[:,1]]


def theta_matrix(theta):
    t=np.asarray(theta)
    delta=np.array([[t[0]/256,t[1]/256,t[2]-t[0]-t[1]],
                    [t[3]/256,t[4]/256,t[5]-t[3]-t[4]],[0,0,0]])
    return np.eye(3)+delta


def fit_normal_affine(design, displacement):
    # scipy's robust loss may rescale a Jacobian in place; retain the fixed
    # geometric design used by the residual and every subsequent evaluation.
    return least_squares(lambda t:design@t-displacement,np.zeros(6),
                         jac=lambda t:design.copy(),loss='soft_l1',f_scale=.5,
                         ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=200)


def line_candidates(values,allowed):
    hxx=gaussian_filter(values,2,order=(0,2));hyy=gaussian_filter(values,2,order=(2,0))
    hxy=gaussian_filter(values,2,order=(1,1))
    h=np.stack([np.stack([hxx,hxy],-1),np.stack([hxy,hyy],-1)],-2)
    eigen,vectors=np.linalg.eigh(h)
    anisotropy=(eigen[...,0]<-1e-5)&(abs(eigen[...,0])>5*abs(eigen[...,1]))
    score=np.where(allowed&anisotropy,-eigen[...,0],0.)
    proposed=[]
    for y in range(40,472,24):
        for x in range(40,472,24):
            region=score[y:y+24,x:x+24]
            iy,ix=np.unravel_index(np.argmax(region),region.shape)
            if region[iy,ix]<=1e-5:continue
            py,px=y+iy,x+ix
            proposed.append((float(region[iy,ix]),np.array([px,py],float),vectors[py,px,:,0]))
    selected=[]
    for item in sorted(proposed,key=lambda x:-x[0]):
        if all(np.linalg.norm(item[1]-old[1])>=16 for old in selected):selected.append(item)
    return selected


def line_shift(raw,real,point_xy,normal_xy):
    p,n=np.asarray(point_xy),np.asarray(normal_xy)
    tangent=np.array([-n[1],n[0]])
    offset=np.arange(-10,11,dtype=float)
    side=np.arange(-8,9,dtype=float)
    def profile(image,shift):
        coords=p+(offset[:,None,None]+shift)*n+side[None,:,None]*tangent
        return map_coordinates(image,[coords[...,1],coords[...,0]],order=1,mode='nearest').mean(1)
    reference=profile(raw,0);reference-=reference.mean();denom=np.linalg.norm(reference)
    if denom<1e-5:return None
    shifts=np.arange(-16,17,dtype=float);scores=[]
    for s in shifts:
        q=profile(real,s);q-=q.mean();scores.append(float(reference@q/(denom*np.linalg.norm(q)+1e-20)))
    scores=np.array(scores);i=int(np.argmax(scores))
    if i in [0,len(scores)-1]:return None
    keep=np.ones(len(scores),bool);keep[max(0,i-3):i+4]=False
    if scores[i]<.8 or scores[i]-scores[keep].max()<.025:return None
    curvature=scores[i-1]-2*scores[i]+scores[i+1]
    refined=shifts[i]+(.5*(scores[i-1]-scores[i+1])/curvature if curvature<0 else 0)
    return float(refined),float(scores[i]),float(scores[i]-scores[keep].max())


def run():
    from local_ceiling_registration import structure
    from light_acceptance import crop_warp
    root=Path('/workspace/filmlab-audit');out=root/'ceiling-line-registration';out.mkdir(exist_ok=True)
    corners=json.loads((root/'local-ceiling-registration/results.json').read_text())
    report={'scope':__doc__,'protocol':'Line normal NCC>=.8, peak margin>=.025, search16. At least18 normals, design condition<=20; >=6 held-out corners, median<=1.5px, p90<=3px and20% baseline improvement. Small affine limits unchanged.',
            'regions':[],'material_or_output_parameters_changed':False}
    for name in ['ceiling_lamp','middle_ceiling']:
        a=np.load(root/'inverse-source-radiance'/(name+'_source.npz'))
        captured=np.load(root/'local-ceiling-registration'/(name+'_geometry.npz'))
        raw,real=captured['raw_structure'],captured['real_structure']
        allowed=captured['allowed'].copy()
        # Exclude line-fit centres within 40px of corner centres. Gaussian,
        # profile and search footprints are not pixel-level independent.
        yy,xx=np.indices(allowed.shape)
        for p in captured['source_xy']:
            allowed&=(xx-p[0])**2+(yy-p[1])**2>40**2
        candidates=line_candidates(raw,allowed)
        observations=[]
        for score,p,n in candidates:
            matched=line_shift(raw,real,p,n)
            if matched is None:continue
            displacement,ncc,margin=matched
            observations.append({'point_xy':p.tolist(),'normal_xy':n.tolist(),'normal_displacement':displacement,'NCC':ncc,'peak_margin':margin})
        row={'name':name,'candidate_lines':len(candidates),'line_observations':observations,'adopt_for_diagnostic':False}
        if len(observations)<18:
            row['status']='insufficient_normal_measurements';report['regions'].append(row);continue
        p=np.array([x['point_xy'] for x in observations]);n=np.array([x['normal_xy'] for x in observations]);d=np.array([x['normal_displacement'] for x in observations])
        D=normal_design(p,n);condition=float(np.linalg.cond(D))
        fit=fit_normal_affine(D,d)
        A=theta_matrix(fit.x);transform=ProjectiveTransform(A)
        pc,qc=captured['source_xy'],captured['target_xy']
        initial=np.linalg.norm(pc-qc,axis=1);after=np.linalg.norm(transform(pc)-qc,axis=1)
        sv=np.linalg.svd(A[:2,:2],compute_uv=False)
        center_shift=transform(np.array([[256.,256.]]))[0]-[256.,256.]
        angle=float(np.degrees(np.arctan2(A[1,0],A[0,0])))
        hull=ConvexHull(p)
        center_inside=bool(np.all(hull.equations[:,:2]@np.array([256.,256.])+hull.equations[:,2]<=1e-8))
        checks={'enough_normal_measurements':len(p)>=18,'well_conditioned_six_parameters':condition<=20,
                'center_inside_fit_hull':center_inside,'at_least_six_withheld_corners':len(pc)>=6,
                'corner_median_at_most_1p5px':bool(len(pc) and np.median(after)<=1.5),
                'corner_p90_at_most_3px':bool(len(pc) and np.percentile(after,90)<=3),
                'corner_error_improves_20percent':bool(len(pc) and np.median(after)<=.8*np.median(initial)),
                'small_affine':bool(np.all((sv>=.97)&(sv<=1.03)) and abs(angle)<=3 and np.linalg.norm(center_shift)<=16)}
        row.update(status='geometry_accepted' if all(checks.values()) else 'geometry_not_accepted',checks=checks,
                   adopt_for_diagnostic=bool(all(checks.values())),matrix_raw_warp_to_scan=A.tolist(),
                   design_condition=condition,solver_success=bool(fit.success),normal_fit_residual=(D@fit.x-d).tolist(),
                   heldout_corner_residual_before=initial.tolist(),heldout_corner_residual_after=after.tolist(),
                   heldout_corner_median_before=float(np.median(initial)),heldout_corner_median_after=float(np.median(after)),
                   center_displacement_xy_scan_pixels=center_shift.tolist(),singular_values=sv.tolist(),rotation_degrees=angle,
                   fixture_parallax='Not independently measured; housing mask is not a set of calibrated unsaturated structural points.')
        folds=[]
        quadrant=(p[:,0]>=256).astype(int)+2*(p[:,1]>=256).astype(int)
        for omitted in range(4):
            keep=quadrant!=omitted
            if keep.sum()<18:continue
            f=fit_normal_affine(D[keep],d[keep]);af=theta_matrix(f.x);tf=ProjectiveTransform(af)
            folds.append({'omitted_quadrant':omitted,'retained_normal_constraints':int(keep.sum()),
                          'matrix':af.tolist(),'center_displacement_xy':(tf(np.array([[256.,256.]]))[0]-[256.,256.]).tolist(),
                          'withheld_corner_median_error_px':float(np.median(np.linalg.norm(tf(pc)-qc,axis=1))),
                          'scope':'Geometric leave-one-block-out sensitivity only; no fold chosen by lamp colour or used instead of full-fit A.'})
        row['spatial_line_fit_sensitivity']=folds
        np.savez_compressed(out/(name+'.npz'),points_xy=p,normals_xy=n,displacements=d,design=D,theta=fit.x,matrix=A,corner_source=pc,corner_target=qc)
        report['regions'].append(row)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in report['regions']:
        print(r['name'],r['status'],r.get('checks'),r.get('heldout_corner_median_before'),r.get('heldout_corner_median_after'),flush=True)
    return report


if __name__=='__main__':run()
