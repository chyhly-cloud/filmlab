"""A shared spectral source constrained to a predeclared catalogue cone.

Optimization coordinates are six direct/return responses, but linear cone
constraints require ONE nonnegative catalogue spectrum to produce all six.
No independently tuned layer/halo response is accepted as a physical candidate.
Known source geometry, corrected background, film, timing and scanner are fixed.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize, linprog, least_squares
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]


def cone_halfspaces(rays, pivot=1):
    """Exact finite-ray cone via a positive-pivot projective cross-section.

    If every ray has positive pivot, its normalized section is a convex hull.
    Multiplying each hull inequality by the nonnegative pivot returns linear
    homogeneous inequalities A @ moment <= 0. No target data enter this step.
    """
    rays = np.asarray(rays, float)
    if rays.ndim != 2 or not np.isfinite(rays).all() or np.any(rays[:, pivot] <= 0):
        raise ValueError('Finite rays with a strictly positive common pivot required')
    other = np.delete(np.arange(rays.shape[1]), pivot)
    section = rays[:, other]/rays[:, pivot, None]
    scale = section.std(0)
    if np.any(scale == 0):
        raise ValueError('Full-dimensional section required')
    hull = ConvexHull(section/scale)
    A = np.zeros((len(hull.equations), rays.shape[1]))
    A[:, other] = hull.equations[:, :-1]/scale
    A[:, pivot] = hull.equations[:, -1]
    A /= np.linalg.norm(A, axis=1)[:, None]
    return A


def spectral_witness(responses, moments, camera=None, required=None, maximize_input_Y=False):
    """Same-spectrum witness; optional Y extrema describe degeneracy, not SI radiance."""
    S, m = np.asarray(responses, float), np.asarray(moments, float)
    scale = np.maximum(abs(m), np.maximum(S.max(0), 1e-12))
    A, b = (None, None) if camera is None else (-np.asarray(camera).T, -np.asarray(required))
    objective = (-1 if maximize_input_Y else 1)*np.ones(len(S))
    result = linprog(objective, A_ub=A, b_ub=b, A_eq=S.T/scale[:, None],
                     b_eq=m/scale, bounds=(0, None), method='highs',
                     options={'primal_feasibility_tolerance': 1e-8,
                              'dual_feasibility_tolerance': 1e-8})
    return result


def apply_accepted_ceiling_geometry(case):
    """Apply a RAW/scan-structure-only accepted affine, with coverage checks."""
    from inverse_source_radiance import observation_operator
    from measurement_aperture_audit import native_support_overlap
    record=json.loads((case.root/'ceiling-line-registration/results.json').read_text())
    accepted=[x for x in record['regions'] if x['name']==case.name and x['adopt_for_diagnostic']]
    if len(accepted)!=1:
        raise ValueError('No independently accepted ceiling geometry for this lamp')
    A=np.array(accepted[0]['matrix_raw_warp_to_scan'])
    matrix=A@case.a['matrix']
    P=observation_operator(case.a['source_profile'].shape,matrix,case.a['points'],25/case.meta['scan_um_per_pixel'])
    overlap,_=native_support_overlap(P,np.isin(case.a['groups'],[0,2]),np.isin(case.a['groups'],[1,3]))
    if not overlap['strictly_disjoint']:
        raise ValueError('Geometry update breaks native measurement separation')
    source=np.load(case.root/'inverse-source-radiance'/(case.name+'_source.npz'))
    if not source['safe'].ravel()[np.unique(P.indices)].all():
        raise ValueError('Geometry update requires pixels outside audited safe crop')
    case.a['matrix']=matrix;case.P=P;case.indices=np.unique(P.indices);case.Ps=P[:,case.indices]
    return {'mode':'independently_validated_ceiling_geometry','affine':A.tolist(),
            'new_native_overlap_audit':overlap,'native_safe_coverage':'All required pixels audited safe',
            'fixture_parallax':'Not independently measured; this aligns the ceiling plane only.'}


class MomentCase:
    def __init__(self, name, ceiling_geometry=False):
        from filmlab import film
        from inverse_source_radiance import engine_for, observation_operator
        from brightness_response_audit import lab
        self.lab = lab
        self.name = name
        self.root = Path('/workspace/filmlab-audit')
        self.a = dict(np.load(self.root/'joint-radius-source-inverse'/(name+'_case.npz')))
        self.e, self.meta = engine_for()
        a, e = self.a, self.e
        profile3 = np.repeat(a['source_profile'][..., None], 3, axis=-1)
        direct = film.emulsion_scatter(profile3, e.px_um)
        returning = film.halation(np.zeros_like(profile3), profile3, e.px_um, e.v,
                                   film.bounce_retention(e.e.stock, e.v))
        self.B = np.array([direct*np.eye(3)[i] for i in range(3)]
                         + [returning*np.eye(3)[i] for i in range(3)])
        self.S10 = np.c_[a['direct_response'], a['return_response']]
        self.factorization_error = float(np.max(abs(np.einsum('km,mhwc->khwc', self.S10, self.B)-a['basis'])))
        if self.factorization_error > 1e-10:
            raise ValueError('Six-response factorization does not reproduce the original source')
        self.m0 = a['baseline_core_coefficients'] @ self.S10
        self.scale = np.maximum(self.m0, 1e-6)
        self.P = observation_operator(a['source_profile'].shape, a['matrix'], a['points'],
                                      25/self.meta['scan_um_per_pixel'])
        self.indices = np.unique(self.P.indices); self.Ps = self.P[:, self.indices]
        self.counter = 0
        self.geometry={'mode':'original_fixed_geometry'}
        if ceiling_geometry:
            self.geometry=apply_accepted_ceiling_geometry(self)

    def render(self, m, full=False):
        self.counter += 1
        a, e = self.a, self.e
        H = a['background'] + np.einsum('m,mhwc->hwc', m, self.B, optimize=True)
        q = e._log_exposure(np.maximum(H, 1e-12))
        state = e._develop(q, blur_px=12/e.px_um); state.dye *= a['factor'][..., None]
        if full:
            from skimage.transform import ProjectiveTransform
            from light_acceptance import crop_warp
            rgb = e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(state)))
            return crop_warp(ProjectiveTransform(a['matrix']))(rgb), q
        sub = replace(state, **{key: getattr(state, key).reshape(-1, 3)[self.indices]
                                for key in ['dye', 'silver', 'halide']})
        rgb = e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(sub)))
        return self.lab(self.Ps @ rgb)

    def residual(self, z):
        prediction = self.render(z*self.scale)
        return np.concatenate([(prediction[self.a['groups'] == group]-self.a['target_lab'][self.a['groups'] == group]).ravel()
                               /np.sqrt(2*3*np.count_nonzero(self.a['groups'] == group)) for group in [0, 2]])

    def jacobian(self, z):
        base = self.residual(z); J = []
        for k in range(6):
            step = max(abs(z[k])*1e-3, 1e-5)
            changed = z.copy(); changed[k] += step
            J.append((self.residual(changed)-base)/step)
        return np.array(J).T

    def stats(self, prediction):
        import colour
        return {label: {'samples': int(np.count_nonzero(self.a['groups'] == group)),
                        'Lab_RMSE': float(np.sqrt(np.mean((prediction[self.a['groups'] == group]-self.a['target_lab'][self.a['groups'] == group])**2))),
                        'median_deltaE00': float(np.median(colour.delta_E(prediction[self.a['groups'] == group], self.a['target_lab'][self.a['groups'] == group], method='CIE 2000'))),
                        'mean_model_minus_real_Lab': (prediction[self.a['groups'] == group]-self.a['target_lab'][self.a['groups'] == group]).mean(0).tolist()}
                for group, label in enumerate(['inner_train','inner_heldout','outer_train','outer_heldout'])}


def fit(name, out, starts=8, relaxed=False, ceiling_geometry=False):
    case = MomentCase(name, ceiling_geometry)
    root = case.root
    library = dict(np.load(root/'spectral-library-coverage/normalized_spectra.npz'))
    a = case.a
    # Reuse independently declared SHAPES, not a different nominal film-roll
    # response or pre-generation white-reference normalization from its audit.
    from filmlab import lights
    from filmlab.incident import spectral_responses
    spectra = library['spectra'].copy()
    spectra /= (lights.camera_rgb(spectra) @ np.array([.2126,.7152,.0722]))[:, None]
    direct, returning = spectral_responses(case.e)
    S = np.c_[spectra @ direct, spectra @ returning] * 2**case.e.p.exposure_ev
    # Spectrum universe was declared without reading real scans; numerical
    # responses are always recomputed for the unchanged actual scene instance.
    if np.max(abs(S[library['original10_indices']]-case.S10)) > 1e-10:
        raise ValueError('Catalogue and original response units differ')
    edge = np.load(root/'lamp-edge-input-audit'/(name+'.npz'))
    C = lights.camera_rgb(spectra) @ np.linalg.inv(edge['cam_to_rgb'])/float(edge['scale'])
    required = np.max(a['source_camera']/a['source_profile'][a['support'], None], axis=0)
    if np.any(C <= 0):
        raise ValueError('These catalogue camera-proxy witnesses require positive broad-source responses')
    A = cone_halfspaces(S/case.scale)
    initial = [np.ones(6), (a['baseline_core_coefficients'].sum()*S.mean(0))/case.scale]
    rng = np.random.default_rng(20261023)
    for i in range(starts-len(initial)):
        mixture = rng.dirichlet(np.ones(len(S))*.2)
        initial.append((a['baseline_core_coefficients'].sum()*2**rng.uniform(-1, 1)*(mixture @ S))/case.scale)
    started = time.time(); trials = []
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/(name+'_scene_specific_library.npz'), spectra=spectra,
                        response6=S, camera_response=C, names=library['names'],
                        nominal_response6_not_used=library['response6'])
    mode = 'relaxed_nonphysical_diagnostic' if relaxed else 'catalogue'
    last = {}

    def objective(z):
        r = case.residual(z)
        last['z'], last['r'] = z.copy(), r
        return float(r@r)

    def gradient(z):
        r = last['r'] if 'z' in last and np.array_equal(z, last['z']) else case.residual(z)
        return 2*r @ case.jacobian(z)

    for i, z0 in enumerate(initial):
        if relaxed:
            sol = least_squares(case.residual, z0, bounds=(np.zeros(6), np.full(6, 4096.)),
                                jac=case.jacobian, x_scale='jac', max_nfev=100,
                                ftol=2e-8, xtol=2e-8, gtol=2e-6)
            value = float(np.sum(case.residual(sol.x)**2))
            iterations = sol.nfev
        else:
            sol = minimize(objective, z0, method='SLSQP', jac=gradient,
                           bounds=[(0., 4096.)]*6,
                           constraints=[{'type':'ineq','fun':lambda z: -A@z,'jac':lambda z: -A}],
                           options={'ftol':1e-8,'maxiter':100})
            value, iterations = float(sol.fun), sol.nit
        z = np.maximum(sol.x, 0.)
        witness = spectral_witness(S, z*case.scale, C, required)
        row = {'start':i,'train_equal_zone_MSE':value,'success':bool(sol.success),
               'message':str(sol.message),'iterations':int(iterations),
               'moments':(z*case.scale).tolist(),'normalized_moments':z.tolist(),
               'maximum_cone_facet_violation':float(max(0.,np.max(A@z))),
               'same_spectrum_camera_proxy_witness_feasible':bool(witness.success),
               'witness_coefficients':np.maximum(witness.x, 0.).tolist() if witness.success else None,
               'raw_LP_minimum_coefficient':float(witness.x.min()) if witness.success else None,
               'numerical_upper_active':bool(np.any(z>4096*.999))}
        trials.append(row)
        (out/(name+'_'+mode+'_partial.json')).write_text(json.dumps({'trials':trials,'forward_evaluations':case.counter},indent=2)+'\n')
        print(name,mode,'start',i,'loss',value,'cone',row['maximum_cone_facet_violation'],'witness',witness.success,'eval',case.counter,flush=True)
    accepted = trials if relaxed else [t for t in trials if t['same_spectrum_camera_proxy_witness_feasible'] and t['maximum_cone_facet_violation'] < 2e-6]
    if not accepted:
        raise ValueError('No camera-proxy-feasible common-spectrum candidate; keep partial numerical results')
    best = min(accepted,key=lambda t:t['train_equal_zone_MSE'])
    m = np.array(best['moments']); prediction = case.render(m)
    old = case.render(case.m0)
    image, q = case.render(m, True)
    result = {'name':name,'mode':mode,'scope':__doc__,
              'geometry':case.geometry,
              'relaxation_is_not_a_physical_candidate':bool(relaxed),
              'library_count':len(S),'cone_facets':len(A),'response_factorization_max':case.factorization_error,
              'scene_response_instance':{'roll':case.e.p.roll,'frame':case.e.p.frame,
                                         'white_reference':lights.WB_REFERENCE,
                                         'original10_response_reproduction_max':float(np.max(abs(S[library['original10_indices']]-case.S10)))},
              'background':'Fixed measured partial minY edge + actual HDR full-measurement body + all other sources.',
              'training':'Same predeclared equal-zone near/outer training objective and independently audited 19px sparse-P separation. Heldout never selects fit.',
              'trials':trials,'selected_by_train_only':best,'inherited_corrected_core_inverse':case.stats(old),
              'fitted':case.stats(prediction),'total_forward_evaluations':case.counter,
              'elapsed_seconds':time.time()-started,
              'source_claim':'Common nonnegative standard-spectrum witness, not measured lamp SPD or calibrated SI radiance.' if not relaxed else 'Free six moments may violate current spectral coupling; diagnostic only, never adopted.',
              'limits':['Same final-RGB picture reused; no external independent scene/material validation.',
                        'Catalogue cone is a finite model family, not all physical light sources.',
                        'Camera lower bounds use the already qualified CIE/inverse-decoder proxy.',
                        'Physical material response and nominal scan geometry remain uncalibrated.',
                        'Old sample/group/radius labels remain conditional image coordinates; updated geometry does not certify new physical annuli or RAW-clipping classes.',
                        'Finite multistart results do not prove global reachability or impossibility.']}
    if best['witness_coefficients'] is not None:
        c = np.array(best['witness_coefficients']); result['source_input_Y'] = float(c.sum())
        ymax = spectral_witness(S, m, C, required, maximize_input_Y=True)
        result['same_moments_inputY_min_max'] = [round(float(c.sum()), 8), round(float(ymax.x.sum()), 8) if ymax.success else None]
        result['inputY_range_scope'] = 'Catalogue/proxy-constrained coefficient-sum range at fixed six moments; not measured brightness or SI radiance.'
        result['nonzero_witness_members'] = [{'name':str(n),'coefficient':float(v)} for n,v in zip(library['names'],c) if v>1e-7]
        result['witness_moment_max_relative_residual'] = float(np.max(abs(c@S-m)/np.maximum(abs(m),1e-10)))
        result['camera_proxy_max_deficit'] = float(np.maximum(required-c@C,0.).max())
    (out/(name+'_'+mode+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(out/(name+'_'+mode+'.npz'),moments=m,image=image,q=q,lab=prediction,points=a['points'],groups=a['groups'],target_lab=a['target_lab'],cone=A)
    if not relaxed:
        from PIL import Image,ImageDraw,ImageFont
        from filmlab.spectral import srgb_encode
        source=np.load(root/'inverse-source-radiance'/(name+'_source.npz'))
        old_image,_=case.render(case.m0,True)
        sheet=Image.new('RGB',(1536,580),(18,18,18));draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
        for i,(title,values) in enumerate([('真实800T扫描',source['real']),('此前·已验证输入修补',old_image),('112谱库·联合近外训练',image)]):
            picture=Image.fromarray(np.rint(srgb_encode(np.nan_to_num(values))*255).astype('uint8'))
            sheet.paste(picture,(i*512,60));draw.text((i*512+10,12),title,font=font,fill='white')
        sheet.save(out/(name+'_catalogue_comparison.jpg'),quality=98)
    print('complete',name,mode,json.dumps(result['fitted'],ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['ceiling_lamp','middle_ceiling'],required=True)
    p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/catalogue-source-inverse'))
    p.add_argument('--starts',type=int,default=8)
    p.add_argument('--relaxed',action='store_true',help='Unphysical six-response diagnostic only; never a correction.')
    p.add_argument('--ceiling-geometry',action='store_true',help='Use independently accepted ceiling-plane geometry only.')
    args=p.parse_args();fit(args.region,args.out,args.starts,args.relaxed,args.ceiling_geometry)
