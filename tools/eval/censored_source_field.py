"""Fixed-total censored source-field inverse, not a material/radiance measurement.

Only fully censored core amplitudes vary, at one previously selected common SPD.
Every amplitude obeys its camera-proxy floor; direct/return use the same field.
Finite trust-region linear proposals require full nonlinear training acceptance.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]


def linear_proposal(J, residual, source, lower, radius):
    """Box trust region plus hard camera floor and unchanged total amplitude."""
    J, residual, source, lower = map(np.asarray, (J, residual, source, lower))
    if np.any(source < lower-1e-8) or radius <= 0:
        raise ValueError('Feasible source and positive numerical radius required')
    low = np.maximum(lower-source, -radius)
    high = np.minimum(source.sum()-lower.sum()+lower-source, radius)
    gram, gradient = J.T@J, J.T@residual
    result = minimize(lambda d: float(gradient@d+.5*d@gram@d), np.zeros(len(source)),
                      jac=lambda d: gradient+gram@d,
                      bounds=list(zip(low, high)),
                      constraints=[{'type':'eq', 'fun':lambda d: d.sum(),
                                    'jac':lambda d: np.ones(len(d))}],
                      method='SLSQP', options={'maxiter':250, 'ftol':1e-10})
    if abs(result.x.sum()) > 1e-7 or np.any(source+result.x < lower-1e-7):
        raise ValueError('Linear solver violated hard amplitude constraints')
    return result


def field_summary(values, positions, lower):
    values, positions = np.asarray(values), np.asarray(positions)
    p = values/values.sum()
    center = p@positions
    return {'peak_to_mean':float(values.max()/values.mean()),
            'maximum_single_cell_fraction':float(p.max()),
            'effective_cells':float(1/(p@p)),
            'centroid_yx_native_pixels':center.tolist(),
            'rms_radius_native_pixels':float(np.sqrt(p@np.sum((positions-center)**2,axis=1))),
            'camera_floor_active_cells':int(np.count_nonzero(values<=lower+1e-7)),
            'camera_floor_fraction_of_total':float(np.asarray(lower).sum()/values.sum())}


class FieldCase:
    def __init__(self, name, swap_sectors=False, ceiling_geometry=False):
        from catalogue_source_inverse import MomentCase
        from filmlab import film
        self.base = MomentCase(name)
        self.a, self.e = self.base.a, self.base.e
        self.root, self.name = self.base.root, name
        self.train_groups = [1,3] if swap_sectors else [0,2]
        self.geometry = {'mode':'original_fixed_geometry'}
        if ceiling_geometry:
            from inverse_source_radiance import observation_operator
            from measurement_aperture_audit import native_support_overlap
            record=json.loads((self.root/'ceiling-line-registration/results.json').read_text())
            accepted=[x for x in record['regions'] if x['name']==name and x['adopt_for_diagnostic']]
            if len(accepted)!=1:raise ValueError('No independently accepted ceiling geometry for this lamp')
            A=np.array(accepted[0]['matrix_raw_warp_to_scan'])
            self.a['matrix']=A@self.a['matrix']
            P=observation_operator(self.a['source_profile'].shape,self.a['matrix'],self.a['points'],25/self.base.meta['scan_um_per_pixel'])
            overlap,_=native_support_overlap(P,np.isin(self.a['groups'],[0,2]),np.isin(self.a['groups'],[1,3]))
            if not overlap['strictly_disjoint']:raise ValueError('Geometry update breaks native measurement separation')
            self.base.P=P;self.base.indices=np.unique(P.indices);self.base.Ps=P[:,self.base.indices]
            self.geometry={'mode':'independently_validated_ceiling_geometry','affine':A.tolist(),'new_native_overlap_audit':overlap,
                           'fixture_parallax':'Not independently measured; this only aligns the ceiling plane.'}
        path = self.root/'catalogue-source-inverse'
        self.fit = json.loads((path/(name+'_catalogue.json')).read_text())
        self.m = np.array(self.fit['selected_by_train_only']['moments'])
        c = np.maximum(self.fit['selected_by_train_only']['witness_coefficients'], 0.)
        data = np.load(path/(name+'_scene_specific_library.npz'))
        self.camera = c@data['camera_response']
        if np.any(self.camera <= 0):
            raise ValueError('Positive common-SPD camera proxy required')
        self.positions = np.argwhere(self.a['support'])
        self.s0 = self.a['source_profile'][self.a['support']].copy()
        self.lower = np.max(self.a['source_camera']/self.camera, axis=1)
        if np.any(self.s0 < self.lower):
            raise ValueError('Selected initial spectrum/field violates censor bounds')
        self.B = np.empty((len(self.positions), *self.a['background'].shape))
        rho = film.bounce_retention(self.e.e.stock, self.e.v)
        for i, (y, x) in enumerate(self.positions):
            direct = np.zeros_like(self.a['background']); returning = direct.copy()
            direct[y, x] = self.m[:3]; returning[y, x] = self.m[3:]
            self.B[i] = film.emulsion_scatter(direct, self.e.px_um)
            self.B[i] += film.halation(np.zeros_like(direct), returning,
                                      self.e.px_um, self.e.v, rho)
        expected = np.einsum('m,mhwc->hwc', self.m, self.base.B)
        self.basis_error = float(np.max(abs(self.linear_H(self.s0)-self.a['background']-expected)))
        if self.basis_error > 1e-9:
            raise ValueError('Per-cell same-SPD source basis does not replay fixed profile')
        self.counter = 0

    def linear_H(self, s):
        return self.a['background']+np.einsum('i,ihwc->hwc', s, self.B, optimize=True)

    def render(self, s, full=False):
        self.counter += 1
        e, a = self.e, self.a
        q = e._log_exposure(np.maximum(self.linear_H(s), 1e-12))
        st = e._develop(q, blur_px=12/e.px_um); st.dye *= a['factor'][..., None]
        if full:
            from skimage.transform import ProjectiveTransform
            from light_acceptance import crop_warp
            rgb = e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(st)))
            return crop_warp(ProjectiveTransform(a['matrix']))(rgb), q
        sub = replace(st, **{key:getattr(st, key).reshape(-1,3)[self.base.indices]
                             for key in ['dye','silver','halide']})
        rgb = e.scanner.to_rgb(e.scanner.normalized_density(e._transmittance(sub)))
        return self.base.lab(self.base.Ps@rgb)

    def residual_from(self, lab, target):
        return np.concatenate([(lab[self.a['groups']==g]-target[self.a['groups']==g]).ravel()
                               /np.sqrt(2*3*np.count_nonzero(self.a['groups']==g)) for g in self.train_groups])

    def residual(self, s, target=None):
        return self.residual_from(self.render(s), self.a['target_lab'] if target is None else target)

    def jacobian(self, s, target=None):
        r = self.residual(s, target); columns = []
        for i in range(len(s)):
            h = max(abs(s[i])*.005, .005)
            changed = s.copy(); changed[i] += h
            columns.append((self.residual(changed, target)-r)/h)
        return np.array(columns).T


def run(name, out, iterations=3, swap_sectors=False, ceiling_geometry=False):
    case = FieldCase(name, swap_sectors, ceiling_geometry); a = case.a
    out.mkdir(parents=True, exist_ok=True); started = time.time()
    initial = case.render(case.s0)
    selected = np.array(case.fit['selected_by_train_only']['moments'])
    replay = float(abs(initial-case.base.render(selected)).max())
    if replay > 1e-6:
        raise ValueError('Nonlinear initial field fails same-SPD candidate replay')
    s = case.s0.copy(); target = a['target_lab']; history = []
    # One known synthetic perturbation checks the numerical method, not 800T.
    rng = np.random.default_rng(20261021)
    shape = rng.dirichlet(np.ones(len(s))*2)
    known = case.lower+(s.sum()-case.lower.sum())*shape
    known = .95*s+.05*known
    known_target = case.render(known)
    J = case.jacobian(s)
    synthetic_r = case.residual_from(initial, known_target)
    sol = linear_proposal(J, synthetic_r, s, case.lower, radius=.25*s.sum())
    synthetic_pred = synthetic_r+J@sol.x
    synthetic_actual = case.residual(s+sol.x, known_target)
    synthetic = {'known_field_max_change':float(abs(known-s).max()),
                 'initial_field_relative_L1_error':float(abs(s-known).sum()/known.sum()),
                 'initial_component_MSE':float(synthetic_r@synthetic_r),
                 'linear_prediction_MSE':float(synthetic_pred@synthetic_pred),
                 'complete_nonlinear_MSE':float(synthetic_actual@synthetic_actual),
                 'recovered_field_relative_L1_error':float(abs(s+sol.x-known).sum()/known.sum()),
                 'scope':'Algorithm-only known-source check; not a measured 800T fixture or true inverse uniqueness.'}
    if synthetic['complete_nonlinear_MSE'] >= synthetic['initial_component_MSE']:
        raise ValueError('Known-source numerical check failed; do not fit real scan')
    radius = .25*s.sum()
    for iteration in range(iterations):
        r = case.residual(s)
        if iteration:
            J = case.jacobian(s)
        proposal = linear_proposal(J, r, s, case.lower, radius)
        step = proposal.x
        row = {'iteration':iteration, 'solver_success':bool(proposal.success),
               'solver_message':str(proposal.message), 'initial_train_MSE':float(r@r),
               'trust_box_radius':float(radius), 'attempts':[], 'accepted':False}
        for alpha in [1., .5, .25, .125, .0625]:
            trial = s+alpha*step
            actual = case.residual(trial)
            forecast = r+alpha*J@step
            predicted_drop = float(r@r-forecast@forecast)
            drop = float(r@r-actual@actual)
            agreement = drop/predicted_drop if predicted_drop > 1e-12 else None
            attempt = {'alpha':alpha, 'predicted_component_MSE':float(forecast@forecast),
                       'full_nonlinear_component_MSE':float(actual@actual),
                       'actual_over_predicted_decrease':agreement}
            row['attempts'].append(attempt)
            if drop > 1e-8 and agreement is not None and agreement >= .1:
                s = trial; row['accepted'] = True
                radius *= (1.5 if agreement > .75 and alpha == 1 else .5 if agreement < .25 else 1.)
                break
        history.append(row)
        (out/(name+'_partial.json')).write_text(json.dumps(history,indent=2)+'\n')
        print(name,'step',iteration,'accepted',row['accepted'],'loss',row['attempts'][-1]['full_nonlinear_component_MSE'],'eval',case.counter,flush=True)
        if not row['accepted']:
            break
    prediction = case.render(s); image, q = case.render(s, True)
    result = {'name':name, 'scope':__doc__, 'source_cells':len(s),
              'geometry':case.geometry,
              'training_groups':case.train_groups,
              'sector_scope':'Role reversal trains original groups1/3 (still labelled heldout in stats) and evaluates original0/2 (still labelled train). Separate cross-fit diagnostic, never new blind evidence.' if swap_sectors else 'Original predeclared groups 0/2 training, groups 1/3 heldout.',
              'fixed_total_amplitude':[float(case.s0.sum()),float(s.sum())],
              'camera_proxy_max_deficit':float(np.maximum(a['source_camera']-s[:,None]*case.camera,0).max()),
              'relative_source_field_L1_change':float(abs(s-case.s0).sum()/case.s0.sum()),
              'initial_field_shape':field_summary(case.s0,case.positions,case.lower),
              'fitted_field_shape':field_summary(s,case.positions,case.lower),
              'same_SPD_linear_replay_max':case.basis_error,'initial_nonlinear_Lab_replay_max':replay,
              'synthetic_known_field_check':synthetic,'iteration_history':history,
              'initial_catalogue':case.base.stats(initial),'fitted_field':case.base.stats(prediction),
              'total_forward_evaluations':case.counter,'elapsed_seconds':time.time()-started,
              'source_spectrum':'Fixed preselected 112-spectrum common witness; not measured lamp SPD.',
              'limits':['Finite three-step trust-region experiment, not a global optimum or physical impossibility proof.',
                        'Target final RGB repeatedly used. Native-isolated 19px sectors are conditional same-frame tests, not external measurements.',
                        'Hard constraints are qualified camera proxies, not calibrated sensor spectral sensitivity.',
                        'Only fully censored core support may change. Unknown external source support/angle/SPD cannot be excluded.',
                        'No spatial smoothness/material constants fitted. Extreme source patterns can be algebraically feasible but unphysical.',
                        'Known synthetic recovery checks the implementation, not actual material or radiance truth.']}
    (out/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    field = np.zeros(a['support'].shape); field[a['support']] = s
    np.savez_compressed(out/(name+'.npz'),field=field,initial_field=a['source_profile'],support=a['support'],lower=case.lower,
                        image=image,q=q,lab=prediction,initial_lab=initial,points=a['points'],groups=a['groups'],target_lab=target)
    from PIL import Image, ImageDraw, ImageFont
    from filmlab.spectral import srgb_encode
    source = np.load(case.root/'inverse-source-radiance'/(name+'_source.npz'))
    old, _ = case.render(case.s0, True)
    sheet = Image.new('RGB',(1536,580),(18,18,18)); draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
    for i,(title,values) in enumerate([('真实800T扫描',source['real']),('固定空间轮廓 · 112谱候选',old),('固定总输入 · 逐单元空间诊断',image)]):
        picture = Image.fromarray(np.uint8(np.rint(srgb_encode(np.nan_to_num(values))*255)))
        sheet.paste(picture,(i*512,60));draw.text((i*512+8,12),title,font=font,fill='white')
    sheet.save(out/(name+'_comparison.jpg'),quality=98)
    print(name,'complete',json.dumps(result['fitted_field'],ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',required=True,choices=['ceiling_lamp','middle_ceiling'])
    p.add_argument('--out',type=Path,default=Path('/workspace/filmlab-audit/censored-source-field'))
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--swap-sectors',action='store_true',help='Reverse sector roles as a stability diagnostic; not a fresh heldout test.')
    p.add_argument('--ceiling-geometry',action='store_true',help='Only independently accepted remote-grid geometry; no lamp-colour geometry selection.')
    args=p.parse_args();run(args.region,args.out,args.iterations,args.swap_sectors,args.ceiling_geometry)
