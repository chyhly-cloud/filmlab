"""Post-fit numerical/identifiability audit; never refits or selects a spectrum.

Reads the fixed 112-spectrum and explicitly nonphysical six-moment experiments.
LP ranges and finite-difference singular values describe this frozen model only.
"""
from pathlib import Path
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'tools/eval')]
from catalogue_source_inverse import MomentCase, spectral_witness


def main():
    root = Path('/workspace/filmlab-audit/catalogue-source-inverse')
    rows = []
    for name in ['ceiling_lamp', 'middle_ceiling']:
        case = MomentCase(name)
        library = np.load(root/(name+'_scene_specific_library.npz'))
        S, C = library['response6'], library['camera_response']
        a = case.a
        required = np.max(a['source_camera']/a['source_profile'][a['support'], None], axis=0)
        path = root/(name+'_catalogue.json')
        fitted = json.loads(path.read_text())
        relaxed = json.loads((root/(name+'_relaxed_nonphysical_diagnostic.json')).read_text())
        ten = json.loads((case.root/'joint-radius-source-inverse'/(name+'.json')).read_text())
        m = np.array(fitted['selected_by_train_only']['moments'])
        ymin = spectral_witness(S, m, C, required)
        ymax = spectral_witness(S, m, C, required, maximize_input_Y=True)
        if not (ymin.success and ymax.success):
            raise ValueError('Fixed-moment input Y extrema unavailable')
        fitted['scene_response_instance'] = {
            'roll': case.e.p.roll, 'frame': case.e.p.frame,
            'white_reference': 'D65',
            'original10_response_reproduction_max': float(abs(S[np.load(case.root/'spectral-library-coverage/normalized_spectra.npz')['original10_indices']]-case.S10).max())}
        fitted['same_moments_inputY_min_max'] = [round(float(ymin.x.sum()), 8), round(float(ymax.x.sum()), 8)]
        fitted['inputY_range_scope'] = 'Only fixed six moments, finite catalogue and qualified camera proxy. This excludes uncertainty in moments/profile/backend and is not SI radiance.'
        # At this fit, 0.1-percent changes of each scaled coordinate probe local
        # sensitivity. These are not exposure tweaks used to improve the photo.
        J = case.jacobian(m/case.scale)
        values = np.linalg.svd(J, compute_uv=False)
        fitted['numerical_response_audit'] = {
            'residual_J_coordinate': 'moments divided by inherited six-moment scale; residual is equal-zone normalized Lab components',
            'residual_J_singular_values': values.tolist(),
            'residual_J_column_norms_directRGB_returnRGB': np.linalg.norm(J, axis=0).tolist(),
            'finite_difference_rule': 'forward step max(abs(z)*1e-3,1e-5); no claim of exact rank or global uniqueness'}
        fitted['source_claim'] = 'Nonnegative catalogue mixture mathematically consistent with the frozen model/proxy; not measured lamp SPD, actual fixture identity, or SI radiance.'
        selected = fitted['selected_by_train_only']
        raw_c = np.array(selected['witness_coefficients'])
        minimum_raw_coefficient = selected.get('raw_LP_minimum_coefficient', float(raw_c.min()))
        c = np.maximum(raw_c, 0.)
        closure = float(np.max(abs(c@S-m)/np.maximum(abs(m), 1e-10)))
        if raw_c.min() < -1e-7 or closure > 1e-8:
            raise ValueError('LP coefficient roundoff exceeds the reporting contract')
        selected['raw_LP_minimum_coefficient'] = minimum_raw_coefficient
        selected['witness_coefficients'] = c.tolist()
        fitted['LP_roundoff_audit'] = {'minimum_raw_coefficient': minimum_raw_coefficient,
                                     'negative_roundoff_clamped_to_zero': True,
                                     'moment_relative_closure_after_clamping': closure,
                                     'interval_report_decimal_places': 8}
        path.write_text(json.dumps(fitted, ensure_ascii=False, indent=2)+'\n')
        physical_loss = fitted['selected_by_train_only']['train_equal_zone_MSE']
        free_loss = relaxed['selected_by_train_only']['train_equal_zone_MSE']
        row = {
            'name': name,
            'physical_fit': fitted['fitted'],
            'inherited_corrected_fit': fitted['inherited_corrected_core_inverse'],
            'ten_spectrum_joint_fit': ten['joint_radius_fit'],
            'catalogue_train_loss': physical_loss,
            'nonphysical_six_moment_train_loss': free_loss,
            'relative_loss_drop_by_relaxation': (physical_loss-free_loss)/physical_loss,
            'nonphysical_fit': relaxed['fitted'],
            'nonphysical_all_same_spectrum_LPs_failed': all(not x['same_spectrum_camera_proxy_witness_feasible'] for x in relaxed['trials']),
            'source_input_Y_fixed_moment_min_max': fitted['same_moments_inputY_min_max'],
            'numerical_response_audit': fitted['numerical_response_audit'],
            'physical_multistart_train_loss_range': [min(x['train_equal_zone_MSE'] for x in fitted['trials']), max(x['train_equal_zone_MSE'] for x in fitted['trials'])],
            'forward_evaluations_physical_and_relaxed': fitted['total_forward_evaluations']+relaxed['total_forward_evaluations']}
        rows.append(row)
    report = {'scope': __doc__, 'regions': rows,
              'decision': 'No candidate passes simultaneous orange-near and outer spatial acceptance; no production promotion.',
              'interpretation': 'Under the fixed source support/profile and frozen backend, this finite experiment finds little additional leverage from relaxing spectral coupling alone. It does not identify transport, development, or scanner as the physical root cause.',
              'limits': ['Same final-RGB scene repeatedly inspected; spatial sectors are not external validation.',
                         'Finite optimizers, finite catalogue and fixed profile do not prove global physical impossibility.',
                         'Six independent nonnegative direct/return coordinates are a diagnostic, never a physical correction.',
                         'Min/max input Y is conditional on fitted moments and can be narrow despite much larger unmodeled uncertainty.']}
    (root/'round_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
