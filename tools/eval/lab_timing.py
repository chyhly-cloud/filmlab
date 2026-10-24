"""T04 preflight gate and measurement primitives (blocked on auto closure).

This stopped delivery is not a completed 21-frame measurement. No production
files are changed. A fresh command directly runs accept_t02.scene_job in a new
process per scene and stops at the first mismatch. Real O1/O2 and bootstraps
must not begin until this prerequisite passes and the measurement is resumed.
"""
import argparse
import ast
from dataclasses import replace
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/eval'), str(ROOT)]
BASE_COMMIT = 'a43c2af'
CLOSURE_TOLERANCE = 1e-6
O1_LOW = np.array([-1.5, -.8, -.8])
O1_HIGH = np.array([1.5, .8, .8])


class DefinitionMismatch(RuntimeError):
    pass


def oracle_fit(render, timing, densities, reference_lab, keep, level=1):
    """Execute the exact nested fit() AST from oracle_timing.job, unchanged.

    render(timing) must render the supplied cell-density array. Kept here for
    synthetic definition/unit tests; no real O1 is run in this stopped delivery.
    """
    import oracle_timing as ot
    from scipy.optimize import least_squares
    tree = ast.parse(inspect.getsource(ot.job))
    node = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    namespace = {'np': np, 'replace': replace, 'least_squares': least_squares,
                 'lab_lin': ot.lab_lin, 'render': render, 't0': timing,
                 'R': np.asarray(reference_lab), 'keep': np.asarray(keep, bool)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(Path(ot.__file__)), 'exec'), namespace)
    return namespace['fit'](level)


def absolute_controls(timing, fitted, x):
    """Density-unit conversion defined before reading any T04 measurement."""
    x = np.asarray(x, float)
    off0 = np.zeros(3) if timing.M is None else np.asarray(timing.M[0], float)
    off1 = np.zeros(3) if fitted.M is None else np.asarray(fitted.M[0], float)
    if not np.allclose(off1-off0, [x[1], 0., x[2]], rtol=0, atol=1e-12):
        raise DefinitionMismatch('O1 M offsets disagree with x density units')
    if not np.allclose(np.asarray(fitted.K)-timing.K, x[0], rtol=0, atol=1e-12):
        raise DefinitionMismatch('O1 K offset disagrees with x[0]')
    k = np.asarray(timing.K, float)
    if not np.allclose(k, k.flat[0], rtol=0, atol=1e-12):
        raise DefinitionMismatch('Automatic K is not a common three-channel exposure')
    ca = off0[[0, 2]]-off0[1]
    return {'K_auto': float(k.flat[0]), 'c_auto': ca.tolist(),
            'K_lab': float(k.flat[0]+x[0]), 'c_lab': (ca+x[1:3]).tolist()}


def boundary_keys(x):
    ratio = (np.asarray(x, float)-O1_LOW)/(O1_HIGH-O1_LOW)
    return [name for name, f in zip(('K', 'R', 'B'), ratio) if min(f, 1-f) < .02]


def boundary_stop(boundary_frames):
    return len(set(boundary_frames)) > 3


def check_closure(sid, full, expected):
    error = abs(float(full)-float(expected))
    return {'scene': sid, 'actual': float(full), 'baseline': float(expected),
            'abs_error': error, 'closed': bool(np.isfinite(error) and error < CLOSURE_TOLERANCE)}


def empty_result():
    return {'frames': {}, 'k': {}, 'lightbox': {}, 'verdict': None, 'ledger': {},
            'metadata': {'base_commit': BASE_COMMIT, 'status': 'preflight_running',
                         'measurement_complete': False, 'closure_tolerance': CLOSURE_TOLERANCE,
                         'auto_checks': [], 'unimplemented_after_stop':
                         ['real O1/O2 measurement', 'cell bootstrap', 'frame bootstrap',
                          'regional ledger', 'colour-regression scatter']}}


def stopped_result(checks):
    out = empty_result()
    out['metadata']['auto_checks'] = checks
    if any(not c['closed'] for c in checks):
        out['metadata'].update(status='stopped_auto_not_closed', stop_rule='auto abs_error >= 1e-6')
    else:
        out['metadata']['status'] = 'preflight_passed_measurement_pending'
    return out


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def measure_preflight(scenes, output):
    from accept_calibration import TRAIN, HOLDOUT
    from accept_t02 import scene_job
    valid = {**TRAIN, **HOLDOUT}
    baseline = json.loads((ROOT/'tools/eval/t03_baseline.json').read_text())['scenes']
    checks = []
    for sid in scenes:
        if sid not in valid:
            raise ValueError(f'Unknown T04 scene: {sid}')
        # apply() mutates process state, so every scene gets a fresh child.
        with mp.get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
            _, measured = pool.apply_async(scene_job, (({}, sid),)).get(timeout=1800)
        checks.append(check_closure(sid, measured['full'], baseline[sid]['full']))
        result = stopped_result(checks)
        atomic_json(output, result)
        print(json.dumps(checks[-1]), flush=True)
        if not checks[-1]['closed']:
            return result
    return stopped_result(checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene_or_output', help='scene ID for single scene, or output JSON for full preflight')
    p.add_argument('output', nargs='?')
    a = p.parse_args()
    from accept_calibration import TRAIN, HOLDOUT
    scenes = [a.scene_or_output] if a.output else list(TRAIN)+list(HOLDOUT)
    output = a.output or a.scene_or_output
    result = measure_preflight(scenes, output)
    # A stopped/preflight-only run must never claim scientific completion.
    return 2 if result['metadata']['status']=='stopped_auto_not_closed' else 3


if __name__ == '__main__':
    raise SystemExit(main())
