"""T01: training-only, reproducible calibration using the acceptance injection.

FILMLAB_DATA=... python tools/eval/calibrate_film.py
Each candidate/scene runs in a fresh spawned process. The JSON is an atomic,
resumable journal; no holdout evaluator or acceptance main is called here.
"""
import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'tools/eval'), str(REPO)]
import accept_calibration as acceptance

NAMES = tuple(acceptance.BOUNDS)
LOW = np.array([acceptance.BOUNDS[k][0] for k in NAMES])
WIDTH = np.array([acceptance.BOUNDS[k][1] - acceptance.BOUNDS[k][0] for k in NAMES])
GAMMA_BOUNDS = np.array([[1.15, 1.49], [1.42, 1.64], [1.29, 1.50]])
RESULT = REPO / 'tools/eval/t01_calibration_result.json'


def params_at(x):
    return dict(zip(NAMES, map(float, LOW + np.asarray(x) * WIDTH)))


def at_boundary(params):
    x = (np.array([params[k] for k in NAMES]) - LOW) / WIDTH
    return bool(np.any((x <= .02) | (x >= .98)))


def gamma_measurement():
    """Maximum d(dye density)/d(log H), full stock exposure axis, 4097 points.

    Neutral: all three layers receive the same log H. Separation: only the
    measured layer is swept, other layers stay at the stock's minimum log H.
    Uses the actual C-41 run, including silver/coupler clipping, without scan.
    """
    from filmlab import chemistry
    from filmlab.materials import emulsion
    e = emulsion('kodak_vision3_500t')
    x = np.linspace(e.log_exposure[0], e.log_exposure[-1], 4097)
    process = chemistry.PROCESSES['C-41']
    neutral = chemistry.run(e, process, np.repeat(x[:, None], 3, axis=1)).dye
    gn = np.max(np.gradient(neutral, x, axis=0), axis=0)
    gs = []
    for c in range(3):
        q = np.full((len(x), 3), x[0]); q[:, c] = x
        d = chemistry.run(e, process, q).dye[:, c]
        gs.append(float(np.max(np.gradient(d, x))))
    ratio = np.array(gs) / gn
    return {'definition': 'max slope on 4097-point stock logH axis; unexposed layers at axis minimum; C-41 dye density',
            'neutral_gamma': gn.tolist(), 'separation_gamma': gs,
            'ratios': dict(zip('RGB', map(float, ratio))),
            'ranges': dict(zip('RGB', GAMMA_BOUNDS.tolist())),
            'passed': bool(np.all((ratio >= GAMMA_BOUNDS[:, 0]) & (ratio <= GAMMA_BOUNDS[:, 1])))}


def gamma_job(params):
    acceptance.apply(params)
    return gamma_measurement()


def scene_job(args):
    params, sid = args
    if sid not in acceptance.TRAIN:
        raise ValueError('Only the fixed TRAIN scenes are permitted')
    acceptance.apply(params)
    import colour
    import param_sensitivity
    r = param_sensitivity.evaluate_scene(sid, roll=None)
    k = r['keep']
    value = float(np.median(colour.delta_E(r['S'][k], r['R'][k], method='CIE 2000')))
    if not np.isfinite(value):
        raise ValueError(f'Nonfinite training metric: {sid}')
    return sid, value


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


class StopSearch(Exception):
    pass


class Search:
    def __init__(self, output, workers):
        self.output, self.workers = output, workers
        self.state = json.loads(output.read_text()) if output.exists() else {
            'schema': 1, 'seed': 0, 'base_commit': 'fa68cb1',
            'train': dict(acceptance.TRAIN), 'bounds': acceptance.BOUNDS,
            'gamma_definition': 'See gamma_measurement docstring; no measured holdout used',
            'budget': 300, 'evaluations': [], 'status': 'running', 'final': None}
        if self.state['train'] != acceptance.TRAIN:
            raise ValueError('Journal training split mismatch')
        self.best = None
        for row in self.state['evaluations']:
            if row['status'] == 'complete' and row['gamma']['passed']:
                if self.best is None or row['train_mean_dE00'] < self.best['train_mean_dE00']:
                    self.best = row

    def evaluate(self, x, baseline=False):
        params = params_at(x)
        rows = self.state['evaluations']
        row = next((r for r in rows if r['params'] == params), None)
        if row is None:
            if len(rows) >= self.state['budget']:
                self.state['status'] = 'budget_exhausted'; raise StopSearch()
            row = {'index': len(rows), 'params': params, 'status': 'pending', 'scenes': {}}
            rows.append(row)
        if row['status'] == 'complete':
            return row['train_mean_dE00']
        if row['status'] == 'physics_rejected':
            return self.penalty(row['gamma'])
        atomic_json(self.output, self.state)
        with mp.get_context('spawn').Pool(self.workers, maxtasksperchild=1) as pool:
            row['gamma'] = pool.apply_async(gamma_job, (params,)).get(timeout=180)
            if not baseline and not row['gamma']['passed']:
                row['status'] = 'physics_rejected'
                row['train_mean_dE00'] = None
                atomic_json(self.output, self.state)
                return self.penalty(row['gamma'])
            pending = {sid: pool.apply_async(scene_job, ((params, sid),))
                       for sid in acceptance.TRAIN if sid not in row['scenes']}
            for sid, job in pending.items():
                _, row['scenes'][sid] = job.get(timeout=900)
                atomic_json(self.output, self.state)
                print(f"candidate {row['index']} scene {sid} complete", flush=True)
        row['train_mean_dE00'] = float(np.mean([row['scenes'][s] for s in acceptance.TRAIN]))
        row['status'] = 'complete'
        improved = row['gamma']['passed'] and (self.best is None or row['train_mean_dE00'] < self.best['train_mean_dE00'])
        if improved:
            self.best = row
        atomic_json(self.output, self.state)
        print(json.dumps({k: row[k] for k in ('index', 'params', 'train_mean_dE00')}), flush=True)
        if improved and at_boundary(params):
            self.state['status'] = 'stopped_boundary'
            raise StopSearch()
        return row['train_mean_dE00']

    @staticmethod
    def penalty(gamma):
        v = np.array(list(gamma['ratios'].values()))
        return float(1000 + np.maximum(GAMMA_BOUNDS[:, 0] - v, 0).sum()
                     + np.maximum(v - GAMMA_BOUNDS[:, 1], 0).sum())

    def run(self):
        if self.state['status'] not in ('running', 'error'):
            return self.state
        self.state['status'] = 'running'
        try:
            baseline = (np.array([acceptance.CURRENT[k] for k in NAMES]) - LOW) / WIDTH
            self.evaluate(baseline, baseline=True)
            # Interior coarse probes; an improved boundary candidate in refinement
            # terminates immediately. No restart or relaxed bound is attempted.
            for x in itertools.product((.1, .5, .9), (.1, .5, .55, .7, .9), (.1, .5, .9)):
                self.evaluate(x)
            if self.best is None:
                self.state['status'] = 'no_physical_candidate'
            else:
                x0 = (np.array([self.best['params'][k] for k in NAMES]) - LOW) / WIDTH
                opt = minimize(self.evaluate, x0, method='Powell', bounds=[(0, 1)] * 3,
                               options={'maxfev': 300 - len(self.state['evaluations']), 'xtol': .002, 'ftol': 1e-5})
                self.state['optimizer'] = {'success': bool(opt.success), 'message': str(opt.message)}
                self.state['status'] = 'finished' if opt.success else 'budget_exhausted'
        except StopSearch:
            pass
        except Exception as exc:
            self.state.update(status='error', error=f'{type(exc).__name__}: {exc}')
            atomic_json(self.output, self.state)
            raise
        self.state['final'] = self.best
        base = self.state['evaluations'][0]
        self.state['baseline_train_mean_dE00'] = base['train_mean_dE00']
        self.state['training_improvement'] = None if self.best is None else base['train_mean_dE00'] - self.best['train_mean_dE00']
        atomic_json(self.output, self.state)
        return self.state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=RESULT)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    np.random.seed(0)
    state = Search(args.output, args.workers).run()
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
