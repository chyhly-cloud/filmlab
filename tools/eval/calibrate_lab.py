"""T02 training-only lab-profile search; fixed split, seed and bounds.

FILMLAB_DATA=... python tools/eval/calibrate_lab.py
Each acceptance.scene_job calls acceptance.apply itself: never inject twice.
Existing terminal journals are returned unchanged; incomplete scenes resume.
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
import accept_t02 as acceptance

RESULT = REPO / 'tools/eval/t02_result.json'
NAMES = tuple(acceptance.BOUNDS)
LOW = np.array([acceptance.BOUNDS[k][0] for k in NAMES])
WIDTH = np.array([acceptance.BOUNDS[k][1] - acceptance.BOUNDS[k][0] for k in NAMES])
SEED = 20261024


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def params_at(x, conditioning):
    return {**dict(zip(NAMES, map(float, LOW + np.asarray(x) * WIDTH))), 'conditioning': conditioning}


def boundary_keys(params):
    x = (np.array([params[k] for k in NAMES]) - LOW) / WIDTH
    return [k for k, v in zip(NAMES, x) if v <= .02 or v >= .98]


def objective(scenes):
    full = float(np.mean([scenes[s]['full'] for s in acceptance.TRAIN]))
    near_scenes = [s for s in acceptance.TRAIN if scenes[s]['near'] is not None]
    if not near_scenes:
        raise ValueError('No training scenes have valid near-source metrics')
    near = float(np.mean([scenes[s]['near'] for s in near_scenes]))
    return {'full': full, 'near': near, 'objective': full + .5 * near, 'near_scenes': near_scenes}


def scene_job(args):
    params, sid = args
    if sid not in acceptance.TRAIN:
        raise ValueError('Only fixed TRAIN scenes are permitted')
    # scene_job directly calls apply() once, internally. Fresh process per scene.
    returned_sid, metrics = acceptance.scene_job((params, sid))
    if returned_sid != sid or not np.isfinite(metrics['full']):
        raise ValueError(f'Invalid training result: {sid}')
    return sid, {'full': float(metrics['full']),
                 'near': float(metrics['near']) if np.isfinite(metrics['near']) else None}


def coarse_grid():
    # Sparse three-level grid: center + axial neighbors + 32 inner corners.
    points = [np.full(5, .5)]
    for c in range(5):
        for v in (.25, .75):
            p = np.full(5, .5); p[c] = v; points.append(p)
    points += [np.array(x) for x in itertools.product((.25, .75), repeat=5)]
    return points


class StopSearch(Exception):
    pass


class Search:
    def __init__(self, output=RESULT, workers=2):
        self.output, self.workers = Path(output), workers
        self.state = json.loads(self.output.read_text()) if self.output.exists() else {
            'schema': 1, 'base_commit': 'c2f2df7', 'seed': SEED, 'train': dict(acceptance.TRAIN),
            'bounds': {k: list(v) for k, v in acceptance.BOUNDS.items()}, 'budget_total': 400, 'budget_per_conditioning': 200,
            'status': 'running', 'selected': None,
            'runs': {c: {'status': 'pending', 'evaluations': [], 'best_index': None, 'boundary_event': None}
                     for c in ('lab', 'none')}}
        if self.state['train'] != acceptance.TRAIN:
            raise ValueError('Journal training split mismatch')

    def evaluate(self, x, conditioning):
        run = self.state['runs'][conditioning]
        params = params_at(x, conditioning)
        rows = run['evaluations']
        row = next((r for r in rows if r['params'] == params), None)
        if row is None:
            if len(rows) >= self.state['budget_per_conditioning']:
                run['status'] = 'budget_exhausted'; raise StopSearch()
            row = {'index': len(rows), 'params': params, 'status': 'pending', 'scenes': {}}
            rows.append(row)
        if row['status'] == 'complete':
            return row['metrics']['objective']
        atomic_json(self.output, self.state)
        with mp.get_context('spawn').Pool(self.workers, maxtasksperchild=1) as pool:
            jobs = {sid: pool.apply_async(scene_job, ((params, sid),))
                    for sid in acceptance.TRAIN if sid not in row['scenes']}
            for sid, job in jobs.items():
                _, row['scenes'][sid] = job.get(timeout=1800)
                atomic_json(self.output, self.state)
                print(f"{conditioning} candidate {row['index']} scene {sid} complete", flush=True)
        row['metrics'] = objective(row['scenes'])
        if row['index'] and row['metrics']['near_scenes'] != rows[0]['metrics']['near_scenes']:
            raise ValueError('Near-source training subset changed across candidates')
        row['status'] = 'complete'
        if run['best_index'] is None or row['metrics']['objective'] < rows[run['best_index']]['metrics']['objective']:
            run['best_index'] = row['index']
        hit = boundary_keys(params)
        if hit:
            run['boundary_event'] = {'index': row['index'], 'keys': hit, 'params': params}
            run['status'] = 'stopped_boundary'
        atomic_json(self.output, self.state)
        print(json.dumps({'conditioning': conditioning, 'index': row['index'], 'params': params,
                          'metrics': row['metrics'], 'boundary': hit}), flush=True)
        if hit:
            raise StopSearch()
        return row['metrics']['objective']

    def fit(self, conditioning):
        run = self.state['runs'][conditioning]
        if run['status'] not in ('pending', 'running'):
            return
        run['status'] = 'running'
        try:
            current = (np.array([acceptance.CURRENT[k] for k in NAMES]) - LOW) / WIDTH
            self.evaluate(current, conditioning)
            for x in coarse_grid():
                self.evaluate(x, conditioning)
            best = run['evaluations'][run['best_index']]
            x0 = (np.array([best['params'][k] for k in NAMES]) - LOW) / WIDTH
            opt = minimize(lambda x: self.evaluate(x, conditioning), x0, method='Powell', bounds=[(0, 1)] * 5,
                           options={'maxfev': 200 - len(run['evaluations']), 'xtol': .002, 'ftol': 1e-5})
            run['optimizer'] = {'success': bool(opt.success), 'message': str(opt.message)}
            run['status'] = 'finished' if opt.success else 'budget_exhausted'
        except StopSearch:
            pass
        atomic_json(self.output, self.state)

    def run(self):
        if self.state['status'] != 'running':
            return self.state
        np.random.seed(SEED)
        try:
            for c in ('lab', 'none'):
                self.fit(c)
            choices = []
            for c, run in self.state['runs'].items():
                if run['best_index'] is not None:
                    row = run['evaluations'][run['best_index']]
                    choices.append({'conditioning': c, 'run_status': run['status'], **row})
            self.state['selected'] = min(choices, key=lambda r: r['metrics']['objective'])
            self.state['status'] = 'complete'
        except Exception as exc:
            self.state.update(status='error', error=f'{type(exc).__name__}: {exc}')
            atomic_json(self.output, self.state)
            raise
        atomic_json(self.output, self.state)
        return self.state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=RESULT)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    state = Search(args.output, args.workers).run()
    print('T02', state['status'], flush=True)


if __name__ == '__main__':
    main()
