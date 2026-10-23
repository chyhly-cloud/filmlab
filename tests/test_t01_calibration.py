import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/eval'))
import calibrate_film as cf
import accept_calibration as ac


def test_training_split_and_boundary_policy():
    assert len(ac.TRAIN) == 11
    assert not cf.at_boundary(ac.CURRENT)
    for key, (lo, hi) in ac.BOUNDS.items():
        for value in (lo, hi, lo + .019 * (hi - lo), hi - .019 * (hi - lo)):
            assert cf.at_boundary({**ac.CURRENT, key: value})
    with pytest.raises(ValueError, match='Only the fixed TRAIN'):
        cf.scene_job((ac.CURRENT, 'not_a_training_scene'))


def test_gamma_measurement_is_repeatable_in_fresh_processes():
    p = {**ac.CURRENT, 'tau': 1.15, 'dir': 1.04}
    with mp.get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
        a, b = pool.map(cf.gamma_job, [p, p])
    assert a == b
    assert a['passed']
    ratios = np.array(list(a['ratios'].values()))
    np.testing.assert_array_equal(ratios, np.array(a['separation_gamma']) / a['neutral_gamma'])


def test_completed_journal_is_not_restarted(tmp_path):
    output = tmp_path / 'result.json'
    s = cf.Search(output, 1)
    s.state['status'] = 'stopped_boundary'
    cf.atomic_json(output, s.state)
    assert cf.Search(output, 1).run()['evaluations'] == []


def test_no_re_evaluation_of_completed_candidates(tmp_path):
    output = tmp_path / 'result.json'
    s = cf.Search(output, 1)
    p = cf.params_at([.5, .55, .5])
    s.state['evaluations'] = [{'params': p, 'status': 'complete', 'train_mean_dE00': 4.5,
                               'gamma': {'passed': True}}]
    cf.atomic_json(output, s.state)
    assert cf.Search(output, 1).evaluate([.5, .55, .5]) == 4.5


def test_next_cal_matches_apply_and_switching_restores_all_defaults():
    # Synthetic input only: none of the real acceptance/holdout images are read.
    code = r'''
import json
from unittest.mock import patch
import numpy as np
from pathlib import Path
from filmlab import generation, chemistry, variation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from tools.eval.accept_calibration import apply
p = {'tau':1.15,'dir':1.04,'diffuse_fraction':.6}
x = np.random.default_rng(0).uniform(.01,.8,(8,8,3))
def render():
    return Engine(Params(**{**recipe('cinestill_800t'), 'roll':None,'grain':0})).render(x)
base={}
for g in ('current','next','legacy'):
    generation.use(g);base[g]=render()
originals=chemistry.run,chemistry.develop_image,variation.sample
apply(p);expected=render()
chemistry.run,chemistry.develop_image,variation.sample=originals
generation.use('next')
read=Path.read_text
def candidate(path,*a,**k):
    if path.name=='next_cal.json':return json.dumps({'enabled':True,'params':p})
    return read(path,*a,**k)
with patch.object(Path,'read_text',candidate):
    for _ in range(2):
        generation.use('next_cal')
        assert generation.active()=='next_cal'
        np.testing.assert_array_equal(render(),expected)
    for g in ('current','next','legacy'):
        generation.use(g)
        np.testing.assert_array_equal(render(),base[g])
        assert (chemistry.run,chemistry.develop_image,variation.sample)==originals
'''
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=180)


def test_improved_boundary_candidate_stops_without_further_search(tmp_path, monkeypatch):
    class Result:
        def __init__(self, value): self.value = value
        def get(self, timeout): return self.value
    class Pool:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def apply_async(self, function, args):
            return Result(function(*args))
    class Context:
        def Pool(self, *args, **kwargs): return Pool()
    monkeypatch.setattr(cf.mp, 'get_context', lambda _: Context())
    monkeypatch.setattr(cf, 'gamma_job', lambda p: {'passed': True})
    monkeypatch.setattr(cf, 'scene_job', lambda task: (task[1], 4.0))
    s = cf.Search(tmp_path / 'result.json', 1)
    with pytest.raises(cf.StopSearch):
        s.evaluate([0.0, .55, .5])
    assert s.state['status'] == 'stopped_boundary'
    assert len(s.state['evaluations']) == 1
    assert s.best['train_mean_dE00'] == 4.0


@pytest.mark.parametrize('status', ['finished', 'stopped_boundary', 'no_physical_candidate', 'error'])
def test_finalization_gates_generation_and_prepares_flat_params(tmp_path, monkeypatch, status):
    import finalize_t01 as ft
    from types import SimpleNamespace
    for folder in ('filmlab/data', 'tools/eval', 'docs'):
        (tmp_path / folder).mkdir(parents=True)
    monkeypatch.setattr(ft.platform, 'platform', lambda: 'test-platform')
    monkeypatch.setattr(ft, 'REPO', tmp_path)
    monkeypatch.setattr(ft, 'RESULT', tmp_path / 'journal.json')
    monkeypatch.setattr(ft.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='tests passed', stderr=''))
    final = {'params': {**ac.CURRENT, 'tau':1.15, 'dir':1.04}, 'train_mean_dE00':4.,
             'gamma': {'passed':True, 'ratios':{'R':1.3,'G':1.6,'B':1.31},
                       'neutral_gamma':[1,1,1], 'separation_gamma':[1.3,1.6,1.31]}}
    if status == 'stopped_boundary': final['params']['tau'] = 1.0
    if status in ('no_physical_candidate', 'error'): final = None
    s = {'status':status, 'final':final, 'baseline_train_mean_dE00':4.879,
         'training_improvement':.879, 'evaluations':[]}
    assert ft.finalize(s) == 0
    candidate = json.loads((tmp_path / 'filmlab/data/next_cal.json').read_text())
    assert candidate['enabled'] == (status == 'finished')
    flat = json.loads((tmp_path / 'tools/eval/t01_calibration_params.json').read_text())
    assert flat == ({} if final is None else final['params'])
