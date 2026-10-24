import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/eval'))
import calibrate_lab as cl


def test_objective_uses_full_train_and_only_valid_near_scenes():
    scenes = {s: {'full': float(i + 1), 'near': 20. if i < 2 else None}
              for i, s in enumerate(cl.acceptance.TRAIN)}
    m = cl.objective(scenes)
    assert m['full'] == 6.
    assert m['near'] == 20.
    assert m['objective'] == 16.
    assert len(m['near_scenes']) == 2


def test_training_guard_and_direct_acceptance_scene_call(monkeypatch):
    calls = []
    def scene(args):
        calls.append(args)
        return args[1], {'full': 1., 'near': np.nan}
    monkeypatch.setattr(cl.acceptance, 'scene_job', scene)
    sid = next(iter(cl.acceptance.TRAIN))
    assert cl.scene_job((cl.acceptance.CURRENT, sid)) == (sid, {'full': 1., 'near': None})
    assert calls == [(cl.acceptance.CURRENT, sid)]
    with pytest.raises(ValueError, match='Only fixed TRAIN'):
        cl.scene_job(({}, 'not_training'))
    assert len(calls) == 1


def test_sparse_grid_is_reproducible_and_interior():
    a, b = cl.coarse_grid(), cl.coarse_grid()
    np.testing.assert_array_equal(a, b)
    assert len(a) == 43
    assert np.all(np.array(a) >= .25) and np.all(np.array(a) <= .75)
    assert not cl.boundary_keys(cl.acceptance.CURRENT)
    for k, (lo, hi) in cl.acceptance.BOUNDS.items():
        for v in (lo, hi, lo + .019 * (hi - lo), hi - .019 * (hi - lo)):
            assert k in cl.boundary_keys({**cl.acceptance.CURRENT, k: v})


def test_both_conditionings_stop_on_any_boundary_candidate_and_resume_stays_terminal(tmp_path, monkeypatch):
    class Result:
        def __init__(self, value): self.value = value
        def get(self, timeout): return self.value
    class Pool:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def apply_async(self, fn, args): return Result(fn(*args))
    class Context:
        def Pool(self, *args, **kwargs): return Pool()
    monkeypatch.setattr(cl.mp, 'get_context', lambda _: Context())
    monkeypatch.setattr(cl, 'coarse_grid', lambda: [np.zeros(5), np.ones(5)])
    def scene(task):
        p, sid = task
        # A worse boundary candidate must still stop the search.
        value = 30. if cl.boundary_keys(p) else 10.
        return sid, {'full': value, 'near': value}
    monkeypatch.setattr(cl, 'scene_job', scene)
    output = tmp_path / 'result.json'
    s = cl.Search(output, 1).run()
    for c in ('lab', 'none'):
        r = s['runs'][c]
        assert r['status'] == 'stopped_boundary'
        assert len(r['evaluations']) == 2
        assert r['best_index'] == 0
        assert r['boundary_event']['index'] == 1
    assert s['status'] == 'complete'
    assert cl.Search(output, 1).run() == s


@pytest.mark.parametrize('conditioning', ['lab', 'none'])
def test_profile_recipe_matches_acceptance_apply_without_global_or_film_changes(conditioning):
    code = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from filmlab import engine,variation,generation
from filmlab.scanning import timing
from filmlab.engine import Engine,Params
from filmlab.recipes import recipe
from tools.eval.accept_t02 import apply
p={'latd_normal':-.23,'latd_limit':.21,'highlight_aim':.72,'shoulder_white':.96,
   'color_strength':.63,'conditioning':sys.argv[1]}
x=np.random.default_rng(9).uniform(.001,.9,(9,11,3));x[4,5]=8.
generation.use('next')
kw={**recipe('cinestill_800t'),'timing':'frame','roll':None,'grain':0}
originals=engine.LATD_NORMAL,timing.HIGHLIGHT_AIM,timing.SHOULDER_WHITE,variation.sample
base=Engine(Params(**kw));before=base.render(x);dye0=base.state.dye.copy()
extra=apply(p)
expected=Engine(Params(**{**kw,**extra})).render(x)
engine.LATD_NORMAL,timing.HIGHLIGHT_AIM,timing.SHOULDER_WHITE,variation.sample=originals
generation.use('next')
read=Path.read_text
def profile(path,*a,**k):
 if path.name=='epfl_lab_profile.json':return json.dumps({'enabled':True,'params':p})
 return read(path,*a,**k)
with patch.object(Path,'read_text',profile):
 eng=Engine(Params(**{**recipe('cinestill_800t_epfl_profile'),'roll':None,'grain':0}))
 actual=eng.render(x)
np.testing.assert_array_equal(actual,expected)
np.testing.assert_array_equal(eng.state.dye,dye0)
assert (engine.LATD_NORMAL,timing.HIGHLIGHT_AIM,timing.SHOULDER_WHITE,variation.sample)==originals
np.testing.assert_array_equal(Engine(Params(**kw)).render(x),before)
# Explicit existing constants with None-compatible fields are bitwise default.
for g in ('current','next','legacy'):
 generation.use(g)
 kw={**recipe('cinestill_800t'),'timing':'frame','roll':None,'grain':0}
 a=Engine(Params(**kw)).render(x)
 b=Engine(Params(**kw,scan_latd_normal=originals[0],scan_latd_limit=.12,
                 scan_highlight_aim=originals[1],scan_shoulder_white=originals[2])).render(x)
 np.testing.assert_array_equal(a,b)
'''
    subprocess.run([sys.executable, '-c', code, conditioning], cwd=ROOT, check=True, timeout=180)


@pytest.mark.parametrize('fit_status', ['finished', 'stopped_boundary', 'budget_exhausted'])
def test_final_report_records_both_fits_and_disables_unqualified_winner(tmp_path, monkeypatch, fit_status):
    import finalize_t02 as ft
    from types import SimpleNamespace
    for name in ('filmlab/data', 'tools/eval', 'docs'):
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(ft, 'REPO', tmp_path)
    monkeypatch.setattr(ft, 'RESULT', tmp_path / 'result.json')
    monkeypatch.setattr(ft.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='passed', stderr=''))
    base = {'full': 12., 'near': 14., 'objective': 19., 'near_scenes': []}
    best = {'full': 10., 'near': 12., 'objective': 16., 'near_scenes': []}
    lab_row = {'index': 0, 'params': dict(cl.acceptance.CURRENT), 'metrics': base}
    p = {**cl.acceptance.CURRENT, 'conditioning': 'none'}
    none_row = {'index': 0, 'params': p, 'metrics': best}
    state = {'status': 'complete', 'selected': {'conditioning': 'none', 'run_status': fit_status, **none_row},
             'runs': {'lab': {'status': 'finished', 'best_index': 0, 'boundary_event': None, 'evaluations': [lab_row]},
                      'none': {'status': fit_status, 'best_index': 0, 'boundary_event': None, 'evaluations': [none_row]}}}
    assert ft.finalize(state) == 0
    profile = json.loads((tmp_path / 'filmlab/data/epfl_lab_profile.json').read_text())
    assert profile['enabled'] == (fit_status == 'finished')
    assert json.loads((tmp_path / 'tools/eval/t02_profile_params.json').read_text()) == p
    text = (tmp_path / 'docs/T02_REPORT.md').read_text()
    assert '| lab |' in text and '| none |' in text
