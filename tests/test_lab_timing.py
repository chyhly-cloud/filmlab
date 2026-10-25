"""T04 tests use only synthetic inputs; never load any real scene."""
from dataclasses import replace
import numpy as np
import pytest
from filmlab.scanning.timing import PrintTiming
from tools.eval.lab_timing import (oracle_fit, absolute_controls, DefinitionMismatch,
    check_closure, boundary_keys, boundary_stop, stopped_result)


def test_oracle_o1_recovers_known_density_units():
    import oracle_timing as ot
    rng = np.random.default_rng(20261025)
    densities = rng.uniform(.6, 1.6, (240, 3))
    t0 = PrintTiming(K=np.full(3, 1.2), s=np.full(3, 1.1),
                     M=(np.array([-.13, 0., .09]), np.array([1.02, 1., .98])))
    truth = np.array([.15, .05, -.06])
    target = replace(t0, K=t0.K+truth[0], M=(t0.M[0]+[truth[1],0,truth[2]], t0.M[1]))
    # Synthetic invertible response, with the real oracle's exact fitter/Lab map.
    render = lambda t: 1/(1+np.exp(-t.paper_exposure(densities)))
    fitted, x = oracle_fit(render, t0, densities, ot.lab_lin(render(target)), np.ones(240,bool), 1)
    np.testing.assert_allclose(x, truth, atol=1e-6, rtol=0)
    values = absolute_controls(t0, fitted, x)
    np.testing.assert_allclose(values['c_auto'], [-.13,.09], atol=1e-12)
    np.testing.assert_allclose(values['c_lab'], [-.08,.03], atol=1e-6)
    assert abs(values['K_lab']-1.35) < 1e-6


def test_definition_mismatch_stops():
    t0=PrintTiming(K=np.ones(3),M=(np.zeros(3),np.ones(3)))
    bad=replace(t0,K=t0.K+.1,M=(np.array([-.2,0,.1]),np.ones(3)))
    with pytest.raises(DefinitionMismatch):
        absolute_controls(t0,bad,[.1,.2,.1])


def test_closure_stops_at_tolerance_and_keeps_result_schema():
    good=check_closure('synthetic', 1.+.5e-6,1.)
    bad=check_closure('synthetic', 1.+2e-6,1.)
    assert good['closed'] and not bad['closed']
    result=stopped_result([bad])
    assert result['metadata']['status']=='stopped_auto_not_closed'
    assert result['frames']==result['ledger']=={}
    assert result['verdict'] is None
    assert check_closure('synthetic',float('nan'),1.)['closed'] is False


def test_boundary_rule_is_strict_and_counts_frames_not_parameters():
    assert boundary_keys([0,0,0])==[]
    assert boundary_keys([1.49,0,0])==['K']
    assert not boundary_stop(['a','b','c','a'])
    assert boundary_stop(['a','b','c','d'])


def synthetic_frames():
    rng = np.random.default_rng(331)
    frames = {}
    for i in range(20):
        ca = rng.normal(size=2)
        ka = float(rng.normal())
        frames[f'frame{i:02}'] = {'group': ('钨丝灯箱','白光灯管','品红灯管','night','card')[i//4],
            'c_auto': ca.tolist(), 'c_lab': (.1+.5*ca).tolist(),
            'K_auto': ka, 'K_lab': -.2+.3*ka, 'c_lab_se': [.01,.02]}
    return frames


def test_frame_bootstrap_known_slopes_and_frame_order_invariance():
    from tools.eval.lab_timing import frame_regressions, classify, lightbox_summary
    frames = synthetic_frames()
    k, omitted = frame_regressions(frames)
    again, _ = frame_regressions(dict(reversed(list(frames.items()))))
    assert k == again
    for name, truth in [('R',.5),('B',.5),('pooled',.5),('K',.3)]:
        np.testing.assert_allclose(k[name], [truth]*3, atol=1e-12, rtol=0)
    assert set(omitted.values()) == {0}
    for estimates in k['leave_group_out'].values():
        np.testing.assert_allclose(estimates,[.5,.5,.3],atol=1e-12)
    assert classify(k['pooled']) == '部分校色，力度 = k'
    light = lightbox_summary(frames)
    assert len(light) == 3
    for values in light.values():
        np.testing.assert_allclose(values['c_lab_within_sd'], .5*np.array(values['c_auto_within_sd']))


@pytest.mark.parametrize('interval,expected', [
    ([0,-.25,.25],'冲印店基本不逐帧校色'),
    ([1,.75,1.25],'校色力度与模型相当，颜色误差主要不在定光'),
    ([.5,.251,.749],'部分校色，力度 = k'),
    ([.5,.25,.7],'数据不足以区分'),
    ([.5,.3,.75],'数据不足以区分'),
    ([.2,.1,.4],'数据不足以区分')])
def test_verdict_exact_predeclared_thresholds(interval, expected):
    from tools.eval.lab_timing import classify
    assert classify(interval) == expected


def test_region_boundaries_signed_ledger_and_empty_regions():
    from tools.eval.lab_timing import regions, ledger_stage, encoded_lab
    real = np.array([[[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[.3,.3,.3],[.6,.6,.6]]])
    lab = encoded_lab(real)
    ok = np.ones((1,5),bool)
    masks = regions(lab,ok,np.array([[0,100,300,1500,1501]]))
    assert [int(masks[k].sum()) for k in ('核心','近','中','远')] == [2,1,1,1]
    masks['empty'] = np.zeros_like(ok)
    same = ledger_stage(real, lab, masks)
    assert same['全图'] == {'n':5,'dE':0.,'dL':0.,'da':0.,'db':0.}
    assert same['empty'] == {'n':0,'dE':None,'dL':None,'da':None,'db':None}
    shifted = real+.01
    delta = encoded_lab(shifted)-lab
    measured = ledger_stage(shifted, lab, masks)['全图']
    np.testing.assert_allclose([measured[k] for k in ('dL','da','db')],np.median(delta,axis=(0,1)))
    assert measured['dL'] > 0


def test_failed_closure_does_not_invoke_real_fitter(monkeypatch):
    import tools.eval.lab_timing as lt
    monkeypatch.setattr(lt,'capture_auto',lambda sid: {'measured':{'full':10.}})
    monkeypatch.setattr(lt,'measure_context',lambda *args: pytest.fail('fit ran after failed closure'))
    result = lt.scene_worker(('synthetic','group',1.,0))
    assert result['status'] == 'stopped_auto_not_closed'
    assert 'ledger' not in result and 'frame' not in result


def test_single_and_batch_identical_values_and_checkpoint(tmp_path):
    import tools.eval.lab_timing as lt
    calls = []
    def synthetic_runner(job, timeout):
        sid, group, baseline, _ = job
        calls.append(sid)
        rng = np.random.default_rng(lt.scene_seed(sid))
        values = rng.normal(size=3).tolist()
        return {'check':lt.check_closure(sid,baseline,baseline),'status':'complete',
                'frame':{'x':values,'group':group},'ledger':{'auto':{'全图':{'dE':values[0]}}},'boundary':[]}
    ids = ['R1-00-1','R1-02-3']
    batch = lt.run_measurement(ids,tmp_path/'batch.json',synthetic_runner)
    single = lt.run_measurement(ids[:1],tmp_path/'single.json',synthetic_runner)
    assert batch['frames'][ids[0]] == single['frames'][ids[0]]
    assert batch['ledger'][ids[0]] == single['ledger'][ids[0]]
    calls.clear()
    resumed = lt.run_measurement(ids,tmp_path/'batch.json',synthetic_runner,resume=True)
    assert calls == [] and resumed == batch
    stopped = lt.stopped_result([lt.check_closure(ids[0],10,1)])
    lt.atomic_json(tmp_path/'stop.json',stopped)
    with pytest.raises(ValueError,match='Scientific stop'):
        lt.run_measurement(ids,tmp_path/'stop.json',synthetic_runner,resume=True)


def test_boundary_stops_on_fourth_frame(tmp_path):
    import tools.eval.lab_timing as lt
    ids = ['R1-00-1','R1-02-3','R1-04-5','R1-06-7','R1-07-8']
    calls=[]
    def runner(job, timeout):
        sid,group,baseline,count = job
        calls.append(sid)
        return {'check':lt.check_closure(sid,baseline,baseline),'frame':{'x':[1.49,0,0]},
                'boundary':['K'],'status':'stopped_boundary_frames' if count>=3 else 'complete'}
    result = lt.run_measurement(ids,tmp_path/'out.json',runner)
    assert result['metadata']['status']=='stopped_boundary_frames'
    assert len(calls)==4 and len(result['metadata']['boundary_frames'])==4
    assert result['verdict'] is None and result['k']=={}


def test_exception_preserves_checkpoint_for_execution_resume(tmp_path):
    import tools.eval.lab_timing as lt
    import json
    def runner(job, timeout):
        raise TimeoutError('synthetic timeout')
    path=tmp_path/'out.json'
    with pytest.raises(TimeoutError):
        lt.run_measurement(['R1-00-1'],path,runner)
    saved=json.loads(path.read_text())
    assert saved['metadata']['status']=='execution_failed'
    assert saved['metadata']['completed_scenes']==[]
    assert saved['metadata']['failed_scene']=='R1-00-1'


def test_synthetic_full_image_fits_bootstraps_and_single_batch_equality():
    """Exercise the complete array pipeline, real 24px cells and 200 refits."""
    from types import SimpleNamespace
    from filmlab.io import srgb_encode
    import tools.eval.lab_timing as lt
    rng=np.random.default_rng(225)
    dc=rng.uniform(.5,1.8,(6,6,3))
    dn=np.repeat(np.repeat(dc,24,axis=0),24,axis=1)
    t0=PrintTiming(K=np.full(3,1.2),s=np.full(3,1.1),
                   M=(np.array([-.13,0,.09]),np.array([1.02,1,.98])))
    truth=np.array([.15,.05,-.06])
    target=replace(t0,K=t0.K+truth[0],M=(t0.M[0]+[truth[1],0,truth[2]],t0.M[1]))
    scanner=SimpleNamespace(timing=t0)
    scanner.to_rgb=lambda d: 1/(1+np.exp(-scanner.timing.paper_exposure(d)))
    auto=srgb_encode(np.clip(scanner.to_rgb(dn),0,1))
    scanner.timing=target
    real=srgb_encode(np.clip(scanner.to_rgb(dn),0,1))
    scanner.timing=t0
    ok=np.ones(dn.shape[:2],bool)
    full=lt.ledger_stage(auto,lt.encoded_lab(real),{'全图':ok})['全图']['dE']
    context={'scanner':scanner,'Dn':dn,'timing':t0,'aligned':auto,'real':real,
             'ok':ok,'H':np.eye(3),'distance_um':None,'measured':{'full':full}}
    first=lt.measure_context(context,'synthetic-grid','synthetic')
    second=lt.measure_context(context,'synthetic-grid','synthetic')
    assert first==second
    assert scanner.timing is t0
    np.testing.assert_allclose(first['frame']['x'],truth,atol=1e-6,rtol=0)
    assert first['frame']['n_cells']==36
    assert len(first['frame']['c_lab_se'])==2
    assert first['ledger']['O1']['全图']['dE']<1e-6
    assert first['ledger']['O2']['全图']['dE']<1e-6
    assert first['ledger']['auto']['全图']['dE']==full
    fourth=lt.measure_context(context,'synthetic-grid','synthetic',boundary_count=3)
    assert fourth['status']=='complete'  # only actual boundary fits count


def test_capture_observes_actual_call_and_restores_profiler(monkeypatch):
    from types import SimpleNamespace
    import sys
    import tools.eval.lab_timing as lt
    from filmlab.engine import Engine
    timing=object()
    engine=SimpleNamespace(scanner=SimpleNamespace(timing=timing))
    def scan(self):
        sc=self.scanner
        shape=(2,2)
        Dn=np.arange(12.).reshape(-1,3)
        return Dn.reshape(*shape,3)
    monkeypatch.setattr(Engine,'scan',scan)
    calls=[]
    def scene_job(args):
        calls.append(args)
        eng=engine
        w=Engine.scan(eng)
        Hm=np.eye(3); real=w; ok=np.ones((2,2),bool)
        core=np.ones((2,2),bool); d=np.zeros((2,2))
        return args[1],{'full':1.}
    original=sys.getprofile()
    captured=lt.capture_auto('synthetic',scene_job)
    assert calls==[({},'synthetic')]
    assert captured['timing'] is timing
    np.testing.assert_array_equal(captured['Dn'],captured['aligned'])
    np.testing.assert_array_equal(captured['distance_um'],np.zeros((2,2)))
    assert sys.getprofile() is original
    def broken(args):
        raise RuntimeError('synthetic capture error')
    with pytest.raises(RuntimeError):
        lt.capture_auto('synthetic',broken)
    assert sys.getprofile() is original


def test_pooled_bootstrap_matches_independent_paired_frame_reference():
    import tools.eval.lab_timing as lt
    frames=synthetic_frames()
    rng=np.random.default_rng(3399)
    for f in frames.values():
        f['c_lab']=(np.array(f['c_lab'])+rng.normal(0,.1,2)).tolist()
    result,_=lt.frame_regressions(frames)
    rows=[frames[s] for s in sorted(frames)]
    x=np.array([f['c_auto'] for f in rows]);y=np.array([f['c_lab'] for f in rows])
    rng=np.random.default_rng(np.random.SeedSequence([lt.SEED,2]))
    indices=rng.integers(0,20,(lt.FRAME_BOOTSTRAPS,20))
    xx=x[indices].reshape(lt.FRAME_BOOTSTRAPS,-1)
    yy=y[indices].reshape(lt.FRAME_BOOTSTRAPS,-1)
    xx=xx-xx.mean(axis=1,keepdims=True)
    slopes=(xx*yy).sum(axis=1)/(xx*xx).sum(axis=1)
    expected=[np.polyfit(x.ravel(),y.ravel(),1)[0],*np.percentile(slopes,[2.5,97.5])]
    np.testing.assert_allclose(result['pooled'],expected,atol=1e-12,rtol=0)


def test_completed_report_uses_measured_data_and_refuses_stop(tmp_path):
    import tools.eval.lab_timing as lt
    from accept_calibration import TRAIN,HOLDOUT
    groups={**TRAIN,**HOLDOUT}
    result=lt.empty_result()
    result['frames']=dict(zip([s for s in groups if s!='cafe'],synthetic_frames().values()))
    for sid,frame in result['frames'].items():frame['group']=groups[sid]
    result['ledger']={sid:{stage:{region:{'n':10,'dE':float(v),'dL':.1,'da':-.2,'db':.3}
                      for region,v in [('全图',3),('暗部',7),('中间调',4),('亮部',6)]}
                      for stage in ('auto','O1','O2')} for sid in groups}
    lt.finalize(result,list(groups))
    lt.write_report(result,tmp_path/'REPORT.md')
    report=(tmp_path/'REPORT.md').read_text()
    assert '合并 k' in report and '**>5**' in report
    assert '| 品红灯管 | 3 | 2 |' in report
    assert '全图' in report and '中间调' in report
    assert (tmp_path/'figures/t04/timing_scatter.png').stat().st_size>1000
    assert (tmp_path/'figures/t04/ledger_heatmap.png').stat().st_size>1000
    with pytest.raises(ValueError):lt.write_report(lt.empty_result(),tmp_path/'STOP.md')


def test_capture_auto_uses_nominal_next_configuration(monkeypatch):
    """Actual Engine configuration, with synthetic scan arrays and no scene IO."""
    import accept_t02
    import tools.eval.lab_timing as lt
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    observed = {}
    original_apply = accept_t02.apply

    def scan(self):
        sc = self.scanner
        shape = (2, 2)
        Dn = np.ones((4, 3))
        return Dn.reshape(*shape, 3)

    def scene_job(args):
        extra = accept_t02.apply({})
        params = Params(**{**recipe('cinestill_800t'), 'timing': 'frame',
                           'roll': None, 'grain': 0, **extra})
        eng = Engine(params)
        observed.update(extra=extra, parameter=params.scan_color_strength,
                        nominal=eng.v.scan_strength, actual=eng.scanner.strength)
        w = eng.scan()
        Hm = np.eye(3)
        real = np.zeros_like(w)
        ok = np.ones((2, 2), bool)
        return args[1], {'full': 1.}

    monkeypatch.setattr(Engine, 'scan', scan)
    monkeypatch.setattr(accept_t02, 'scene_job', scene_job)
    lt.capture_auto('synthetic-nominal')
    assert observed == {'extra': {}, 'parameter': None, 'nominal': .75, 'actual': .75}
    assert accept_t02.apply is original_apply
    assert 'scan_color_strength=None' in lt.method_metadata()['configuration']
