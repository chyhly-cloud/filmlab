"""T04 measurement, with a mandatory auto-baseline gate before every O1.

The auto entry point uses T03's nominal next configuration with the unchanged
accept_t02.scene_job renderer and alignment.
A mismatch stops the run; it is never converted into a scientific verdict.
Single and batch runs use identical scene seeds and fresh worker processes.
"""
import argparse
import ast
from dataclasses import replace
import inspect
import json
import multiprocessing as mp
import hashlib
import zlib
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/eval'), str(ROOT)]
BASE_COMMIT = 'a43c2af'
CLOSURE_TOLERANCE = 1e-6
O1_LOW = np.array([-1.5, -.8, -.8])
O1_HIGH = np.array([1.5, .8, .8])
CELL_BOOTSTRAPS = 200
FRAME_BOOTSTRAPS = 10000
SEED = 20261025
LIGHTBOX_GROUPS = ('钨丝灯箱', '白光灯管', '品红灯管')


class DefinitionMismatch(RuntimeError):
    pass


@lru_cache(maxsize=1)
def oracle_fit_code():
    import oracle_timing as ot
    tree = ast.parse(inspect.getsource(ot.job))
    node = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    return compile(ast.Module(body=[node], type_ignores=[]), str(Path(ot.__file__)), 'exec')


def oracle_fit(render, timing, densities, reference_lab, keep, level=1):
    """Execute the exact nested fit() AST from oracle_timing.job, unchanged.

    render(timing) must render the supplied cell-density array. The densities
    argument documents that contract; the callback closes over the array.
    """
    import oracle_timing as ot
    from scipy.optimize import least_squares
    namespace = {'np': np, 'replace': replace, 'least_squares': least_squares,
                 'lab_lin': ot.lab_lin, 'render': render, 't0': timing,
                 'R': np.asarray(reference_lab), 'keep': np.asarray(keep, bool)}
    exec(oracle_fit_code(), namespace)
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
                         'auto_checks': [], 'boundary_frames': {},
                         'method': method_metadata()}}


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


def method_metadata():
    return {'cell_bootstraps': CELL_BOOTSTRAPS, 'frame_bootstraps': FRAME_BOOTSTRAPS,
            'seed': SEED, 'cell_seed': 'SeedSequence([20261025, crc32(scene), 1])',
            'frame_seed': 'SeedSequence([20261025, 2])',
            'pooled': 'OLS with one intercept on flattened paired R/B frame values',
            'frame_sampling': 'resample frames jointly for R/B/K; percentile 2.5/97.5',
            'undefined_bootstrap_slopes': 'omit zero-variance draws; record count',
            'within_sd': 'sample standard deviation, ddof=1',
            'scene_order': 'TRAIN insertion order then HOLDOUT insertion order',
            'configuration': 'accept_t02.scene_job(({}, sid)) with apply = lambda params: '
                             'accept_t03._use("next"); nominal Params.scan_color_strength=None, '
                             'Variation.scan_strength=0.75 (T04 clarification v2)'}


def scene_seed(sid):
    return np.random.SeedSequence([SEED, zlib.crc32(sid.encode()), 1])


def capture_auto(sid, scene_job=None):
    """Observe return locals of the *actual* scene_job and Engine.scan.

    Profiling only observes Python return events. No render/alignment function
    is replaced, no trace flag is injected, and no random calls are added.
    The profile is restored even on failure. This runs in a disposable process.
    """
    from filmlab.engine import Engine
    import accept_t02
    import accept_t03
    if scene_job is None:
        scene_job = accept_t02.scene_job
    original_apply = accept_t02.apply
    scans, state = {}, {}
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == 'return' and frame.f_code is Engine.scan.__code__:
            loc = frame.f_locals
            if arg is not None:
                scans[id(loc['self'])] = (loc['Dn'].reshape(*loc['shape'], 3).copy(),
                                          loc['sc'].timing)
        elif event == 'return' and frame.f_code is scene_job.__code__:
            state.update(frame.f_locals)

    try:
        # Same entry-point override as accept_t03.scene_job; renderer unchanged.
        accept_t02.apply = lambda params: accept_t03._use("next")
        sys.setprofile(observe)
        measured_sid, measured = scene_job(({}, sid))
    finally:
        sys.setprofile(previous)
        accept_t02.apply = original_apply
    if measured_sid != sid or 'eng' not in state or id(state['eng']) not in scans:
        raise RuntimeError('scene_job/Engine.scan observation contract changed')
    density, timing = scans[id(state['eng'])]
    return {'sid': sid, 'measured': measured, 'Dn': density, 'timing': timing,
            'scanner': state['eng'].scanner, 'aligned': state['w'],
            'real': state['real'], 'ok': state['ok'], 'H': state['Hm'],
            'distance_um': state.get('d') if np.any(state.get('core', False)) else None}


def encoded_lab(rgb):
    import colour
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1)))


def regions(real_lab, ok, distance_um=None):
    masks = {'全图': ok, '暗部': ok & (real_lab[..., 0] < 25),
             '中间调': ok & (real_lab[..., 0] >= 25) & (real_lab[..., 0] <= 70),
             '亮部': ok & (real_lab[..., 0] > 70)}
    if distance_um is not None:
        d = distance_um
        masks.update({'核心': ok & (d <= 100), '近': ok & (d > 100) & (d <= 300),
                      '中': ok & (d > 300) & (d <= 1500), '远': ok & (d > 1500)})
    return masks


def ledger_stage(encoded, real_lab, masks):
    import colour
    ok = masks['全图']
    lab = encoded_lab(np.where(ok[..., None], encoded, 0))
    de = colour.delta_E(lab, real_lab, method='CIE 2000')
    delta = lab-real_lab
    out = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        out[name] = {'n': n, 'dE': float(np.median(de[mask])) if n else None,
                     **{key: float(np.median(delta[..., i][mask])) if n else None
                        for i, key in enumerate(('dL', 'da', 'db'))}}
    return out


def cell_bootstrap(scanner, timing, densities, reference, keep, sid):
    """Resample paired 24px cells, including their fixed L*>5 eligibility."""
    rng = np.random.default_rng(scene_seed(sid))
    values = []
    n = len(densities)
    for _ in range(CELL_BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        if not np.any(keep[idx]):
            raise RuntimeError('Bootstrap draw has no eligible cells; measurement undefined')
        dc = densities[idx]
        def render(t):
            scanner.timing = t
            return scanner.to_rgb(dc)
        fitted, x = oracle_fit(render, timing, dc, reference[idx], keep[idx])
        v = absolute_controls(timing, fitted, x)
        values.append([v['K_lab'], *v['c_lab']])
        if len(values) % 20 == 0:
            print(f'{sid}: cell bootstrap {len(values)}/{CELL_BOOTSTRAPS}', flush=True)
    se = np.std(values, axis=0, ddof=1)
    return float(se[0]), se[1:].tolist()


def measure_context(context, sid, group, boundary_count=0):
    """Fit only after caller verified closure; operates on captured auto arrays."""
    import colour
    import oracle_timing as ot
    from skimage.transform import warp, ProjectiveTransform
    from filmlab.io import srgb_encode
    c = context
    real_lab = encoded_lab(c['real'])
    masks = regions(real_lab, c['ok'], c['distance_um'])
    ledger = {'auto': ledger_stage(c['aligned'], real_lab, masks)}
    # Catch instrumentation/metric drift before a real oracle fit.
    if abs(ledger['auto']['全图']['dE']-c['measured']['full']) >= CLOSURE_TOLERANCE:
        raise DefinitionMismatch('Captured auto ledger disagrees with direct scene_job')
    tf = ProjectiveTransform(c['H'])
    wp = lambda a: warp(a, tf.inverse, output_shape=c['real'].shape[:2], order=1, cval=np.nan)
    dw = wp(c['Dn'])
    ok = c['ok'] & np.all(np.isfinite(dw), -1)
    dc, indices = ot.cells(dw, ok)
    rc, reference_indices = ot.cells(c['real'], ok)
    if indices != reference_indices or not len(dc):
        raise RuntimeError('No consistently paired 24px cells')
    reference = ot.lab_lin(colour.models.eotf_sRGB(rc))
    keep = reference[:, 0] > 5
    if not np.any(keep):
        raise RuntimeError('No L*>5 cells for O1/O2')
    scanner, t0 = c['scanner'], c['timing']
    def render(t):
        scanner.timing = t
        return scanner.to_rgb(dc)
    try:
        t1, x = oracle_fit(render, t0, dc, reference, keep, 1)
        values = absolute_controls(t0, t1, x)
        boundaries = boundary_keys(x)
        frame = {**values, 'group': group, 'x': x.tolist(),
                 'n_cells': int(keep.sum()), 'n_grid_cells': len(dc),
                 'boundary': boundaries, 'c_lab_se': None, 'K_lab_se': None}
        if boundaries and boundary_count >= 3:
            return {'frame': frame if sid != 'cafe' else None, 'ledger': ledger,
                    'boundary': boundaries, 'status': 'stopped_boundary_frames'}
        if sid != 'cafe':
            frame['K_lab_se'], frame['c_lab_se'] = cell_bootstrap(scanner, t0, dc, reference, keep, sid)
        t2, x2 = oracle_fit(render, t0, dc, reference, keep, 2)
        for stage, timing in [('O1', t1), ('O2', t2)]:
            scanner.timing = timing
            # As auto: render unwarped density, clip+encode, then warp RGB.
            linear = scanner.to_rgb(c['Dn'].reshape(-1, 3)).reshape(c['Dn'].shape)
            aligned = wp(srgb_encode(np.clip(linear, 0, 1)))
            if not np.all(np.isfinite(aligned[c['ok']])):
                raise RuntimeError(f'{stage}: fitted image changed valid-pixel support')
            ledger[stage] = ledger_stage(aligned, real_lab, masks)
        return {'frame': frame if sid != 'cafe' else None, 'ledger': ledger,
                'boundary': boundaries, 'O2_x': x2.tolist(), 'status': 'complete'}
    finally:
        scanner.timing = t0


def scene_worker(job):
    sid, group, baseline, boundary_count = job
    c = capture_auto(sid)
    check = check_closure(sid, c['measured']['full'], baseline)
    if not check['closed']:
        return {'check': check, 'status': 'stopped_auto_not_closed'}
    measured = measure_context(c, sid, group, boundary_count)
    return {'check': check, **measured}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x-x.mean(axis=0)
    denom = np.sum(xc*xc, axis=0)
    return np.divide(np.sum(xc*(y-y.mean(axis=0)), axis=0), denom,
                     out=np.full_like(denom, np.nan), where=denom > 0)


def frame_regressions(frames):
    """Sorted frame order; R/B are sampled together, never as separate frames."""
    ids = sorted(frames)
    rows = [frames[s] for s in ids]
    ca = np.asarray([v['c_auto'] for v in rows])
    cl = np.asarray([v['c_lab'] for v in rows])
    ka = np.asarray([v['K_auto'] for v in rows])
    kl = np.asarray([v['K_lab'] for v in rows])
    def estimates(idx):
        pair = slope(ca[idx], cl[idx])
        return np.array([*pair, slope(ca[idx].ravel(), cl[idx].ravel()), slope(ka[idx], kl[idx])])
    point = estimates(np.arange(len(rows)))
    if not np.all(np.isfinite(point)):
        raise RuntimeError('Regression is undefined: insufficient predictor variance')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    draws = np.asarray([estimates(rng.integers(0, len(rows), len(rows))) for _ in range(FRAME_BOOTSTRAPS)])
    out, omitted = {}, {}
    for j, name in enumerate(('R', 'B', 'pooled', 'K')):
        valid = np.isfinite(draws[:, j])
        if not valid.any():
            raise RuntimeError(f'All bootstrap slopes undefined: {name}')
        lo, hi = np.percentile(draws[valid, j], [2.5, 97.5])
        out[name] = [float(point[j]), float(lo), float(hi)]
        omitted[name] = int((~valid).sum())
    out['leave_group_out'] = {}
    for group in sorted({v['group'] for v in rows}):
        idx = np.array([i for i, v in enumerate(rows) if v['group'] != group], int)
        v = estimates(idx)
        if not np.all(np.isfinite(v[[0, 1, 3]])):
            raise RuntimeError(f'Leave-group-out regression undefined: {group}')
        out['leave_group_out'][group] = v[[0, 1, 3]].tolist()
    return out, omitted


def classify(interval):
    _, lo, hi = interval
    if -.25 <= lo <= hi <= .25:
        return '冲印店基本不逐帧校色'
    if .75 <= lo <= hi <= 1.25:
        return '校色力度与模型相当，颜色误差主要不在定光'
    if .25 < lo <= hi < .75:
        return '部分校色，力度 = k'
    return '数据不足以区分'


def lightbox_summary(frames):
    out = {}
    for group in LIGHTBOX_GROUPS:
        rows = [frames[s] for s in sorted(frames) if frames[s]['group'] == group]
        if not rows:
            continue
        out[group] = {}
        for key in ('c_lab', 'c_auto'):
            values = np.asarray([v[key] for v in rows])
            out[group][key+'_mean'] = np.mean(values, axis=0).tolist()
            out[group][key+'_within_sd'] = np.std(values, axis=0, ddof=1).tolist() if len(rows)>1 else None
    return out


def finalize(out, requested):
    out['metadata']['status'] = 'complete' if len(requested) == 21 else 'single_scene_complete'
    out['metadata']['measurement_complete'] = len(requested) == 21
    if len(requested) == 21:
        out['k'], omitted = frame_regressions(out['frames'])
        out['lightbox'] = lightbox_summary(out['frames'])
        out['verdict'] = classify(out['k']['pooled'])
        out['metadata']['undefined_frame_bootstrap_draws'] = omitted
        groups = list(out['lightbox'])
        out['metadata']['lightbox_between_mean_differences'] = {
            a+' - '+b: {key: (np.array(out['lightbox'][a][key+'_mean'])-
                                  out['lightbox'][b][key+'_mean']).tolist()
                        for key in ('c_lab', 'c_auto')}
            for i, a in enumerate(groups) for b in groups[i+1:]}
    return out


def worker_call(job, timeout):
    # accept_t02.apply patches process state; never reuse a scene process.
    with mp.get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
        return pool.apply_async(scene_worker, (job,)).get(timeout=timeout)


def run_measurement(scenes, output, runner=worker_call, timeout=14400, resume=False):
    from accept_calibration import TRAIN, HOLDOUT
    groups = {**TRAIN, **HOLDOUT}
    if len(set(scenes)) != len(scenes) or not set(scenes) <= groups.keys():
        raise ValueError('Unknown or duplicate T04 scene')
    baseline = json.loads((ROOT/'tools/eval/t03_baseline.json').read_text())['scenes']
    fingerprint = {'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'baseline_sha256': hashlib.sha256((ROOT/'tools/eval/t03_baseline.json').read_bytes()).hexdigest()}
    if resume:
        out = json.loads(Path(output).read_text())
        meta = out['metadata']
        if meta['status'].startswith('stopped_'):
            raise ValueError('Scientific stop cannot be resumed; resolve the specification first')
        if meta.get('fingerprint') != fingerprint or meta.get('requested_scenes') != scenes:
            raise ValueError('Checkpoint code/baseline/scene list differs')
    else:
        out = empty_result()
        out['metadata'].update(requested_scenes=scenes, fingerprint=fingerprint, completed_scenes=[])
    for sid in scenes:
        if sid in out['metadata']['completed_scenes']:
            continue
        job = (sid, groups[sid], baseline[sid]['full'], len(out['metadata']['boundary_frames']))
        try:
            result = runner(job, timeout)
            out['metadata']['auto_checks'].append(result['check'])
            if 'frame' in result and result['frame'] is not None:
                out['frames'][sid] = result['frame']
            if 'ledger' in result:
                out['ledger'][sid] = result['ledger']
            if result.get('boundary'):
                out['metadata']['boundary_frames'][sid] = result['boundary']
            if 'O2_x' in result:
                out['metadata'].setdefault('O2_x', {})[sid] = result['O2_x']
            out['metadata']['status'] = result['status']
            if result['status'] == 'complete':
                out['metadata']['completed_scenes'].append(sid)
            atomic_json(output, out)
            print(json.dumps({'scene': sid, 'status': result['status']}, ensure_ascii=False), flush=True)
            if result['status'] != 'complete':
                return out
        except Exception as exc:
            out['metadata'].update(status='stopped_definition_mismatch' if isinstance(exc, DefinitionMismatch)
                                   else 'execution_failed', failed_scene=sid,
                                   error=f'{type(exc).__name__}: {exc}')
            atomic_json(output, out)
            raise
    try:
        finalize(out, scenes)
    except Exception as exc:
        out['metadata'].update(status='analysis_failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(output, out)
        raise
    atomic_json(output, out)
    return out


def write_report(result, path):
    """Render actual completed measurements only; never invent missing plots."""
    if not result['metadata']['measurement_complete']:
        raise ValueError('Full scientific report requires a complete measurement')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figdir = path.parent/'figures/t04'
    figdir.mkdir(parents=True, exist_ok=True)
    frames, k = result['frames'], result['k']
    groups = sorted({v['group'] for v in frames.values()})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for j, ax in enumerate(axes):
        for i, group in enumerate(groups):
            rows = [frames[s] for s in sorted(frames) if frames[s]['group'] == group]
            ax.errorbar([r['c_auto'][j] for r in rows], [r['c_lab'][j] for r in rows],
                        yerr=[r['c_lab_se'][j] for r in rows], fmt='o',
                        color=plt.cm.tab10(i), label=f'G{i+1}')
        ax.set(xlabel='c_auto (density)', ylabel='c_lab (density)', title=('R-G', 'B-G')[j])
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(figdir/'timing_scatter.png', dpi=160); plt.close(fig)
    from accept_calibration import TRAIN, HOLDOUT
    scene_groups = {**TRAIN, **HOLDOUT}
    ledger_groups = sorted({scene_groups[s] for s in result['ledger']})
    region_order = ['全图', '核心', '近', '中', '远', '暗部', '中间调', '亮部']
    summaries, labels, data = {}, [], []
    for group in ledger_groups:
        for stage in ('auto', 'O1', 'O2'):
            row = []
            for region in region_order:
                values = [stages[stage][region]['dE'] for sid, stages in result['ledger'].items()
                          if scene_groups[sid] == group and region in stages[stage]
                          and stages[stage][region]['n'] > 0]
                row.append(float(np.mean(values)) if values else np.nan)
            key = f'G{ledger_groups.index(group)+1}/{stage}'
            labels.append(key); data.append(row); summaries[key] = row
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=(11, max(5, len(labels)*.3)))
    im = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(region_order)),
                  ['Full', 'Core', 'Near', 'Middle', 'Far', 'Dark', 'Midtones', 'Highlights'], rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(region_order)):
            if np.isfinite(data[i, j]):
                mark = '*' if labels[i].endswith('/O2') and data[i, j] > 5 else ''
                ax.text(j, i, f'{data[i,j]:.2f}{mark}', ha='center', va='center', fontsize=8,
                        color='white' if data[i,j] < np.nanmedian(data) else 'black')
    fig.colorbar(im, ax=ax, label='Mean of scene median Delta E00')
    ax.set_title('Regional ledger (* O2 > 5; blank = no eligible pixels)')
    fig.tight_layout(); fig.savefig(figdir/'ledger_heatmap.png', dpi=160); plt.close(fig)
    lines = ['# T04 冲印店逐帧校色力度测量', '', f"结论：**{result['verdict']}**。合并 k = {k['pooled'][0]:.6f}。",
             '', '仅作光号测量；没有更改生产参数。', '', '| 回归 | k | 95% 下限 | 95% 上限 |', '|---|---:|---:|---:|']
    for name in ('R', 'B', 'pooled', 'K'):
        lines.append('| '+name+' | '+' | '.join(f'{x:.6f}' for x in k[name])+' |')
    lines += ['', '格子 bootstrap 200 次；帧 bootstrap 10000 次。固定种子、成对重采样、合并回归定义见 JSON metadata.method。',
              '', '| 剔除组 | kR | kB | kK |', '|---|---:|---:|---:|']
    for group, values in k['leave_group_out'].items():
        lines.append('| '+group+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines += ['', '| 灯箱组 | 帧数 | SD 自由度 | c_lab 均值 R/B | 组内 SD R/B | c_auto 均值 R/B | 组内 SD R/B |', '|---|---:|---:|---|---|---|---|']
    for group, values in result['lightbox'].items():
        n = sum(f['group'] == group for f in frames.values())
        lines.append(f'| {group} | {n} | {n-1} | '+' | '.join(str(values[key]) for key in
                      ('c_lab_mean','c_lab_within_sd','c_auto_mean','c_auto_within_sd'))+' |')
    lines += ['', '灯箱组实际为4 / 4 / 3帧。组内样本SD使用ddof=1；品红灯管3帧，自由度为2。',
              '', '组间均值差（前组减后组，R/B）：', '', '```json',
              json.dumps(result['metadata']['lightbox_between_mean_differences'], ensure_ascii=False, indent=2), '```',
              '', '散点颜色：'+', '.join(f'G{i+1}={g}' for i,g in enumerate(groups))+'。',
              '', '![颜色光号与格子标准误](figures/t04/timing_scatter.png)',
              '', '账本组：'+', '.join(f'G{i+1}={g}' for i,g in enumerate(ledger_groups))+'。',
              '', '![账本热图](figures/t04/ledger_heatmap.png)',
              '', '| 组/阶段 | '+' | '.join(region_order)+' |', '|---|'+'---:|'*len(region_order)]
    for label, row in summaries.items():
        lines.append('| '+label+' | '+' | '.join('—' if not np.isfinite(v) else
                     f'{v:.3f}'+(' **>5**' if label.endswith('/O2') and v>5 else '') for v in row)+' |')
    lines += ['', '触边帧（O1 任一参数距上下限 <2%）：', '',
              '```json', json.dumps(result['metadata']['boundary_frames'], ensure_ascii=False, indent=2), '```',
              '', '完整逐场景、阶段、区域 ΔE/ΔL/Δa/Δb 与光号在结果 JSON 中。未运行 accept_t04.py；独立验收仍由 Claude 执行。', '']
    path.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene_or_output', help='scene ID for single scene, or output JSON for all 21')
    p.add_argument('output', nargs='?')
    p.add_argument('--resume', action='store_true', help='Resume interrupted execution only, never a scientific stop')
    p.add_argument('--report', type=Path, help='Generate Markdown/figures from completed measurement')
    a = p.parse_args()
    from accept_calibration import TRAIN, HOLDOUT
    scenes = [a.scene_or_output] if a.output else list(TRAIN)+list(HOLDOUT)
    output = a.output or a.scene_or_output
    # A stopped output is an explicit checkpoint, not permission to rerun it.
    if Path(output).exists() and not a.resume:
        p.error('Output already exists; preserve the recorded run and use a new output path')
    result = run_measurement(scenes, output, resume=a.resume)
    if a.report and result['metadata']['measurement_complete']:
        write_report(result, a.report)
    return 0 if result['metadata']['status'] in ('complete', 'single_scene_complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
