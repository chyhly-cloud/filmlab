"""Finalize the persistent T02 run; prepare an exact-tree GitHub connector delivery.

Waits for both conditioning fits; never reads holdout data or calls acceptance
main/evaluate. Run inside the dedicated codex/t02-lab-profile checkout.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools/eval'))
from calibrate_lab import RESULT, atomic_json, boundary_keys

FILES = ['tools/eval/calibrate_lab.py', 'tools/eval/finalize_t02.py', 'tools/eval/t02_result.json',
         'tools/eval/t02_profile_params.json', 'tools/eval/t02_test_result.json',
         'filmlab/engine.py', 'filmlab/scanner.py', 'filmlab/scanning/timing.py', 'filmlab/recipes.py',
         'filmlab/data/epfl_lab_profile.json', 'tests/test_t02_lab_profile.py', 'docs/T02_REPORT.md']


def command(args):
    return subprocess.run(args, cwd=REPO, check=True, capture_output=True, text=True, timeout=180)


def finalize(state):
    state['environment'] = {'python': platform.python_version(), 'platform': platform.system(),
                            'packages': {k: importlib.metadata.version(k) for k in
                                         ('numpy', 'scipy', 'colour-science', 'rawpy', 'scikit-image', 'pillow')}}
    selected = state['selected']
    runs = state['runs']
    baseline = runs['lab']['evaluations'][0].get('metrics') if runs['lab']['evaluations'] else None
    enabled = bool(state['status'] == 'complete' and selected and baseline
                   and selected['run_status'] == 'finished' and not boundary_keys(selected['params'])
                   and selected['metrics']['objective'] < baseline['objective'])
    atomic_json(RESULT, state)
    atomic_json(REPO / 'tools/eval/t02_profile_params.json', {} if selected is None else selected['params'])
    atomic_json(REPO / 'filmlab/data/epfl_lab_profile.json', {
        'enabled': enabled, 'params': None if selected is None else selected['params'],
        'status': 'no_candidate' if selected is None else selected['run_status'],
        'acceptance': 'pending independent Claude validation and user visual approval'})
    regression = subprocess.run([sys.executable, str(REPO / '.local/default_regression.py'), 'check'],
                                cwd=REPO, capture_output=True, text=True, timeout=1200)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=REPO,
                           capture_output=True, text=True, timeout=1200)
    atomic_json(REPO / 'tools/eval/t02_test_result.json', {
        'default_regression': {'exit_code': regression.returncode, 'stdout': regression.stdout, 'stderr': regression.stderr},
        'pytest': {'exit_code': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}})
    lines = ['# T02 EPFL 冲印店扫描档案校准', '',
             '- 基线：`handoff/recovered-research-2026-10-23` / `c2f2df7`。',
             f"- 执行状态：`{state['status']}`；只使用固定 TRAIN 11 场景，22/22 输入文件已重新核验 SHA256。",
             '- 只拟合 latd_normal、latd_limit、highlight_aim、shoulder_white、color_strength，'
             'conditioning=lab/none 分别拟合，按训练目标选较小者；不依据留出结果做选择。',
             '- 目标 = 11 场景全图 ΔE00 中位的均值 + 0.5 × 有有效光源邻域的训练场景 ΔE00 中位的均值。'
             '无有效邻域记 null、不当成零；邻域场景集合必须与该组基准一致。',
             '- 评估直接调用 accept_t02.scene_job()，由其内部恰好调用一次 apply()；实际自动定光，'
             'next、roll=None、grain=0。每场景新进程，不使用 O1 预言机。',
             '- 固定种子 20261024；每组：现值基准 + 43 点稀疏内部网格（中心、各轴四分位点、内部角点），'
             '再做原范围内有界 Powell。每组最多 200 次，总计 ≤400；所有场景和候选原子存盘、可续跑。',
             '- 任一已评估候选触及 2% 边界区域，不论目标是否改善，立即停止该组；仍独立完成另一个 conditioning。'
             '不扩大范围、不换参数、不重启已终止搜索。', '',
             '| conditioning | 状态 | 候选数 | 最佳全图 | 最佳光源旁 | 最佳目标 |',
             '|---|---|---:|---:|---:|---:|']
    for c, run in runs.items():
        if run['best_index'] is None:
            lines += [f"| {c} | {run['status']} | {len(run['evaluations'])} | — | — | — |"]
        else:
            best = run['evaluations'][run['best_index']]
            m = best['metrics']
            lines += [f"| {c} | {run['status']} | {len(run['evaluations'])} | {m['full']:.6f} | {m['near']:.6f} | {m['objective']:.6f} |"]
    lines += ['', '每组最佳参数与停止事件：', '']
    for c, run in runs.items():
        if run['best_index'] is not None:
            best = run['evaluations'][run['best_index']]
            lines += [f"- {c} 最佳参数：`{json.dumps(best['params'])}`；最佳点是否贴边：{bool(boundary_keys(best['params']))}。"]
        lines += [f"- {c} 触边停止事件：`{json.dumps(run['boundary_event'])}`。"]
    if baseline:
        lines += ['', f"- 本机 lab 现值基准：全图 {baseline['full']:.6f}，光源旁 {baseline['near']:.6f}，目标 {baseline['objective']:.6f}。",
                  '- 规格的训练基准为全图 12.308、光源旁 14.873；不同环境/配准引起的差异应由 Claude 独立复核，'
                  '没有为贴合基准修改数据、评估器或参数。']
    if selected:
        lines += [f"- 训练选择：{selected['conditioning']}，来源拟合状态 `{selected['run_status']}`；"
                  f"参数 `t02_profile_params.json` 是该组最佳已评估点，不保证该组已正常收敛或没有触边事件。"]
        if baseline:
            lines += [f"- 训练目标相对 lab 现值改善：{baseline['objective'] - selected['metrics']['objective']:.6f}。"]
    if state.get('error'):
        lines += [f"- 执行错误：{state['error']}。"]
    lines += [f'- 可选配方 `cinestill_800t_epfl_profile` 的 enabled={enabled}。触边或未正常完成的来源组不得启用，'
              '即使它的最佳点在内部且训练目标更小。最终采用仍需 Claude 独立验收和用户目视批准。',
              '- 正式接入使用 Params 的四个可选扫描覆盖字段，默认 None；颜色力度与 conditioning 使用现有字段。'
              '不修改全局常数，不污染其他引擎；胶片/显影/光晕及矩阵/tint/P/LED 数值保持原样。',
              f'- 改动前后合成输入逐位回归退出码：{regression.returncode}；三代际各 42 项，共126项，'
              '这是6幅合成输入×7种配置，不冒称 Claude 的真实回归夹具。',
              f'- 全量测试退出码：{tests.returncode}；完整输出见 `t02_test_result.json`。',
              '- 未查看或运行留出数据、缓存或验收结果，未自行报告留出指标。', '',
              '## 复现与独立验收', '',
              '拟合命令（已终止的完整日志不会自动重开搜索）：', '', '```bash',
              'FILMLAB_DATA=… OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/eval/calibrate_lab.py',
              '```', '', '由 Claude 独立执行：', '', '```bash',
              'FILMLAB_DATA=… python tools/eval/accept_t02.py tools/eval/t02_profile_params.json 输出.json',
              '```', '', '请先审查两组停止状态、边界事件和完整评估记录，再核验灰卡、默认输出逐位回归、'
              '留出验收及他店隔离。完整画面与光源局部对比图应交用户目视；不放宽范围、不根据留出结果重新拟合。']
    (REPO / 'docs/T02_REPORT.md').write_text('\n'.join(lines) + '\n')
    return regression.returncode or tests.returncode


def main():
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        state = json.loads(RESULT.read_text())
        if state['status'] != 'running':
            break
        time.sleep(30)
    else:
        raise TimeoutError('T02 did not finish within 72 hours')
    if command(['/usr/bin/git', 'rev-parse', '--abbrev-ref', 'HEAD']).stdout.strip() != 'codex/t02-lab-profile':
        raise RuntimeError('Unexpected branch; refusing delivery commit')
    test_status = finalize(state)
    command(['/usr/bin/git', 'add', '--', *FILES])
    command(['/usr/bin/git', 'commit', '-m', 'Complete both T02 conditioning fits and report training-only outcome'])
    commit = command(['/usr/bin/git', 'rev-parse', 'HEAD']).stdout.strip()
    tree = command(['/usr/bin/git', 'rev-parse', 'HEAD^{tree}']).stdout.strip()
    atomic_json(REPO / '.local/t02-delivery-ready.json', {
        'status': state['status'], 'local_commit': commit, 'tree': tree, 'files': FILES,
        'tests_exit_code': test_status, 'upload_status': 'pending GitHub connector completion follow-up'})
    print(json.dumps({'local_commit': commit, 'tree': tree, 'tests_exit_code': test_status}), flush=True)
    if test_status:
        raise SystemExit(test_status)


if __name__ == '__main__':
    main()
