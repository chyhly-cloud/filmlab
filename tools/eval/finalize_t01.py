"""Finish the authorized persistent T01 run: report, test, commit and prepare connector upload.

Run from the T01 checkout, after starting calibrate_film.py. No holdout data or
acceptance results are read. The completion follow-up uploads only the named T01 branch without force.
"""
import json
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools/eval'))
from calibrate_film import RESULT, at_boundary, atomic_json

FILES = ['tools/eval/calibrate_film.py', 'tools/eval/finalize_t01.py',
         'tools/eval/t01_calibration_result.json', 'tools/eval/t01_calibration_params.json',
         'tools/eval/t01_test_result.json', 'filmlab/generation.py', 'filmlab/data/next_cal.json',
         'tests/test_t01_calibration.py', 'docs/T01_REPORT.md']


def command(args, **kw):
    return subprocess.run(args, cwd=REPO, check=True, text=True, timeout=180, **kw)


def finalize(state):
    state['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
                            'packages': {k: importlib.metadata.version(k) for k in
                                         ('numpy', 'scipy', 'colour-science', 'rawpy', 'scikit-image', 'pillow')}}
    atomic_json(RESULT, state)
    final = state['final']
    params = None if final is None else final['params']
    boundary = None if params is None else at_boundary(params)
    enabled = bool(state['status'] == 'finished' and final and final['gamma']['passed']
                   and not boundary and state['training_improvement'] > 0)
    atomic_json(REPO / 'filmlab/data/next_cal.json', {
        'enabled': enabled, 'params': params, 'status': state['status'],
        'acceptance': 'pending independent Claude validation'})
    # The acceptance program expects a flat parameter object, not the journal.
    if params is not None:
        atomic_json(REPO / 'tools/eval/t01_calibration_params.json', params)
    else:
        atomic_json(REPO / 'tools/eval/t01_calibration_params.json', {})
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=REPO,
                          capture_output=True, text=True, timeout=1200)
    atomic_json(REPO / 'tools/eval/t01_test_result.json', {
        'exit_code': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr})
    baseline = state.get('baseline_train_mean_dE00')
    rows = state['evaluations']
    lines = ['# T01 胶片参数校准', '',
             '- 基线：`handoff/recovered-research-2026-10-23` / `fa68cb1`。',
             f"- 搜索状态：`{state['status']}`；共记录 {len(rows)} 个候选（含物理约束拒绝项），"
             f"完整训练评估 {sum(r['status'] == 'complete' for r in rows)} 个。",
             '- 只使用 `accept_calibration.TRAIN` 的固定 11 场景；没有读取留出图像、缓存或结果。',
             '- 注入直接调用 `accept_calibration.apply()`；每场景直接调用 '
             '`param_sensitivity.evaluate_scene(sid, roll=None)`，新进程隔离补丁，O1、ΔE00 中位再取场景平均。',
             '- 固定种子 0；粗搜索 3×5×3 个内部点，再在原始范围内 Powell 细化；总候选不超过 300，'
             '物理不合格候选记为拒绝，不计算训练误差；改进候选贴边时立即停止。',
             '- 训练文件由 `epfl_800t_manifest.csv` 指定，22/22 文件 SHA256 校验通过。',
             f'- 本机训练基准均值：{baseline}（规格记录 4.879；环境/重算配准差异应独立复核）。']
    if state.get('error'):
        lines += [f"- 执行失败：{state['error']}。"]
    if final:
        lines += [f'- 候选参数：`{json.dumps(params)}`。',
                  f"- 训练平均 ΔE00：{final['train_mean_dE00']:.6f}；相对本机基准改善：{state['training_improvement']:.6f}。",
                  f'- 是否贴边：{boundary}（任一参数距边界 ≤ 范围 2%）；没有放宽范围或更换拆分。',
                  f"- DIR 分色/中性 gamma 比：`{json.dumps(final['gamma']['ratios'])}`；物理检查通过：{final['gamma']['passed']}。",
                  f"- 中性 gamma：{final['gamma']['neutral_gamma']}；分色 gamma：{final['gamma']['separation_gamma']}。"]
    else:
        lines += ['- 未取得物理约束合格的拟合候选；没有可交付的校准参数。',
                  '- 参数输入 JSON 为 `{}`，仅表示现值，不得作为校准成功结果。']
    lines += ['- DIR 定义：C-41 实际化学 `run()` 生成的染料密度，对库存 logH 全轴 4097 点求最大斜率；'
              '中性时三层同步曝光，分色时仅目标层曝光、其余固定库存曝光轴最低值。'
              '专利范围来自规格/现有 `_dir_matrix` 注释；此定义在规格中未细化，供 Claude 独立审查。',
              f'- `next_cal` 为显式可选代际，当前 enabled={enabled}；只有搜索正常完成、'
              '物理合格、未贴边且训练改善时启用。其存在不表示独立验收已通过。',
              '- `next_cal` 直接复用同一个 `apply()`；退出该代际恢复原函数并清缓存。'
              '`current/next/legacy` 默认值与成像算法保持，合成输入验证逐位恢复；真实 42/42 回归留给验收方。',
              f'- 全部测试退出码：{test.returncode}；完整测试摘要见 `tools/eval/t01_test_result.json`。',
              '- 没有新增 RGB 调色、扫描拟合或额外物理参数。未自行运行留出验收。', '',
              '## 复现与独立验收', '', '拟合：', '',
              '```bash', 'FILMLAB_DATA=… OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/eval/calibrate_film.py', '```', '',
              '拟合日志在结果 JSON 中，重跑按参数和已完成场景恢复；已停止的边界搜索不会自动重启。', '',
              '由 Claude 独立执行（仅当上面确有候选）：', '',
              '```bash', 'FILMLAB_DATA=… python tools/eval/accept_calibration.py tools/eval/t01_calibration_params.json 输出.json',
              '```', '', '请独立检查 DIR 定义、灰卡、ECN-2 中性密度、留出改善与分组退步、三代际严格回归，并制作目视对比图。']
    (REPO / 'docs/T01_REPORT.md').write_text('\n'.join(lines) + '\n')
    return test.returncode


def main():
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        state = json.loads(RESULT.read_text())
        if state['status'] != 'running':
            break
        time.sleep(30)
    else:
        raise TimeoutError('Calibration did not finish within 24 hours')
    if command(['/usr/bin/git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True).stdout.strip() != 'codex/t01-film-calibration':
        raise RuntimeError('Refusing to commit on an unexpected branch')
    test_status = finalize(state)
    command(['/usr/bin/git', 'add', '--', *FILES])
    command(['/usr/bin/git', 'commit', '-m', f"Complete T01 training calibration: {state['status']}"])
    commit = command(['/usr/bin/git', 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
    tree = command(['/usr/bin/git', 'rev-parse', 'HEAD^{tree}'], capture_output=True).stdout.strip()
    atomic_json(REPO / '.local/t01-delivery-ready.json', {
        'status': state['status'], 'local_commit': commit, 'tree': tree,
        'files': FILES, 'tests_exit_code': test_status,
        'upload_status': 'pending GitHub connector completion follow-up'})
    print(json.dumps({'status': state['status'], 'local_commit': commit,
                      'tests_exit_code': test_status, 'upload': 'pending connector'}), flush=True)
    if test_status:
        raise SystemExit(test_status)


if __name__ == '__main__':
    main()
