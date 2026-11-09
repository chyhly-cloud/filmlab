"""颗粒的银盐物理模型: filmlab 对独立包 silvergrain 的薄接口 (T55, docs/tasks/T55_SILVERGRAIN_STANDALONE.md)。

物理全部在 silvergrain (仓库顶层 silvergrain/, 不依赖 filmlab):
  silvergrain.dye     T45 / T46 染料云与 κ 银量路径 (GrainSpec, layer_curves, derive_spec, scale_grain, …)
  silvergrain.silver  T47–T53 显式银粒路径 (SilverSpec, render_silver, silver_granularity, silver_derive, scale_silver, …)
  silvergrain.constants  通用常数 (BAND_PHOTONS, ETA_REF, GRAIN_SCALE_RANGE, …)
本模块重新导出这些名字 (原 filmlab.grain 的全部接口, 含下划线开头的内部函数), 只保留与 filmlab 胶片状态 (FilmState) 有关的
两个入口 apply / apply_silver, 以及进程级的 SILVER_WORKERS。T55 之前的 filmlab/grain.py 与这些代码逐位相同 (T55 逐位核对)。
"""
from silvergrain import constants as _sg_constants, dye as _sg_dye, silver as _sg_silver

for _m in (_sg_constants, _sg_dye, _sg_silver):
    globals().update({_k: _v for _k, _v in vars(_m).items() if not _k.startswith("__")})
del _m

SILVER_WORKERS = 1              # 整帧渲染的进程数 (按像素行分块; 结果与进程数无关); 评估脚本可改这里


def render_silver(spec, H, px_um, seed, geometry="diffuse", sub=16, count_sub=4, nphase=4, psf_um=0.0,
                  rows_per_task=16, workers=None):
    """silvergrain.silver.render_silver; workers 为 None 时取本模块的 SILVER_WORKERS。"""
    return _sg_silver.render_silver(spec, H, px_um, seed, geometry=geometry, sub=sub, count_sub=count_sub, nphase=nphase,
                                    psf_um=psf_um, rows_per_task=rows_per_task,
                                    workers=SILVER_WORKERS if workers is None else workers)


def apply(e, spec, st, px_um, rng, strength=1.0):
    """在显影后的状态上实现颗粒 (就地修改 st.dye 或 st.silver); 物理见 silvergrain.dye.apply_dye_grain。"""
    tgt = st.dye if e.has_couplers and _sg_dye.np.any(st.dye != 0) else st.silver
    if tgt.ndim < 3:
        return
    _sg_dye.apply_dye_grain(spec, tgt, px_um, rng, reversal=(e.design == "reversal"), strength=strength,
                            factor=getattr(st, "grain_factor", 1.0))


def apply_silver(e, spec, st, px_um, seed, strength=1.0):
    """T47: 黑白片的显式银粒 (就地修改 st.silver); 物理见 silvergrain.silver.silver_grain_density。"""
    tgt = st.silver
    if tgt.ndim < 3:
        return
    D = _sg_silver.silver_grain_density(spec, tgt[..., 1], px_um, seed, strength=strength, render=render_silver)
    for c in range(3):
        tgt[..., c] = D
