"""T51: 直边 ESF 的模型与联合拟合 (冲印店锐化的器件参数)。

每条边 (法向距离 d, 像素): ESF(d) = lo + step·U[E0](d)
  E0 = 台阶 ⊗ 高斯 σ_e ⊗ 像素方盒      (σ_e: 镜头 / 离焦 / 胶片 / 扫描光学合成, 每条边自由)
  U[E] = E + a·(E − G_σu ⊗ E)           (冲印店 USM, 所有边共用 a、σ_u)
  另加两侧各一条线性斜坡 b_lo·(d − d0)[d < d0]、b_hi·(d − d0)[d > d0] (光晕 / 杂光长尾在拟合窗内的近似; 斜率 ≥ 0: 单调, 不产生过冲)。
  只拟合边附近 |d| ≤ 拟合半宽 (默认 5 像素): 长尾在码值域不对称 (编码非线性), 不作为 USM 的证据;
  半径大于约 2 像素的弱 USM 与长尾不可分, 由颗粒功率谱定。
光学模糊 (任何正的点扩散) 只能把台阶变宽, 不能产生过冲; 过冲 / 下冲的幅度与离边距离只由 (a, σ_u) 决定 —— 所以 USM 可由直边
测出, 不需要知道镜头与离焦。
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.special import erf

_TT = (np.arange(5) + 0.5) / 5 - 0.5          # 像素方盒 (宽 1 像素) 的 5 点中点积分


def _box_gauss_step(u, s):
    """台阶 ⊗ 高斯 σ ⊗ 像素方盒 在 u 处 (解析)。"""
    s = max(s, 0.05)
    return np.mean([0.5 * (1 + erf((u + t) / (np.sqrt(2) * s))) for t in _TT], 0)


def esf_shape(d, d0, se, w, lam, a, su):
    """归一化 ESF (0 → 1) 在采样点 d (像素) 上。w、lam: 暗侧 / 亮侧线性斜坡 (每像素, 相对台阶)。
    G_σu ⊗ (台阶 ⊗ G_σe) = 台阶 ⊗ G_√(σe² + σu²), 所以 U[E] = (1 + a)·E_σe − a·E_√(σe²+σu²) (解析, 无需数值卷积)。"""
    u = np.asarray(d, float) - d0
    E = _box_gauss_step(u, se)
    if a > 0 and su > 0.05:
        E = (1 + a) * E - a * _box_gauss_step(u, np.hypot(se, su))
    return E + w * np.minimum(u, 0.0) + lam * np.maximum(u, 0.0)


def _unpack(p, ne, shared_fixed):
    if shared_fixed is None:
        a, su = p[0], np.exp(p[1])
        q = p[2:]
    else:
        a, su = shared_fixed
        q = p
    return a, su, q.reshape(ne, 6)


def fit_edges(edges, shared_fixed=None, a0=0.5, su0=1.0, fit_halfwidth=5.0):
    """edges: [(d, esf_codes)], 返回 dict(a, sigma_u_px, per-edge 参数, 残差)。shared_fixed=(a, σu) 时只拟合每条边。"""
    ne = len(edges)
    ds, ys, idx = [], [], []
    p_edge = []
    for i, (d, y) in enumerate(edges):
        d = np.asarray(d, float); y = np.asarray(y, float)
        ok = np.isfinite(y)
        if fit_halfwidth is not None:
            ok &= np.abs(d) <= fit_halfwidth
        ds.append(d[ok]); ys.append(y[ok]); idx.append(i)
        n4 = max(4, ok.sum() // 8)
        lo, hi = np.nanmean(y[ok][:n4]), np.nanmean(y[ok][-n4:])
        p_edge.append([lo, hi - lo, 0.0, np.log(1.0), 0.0, 0.0])
    p0 = np.array(p_edge).ravel()
    if shared_fixed is None:
        p0 = np.concatenate([[a0, np.log(su0)], p0])
    off = 0 if shared_fixed is not None else 2
    nres = [len(d) for d in ds]
    starts = np.concatenate([[0], np.cumsum(nres)]).astype(int)

    def res(p):
        a, su, q = _unpack(p, ne, shared_fixed)
        r = np.empty(starts[-1])
        for i in range(ne):
            lo, st, d0, lse, bl, bh = q[i]
            m = lo + st * esf_shape(ds[i], d0, np.exp(lse), bl, bh, a, su)
            r[starts[i]:starts[i + 1]] = (m - ys[i]) / max(abs(st), 1.0)
        return r

    S = lil_matrix((starts[-1], len(p0)), dtype=int)
    for i in range(ne):
        S[starts[i]:starts[i + 1], off + 6 * i: off + 6 * i + 6] = 1
        if off:
            S[starts[i]:starts[i + 1], :2] = 1
    lb = np.full(len(p0), -np.inf); ub = np.full(len(p0), np.inf)
    for i in range(ne):
        j = off + 6 * i
        lb[j + 3], ub[j + 3] = np.log(0.1), np.log(12.0)
        lb[j + 4], ub[j + 4] = 0.0, 0.05
        lb[j + 5], ub[j + 5] = 0.0, 0.05
        lb[j + 2], ub[j + 2] = -3.0, 3.0
    if off:
        lb[0], ub[0] = 0.0, 8.0
        lb[1], ub[1] = np.log(0.2), np.log(8.0)
    p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
    sol = least_squares(res, p0, jac_sparsity=S, bounds=(lb, ub), x_scale="jac", max_nfev=400)
    a, su, q = _unpack(sol.x, ne, shared_fixed)
    r = sol.fun
    per = []
    for i in range(ne):
        lo, st, d0, lse, bl, bh = q[i]
        rr = r[starts[i]:starts[i + 1]]
        per.append(dict(lo=float(lo), step=float(st), d0=float(d0), sigma_e=float(np.exp(lse)), b_lo=float(bl),
                        b_hi=float(bh), rms=float(np.sqrt(np.mean(rr ** 2)))))
    cost = float(np.mean(r ** 2))
    return dict(a=float(a), sigma_u_px=float(su), edges=per, cost=cost, n=ne)


def overshoot_metrics(d, y, lo, step, d0):
    """直接从测得 ESF 读出: 亮侧过冲 (峰值超出平台, 相对台阶) 与暗侧下冲 (相对台阶), 只看 |d − d0| ≤ 5 像素。"""
    d = np.asarray(d) - d0; y = (np.asarray(y) - lo) / step
    hi_plat = np.nanmedian(y[d > 6]); lo_plat = np.nanmedian(y[d < -6])
    near_hi = (d > 0) & (d <= 5); near_lo = (d < 0) & (d >= -5)
    return float(np.nanmax(y[near_hi]) - hi_plat), float(lo_plat - np.nanmin(y[near_lo]))
