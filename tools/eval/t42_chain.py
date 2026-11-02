"""T42: 从存档的 S1 logH 逐级重算模型链 (不做空间扩散), 并把每一级换算成"等效曝光" (中性灰阶反查)。

每一级 Q (三通道) 都有一条由中性灰阶 (三层同一 logH, 以 3200K 中灰 G0 为中性) 得到的单通道传递 F_c;
等效曝光 e_c = F_c^{-1}(Q_c) − G0_c。若这一级之前的整条链是"每通道一条单调曲线", e = logH − G0;
e 的通道极差相对 logH 的通道极差 (彩度比) 就是这一级之前累计的跨通道增 / 减饱和。

级 (按处理顺序):
  dye    S2 DIR 之后的三层染料量 (负片密度, 层单位)
  sm     S3 负片透射率的 Status M 积分密度 (Kodak H-1-5219 数据手册的 Status M 响应, 数字化 csv), 减片基
  scan   S4a 扫描仪窄带密度, 减片基
  dn     S4b 印放矩阵 P 与档案 gain 之后的 Dn
  dp     S4d 数字相纸密度 (逐通道; 用本帧的定光)
  out    S4d 输出线性 sRGB (相纸染料 + D50 观看 + sRGB 截断)
反事实 (只诊断): dir_fn 替换 DIR 矩阵, dye_fn 替换染料光谱, P 替换印放矩阵。
"""
import os, sys, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools/eval"), str(REPO)]
G0 = np.array([-0.83960607, -0.84922118, -0.8927258])
RAW5219 = os.environ.get("KODAK5219_RAW", "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/mask/raw5219")
RAMP = np.linspace(-3.5, 3.5, 701)


def status_m_kodak():
    from filmlab.stocks import WAVELENGTHS as W
    F = []
    for c in "rgb":
        a = np.loadtxt(f"{RAW5219}/statusM_{c}.csv", delimiter=","); a = a[np.argsort(a[:, 0])]
        F.append(np.clip(np.interp(W, a[:, 0], a[:, 1], left=0, right=0), 0, None))
    F = np.stack(F, 1)
    return F / F.sum(0)


def engine(gen, sid):
    from filmlab import generation
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    generation.use(gen)
    seed = zlib.crc32(sid.encode()) % 1000
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": seed % 36, "grain": 0}))


class Chain:
    """eng: 已按代际构造的 Engine; timing: (K, s, M_off, M_r) 本帧定光 (输出级用)。"""

    def __init__(self, eng, timing=None, dir_fn=None, dye_spectra=None, P=None, neutral=None):
        from filmlab.scanning.timing import PrintTiming
        self.eng, self.sc = eng, eng.scanner
        self.dir_fn, self.dye_spectra, self.P = dir_fn, dye_spectra, P
        self.F = status_m_kodak()
        if timing is not None:
            K, s, mo, mr = timing
            self.timing = PrintTiming(K=np.asarray(K, float), s=np.asarray(s, float), M=(np.asarray(mo, float), np.asarray(mr, float)))
        else:
            self.timing = None
        self.base_T = eng._transmittance_base()
        self.neutral = np.zeros(3) if neutral is None else np.asarray(neutral, float)
        self.ramp = self.stages(G0[None, :] + self.neutral[None, :] + RAMP[:, None])

    def develop(self, logH):
        from filmlab import chemistry
        eng = self.eng
        if self.dir_fn is None:
            return eng._develop(logH, 0.0)
        A0 = chemistry._dir_matrix
        chemistry._dir_matrix = self.dir_fn
        try:
            return eng._develop(logH, 0.0)
        finally:
            chemistry._dir_matrix = A0

    def transmittance(self, st):
        eng = self.eng
        if self.dye_spectra is None:
            return eng._transmittance(st)
        e = eng.e
        dens = st.dye @ self.dye_spectra.T + e.stock.base_density
        return 10.0 ** -dens

    def stages(self, logH, timing=None, chunk=1 << 15):
        from filmlab.scanning.optics import measure
        sc = self.sc
        st = self.develop(logH)
        n = logH.shape[0]
        out = {k: np.empty((n, 3)) for k in ("sm", "scan", "dn", "dp", "out")}
        out["dye"] = np.asarray(st.dye, float)
        bsm = -np.log10(self.base_T @ self.F)
        bsc = measure(self.base_T[None], sc.m)[0]
        P = sc.conversion.P if self.P is None else self.P
        tm = timing or self.timing
        from filmlab.chemistry import FilmState
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            sub = FilmState(latent=None, halide=st.halide[sl], silver=st.silver[sl], dye=st.dye[sl], coupler=None,
                            remjet=st.remjet, dye_agent=st.dye_agent)
            T = self.transmittance(sub)
            out["sm"][sl] = -np.log10(np.clip(T @ self.F, 1e-12, None)) - bsm
            ds = measure(T, sc.m)
            out["scan"][sl] = ds - bsc
            dn = ((ds - sc.conversion.base) @ P) * sc.conversion.gain
            out["dn"][sl] = dn
            if tm is not None:
                old = sc.timing; sc.timing = tm
                out["dp"][sl] = sc.digital_paper.density(tm.paper_exposure(dn))
                out["out"][sl] = sc.to_rgb(dn).reshape(-1, 3)
                sc.timing = old
        return out

    def equiv(self, q, stage):
        """级 stage 的三通道量 q (n,3) → 等效曝光 (相对 G0 + neutral)。超出灰阶有效段 → NaN。"""
        r = self.ramp[stage]
        e = np.full(q.shape, np.nan)
        for c in range(3):
            y = r[:, c]
            if stage == "dp":           # 相纸密度随曝光减小
                y = -y; qq = -q[:, c]
            elif stage == "out":
                y = np.log(np.clip(y, 1e-9, None)); qq = np.log(np.clip(q[:, c], 1e-9, None))
            else:
                qq = q[:, c]
            # 只用斜率不小于最大斜率 5% 的连续段 (趾部 / 肩部平台里反查不唯一)
            sl = np.gradient(y, RAMP)
            good = sl > 0.05 * sl.max()
            best, i0 = (0, 0), None
            for i, g in enumerate(np.r_[good, False]):
                if g and i0 is None: i0 = i
                if not g and i0 is not None:
                    if i - i0 > best[1] - best[0]: best = (i0, i)
                    i0 = None
            xs, ys = RAMP[best[0]:best[1]], np.maximum.accumulate(y[best[0]:best[1]])
            v = np.interp(qq, ys, xs)
            v[(qq < ys[0]) | (qq > ys[-1])] = np.nan
            e[:, c] = v
        return e


def chroma(v):
    """通道极差 (log10)。"""
    return np.ptp(v, -1)


def hue(v):
    return np.degrees(np.arctan2(np.sqrt(3) / 2 * (v[..., 1] - v[..., 2]), v[..., 0] - 0.5 * (v[..., 1] + v[..., 2]))) % 360
