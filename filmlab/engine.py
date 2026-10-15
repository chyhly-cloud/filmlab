"""色彩科学引擎: 七个物理阶段。

  S0 场景     线性 RGB → 光谱 (基光谱; 可扩展近红外估算)
  S1 曝光     光源 × 滤镜 × 乳剂光谱感光度 → 每层 log H; 测光、老化、光晕、散射
  S2 冲洗     化学步骤序列 → 银 / 染料 / 卤化银 (chemistry.run); 颗粒
  S3 底片     染料光谱 + 银 + 片基 → 光谱透射率
  S4 印放     负像: 放大机光源 → 相纸感光 (光号自动定时); 正像: 直接观看
  S5 相纸     相纸也走自己的冲洗 (RA-4 / 黑白) → 反射 (或拷贝片透射) 光谱
  S6 观看     观看光源 × CIE 观察者 → XYZ → 显示器

校色原则 (和现实一致):
  · 负片: 冲印店按灰卡调整每张底片的印放光号 → 自动校色, 光号会报告出来
  · 反转片: 只有一次"材料校准" (标准冲洗、标准光源下让中灰中性, 用来抵消数据误差),
    之后任何偏离标准的条件 (交叉冲洗、错误光源、推冲) 造成的偏色都保留
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from . import chemistry, film, spectral
from .chemistry import PROCESSES, Process
from .materials import Emulsion, emulsion
from . import variation

PRINTER_POINT = 0.025      # 1 个印放光号 = 0.025 logE (电影 / 冲印行业惯例)
GRAY = 0.184


@dataclass
class Params:
    # ---- 材料
    film: str | Emulsion = "kodak_portra_400"
    paper: str | Emulsion | None = None        # None = 胶片推荐的相纸 / 拷贝片
    # ---- S0/S1 拍摄
    exposure_ev: float = 0.0                   # 相对于按包装感光度测光
    # 输入 RGB 的参考光源。默认 D55: RAW 以固定日光白平衡读入, 场景光色已包含在 RGB 里
    # (钨丝灯场景本身就是橙色的)。若输入已用相机白平衡"拉正", 设为 None = 胶片自己的平衡光源。
    scene_illuminant: str | None = "D55"
    lens_filter: np.ndarray | None = None      # (81,) 镜头前滤镜光谱透射率
    frame_width_mm: float = 36.0
    exposure_hook: object = None               # (rgb, H) -> H, 例: 红外伪彩的近红外估算
    # ---- S2 冲洗
    process: str | Process | None = None       # None = 原生工艺
    push: float = 0.0                          # 推 (+) / 拉 (−) 档数
    bleach: float | None = None                # 漂白效率, 0 = 漂白旁路
    # ---- S4 印放
    # 冲印店定光: frame = 按整张底片的平均密度 (自动冲印机 LATD 方式, 偏色基本被校正)
    #            gray  = 按拍摄光下的灰卡
    #            standard = 按标准光源下的灰卡, 不做逐张校正 (光源偏色会保留)
    timing: str = "gray"
    printer_points: tuple = (0.0, 0.0, 0.0)    # 在自动光号上手动加减 (R G B, +号 = 该层多曝光)
    print_filter: tuple = (0.0, 0.0, 0.0)      # Durst 分色滤镜 C M Y
    mid_gray: float = 0.18
    # ---- 颗粒
    grain: float = 1.0                         # 颗粒倍数 (1 = 按感光度估算)
    # ---- 物理随机性 (variation.py): 哪一卷胶片 / 哪一次冲洗 / 第几张。roll=None 为标称值
    roll: int | None = 0
    lab: int | None = None
    frame: int = 0


def _resolve(x, default=None):
    if x is None:
        x = default
    return emulsion(x) if isinstance(x, str) else x


def _process(e: Emulsion, p: Params) -> Process:
    proc = p.process or e.native_process
    proc = PROCESSES[proc] if isinstance(proc, str) else proc
    if p.push:
        proc = proc.push(p.push)
    if p.bleach is not None:
        proc = proc.with_bleach(p.bleach)
    return proc


def _iso(e: Emulsion):
    import re
    m = re.search(r"(\d{2,4})", e.name)
    return float(m.group(1)) if m else 200.0


# --------------------------------------------------------------- S1 --
def _speed_point(e: Emulsion):
    """按包装感光度测光时, 中灰在原生冲洗的成像银曲线上的位置。
    负片: 感光点 (灰雾上 0.1) 再往上 0.9 logH; 反转片: 正像 D≈1.0 处 (第一显影银 = Dmax − 1.0)。"""
    x, S = e.log_exposure, e.S_nom[:, 1]
    if e.design == "reversal":
        return x[np.argmin(np.abs((e.halide_cap[1] - S) - 1.0))]
    return x[np.argmax(S > S.min() + 0.1)] + 0.9


class Engine:
    def __init__(self, p: Params):
        self.p = p
        self.e = e = _resolve(p.film)
        self.proc = _process(e, p)
        st = e.stock
        ref = st.reference_illuminant
        self.v = variation.sample(p.roll, p.frame, p.lab)
        self.M_ref = film.exposure_response(st, ref, p.lens_filter)
        self.M = film.exposure_response(st, p.scene_illuminant or ref, p.lens_filter)
        self.halo = film.halation_response(st, p.scene_illuminant or ref, p.lens_filter, self.v, e.halation) \
            if e.halation > 0 and p.exposure_hook is None else None
        g = self._expose_lin(np.full((1, 1, 3), GRAY), self.M_ref, compensation=False)[0, 0]
        self.log_k = _speed_point(e) - np.log10(g[1])        # 测光 (透过滤镜, 含滤镜因数)
        self.layer_cal = np.zeros(3)
        if e.design == "reversal" and isinstance(p.film, str):
            # 有 S0 扩展 (如红外层映射) 时, 它和滤镜一起构成一种独立的乳剂, 需要整体校准
            self.layer_cal = _calibrate_reversal(self) if p.exposure_hook is not None else _material_calibration(e.key)
        self.px_um = 10.0
        # 正像还是负像由化学结果决定, 不由胶片类型决定
        self.polarity = self._polarity()
        self.paper = None
        if self.polarity == "negative":
            if p.paper is not None:
                self.paper = _resolve(p.paper)
            elif e.design == "bw":
                self.paper = _bw_paper()
            else:
                self.paper = _resolve(e.stock.target_print or "kodak_portra_endura")
            self.paper_proc = PROCESSES[self.paper.native_process]
            self.enl_light = spectral.illuminant("TH-KG3") * spectral.enlarger_filtration(np.asarray(p.print_filter))
            self.paper_sens = 10.0 ** self.paper.stock.log_sensitivity
            self.print_k = np.zeros(3)
        transmissive = self.paper is not None and self.paper.stock.kind != "paper"
        self.view_illum = spectral.illuminant("K75P" if transmissive else "D50")
        self.white_xy = spectral.white_xy(self.view_illum)
        if self.paper is not None:
            self._timing()

    # ---- S0/S1 ----
    def _expose_lin(self, rgb, M, compensation=True):
        if compensation:
            rgb = rgb * 2.0 ** self.p.exposure_ev
        H = film.expose(rgb, M)
        if self.p.exposure_hook is not None:
            H = np.clip(self.p.exposure_hook(rgb, H), 1e-9, None)
        return H

    def _log_exposure(self, H_lin):
        e = self.e
        H = H_lin * 10.0 ** (self.log_k - e.speed_loss + self.layer_cal)
        H = H + 10.0 ** e.fog_exposure                    # 老化潜影灰雾
        return np.log10(H)

    # ---- S2/S3 ----
    def _develop(self, logH, blur_px=0.0):
        return chemistry.run(self.e, self.proc, logH, blur_px, tau_factor=self.v.tau_factor)

    def _transmittance(self, st):
        e = self.e
        dens = st.dye @ e.stock.dye_density.T + e.stock.base_density
        neutral = (st.silver * e.silver_covering).mean(-1) + 0.3 * st.halide.mean(-1)
        if st.remjet:    # 没有去除的 remjet 残留: 近似为中性灰雾
            neutral = neutral + 0.4
        return 10.0 ** -(dens + neutral[..., None])

    def _polarity(self):
        lo = self._develop(self._log_exposure(np.full((1, 3), 1e-4)))
        hi = self._develop(self._log_exposure(np.full((1, 3), 5.0)))
        t = lambda s: self._transmittance(s).mean()
        return "negative" if t(hi) < t(lo) else "positive"

    # ---- S4/S5/S6 ----
    def _view(self, st):
        T = self._transmittance(st)
        if self.paper is None:
            xyz = spectral.spectrum_to_xyz(T, self.view_illum)
            white = spectral.spectrum_to_xyz(10.0 ** -self.e.stock.base_density, self.view_illum)
            return xyz / white[1]
        Hp = T @ (self.enl_light[:, None] * self.paper_sens)
        self._last_Hp = Hp.reshape(-1, 3)[0].copy()
        pts = np.asarray(self.p.printer_points) * PRINTER_POINT
        logHp = np.log10(np.clip(Hp, 1e-12, None)) + self.print_k + pts
        pst = chemistry.run(self.paper, self.paper_proc, logHp)
        ps = self.paper.stock
        dens = pst.dye @ ps.dye_density.T + ps.base_density + (pst.silver * self.paper.silver_covering).mean(-1)[..., None]
        R = 10.0 ** -dens
        xyz = spectral.spectrum_to_xyz(R, self.view_illum)
        white = spectral.spectrum_to_xyz(10.0 ** -ps.base_density, self.view_illum)
        return xyz / white[1]

    def view_xyz(self, st, chunk=1 << 18):
        """S3 → S6, 分块计算。返回相对 (相纸白 / 片基 = 1) 的 XYZ。"""
        shape = st.dye.shape
        n = int(np.prod(shape[:-1]))
        if n <= chunk:
            return self._view(st)
        flat = lambda a: a.reshape(-1, 3)
        out = np.empty((n, 3))
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            sub = chemistry.FilmState(latent=None, halide=flat(st.halide)[sl], silver=flat(st.silver)[sl],
                                      dye=flat(st.dye)[sl], coupler=None, remjet=st.remjet)
            out[sl] = self._view(sub)
        return out.reshape(shape)

    def _gray_state(self, actual_light=False, rgb=None):
        """灰卡 (或任意 rgb) 的冲洗结果。actual_light=True: 用实际拍摄光源。"""
        rgb = np.full((1, 3), GRAY) if rgb is None else np.asarray(rgb, float).reshape(1, 3)
        H = self._expose_lin(rgb, self.M if actual_light else self.M_ref)
        return self._develop(self._log_exposure(H))

    def _timing(self, target_rgb=None):
        """冲印店定光号: 调整三层印放曝光, 让目标 (灰卡或整张底片的平均) 印成中性 18% 灰。"""
        import colour
        if self.p.timing == "standard":
            st = self._gray_state(actual_light=False)
        else:
            st = self._gray_state(actual_light=True, rgb=target_rgb)
        self._view(st)
        pc = self.paper.S_nom
        k0 = np.array([self.paper.log_exposure[np.argmax(pc[:, c] > 0.8)] for c in range(3)])
        k0 -= np.log10(self._last_Hp)
        self.print_k = k0

        def resid(v):
            self.print_k = np.asarray(v)
            xyz = self._view(st)[0]
            xy = colour.XYZ_to_xy(np.clip(xyz, 1e-9, None))
            return [(xy[0] - self.white_xy[0]) * 20, (xy[1] - self.white_xy[1]) * 20,
                    np.log10(max(xyz[1], 1e-9) / self.p.mid_gray)]
        least_squares(resid, k0, bounds=(k0 - 3, k0 + 3))

    # ---- 主入口 ----
    def render_xyz(self, rgb_lin):
        p, e = self.p, self.e
        h, w = rgb_lin.shape[:2]
        self.px_um = p.frame_width_mm * 1000.0 / max(h, w)
        if self.paper is not None and p.timing == "frame":
            # 自动冲印机测整张底片的平均密度 (LATD), 以此定光
            lin = np.clip(rgb_lin.reshape(-1, 3), 1e-4, None)
            avg = np.exp(np.log(lin).mean(0))
            avg *= GRAY / (avg @ np.array([0.2126, 0.7152, 0.0722]))
            self._timing(avg / 2.0 ** p.exposure_ev)
        H = self._expose_lin(rgb_lin, self.M)
        H = film.emulsion_scatter(H, self.px_um)
        if self.halo is not None:
            H = film.halation(H, self.halo(rgb_lin * 2.0 ** p.exposure_ev), self.px_um, self.v)
        st = self._develop(self._log_exposure(H), blur_px=12.0 / self.px_um)
        self._agitation(st)
        self._grain(st)
        self.state = st
        return self.view_xyz(st)

    def render(self, rgb_lin):
        """→ 线性 sRGB (D65)。"""
        return spectral.xyz_to_srgb(self.render_xyz(rgb_lin), self.white_xy)

    def _grain(self, st):
        """颗粒: 成像银的离散性。σ 按 datasheet 惯例 (48µm 孔径, Selwyn 定律) 换算到像素。
        染料云跟随银颗粒 → 加在染料上; 黑白则加在银上。"""
        if self.p.grain <= 0 or st.image_silver is None:
            return
        e = self.e
        iso = _iso(e)
        rms48 = 0.0038 * np.sqrt(iso / 100.0) * (1.6 if e.design == "reversal" else 1.0) \
            * (1.4 if e.design == "bw" else 1.0) * st.grain_factor * self.p.grain
        sigma = rms48 * 48.0 / self.px_um
        blur = (1.0 + iso / 100.0) / self.px_um
        rng = np.random.default_rng(self.v.frame_seed)
        from scipy.ndimage import gaussian_filter
        tgt = st.dye if e.has_couplers and np.any(st.dye > 0) else st.silver
        for c in range(3):
            n = rng.standard_normal(tgt.shape[:2])
            if blur > 0.35:
                n = gaussian_filter(n, blur)
                n /= n.std() + 1e-9
            d = np.clip(tgt[..., c], 0, None)
            sat = np.clip(1 - d / (e.halide_cap[c] * 1.05), 0.05, 1)
            tgt[..., c] = np.clip(tgt[..., c] + n * sigma * np.sqrt(d * sat), 0, None)

    def _agitation(self, st):
        """冲洗时搅拌不均: 显影液在片面上的补充不完全均匀 → 低频 (毫米级) 的密度起伏。"""
        a = self.v.agitation_density
        if a <= 0 or st.dye.ndim < 3 or min(st.dye.shape[:2]) < 32:
            return
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(self.v.frame_seed + 7)
        n = gaussian_filter(rng.standard_normal(st.dye.shape[:2]), 4000.0 / self.px_um)
        n = n / (n.std() + 1e-12) * a
        tgt = st.dye if np.any(st.dye > 0) else st.silver
        tgt *= (1.0 + n)[..., None]

    def describe(self):
        """把整条物理链条写出来 (含冲洗日志与印放光号)。"""
        st = self._gray_state()
        lines = [f"胶片: {self.e.name}  ({self.e.design}, 原生工艺 {self.e.native_process})",
                 f"冲洗: {self.proc.name}"] + [f"  · {l}" for l in st.log]
        v = self.v
        if self.p.roll is not None:
            lines.append(f"第 {self.p.roll} 卷 / 第 {self.p.frame} 张: 片基 {v.base_um:.0f}µm, "
                         f"冲洗偏差 τ×{v.tau_factor:.3f} (时间 {v.dev_time_factor:.3f}, 温度 {v.dev_temp_offset:+.2f}°C, "
                         f"药液 {v.activity_factor:.3f})")
        lines.append(f"成像: {'负像' if self.polarity == 'negative' else '正像'}")
        if self.paper is not None:
            try:
                ref = _reference_timing(self.e.key, self.paper.key)
            except FileNotFoundError:   # 派生材料没有标准底片可比
                ref = None
            if ref is None:             # 例: 反转片走负片工艺, 没有"标准底片"可比
                ref = self.print_k
            pts = 25 + (self.print_k - ref) / PRINTER_POINT
            lines.append(f"印放: {self.paper.name}, 光号 R{pts[0]:.0f} G{pts[1]:.0f} B{pts[2]:.0f} (标准底片 = 25-25-25)")
        else:
            lines.append("观看: 透射 (灯箱 / 放映)")
        return "\n".join(lines)


@lru_cache(maxsize=None)
def _bw_paper():
    from .materials import bw_paper
    return bw_paper()


def _solve_neutral(eng):
    import colour

    def resid(v):
        eng.layer_cal = np.asarray(v)
        xyz = eng._view(eng._gray_state())[0]
        xy = colour.XYZ_to_xy(np.clip(xyz, 1e-9, None))
        return [(xy[0] - eng.white_xy[0]) * 20, (xy[1] - eng.white_xy[1]) * 20,
                np.log10(max(xyz[1], 1e-9) / 0.18)]
    eng.layer_cal = np.zeros(3)
    return least_squares(resid, np.zeros(3), bounds=([-1.5] * 3, [1.5] * 3)).x


def _calibrate_reversal(eng):
    eng.paper = None
    eng.view_illum = spectral.illuminant("D50")
    eng.white_xy = spectral.white_xy(eng.view_illum)
    ev = eng.p.exposure_ev
    eng.p.exposure_ev = 0.0
    try:
        return _solve_neutral(eng)
    finally:
        eng.p.exposure_ev = ev


@lru_cache(maxsize=None)
def _material_calibration(key: str):
    """反转片材料校准: 标准 E-6、参考光源下, 求三层的曝光偏移让中灰中性 (只做一次)。"""
    eng = Engine.__new__(Engine)
    eng.p = Params(film=emulsion(key), grain=0, roll=None)
    eng.e = e = emulsion(key)
    eng.proc = PROCESSES[e.native_process]
    eng.v = variation.NOMINAL
    eng.M_ref = eng.M = film.exposure_response(e.stock, e.stock.reference_illuminant)
    g = film.expose(np.full((1, 3), GRAY), eng.M_ref)[0]
    eng.log_k = _speed_point(e) - np.log10(g[1])
    eng.paper = None
    eng.view_illum = spectral.illuminant("D50")
    eng.white_xy = spectral.white_xy(eng.view_illum)
    return _solve_neutral(eng)


@lru_cache(maxsize=None)
def _reference_timing(film_key: str, paper_key: str):
    paper = _bw_paper() if paper_key == "bw_paper" else paper_key
    eng = Engine(Params(film=film_key, paper=paper, grain=0, roll=None))
    return eng.print_k.copy() if eng.paper is not None else None
