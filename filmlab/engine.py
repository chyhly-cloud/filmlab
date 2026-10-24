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
from .stocks import WAVELENGTHS
from . import variation
from .negative import Negative

PRINTER_POINT = 0.025      # 1 个印放光号 = 0.025 logE (电影 / 冲印行业惯例)
GRAY = 0.184
# 默认的计算修正 (Params.corrections), 由 filmlab.generation.use() 按管线代际设置
DEFAULT_CORRECTIONS = ("gray_halo", "grain_aperture")
HALATION_MEASURED = 0.1     # 历史实验值，仅供复现；旧 halo_edge 重复解码，倍率证据撤销，见 NONNEUTRAL_CHAIN_AUDIT.md
# 正常曝光的日光场景, 整张平均密度比灰卡低多少 (Portra 400, 39 张日光 RAW 按胶片测光模型、
# 不加补偿曝光的中位数, 四分位距 0.11; tools/eval 的 latd 测量)
LATD_NORMAL = -0.14


@dataclass
class Params:
    # ---- 材料
    film: str | Emulsion = "kodak_portra_400"
    paper: str | Emulsion | None = None        # None = 胶片推荐的相纸 / 拷贝片
    # ---- S0/S1 拍摄
    exposure_ev: float = 0.0                   # 相对于按包装感光度测光
    # 输入 RGB 的参考光源。默认 D55: RAW 以固定日光白平衡读入, 场景光色已包含在 RGB 里
    # (钨丝灯场景本身就是橙色的)。若输入已用相机白平衡"拉正", 设为 None = 胶片自己的平衡光源。
    # "raw" = RAW 日光白平衡下视为中性的光源 (lights.WB_REFERENCE: current 为 D55, next 为 LibRaw 实际的 D65)
    scene_illuminant: str | None = "raw"
    lens_filter: np.ndarray | None = None      # (81,) 镜头前滤镜光谱透射率
    # 场景照明: auto = 估计每个区域的光源类别 (日光 / 白炽 / 钠灯 / 荧光 / LED / 金卤),
    # 用真实光源光谱计算胶片曝光 (lights.py); None = 整个场景按 scene_illuminant 一种光源
    scene_lights: str | None = "auto"
    # 每类光源里用哪一款计算曝光: "random" = 按底片种子随机; "nearest" = 与该类所照区域的照明色度最接近的一款
    # (与光源识别用的同一判据, 见 lights.class_weights)
    light_member: str = "random"
    # 已核实的计算修正 (见 tools/eval/CRITERIA.md "计算审计"):
    #   "gray_halo"      定光用的中灰: 光晕与直接曝光用同一光源, 总量按画面用的多次反射核   [默认]
    #   "grain_aperture" 颗粒 RMS 的 48 µm 圆孔径按等面积换算                               [默认]
    #   "linear_white"   非日光光源的线性曝光映射保白点 (中性反射率的曝光精确)
    #   "paper_white"    数字相纸以相纸白为白 (介质相对渲染; 呈现约定)
    #   "halation_measured" 历史实验强度 ×HALATION_MEASURED（非实测，旧指标证据已撤销，仅供复现）
    #   "halo_oblique"   逐层、均匀深度平均的额外斜程衰减 (无自由参数; 可选)
    # corrections=() 逐位复现 2026-10-18 之前的基线 (分支 baseline-legacy)。
    corrections: tuple = field(default_factory=lambda: DEFAULT_CORRECTIONS)
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
    # ---- S4' 输出方式 (负片): scan = 冲印店扫描 (默认, 也是如今大多数胶片照片的样子); print = 放大机印到相纸
    output: str = "scan"
    scanner: str = "frontier"
    scan_profile: str | None = None            # None = 有该胶片的档案就用, 没有就用通用档案
    # 扫描的输出调整 (S4e, 冲印店风格的经验部分): "lab" = 饱和度矩阵 + 型号的输出 tint; None = 不做 (物理参考)
    output_conditioning: str | None = "lab"
    # 扫描定光 (S4c, 操作员变量)。scan_neutral: 不做逐帧颜色校正时以什么为中性 ——
    #   "profile" = 扫描仪通用档案 (Portra 800) 的中灰; "film" = 该胶片自己在设计光源下的中灰
    # scan_color_strength: 逐帧颜色校正力度; None = 按冲印店的随机范围 (variation.py)
    # scan_gray_balance: 逐帧灰平衡方法; None = scanning.timing.GRAY_BALANCE
    scan_neutral: str = "profile"
    scan_color_strength: float | None = None
    scan_gray_balance: str | None = None
    # 实验扫描影调: paper_only 只移除额外指数肩部, 保留同一定光和数字相纸。
    # 非实测扫描响应, 不改变默认或 next; 不保证高光/颜色改善。
    scan_shoulder: str = "exponential"
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
        # 参考光源下的测光 / 灰卡: M_ref 接收"反射率" RGB —— 在日光白平衡的输入里,
        # 该光源照亮的中性灰卡本身带着光源的颜色 (钨丝灯下是橙色的), 先乘上它再计算曝光
        from . import lights
        _m_ref = film.exposure_response(st, ref, p.lens_filter)
        _ill_ref = self._ill_ref = lights.light_rgb(spectral.illuminant(ref))
        self.M_ref = lambda rgb: _m_ref(np.asarray(rgb) * _ill_ref)
        scene_ill = lights.WB_REFERENCE if p.scene_illuminant == "raw" else p.scene_illuminant
        # 拍摄光下的灰卡在相机 (RAW 日光白平衡) 中的颜色: 拍摄光源 = 白平衡参考 (或输入已白平衡) 时为中性
        self._ill_scene = lights.light_rgb(spectral.illuminant(scene_ill)) \
            if p.scene_illuminant not in ("raw", None) else np.ones(3)
        self.M = film.exposure_response(st, scene_ill or ref, p.lens_filter)
        self.halo = film.halation_response(st, scene_ill or ref, p.lens_filter, self.v, self._back_reflect) \
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
        self.scanner = None
        if self.polarity == "negative" and p.output == "scan":
            from .scanner import Scanner
            self.scanner = Scanner(self, p.scanner, p.scan_profile,
                                   self.v.scan_strength if p.scan_color_strength is None else p.scan_color_strength,
                                   self.v.scan_stretch, p.output_conditioning, p.scan_gray_balance,
                                   paper_white="paper_white" in p.corrections, shoulder=p.scan_shoulder)
            if p.scan_neutral == "film" and not self.scanner.mono:
                g = self.scanner.normalized_density(self._transmittance(self._gray_state(actual_light=False)))[0]
                self.scanner.conversion.gray0 = g - g[1]
            if p.timing != "frame":
                st = self._gray_state(actual_light=(p.timing != "standard"))
                self.scanner.balance(self.scanner.normalized_density(self._transmittance(st))[0], p.mid_gray)

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
        dens = st.dye @ self._dye_spectra(st.dye_agent).T + e.stock.base_density
        neutral = (st.silver * e.silver_covering).mean(-1) + 0.3 * st.halide.mean(-1)
        if st.remjet:    # 没有去除的 remjet 残留: 近似为中性灰雾
            neutral = neutral + 0.4
        return 10.0 ** -(dens + neutral[..., None])

    def _dye_spectra(self, agent):
        """染料吸收光谱: 由形成染料的显影剂决定 (交叉冲洗时与设计不同)。"""
        cache = self.__dict__.setdefault("_dye_cache", {})
        if agent not in cache:
            e = self.e
            cache[agent] = chemistry.dye_spectra(e.stock.dye_density, WAVELENGTHS, chemistry.native_dye_agent(e),
                                                 agent, self.v.dye_shift_nm, self.v.dye_broaden,
                                                 self.v.dye_shift_other_nm)
        return cache[agent]

    def _transmittance_base(self):
        """未曝光片头 (片基 + 遮罩 + 残留 remjet) 的透射率, 扫描仪用它去片基。"""
        z = np.zeros((1, 3))
        st = chemistry.FilmState(latent=None, halide=z, silver=z, dye=z, coupler=None,
                                 remjet=self.e.remjet and not any(isinstance(s, chemistry.RemjetRemoval)
                                                                  for s in self.proc.steps))
        return self._transmittance(st)[0]

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
                                      dye=flat(st.dye)[sl], coupler=None, remjet=st.remjet,
                                      dye_agent=st.dye_agent)
            out[sl] = self._view(sub)
        return out.reshape(shape)

    def _gray_state(self, actual_light=False, rgb=None):
        """灰卡 (或任意 rgb) 的冲洗结果。actual_light=True: 用实际拍摄光源。"""
        if rgb is None:      # 灰卡: 参考光源下 M_ref 接收反射率; 拍摄光下是灰卡在相机中的颜色
            rgb = np.full((1, 3), GRAY) * (self._ill_scene if actual_light else 1.0)
        rgb = np.asarray(rgb, float).reshape(1, 3)
        H = self._expose_lin(rgb, self.M if actual_light else self.M_ref)
        if getattr(self, "halo", None) is not None:
            # 大面积均匀区域也会接收到自身的光晕 (整个光环的能量), 冲印店看到的底片包含这部分
            if "gray_halo" in self.p.corrections:
                # 光晕与直接曝光用同一光源 (参考光源下 M_ref 接收的是反射率, 先乘上光源颜色); 总量用画面的多次反射核
                halo = self.halo if actual_light else self._halo_ref()
                src = rgb if actual_light else rgb * self._ill_ref
                H = H + halo(src * 2.0 ** self.p.exposure_ev) * self._halo_total_multi()
            else:
                H = H + self.halo(rgb * 2.0 ** self.p.exposure_ev) * self._halo_total()
        return self._develop(self._log_exposure(H))

    @property
    def _back_reflect(self):
        """片基背面的反射代理; halation_measured 使用未标定的历史实验倍数。"""
        k = HALATION_MEASURED if "halation_measured" in self.p.corrections else 1.0
        return self.e.halation * k

    def _halo_ref(self):
        if not hasattr(self, "_halo_ref_cache"):
            st = self.e.stock
            self._halo_ref_cache = film.halation_response(st, st.reference_illuminant, self.p.lens_filter, self.v,
                                                          self._back_reflect)
        return self._halo_ref_cache

    def _halo_oblique_densities(self):
        if "halo_oblique" not in self.p.corrections:
            return (0.0, 0.0, 0.0)
        if not hasattr(self, "_halo_oblique_density_cache"):
            self._halo_oblique_density_cache = film.oblique_densities(self.e.stock, self.v)
        return self._halo_oblique_density_cache

    def _halo_total_multi(self):
        """每层光晕核的总量, 与 film.halation 对画面用的核一致 (红层含多次反射)。"""
        if not hasattr(self, "_halo_total_multi_cache"):
            rho = film.bounce_retention(self.e.stock, self.v)
            densities = self._halo_oblique_densities()
            self._halo_total_multi_cache = np.array([
                film.layer_halation_kernel(10.0, self.v, rho[c], densities[c]).sum()
                for c in range(3)])
        return self._halo_total_multi_cache

    def _halo_total(self):
        if not hasattr(self, "_halo_total_cache"):
            if "halo_oblique" in self.p.corrections:
                self._halo_total_cache = np.array([
                    film.layer_halation_kernel(10.0, self.v, oblique_density=d).sum()
                    for d in self._halo_oblique_densities()])
            else:
                self._halo_total_cache = film.halation_kernel(10.0, base_um=self.v.base_um, forward=self.v.forward).sum()
        return self._halo_total_cache

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
    def negative(self, rgb_lin):
        """S0–S3: 拍摄并冲洗, 得到底片状态 (FilmState)。"""
        return self.develop_negative(rgb_lin).state

    def develop_negative(self, rgb_lin, *, incident_sources=(), trace=None):
        """S0–S3 → Negative. incident_sources: explicit opt-in spectral-ray replacements.

        Empty sources preserve the historical path. trace optionally exposes S1
        arrays before/after optical transport; these are engine units, not watts.
        """
        p, e = self.p, self.e
        self.light_W = None
        h, w = rgb_lin.shape[:2]
        self.px_um = p.frame_width_mm * 1000.0 / max(h, w)
        if p.scene_lights == "auto" and p.exposure_hook is None and min(rgb_lin.shape[:2]) >= 32:
            H, halo_src = self._expose_mixed(rgb_lin * 2.0 ** p.exposure_ev)
        else:
            H = self._expose_lin(rgb_lin, self.M)
            halo_src = self.halo(rgb_lin * 2.0 ** p.exposure_ev) if self.halo is not None else None
        if incident_sources:
            if p.exposure_hook is not None:
                raise ValueError('Explicit incident spectra cannot be combined with exposure_hook')
            from .incident import apply_incident
            H, halo_src = apply_incident(self, H, halo_src, incident_sources)
        if trace is not None:
            trace['direct_exposure'] = H.copy()
            trace['halo_source'] = None if halo_src is None else halo_src.copy()
        H = film.emulsion_scatter(H, self.px_um)
        if trace is not None:
            trace['scattered_exposure'] = H.copy()
        if halo_src is not None:
            H = film.halation(H, halo_src, self.px_um, self.v, film.bounce_retention(e.stock, self.v),
                              oblique_densities=self._halo_oblique_densities())
            H = np.clip(H, 1e-9, None)                   # 曝光量物理上非负
        logH = self._log_exposure(H)
        if trace is not None:
            trace['total_exposure'] = H
        st = self._develop(logH, blur_px=12.0 / self.px_um)
        self._agitation(st)
        self._grain(st)
        self.state = st
        return Negative(film=e, state=st, spectral=self._transmittance, log_exposure=logH,
                        light_weights=self.light_W, light_mix=getattr(self, "light_mix", None) if self.light_W is not None
                        else None, base=self._transmittance_base())

    def _expose_mixed(self, rgb):
        """S0/S1: 按区域估计光源类别, 用各类真实光源光谱计算曝光与光晕源, 按软权重混合。"""
        from scipy.ndimage import zoom
        from . import lights
        e, p = self.e, self.p
        h, w = rgb.shape[:2]
        f = max(1, int(max(h, w) / 400))                     # 在低分辨率上估计照明 (照明是低频的)
        small = rgb[::f, ::f]
        scale = 0.08 * max(small.shape[:2])
        ill = lights.illumination_map(small, scale)
        W = lights.class_weights(ill, feas=lights.feasibility(small, scale))
        self.light_W = W_small = W
        W = zoom(W, (h / W.shape[0], w / W.shape[1], 1), order=1)[:h, :w]
        members = lights.pick_members(self.v.frame_seed)
        if p.light_member == "nearest":
            members = lights.nearest_members(ill, W_small)
        H = np.zeros(rgb.shape)
        src = np.zeros(rgb.shape) if self.halo is not None else None
        self.light_mix = {}
        for k, c in enumerate(lights.classes()):
            wk = W[..., k]
            share = float(wk.mean())
            self.light_mix[c.name] = (members[c.key][0], share)
            if share < 0.003:
                continue
            spec = members[c.key][1]
            method = "spectral" if c.key == "daylight" else \
                ("linear_wp" if "linear_white" in p.corrections else "linear")   # 见 tools/check_lights.py
            H += wk[..., None] * film.expose(rgb, film.exposure_response(e.stock, spec, p.lens_filter, method))
            if src is not None:
                src += wk[..., None] * film.halation_response(e.stock, spec, p.lens_filter, self.v,
                                                              self._back_reflect, method)(rgb)
        return np.clip(H, 1e-9, None), (np.clip(src, 0, None) if src is not None else None)

    def scan(self, st, chunk=1 << 17, trace=None):
        """S4': 扫描底片 (Negative 或 FilmState) → 线性 sRGB。trace: dict 时记录各级中间量
        (Dn, 定光 PrintTiming, 相纸密度在 to_rgb 里不单独保留)。"""
        sc = self.scanner
        neg = st if isinstance(st, Negative) else Negative(self.e, st, self._transmittance)
        shape = neg.shape
        # S4a + S4b: 光学测量 → 去片基 / 印放密度 / 档案 (按块, 直接积分到扫描通道)
        Dn = neg.measure(sc.normalized_density, chunk).reshape(-1, 3)
        # S4c: 定光
        if self.p.timing == "frame":           # 按整张底片的 LATD 自动校正 (亮度校正有上限)
            # 不做逐张校正时的平衡 = 扫描仪档案的标准: 胶片在其设计光源下 (灯光片 3200K) 的中灰。
            # 冲印店不知道场景用的什么光; RAW 的日光白平衡与此无关。
            gray = sc.normalized_density(self._transmittance(self._gray_state(actual_light=False)))[0]
            # 数码冲印机按整张底片的平均密度 (密度直方图) 定亮度; 夜景底片大部分接近片基, 平均密度很低
            # → 冲印店提亮, 但有上限。按"正常场景印成正常"标定, 正常场景的平均密度比灰卡低 LATD_NORMAL。
            sc.balance_frame(Dn, gray, self.p.mid_gray, density_limit=self.v.latd_limit, latd_normal=LATD_NORMAL)
        if trace is not None:
            trace["Dn"] = Dn.reshape(*shape, 3).copy()
            trace["timing"] = sc.timing
        # S4d + S4e: 数字相纸 → 输出调整
        return sc.to_rgb(Dn).reshape(*shape, 3)

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
        return self.view_xyz(self.negative(rgb_lin))

    def render(self, rgb_lin):
        """→ 线性 sRGB (D65)。负片默认经扫描仪输出, 其余经相纸 / 透射观看。"""
        if self.scanner is not None:
            return self.scan(self.negative(rgb_lin))
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
        # RMS 颗粒度按 48 µm 直径圆孔径测得; 等面积正方形边长 √(π·24²) = 42.5 µm
        sigma = rms48 * (42.5 if "grain_aperture" in self.p.corrections else 48.0) / self.px_um
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
        s = 4000.0 / self.px_um
        # 按滤波后白噪声的理论标准差 1/(2√π·σ) 归一 (不用样本标准差: 画面小于起伏尺度时
        # 样本几乎是常数, 除以它会把均值放大成巨大的偏移)
        n = gaussian_filter(rng.standard_normal(st.dye.shape[:2]), s, mode="wrap") * (2.0 * np.sqrt(np.pi) * s) * a
        tgt = st.dye if np.any(st.dye > 0) else st.silver
        tgt *= np.clip(1.0 + n, 0.0, None)[..., None]

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
        if getattr(self, "light_mix", None):
            mix = ", ".join(f"{k}({n}) {v:.0%}" for k, (n, v) in self.light_mix.items() if v >= 0.01)
            lines.append(f"场景照明估计: {mix}")
        lines.append(f"成像: {'负像' if self.polarity == 'negative' else '正像'}")
        if self.scanner is not None:
            lines.append(self.scanner.describe())
        elif self.paper is not None:
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
    from . import lights
    _m = film.exposure_response(e.stock, e.stock.reference_illuminant)
    _ill = lights.light_rgb(spectral.illuminant(e.stock.reference_illuminant))
    eng.M_ref = eng.M = lambda rgb: _m(np.asarray(rgb) * _ill)
    g = film.expose(np.full((1, 3), GRAY), eng.M_ref)[0]
    eng.log_k = _speed_point(e) - np.log10(g[1])
    eng.paper = None
    eng.view_illum = spectral.illuminant("D50")
    eng.white_xy = spectral.white_xy(eng.view_illum)
    return _solve_neutral(eng)


def _frame_levels(rgb, strength, black_pct=0.5, white_pct=99.5, white_min=0.6,
                  black_out=0.003, white_out=0.92):
    """扫描仪的自动设置把每张的暗部侧与高光侧放进输出范围 (Fuji: 高光与暗部的灰阶做非线性处理,
    把整张的影调范围放进相纸的再现范围)。在线性光里做黑点 / 白点 (同色阶), 三通道相同:
    加深黑、提亮偏暗的高光, 而不像提高反差那样同时放大饱和度与噪声。
    暗部侧 (最暗 black_pct%) → black_out; 高光侧 (white_pct%, 不低于 white_min, 免得把很暗的
    底片拉成白天) → white_out。冲印店扫描的黑不是 0、高光侧也留有余量不截断 (真实 800T 扫描:
    最暗 1% 的 8 位码值中位数约 10, 很少有大面积截断的白)。力度 strength 在范围内随机。"""
    Y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    ok = np.isfinite(Y)
    b, w = np.percentile(Y[ok], [black_pct, white_pct])
    w = min(max(w, white_min), 1.0)
    b0 = (1 - strength) * black_out + strength * max(b, 0.0)    # 输入中被映射到 black_out 的亮度
    lo = b0 + (black_out - b0)                                   # = black_out (仅为可读)
    hi = (1 - strength) * white_out + strength * white_out      # 高光侧的目标
    w0 = (1 - strength) * hi + strength * w                      # 输入中被映射到 hi 的亮度
    return lo + (rgb - b0) * (hi - lo) / max(w0 - b0, 1e-3)


@lru_cache(maxsize=None)
def _reference_timing(film_key: str, paper_key: str):
    paper = _bw_paper() if paper_key == "bw_paper" else paper_key
    eng = Engine(Params(film=film_key, paper=paper, grain=0, roll=None))
    return eng.print_k.copy() if eng.paper is not None else None
