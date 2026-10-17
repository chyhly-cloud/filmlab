"""S3 物理核心的终点: 冲洗后的底片。

底片是一个物理对象, 不是 RGB。它由三层有效曝光 (S1)、三层染料量 / 银 / 卤化银 (S2) 和
材料的光谱数据 (染料吸收光谱、片基 + 遮罩) 决定, 任何仪器 (扫描仪、密度计、相纸) 都只是
用自己的光谱响应去测量它的光谱透射率 T(λ)。

    D(λ) = D_base(λ) + Σ_i c_i · ε_i(λ) + 中性银 + 乳白,   T(λ) = 10^−D(λ)

整幅 T(λ) 有 81 个波长, 全分辨率放不进内存, 所以按块计算, 并直接积分到仪器的通道上 (`measure`)。
"""
from dataclasses import dataclass

import numpy as np

from .chemistry import FilmState

CHUNK = 1 << 17


def _flat(a):
    return a.reshape(-1, a.shape[-1])


@dataclass
class Negative:
    film: object                          # 材料 (Emulsion)
    state: FilmState                      # 三层染料量 c、银、卤化银
    spectral: object                      # FilmState → 光谱透射率 (…, 81); 由材料光谱数据决定 (Engine._transmittance)
    log_exposure: np.ndarray | None = None   # S1: 三层有效曝光 log10 H (含测光、散射、光晕、潜影灰雾)
    light_weights: np.ndarray | None = None  # S0: 低分辨率的光源类别权重图 (…, 类别), 见 lights.classes()
    light_mix: dict | None = None            # 光源类别 → (代表光源, 画面占比)
    base: np.ndarray | None = None           # 未曝光片头的光谱透射率 (81,)

    @property
    def shape(self):
        return self.state.dye.shape[:-1]

    @property
    def dye(self):
        """三层染料量 (相对材料的染料光谱单位)。"""
        return self.state.dye

    def chunks(self, chunk=CHUNK):
        """按块 (切片, 子 FilmState) 遍历展平后的像素。"""
        st = self.state
        n = int(np.prod(self.shape))
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            yield sl, FilmState(latent=None, halide=_flat(st.halide)[sl], silver=_flat(st.silver)[sl],
                                dye=_flat(st.dye)[sl], coupler=None, remjet=st.remjet, dye_agent=st.dye_agent)

    def transmittance(self, index=None):
        """光谱透射率 (…, 81)。index: 展平后的像素下标 (取样用); None = 整幅 (只用于小图)。"""
        st = self.state
        if index is None:
            return self.spectral(st)
        sub = FilmState(latent=None, halide=_flat(st.halide)[index], silver=_flat(st.silver)[index],
                        dye=_flat(st.dye)[index], coupler=None, remjet=st.remjet, dye_agent=st.dye_agent)
        return self.spectral(sub)

    def measure(self, instrument, chunk=CHUNK):
        """用仪器测量底片: instrument(T (n, 81)) → (n, k)。返回 (…, k)。"""
        out = None
        for sl, sub in self.chunks(chunk):
            r = instrument(self.spectral(sub))
            if out is None:
                out = np.empty((int(np.prod(self.shape)), r.shape[-1]))
            out[sl] = r
        return out.reshape(*self.shape, -1)

    def status_m(self, chunk=CHUNK):
        """Status M 积分密度, 减去片基 (ISO 5-3 近似; 负片冲洗厂的控制读数)。"""
        from . import densitometry
        d = self.measure(lambda T: densitometry.measure(T, "M"), chunk)
        if self.base is not None:
            d = d - densitometry.measure(self.base, "M")
        return d
