"""沙箱: RAW 日光白平衡 = D65 中性 (dcraw/libraw 的 pre_mul 由 sRGB(D65) 矩阵求得), 而不是 D55。"""
def install():
    import colour, numpy as np
    from functools import lru_cache
    from filmlab import lights, spectral
    @lru_cache(maxsize=None)
    def wb():
        w = spectral.illuminant("D65") @ spectral.cmfs()
        return lights._M_SRGB, w[1]
    lights._daylight_wb = wb
