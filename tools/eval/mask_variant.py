"""沙箱: 不改正式代码, 在进程内替换 kodak_vision3_500t 的染料谱表示。
mode: clip (现行) | signed (带符号, C-41 时整体平移) | signed_pos (带符号, C-41 时只平移正部分, 成色剂负瓣不动)"""
import json, numpy as np
from functools import lru_cache
def install(mode):
    if mode in (None, "", "clip"):
        return
    from filmlab import stocks, materials, chemistry
    orig = stocks.load_stock.__wrapped__
    @lru_cache(maxsize=None)
    def load(key):
        s = orig(key)
        if key == "kodak_vision3_500t":
            d = json.loads((stocks.PROFILE_DIR / f"{key}.json").read_text())["data"]
            s = s.with_(dye_density=stocks._clean(d["channel_density"]))
        return s
    stocks.load_stock = load
    materials.load_stock = load
    if mode == "signed_pos":
        f = chemistry.dye_spectra
        def ds(dye_density, wavelengths, native, actual, shift_nm, broaden, shift_other_nm=None):
            pos, neg = np.clip(dye_density, 0, None), np.minimum(dye_density, 0)
            return f(pos, wavelengths, native, actual, shift_nm, broaden, shift_other_nm) + neg
        chemistry.dye_spectra = ds
