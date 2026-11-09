"""silvergrain: 银盐颗粒的第一性原理模型 (卤化银晶体 → 光子统计显影 → 银丝团 / 染料云 → 密度 → 透射率)。

独立的包, 只依赖 numpy / scipy (读写图片另用 Pillow, 可选 imageio); 不依赖 filmlab。见 README.md。

    import silvergrain as sg
    em = sg.derive("kodak_trix_400", "d76", grain_scale=1.0)        # 胶片 + 显影液 + 颗粒粗细 → 晶体群
    r = sg.render(em, exposure=H, px_um=6.0, seed=1)                  # 每像素颗粒密度 (r.density) 与闭式平均 (r.mean)
    sg.granularity(em, density=1.0, aperture_um=48)                   # 闭式 RMS 颗粒度
    out = sg.apply_to_image("photo.jpg", "kodak_trix_400", "rodinal")  # 给照片加颗粒 (默认 mode="grain")
    out.save("photo_trix.jpg")
"""
from .constants import GRAIN_SCALE_RANGE
from .emulsion import Emulsion, derive, universal
from .films import Developer, Film, developer, developers, film, films
from .formats import FILM_FORMATS, crop_to_format, px_um_for
from .image import GrainImage, apply_to_image, expose, finish, load_image, phone_size, save_image
from .render import RenderResult, granularity, render

__version__ = "0.1.0"
__all__ = ["Film", "Developer", "Emulsion", "RenderResult", "GrainImage", "films", "developers", "film", "developer",
           "derive", "universal", "render", "granularity", "apply_to_image", "expose", "finish", "load_image", "save_image",
           "phone_size", "FILM_FORMATS", "crop_to_format", "px_um_for", "GRAIN_SCALE_RANGE"]
