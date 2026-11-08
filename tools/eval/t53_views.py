"""T53 出图: 颗粒粗细 k 的整张图 (docs/tasks/T53_GRAIN_COARSENESS.md)。出图方式同 T52 (tools/eval/t52_views.py):
线性 → sRGB 编码 → 8 位, 不加任何处理; 整张原尺寸 JPEG 质量 95; 手机尺寸 = 整张 Lanczos 缩到长边 1200, 质量 92;
并排图每格都是手机尺寸; 诊断局部 = 手机尺寸图上同一位置 110 × 110 像素最近邻 3 倍, 左侧缩略图红框标位置。
用法: python3 -I t53_views.py 帧目录 出图目录 [T52 帧目录]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json  # noqa: E402

from t52_views import DIAG, diag, load, phone, row, titled  # noqa: E402

SCENES = {"day_9195": "日景", "people_61227": "人像"}
KS = ("0.5", "1", "1.5", "2", "3")


def kname(k):
    return f"k{k}"


def main(fdir, vdir, f52=None):
    fdir, vdir = Path(fdir), Path(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    for scene, zh in SCENES.items():
        portrait = scene == "people_61227"
        sets = {}
        for film, fz, lab, ks in (("trix_d76_35", "TriX400_D76_35mm", "Tri-X 400 / D-76 / 35 mm", KS),
                                  ("portra400_35", "Portra400_C41_35mm", "Portra 400 / C-41 / 35 mm", ("1", "2"))):
            ims, metas, labs = [], [], []
            for k in ks:
                p = fdir / f"{scene}_{film}_k{k}.npy"
                if not p.exists():
                    print("缺", p)
                    continue
                im = load(p)
                ph = phone(im)
                im.save(vdir / f"{zh}_{fz}_颗粒粗细k{k}_整张原尺寸{im.width}x{im.height}.jpg", quality=95)
                ph.save(vdir / f"{zh}_{fz}_颗粒粗细k{k}_手机尺寸1200.jpg", quality=92)
                ims.append(ph); metas.append(json.load(open(str(p).replace(".npy", "_meta.json"))))
                labs.append(f"{lab} · 颗粒粗细 k = {k}")
                print(scene, film, k, im.size, flush=True)
            sets[film] = (ims, metas, labs, ks)
            if len(ims) >= 2:
                tag = "_vs_".join(f"k{k}" for k in ks)
                row([titled(i, l) for i, l in zip(ims, labs)], vertical=not portrait).save(
                    vdir / f"{zh}_对照_手机尺寸_{fz}_颗粒粗细_{tag}.jpg", quality=92)
                diag(ims, metas, labs, DIAG[scene], vdir / f"{zh}_诊断_手机尺寸局部3倍_{fz}_颗粒粗细.jpg")
        # 颗粒粗细与"真实"的变粗方式 (T52 的帧): Tri-X D-76 k = 1 | D-76 k = 1.5 | Rodinal k = 1 (35 mm) | Rodinal 半格
        if f52 is not None:
            f52 = Path(f52)
            ims, metas, labs = [], [], []
            for p, lab in ((fdir / f"{scene}_trix_d76_35_k1.npy", "Tri-X / D-76 / 35 mm · k = 1"),
                           (fdir / f"{scene}_trix_d76_35_k1.5.npy", "Tri-X / D-76 / 35 mm · k = 1.5"),
                           (f52 / f"{scene}_trix_rodinal_35.npy", "Tri-X / Rodinal / 35 mm · k = 1 (T52)"),
                           (f52 / f"{scene}_trix_rodinal_half.npy", "Tri-X / Rodinal / 半格 · k = 1 (T52)")):
                if p.exists():
                    ims.append(phone(load(p))); metas.append(json.load(open(str(p).replace(".npy", "_meta.json"))))
                    labs.append(lab)
            if len(ims) == 4:
                row([titled(i, l) for i, l in zip(ims, labs)], vertical=not portrait).save(
                    vdir / f"{zh}_对照_手机尺寸_颗粒粗细与显影液画幅_D76k1_vs_D76k1.5_vs_Rodinal_vs_Rodinal半格.jpg",
                    quality=92)


if __name__ == "__main__":
    main(*sys.argv[1:4])
