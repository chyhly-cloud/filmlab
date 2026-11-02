"""T37 锚: 我们自己的夜景 RAW 里"场景亮度分位 → 截断电平"的档数 (rawio 同一亮度 Y、同一截断判定与块划分)。
截断电平 C = 灯具类块 base (块内 Y 的第 90 百分位, 同 reconstruct_highlights) 的中位;
场景分位 P_q = 全画面 Y 的第 q 百分位 (q = 50 / 75 / 90 / 99)。锚 a_q = log2(C / P_q)。
另记 EXIF 焦距与 35 mm 等效焦距 (换算 HDRI 角尺寸 → 胶片 µm)。
用法: python3 t37_anchor.py 输出.tsv RAW…   (长边按 t19_model 的 1600 像素下采样)"""
import sys, warnings
from fractions import Fraction
from pathlib import Path
import numpy as np
import rawpy
from scipy import ndimage
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import rawio

Q = (50, 75, 90, 99)


def focal(path):
    try:
        t = rawio._exif(path)
        fl = float(Fraction(str(t["EXIF FocalLength"].values[0]))) if "EXIF FocalLength" in t else None
        f35 = float(str(t["EXIF FocalLengthIn35mmFilm"].values[0])) if "EXIF FocalLengthIn35mmFilm" in t else None
        return fl, f35
    except Exception:
        return None, None


def one(path, max_size=1600):
    with rawpy.imread(str(path)) as raw:
        wb = list(raw.daylight_whitebalance)
        kw = dict(use_camera_wb=False, use_auto_wb=False)
        if not np.all(np.isfinite(wb[:3])) or min(wb[:3]) <= 0:
            kw["use_camera_wb"] = True
        else:
            kw["user_wb"] = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        common = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None)
        xyz = raw.postprocess(output_color=rawpy.ColorSpace.XYZ, **common, **kw).astype(np.float32) / 65535.0
        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
        mx = cam.reshape(-1, 3).max(0).astype(np.float32)
        clip = (cam >= 0.97 * mx).any(-1).astype(np.float32)
        half = xyz.shape[:2]
    step = max(1, int(round(max(xyz.shape[:2]) / max_size)))
    xyz = rawio._downsample(xyz, step)
    cf = rawio._downsample(clip, step)
    Y = (xyz @ rawio.XYZ_TO_SRGB.T) @ rawio.LUMA
    mask = ndimage.binary_dilation(cf > 0.25)
    lab, n = ndimage.label(mask)
    bases, cls = [], []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        b = lab[sl] == k
        area = b.sum()
        bases.append(np.percentile(Y[sl][b], 90))
        c = "large" if area / mask.size >= rawio.HIGHLIGHT_PRIOR[0][0] else ("line" if rawio._line_like(cf[sl] * b > 0.25) else "lamp")
        cls.append(c)
    bases, cls = np.array(bases), np.array(cls)
    lampb = bases[cls == "lamp"] if (cls == "lamp").any() else bases
    C = float(np.median(lampb)) if len(lampb) else float("nan")
    P = np.percentile(Y, Q)
    return dict(half=half, step=step, work=Y.shape, n=n, nlamp=int((cls == "lamp").sum()), C=C,
                Cspread=float(np.log2(np.percentile(lampb, 75) / np.percentile(lampb, 25))) if len(lampb) > 3 else float("nan"),
                clipfrac=float(cf.mean()), P=P, fl=focal(path))


if __name__ == "__main__":
    out = open(sys.argv[1], "w")
    out.write("raw\thalf_px\twork_px\tfocal_mm\tf35_mm\tclip_frac\tn_blobs\tn_lamp\tC\tC_iqr_stops\t" + "\t".join(f"a{q}" for q in Q) + "\n")
    for p in sys.argv[2:]:
        r = one(p)
        a = [np.log2(r["C"] / max(x, 1e-9)) for x in r["P"]]
        row = [Path(p).name, "x".join(map(str, r["half"])), "x".join(map(str, r["work"])), r["fl"][0], r["fl"][1],
               f"{r['clipfrac']:.2e}", r["n"], r["nlamp"], f"{r['C']:.4f}", f"{r['Cspread']:.2f}"] + [f"{x:.2f}" for x in a]
        out.write("\t".join(map(str, row)) + "\n"); out.flush()
        print("\t".join(map(str, row)), flush=True)
