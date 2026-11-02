"""T37: SID (Learning-to-See-in-the-Dark, Sony A7S II) 长 / 短曝光配对上的"截断块高出截断电平"。
同一机位、同 ISO / 光圈, 曝光比由两张都未截断的像素实测 (与 EXIF 快门比对照)。
辐亮度图 R (长曝光单位): 长曝光未截断处用长曝光, 截断处用 短曝光 × 曝光比; 短曝光也截断的块只给下界。
(A) 按 SID 自己的截断 (rawio 同一判定: 相机原始通道 ≥ 0.97 × 通道最大值; 块 = 膨胀一圈后连通; base = 块内 Y 第 90 百分位),
    flux = log2(Σ f·R / Σ f·base)  (= rawio STOPS_LOG 的 flux)。
(B) 按我们 RAW 的锚重新设截断: C = 帧内 R 的第 90 百分位 × 2^3.71 (t37_anchor.py), 块 = R > C。
分类同 rawio (HIGHLIGHT_PRIOR 面积阈、_line_like)。半尺寸像素 (长边 2128) 即评估分辨率附近 (t19_model 1600 时 step = 1)。
用法: python3 t37_sid.py 输出.tsv 长曝光.ARW:短曝光.ARW …"""
import sys, warnings
from fractions import Fraction
from pathlib import Path
import numpy as np
import rawpy
from scipy import ndimage
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import rawio

A90 = 3.71


def load(p):
    with rawpy.imread(str(p)) as raw:
        wb = list(raw.daylight_whitebalance)
        common = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None)
        xyz = raw.postprocess(output_color=rawpy.ColorSpace.XYZ, user_wb=wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]], **common)
        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **common)
    mx = cam.reshape(-1, 3).max(0).astype(np.float32)
    clip = (cam >= 0.97 * mx).any(-1)
    Y = ((xyz.astype(np.float32) / 65535.0) @ rawio.XYZ_TO_SRGB.T) @ rawio.LUMA
    t = rawio._exif(p)
    et = float(Fraction(str(t["EXIF ExposureTime"].values[0])))
    return Y, clip, et


def blobs(mask, R, base_of, f, frame, short_clip):
    lab, n = ndimage.label(ndimage.binary_dilation(mask))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        b = lab[sl] == k
        area = b.sum()
        base = base_of(sl, b)
        ff = f[sl][b]
        if ff.sum() <= 0:
            continue
        flux = float(np.log2(np.sum(ff * R[sl][b]) / (ff.sum() * base)))
        peak = float(np.log2(R[sl][b][ff > 0].max() / base))
        cls = "large" if area / frame >= rawio.HIGHLIGHT_PRIOR[0][0] else ("line" if rawio._line_like(f[sl] * b > 0.25) else "lamp")
        lower = bool(short_clip[sl][b & (f[sl] > 0)].any())
        out.append(dict(area=int(area), r_px=float(np.sqrt(ff.sum() / np.pi)), cls=cls, flux=flux, peak=peak, lower=lower,
                        cy=int(sl[0].start + np.nonzero(b)[0].mean()), cx=int(sl[1].start + np.nonzero(b)[1].mean())))
    return out


def main(dst, pairs):
    fo = open(dst, "w")
    fo.write("scene\tmode\tratio_meas\tratio_exif\ta50\ta90\tC\tblob\tarea_px\tr_px\tr35_um\tcls\tflux\tpeak\tlower_bound\tcy\tcx\n")
    for pr in pairs:
        pl, ps = pr.split(":")
        YL, cL, etL = load(pl); YS, cS, etS = load(ps)
        scene = Path(pl).stem.split("_")[0]
        cl_lv = np.percentile(YL[cL], 50) if cL.any() else YL.max()
        ok = (~cL) & (~cS) & (YL > 0.05 * cl_lv) & (YL < 0.7 * cl_lv) & (YS > 0)
        ratio = float(np.median(YL[ok] / YS[ok])) if ok.sum() > 100 else etL / etS
        R = np.where(cL, YS * ratio, YL)
        f = cL.astype(np.float32)
        frame = YL.size
        a50 = float(np.log2(cl_lv / np.percentile(YL, 50))); a90 = float(np.log2(cl_lv / np.percentile(YL, 90)))
        um = 36000.0 / max(YL.shape)   # Sony A7S II 全画幅, 半尺寸像素的胶片 µm
        rows = [("own", cl_lv, b) for b in blobs(cL, R, lambda sl, b: np.percentile(YL[sl][b], 90), f, frame, cS)]
        Cn = float(np.percentile(R, 90) * 2.0 ** A90)
        mB = R > Cn
        rows += [("anchor", Cn, b) for b in blobs(mB, R, lambda sl, b: Cn, mB.astype(np.float32), frame, cS)]
        for mode, C, b in rows:
            fo.write("\t".join(map(str, [scene, mode, f"{ratio:.1f}", f"{etL / etS:.1f}", f"{a50:.2f}", f"{a90:.2f}", f"{C:.4g}", 0, b["area"],
                                         f"{b['r_px']:.2f}", f"{b['r_px'] * um:.1f}", b["cls"], f"{b['flux']:.3f}", f"{b['peak']:.3f}",
                                         int(b["lower"]), b["cy"], b["cx"]])) + "\n")
        print(scene, f"ratio meas {ratio:.1f} exif {etL / etS:.1f}; a50 {a50:.2f} a90 {a90:.2f}; own blobs {sum(r[0]=='own' for r in rows)}, "
              f"anchor blobs {sum(r[0]=='anchor' for r in rows)}; anchor C / own C = {np.log2(Cn / cl_lv):+.2f} 档", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
