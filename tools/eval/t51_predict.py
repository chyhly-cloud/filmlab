"""T51: 用实测的冲印店空间传递 (filmlab.scanning.spatial.LABS) 预测颗粒的三个量, 与真实扫描比较:
  1. T46 第 7.2 节的通道相关随空间尺度 (4 个高斯差分频带; 64 像素块, 与真实同一代码 t51_scan_measure);
  2. 48 µm (42.5 µm 等面积方孔) RMS 颗粒的变化 (T45 的 2–3 倍差距是否由冲印店空间处理补上);
  3. 像素级颗粒 (每像素 Do 标准差) 的变化 —— 整图观感。
模型侧: 胶片侧颗粒参照 (t51_model_ref.py) → 扫描光学 σ_o → 冲印店像素方盒 → (无 / 有) 冲印店软件。
用法: python3 -I t51_predict.py 工作目录 输出.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import t51_scan_measure as SM  # noqa: E402
from filmlab.scanning import spatial  # noqa: E402


def lab_output(npz_key_arr, fine_um, pitch_um, lab, software=True, mono=False):
    o = npz_key_arr.astype(float)
    s = lab.optics_px * pitch_um / fine_um              # 光学 σ 以冲印店输出像素计
    o = np.stack([gaussian_filter(o[..., c], s, mode="wrap") for c in range(3)], -1)
    k = int(round(pitch_um / fine_um))
    h = (o.shape[0] // k) * k
    o = o[:h, :h].reshape(h // k, k, h // k, k, 3).mean((1, 3))
    if software:
        o = spatial.apply(o, replace(lab, pitch_um=pitch_um), pitch_um, mono=mono)
    return o


def sigma_box(o, pitch_um, side_um=42.5):
    D = -np.log10(np.clip(o, 1e-6, None))
    b = max(1, int(round(side_um / pitch_um)))
    v = SM.boxmean(D - D.mean((0, 1)), b)
    return (v.std((0, 1)) * b * pitch_um / side_um).tolist(), D.std((0, 1)).tolist()


def predict(npz, fine_um, pitch_um, lab, mono=False):
    z = np.load(npz)
    out = {}
    for key in z.files:
        r = {}
        for sw in (False, True):
            o = lab_output(z[key], fine_um, pitch_um, lab, sw, mono)
            m = SM.measure_array(spatial.encode(o) * 255.0, pitch_um, synthetic=True)
            s48, s1 = sigma_box(o, pitch_um)
            r["with" if sw else "without"] = dict(D=m["D_med"], band_corr=m["band_corr"], sigma48=s48, sigma_px=s1)
        out[key] = r
    return out


def main():
    work, dst = Path(sys.argv[1]), Path(sys.argv[2])
    L = spatial.LABS
    res = {
        "frontier_800t_6.6um": predict(work / "ref_800t.npz", 2.2, 6.6, L["frontier_epfl"]),
        "noritsu_portra400_20um": predict(work / "ref_p400.npz", 5.0, 20.0, L["noritsu_rpl"]),
        "noritsu_trix_20um": predict(work / "ref_trix.npz", 5.0, 20.0, L["noritsu_rpl"], mono=True),
    }
    if (work / "ref_800t_20.npz").exists():
        res["noritsu_800t_20um"] = predict(work / "ref_800t_20.npz", 5.0, 20.0, L["noritsu_rpl"])
    json.dump(res, open(dst, "w"), indent=1)
    for k, v in res.items():
        for lev, r in v.items():
            a, b = r["without"], r["with"]
            print(k, lev, "D", np.round(a["D"], 2), "σ48 ×", np.round(np.array(b["sigma48"]) / np.array(a["sigma48"]), 2),
                  "σpx ×", np.round(np.array(b["sigma_px"]) / np.array(a["sigma_px"]), 2))
            print("   bands RG/GB/RB without", np.round(a["band_corr"], 2).tolist())
            print("   bands RG/GB/RB with   ", np.round(b["band_corr"], 2).tolist())


if __name__ == "__main__":
    main()
