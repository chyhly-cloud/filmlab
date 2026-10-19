"""RAW 截断判定审计: rawio.load_raw 的截断掩膜 (每张图各通道最大值的 97%) 与传感器实际白电平的比较,
以及 libraw 输出的缩放 (1.0 是否等于传感器截断)。

用法: python3 -I raw_clip_audit.py RAW…
"""
import sys
import numpy as np
import rawpy

COMMON = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True, user_flip=None)


def audit(path):
    with rawpy.imread(path) as raw:
        wl = float(raw.white_level)
        bl = np.array(raw.black_level_per_channel, float)
        data = raw.raw_image_visible.astype(np.float64)
        col = raw.raw_colors_visible
        sat = np.zeros(data.shape, bool)
        dmax = []
        for c in range(4):
            m = col == c
            sat[m] = data[m] >= wl - 2.0 * (wl - bl[c]) / 4096.0    # 白电平以下 2 个 12 位码值内 = 饱和
            dmax.append(data[m].max())
        h2, w2 = data.shape[0] // 2, data.shape[1] // 2
        true_clip = sat[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2).any((1, 3))   # 半尺寸像素里有饱和传感元
        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, user_wb=[1, 1, 1, 1], **COMMON)
        mx = cam.reshape(-1, 3).max(0).astype(np.float32)
        cur_clip = (cam >= 0.97 * mx).any(-1)
        # 缩放: 未饱和的亮像素, libraw 输出 / 65535 与 (数据 − 黑) / (白 − 黑) 之比
        G = data[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2)
        cc = col[:2 * h2, :2 * w2].reshape(h2, 2, w2, 2)
        bl_img = bl[cc]
        lin = ((G - bl_img) / (wl - bl_img))
        green = (cc == 1) | (cc == 3)
        g_raw = (lin * green).sum((1, 3)) / green.sum((1, 3))
    H, W = min(cam.shape[0], h2), min(cam.shape[1], w2)
    g_out = cam[:H, :W, 1] / 65535.0
    g_raw = g_raw[:H, :W]
    ok = (g_raw > 0.05) & (g_raw < 0.8) & ~true_clip[:H, :W]
    scale = float(np.median(g_out[ok] / g_raw[ok])) if ok.any() else np.nan
    tc, cu = true_clip[:H, :W], cur_clip[:H, :W]
    return dict(white=wl, black=bl.tolist(), data_max=dmax, out_max=(mx / 65535).round(3).tolist(), scale=scale,
                true=int(tc.sum()), cur=int(cu.sum()), false_pos=int((cu & ~tc).sum()), missed=int((tc & ~cu).sum()))


if __name__ == "__main__":
    print(f"{'文件':28s} {'白电平':>6s} {'数据最大 (4 通道)':>26s} {'输出缩放':>6s} {'真截断':>7s} {'现判定':>7s} {'误判':>7s} {'漏判':>6s}")
    for p in sys.argv[1:]:
        r = audit(p)
        print(f"{p.split('/')[-1]:28s} {r['white']:6.0f} {str([int(x) for x in r['data_max']]):>26s} {r['scale']:6.3f} "
              f"{r['true']:7d} {r['cur']:7d} {r['false_pos']:7d} {r['missed']:6d}", flush=True)
