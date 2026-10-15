"""RAW 解码: 只做解马赛克和色彩矩阵, 输出场景线性光 (线性 sRGB 原色, 允许超出色域的负值)。"""
import numpy as np
import rawpy

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])


def load_raw(path, max_size=2400, white_balance="camera"):
    with rawpy.imread(str(path)) as raw:
        xyz = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16,
            output_color=rawpy.ColorSpace.XYZ,
            use_camera_wb=(white_balance == "camera"),
            use_auto_wb=(white_balance == "auto"),
            half_size=True, user_flip=None,
        ).astype(np.float32) / 65535.0
    step = max(1, int(round(max(xyz.shape[:2]) / max_size)))
    if step > 1:  # 盒式下采样 (线性光下平均, 物理正确)
        h, w = (xyz.shape[0] // step) * step, (xyz.shape[1] // step) * step
        xyz = xyz[:h, :w].reshape(h // step, step, w // step, step, 3).mean((1, 3))
    return xyz @ XYZ_TO_SRGB.T


def auto_expose(rgb, key=0.18):
    """平均测光: 让画面的几何平均亮度 = 中灰。"""
    Y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    gm = np.exp(np.mean(np.log(np.clip(Y, 1e-5, None))))
    return rgb * (key / gm)
