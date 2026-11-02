"""T37: 夜景 HDRI (等距柱状全景, 灯未裁切) 上测"截断块高出截断电平"的分布, 物理量与 rawio.reconstruct_highlights 一致。

假想相机: 36×24 mm 横幅, 35 mm 焦距, 偏航 0, 45, …, 315°, 俯仰 0 / +20° (共 16 帧; 每帧 = 一张假想照片)。
  帧内像素按投影到胶片的面积加权 (cos(lat) / z_c³; z_c = 与光轴夹角的余弦)。
假想截断电平: C = 帧内亮度 Y (rawio.LUMA) 的分位 P_q × 2^(a_q + δ), a_q 由我们自己的夜景 RAW 测得 (t37_anchor.py)。
截断块: Y > C 的像素, 与 rawio 一样先膨胀一圈再连通标记; 分类同 rawio:
  面积 (含膨胀圈, 胶片面积) / 画幅 ≥ HIGHLIGHT_PRIOR[0][0] → large; 否则 _line_like (同一判据, 以评估分辨率 22.5 µm/像素计) → line; 其余 lamp。
每块记录 (只在截断像素内积分, HDRI 相机自身的眩光裙边不计):
  flux  = log2(Σ w·Y / (Σ w · C))  (截断区平均辐亮度 / 截断电平; = rawio STOPS_LOG 的 flux)
  peak  = log2(max Y / C)          (受全景像素平均压低, 只作参考)
  r35 / r50: 截断面积的等效半径换算到胶片 µm (35 / 50 mm)
  dome: 块内距边界距离 d (边界像素 d=1) 归一化 ρ = d/d_max, 按 ρ 五档平均 log2(Y/C); d_max ≥ 3 时拟合 log2(Y/C) = s·ρ^p 的 p
  s07:  用该块自己的 d 分布, 反解使 f 加权 2^(s·ρ^0.7) 平均 = flux 的 s (rawio 穹顶 0.7)
用法: python3 -I t37_hdri_blobs.py 名称 URL 亮度.y 输出前缀 [bin]
  亮度.y 由 t37_hdr2y.c 生成; bin = 2 时先 2×2 盒式平均 (16k → 8k 对照)。输出 前缀_blobs.tsv、前缀_frames.tsv。"""
import sys
import numpy as np
from scipy import ndimage

LUMA_NOTE = "Y = 0.2126 R + 0.7152 G + 0.0722 B (rawio.LUMA)"
HP0 = 5e-3                         # rawio.HIGHLIGHT_PRIOR[0][0]
PITCH_UM = 22.5                    # 评估分辨率 (t19_model 长边 1600 像素)
FOCAL = 35.0
TX, TY = 18.0 / FOCAL, 12.0 / FOCAL
YAWS = np.arange(0, 360, 45)
PITCHES = (0.0, 20.0)
ANCHORS = {"p50": (50, 5.63), "p90": (90, 3.71)}   # t37_anchor.py: 25 张有截断灯的夜景 RAW 的中位
DELTAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RHO_BINS = np.linspace(0, 1, 6)


def wpct(v, w, q):
    o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, q / 100.0 * cw[-1])])


def solve_s(flux, rho, wts, p=0.7):
    dome = rho ** p
    F = lambda s: np.log2(np.sum(wts * 2.0 ** (s * dome)) / np.sum(wts)) - flux
    lo, hi = 0.0, 40.0
    if F(hi) < 0:
        return hi
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if F(m) < 0 else (lo, m)
    return 0.5 * (lo + hi)


def main(name, url, ypath, prefix, binf=1):
    H, W = np.fromfile(ypath, dtype=np.int32, count=2)
    Y = np.fromfile(ypath, dtype=np.float32, offset=8).reshape(H, W)
    if binf > 1:
        Y = Y[:H // binf * binf, :W // binf * binf].reshape(H // binf, binf, W // binf, binf).mean((1, 3))
        H, W = Y.shape
    ymax = float(Y.max()); nsat = int((Y >= 0.98 * ymax).sum())
    lat = np.deg2rad(90.0 - 180.0 * (np.arange(H) + 0.5) / H)
    lon = np.deg2rad(360.0 * (np.arange(W) + 0.5) / W - 180.0)
    dOm = (2 * np.pi / W) * (np.pi / H)
    fb = open(f"{prefix}_blobs.tsv", "w"); ff = open(f"{prefix}_frames.tsv", "w")
    fb.write("hdri\turl\tres\tyaw\tpitch\tanchor\tdelta\tblob\tlat\tlon\tn_clip\tedge\tfrac35\tcls35\tcls50\tr35_um\tr50_um\t"
             "flux\tpeak\tdmax\tp_fit\ts_fit\ts07\t" + "\t".join(f"rho{i}" for i in range(5)) + "\n")
    ff.write("hdri\tres\tyaw\tpitch\tP50\tP90\tYmax\tn_ge98max\tanchor\tdelta\tC\tn_blobs\tn_lamp\tn_line\tn_large\n")
    res = f"{W}x{H}"
    for pitch in PITCHES:
        th = np.deg2rad(pitch)
        r0 = max(0, int((90 - (pitch + 25)) / 180 * H)); r1 = min(H, int((90 - (pitch - 25)) / 180 * H) + 1)   # 帧的纬度跨度 ±21°
        for yaw in YAWS:
            ps = np.deg2rad(yaw)
            c0 = int(((yaw - 40 + 180) % 360) / 360 * W); cols = (c0 + np.arange(int(80 / 360 * W))) % W   # 帧的经度跨度 ±32°
            Ys = Y[r0:r1][:, cols]
            la, lo = lat[r0:r1][:, None], lon[cols][None, :]
            vx, vy, vz = np.cos(la) * np.sin(lo), np.sin(la) * np.ones_like(lo), np.cos(la) * np.cos(lo)
            zc = vx * np.cos(th) * np.sin(ps) + vy * np.sin(th) + vz * np.cos(th) * np.cos(ps)
            xc = vx * np.cos(ps) - vz * np.sin(ps)
            yc = -vx * np.sin(th) * np.sin(ps) + vy * np.cos(th) - vz * np.sin(th) * np.cos(ps)
            with np.errstate(divide="ignore", invalid="ignore"):
                u, w = xc / zc, yc / zc
            inf = (zc > 0) & (np.abs(u) <= TX) & (np.abs(w) <= TY)
            wt = np.where(inf, np.cos(la) / np.maximum(zc, 1e-6) ** 3, 0.0)        # 胶片面积权 (相对)
            film_area = wt * dOm * FOCAL ** 2                                      # mm² / 像素
            U, Wm = u * FOCAL * 1000.0, w * FOCAL * 1000.0                          # 胶片坐标 µm
            yv, wv = Ys[inf], wt[inf]
            P = {50: wpct(yv, wv, 50), 90: wpct(yv, wv, 90)}
            frame_mm2 = float(film_area.sum())                                     # ≈ 864
            ones = np.ones_like(Ys)
            edge_px = (ndimage.binary_dilation(~inf) & inf).astype(np.uint8)       # 帧边界像素
            for an, (q, a) in ANCHORS.items():
                for dl in DELTAS:
                    C = P[q] * 2.0 ** (a + dl)
                    clip = (Ys > C) & inf
                    lab, n = ndimage.label(ndimage.binary_dilation(clip) & inf)
                    cnt = {"lamp": 0, "line": 0, "large": 0}
                    if n:
                        idx = np.arange(1, n + 1)
                        labc = np.where(clip, lab, 0)
                        ncl = ndimage.sum(ones, labc, idx)
                        fa_dil = ndimage.sum(film_area, lab, idx)
                        fa_cl = ndimage.sum(film_area, labc, idx)
                        swy = ndimage.sum(wt * Ys, labc, idx); sw = ndimage.sum(wt, labc, idx)
                        pk = ndimage.maximum(Ys, labc, idx)
                        edge = ndimage.maximum(edge_px, lab, idx)      # 膨胀块碰到帧边界
                        objs = ndimage.find_objects(labc)
                    for i in range(n):
                        if ncl[i] < 1:
                            continue
                        k = i + 1
                        flux = float(np.log2(swy[i] / (sw[i] * C)))
                        peak = float(np.log2(pk[i] / C))
                        frac35 = fa_dil[i] / frame_mm2
                        r35 = float(np.sqrt(fa_cl[i] / np.pi) * 1000.0)
                        sl = objs[i]
                        m = labc[sl] == k
                        uu, ww = U[sl][m], Wm[sl][m]
                        cl35 = "large" if frac35 >= HP0 else None
                        cl50 = "large" if frac35 * (50 / 35) ** 2 >= HP0 else None
                        isline = False
                        if ncl[i] >= 3:
                            # 宽度 = 面积 / 主轴长度, 均以评估像素计; 35 / 50 mm 各自换算
                            for fl_mm in (35.0, 50.0):
                                s = fl_mm / 35.0
                                ev = np.linalg.eigvalsh(np.cov(np.stack([uu, ww]) * s / PITCH_UM) + 1e-9 * np.eye(2))
                                length = np.sqrt(12.0 * ev[-1]); width = fa_cl[i] * 1e6 * s * s / PITCH_UM ** 2 / max(length, 1.0)
                                ll = width <= 2.0 and length >= 4.0 * width
                                if fl_mm == 35.0 and cl35 is None:
                                    cl35 = "line" if ll else "lamp"
                                if fl_mm == 50.0 and cl50 is None:
                                    cl50 = "line" if ll else "lamp"
                        cl35 = cl35 or "lamp"; cl50 = cl50 or "lamp"
                        cnt[cl35] += 1
                        # 块内剖面
                        d = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
                        dv = d[m]; dmax = float(dv.max()); rho = dv / dmax
                        ly = np.log2(np.maximum(Ys[sl][m], 1e-30) / C)
                        prof = [float(np.mean(ly[(rho > RHO_BINS[j]) & (rho <= RHO_BINS[j + 1])])) if np.any((rho > RHO_BINS[j]) & (rho <= RHO_BINS[j + 1])) else np.nan for j in range(5)]
                        pfit = sfit = np.nan
                        if dmax >= 3:
                            best = None
                            for p in np.arange(0.1, 3.01, 0.05):
                                g = rho ** p; s_ = float(np.dot(g, ly) / np.dot(g, g)); e = float(np.sum((ly - s_ * g) ** 2))
                                if best is None or e < best[0]:
                                    best = (e, p, s_)
                            _, pfit, sfit = best
                        wl = wt[sl][m]
                        s07 = solve_s(flux, rho, wl)
                        cy, cx = [int(np.mean(t)) for t in np.nonzero(m)]
                        clat = np.rad2deg(lat[r0 + sl[0].start + cy]); clon = np.rad2deg(lon[cols[sl[1].start + cx]])
                        fb.write("\t".join(map(str, [name, url, res, yaw, pitch, an, dl, k,
                                 f"{clat:.2f}", f"{clon:.2f}",
                                 int(ncl[i]), int(edge[i]), f"{frac35:.3e}", cl35, cl50, f"{r35:.1f}", f"{r35 * 50 / 35:.1f}",
                                 f"{flux:.3f}", f"{peak:.3f}", f"{dmax:.1f}", f"{pfit:.2f}", f"{sfit:.2f}", f"{s07:.3f}"]
                                 + [f"{x:.3f}" for x in prof])) + "\n")
                    ff.write("\t".join(map(str, [name, res, yaw, pitch, f"{P[50]:.5g}", f"{P[90]:.5g}", f"{ymax:.5g}", nsat, an, dl,
                                                 f"{C:.5g}", n, cnt["lamp"], cnt["line"], cnt["large"]])) + "\n")
    fb.close(); ff.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1)
