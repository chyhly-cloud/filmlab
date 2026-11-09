"""T55b: 冲印店影调 (输出码值 ↔ 底片密度) 的实测曲线 → silvergrain/data/lab_tone.json (只存测量与来源)。

来源 (扫描与同一底片的灯箱翻拍配准, 在底片密度上分档取扫描码值中位; 图像不进仓库):
  · 黑白 (T51b, tools/eval/t51/bw_level_real.json.gz): RPL Noritsu, Tri-X 7 帧 + HP5+ 7 帧 (−3 … +4 EV), 灰度 JPEG;
    底片密度 D = −log10(T / T_片基) (翻拍按 sRGB 解码当线性; 密度标度对数据表曲线 ±15%)。
  · 彩色 (T54, tools/eval/t54/rpl_real.json.gz): RPL Noritsu, Portra 400 / 160 / 800、Ektar 100、Pro 400H 各 6–7 帧, 每通道;
    翻拍密度 D_c (减片基与橙色遮罩) 按 u_c = γ_翻拍 / γ_模型 (T54 2.2 节, tools/eval/t54/summary.json) 换到模型染料量 d_c = D_c / u_c。
冲印店逐帧自动定光 (只平移密度轴), 所以每帧取锚点 D_118 (该帧曲线上码值 118 = sRGB 中灰处的密度), 曲线写成码值对 u = D − D_118。
帧间合并在码值域做: 每帧用三次多项式拟合 (与 T51b / T54 求真实斜率的同一拟合; 只取该帧实测范围内斜率为正的一段, 两端各去 4 码值),
每个码值上取各帧局部斜率 du/dCode 的中位 (至少 4 帧), 再从 118 积分得 u(码值)。这样保留每帧在该码值处的
局部斜率 (颗粒只看局部斜率); 若在 u 网格上直接取码值的中位, 各帧锚点附近形状的差异会把合并曲线压平约 10–20%。
另存各码值的斜率四分位 (帧间分散) 与 u 网格上的码值中位 (对照)。码值不单调的端点 (底片翻拍的截断 / 配准外点) 去掉。
EPFL Frontier (T54) 的配准是扫描对同场景数码 RAW (景物曝光), 不是对底片密度, 含 800T 自己的曲线, 不能作冲印店影调, 不导出。
用法: python3 tools/export_silvergrain_labtone.py"""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = np.round(np.arange(-1.5, 1.5001, 0.05), 3)


def monotone(D, C, tol=4.0):
    """按 D 升序; 从码值最低处起, 保留码值不比已达最大值低 tol 以上的点 (去掉端点外点)。"""
    o = np.argsort(D); D, C = np.asarray(D, float)[o], np.asarray(C, float)[o]
    keep, mx = [], -1e9
    for d, c in zip(D, C):
        if c >= mx - tol:
            keep.append((d, c)); mx = max(mx, c)
    D, C = map(np.array, zip(*keep))
    return D, np.maximum.accumulate(C)


def anchor(D, C, code=118.0):
    if not (C.min() <= code <= C.max()):
        return None
    i = np.argsort(C + 1e-6 * np.arange(len(C)))
    return float(np.interp(code, C[i], D[i]))


def pava(y):
    """单调不减的保序回归 (相邻违例合并)。"""
    blocks = [[float(v), 1] for v in y]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, n2 = out.pop(); v1, n1 = out.pop()
            out.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([[v] * n for v, n in out])


def pooled(frames, min_frames=4):
    """frames: [(u, code)] → 网格上每点的帧间中位、四分位、帧数 (只用覆盖该点的帧, 不外推; 至少 min_frames 帧)。
    code = 中位经保序回归 (单调) 与三点平滑 (影调曲线用); code_median 为原始中位。"""
    M = np.full((len(frames), len(GRID)), np.nan)
    for j, (u, c) in enumerate(frames):
        ok = (GRID >= u.min()) & (GRID <= u.max())
        M[j, ok] = np.interp(GRID[ok], u, c)
    n = np.sum(np.isfinite(M), 0)
    keep = n >= min_frames
    raw = np.nanmedian(M[:, keep], 0); q25 = np.nanpercentile(M[:, keep], 25, 0); q75 = np.nanpercentile(M[:, keep], 75, 0)
    med = pava(raw)
    if len(med) >= 3:
        sm = med.copy(); sm[1:-1] = (med[:-2] + med[1:-1] + med[2:]) / 3.0
        med = sm
    i0 = int(np.argmin(np.abs(GRID[keep])))            # 平滑后锚点仍为 118
    med = med - med[i0] + 118.0
    return dict(u=GRID[keep].tolist(), code=np.round(med, 2).tolist(), code_median=np.round(raw, 2).tolist(), q25=np.round(q25, 2).tolist(),
                q75=np.round(q75, 2).tolist(), n_frames=n[keep].astype(int).tolist())


CODES = np.arange(8.0, 236.0, 2.0)


def cubic_frame(D, coef, a, trim=4.0):
    """一帧的三次多项式影调 (T51b / T54 求斜率用的同一拟合) 在该帧实测密度范围内、斜率为正的一段, 采样成 (u, code)。"""
    d = np.linspace(D.min(), D.max(), 400)
    c = np.polyval(coef, d)
    g = np.polyval(np.polyder(coef), d)
    ok = g > 0
    i = np.argmax(ok); j = len(ok) - np.argmax(ok[::-1])           # 最长连续正斜率段的近似: 首末正斜率点之间
    d, c = d[i:j], c[i:j]
    m = (c >= c.min() + trim) & (c <= c.max() - trim)
    return d[m] - a, c[m]


def pooled_by_code(frames, min_frames=4):
    """码值域合并: 每帧 u(码值) 在码值网格上的导数 du/dc → 帧间中位 → 从 118 积分。返回网格 (u, code) 与斜率分散。"""
    S = np.full((len(frames), len(CODES)), np.nan)
    for j, (u, c) in enumerate(frames):
        c = c + 1e-6 * np.arange(len(c))
        ok = (CODES >= c.min()) & (CODES <= c.max())
        if ok.sum() < 3:
            continue
        uc = np.interp(CODES[ok], c, u)
        S[j, ok] = np.gradient(uc, CODES[ok])                       # du/dCode
    n = np.sum(np.isfinite(S), 0)
    keep = n >= min_frames
    cs = CODES[keep]
    med = np.nanmedian(S[:, keep], 0)
    med = np.maximum(med, 1e-4)
    i0 = int(np.argmin(np.abs(cs - 118.0)))
    u = np.concatenate([[0.0], np.cumsum(0.5 * (med[1:] + med[:-1]) * np.diff(cs))])
    u = u - u[i0] - (118.0 - cs[i0]) * med[i0]
    slope = 1.0 / np.nanpercentile(S[:, keep], [75, 50, 25], 0)        # dCode/du 的四分位 (帧间)
    return dict(code=np.round(cs, 2).tolist(), u=np.round(u, 4).tolist(), slope_q25=np.round(slope[0], 1).tolist(),
                slope_median=np.round(slope[1], 1).tolist(), slope_q75=np.round(slope[2], 1).tolist(),
                n_frames=n[keep].astype(int).tolist())


def main():
    bw = json.load(gzip.open(ROOT / "tools/eval/t51/bw_level_real.json.gz"))
    bw_frames, bw_raw, bw_fits = [], {}, []
    for k, v in bw.items():
        if "tone_D" not in v:
            continue
        D, C = monotone(v["tone_D"], v["tone_code"])
        a = anchor(D, C)
        if a is None:
            continue
        bw_frames.append((D - a, C))
        bw_fits.append(cubic_frame(np.asarray(v["tone_D"]), np.polyfit(v["tone_D"], v["tone_code"], 3), a))
        bw_raw[k] = dict(D=np.round(D, 3).tolist(), code=np.round(C, 1).tolist(), D_118=round(a, 3))
    col = json.load(gzip.open(ROOT / "tools/eval/t54/rpl_real.json.gz"))
    U = {k: v["u"] for k, v in json.load(open(ROOT / "tools/eval/t54/summary.json"))["rpl"].items()}
    col_frames, col_fits = [[], [], []], [[], [], []]
    col_raw = {}
    for film, v in col.items():
        for fr, x in v["frames"].items():
            for c in range(3):
                t = x["tone"][c]
                D, C = monotone(np.asarray(t["D"]) / U[film][c], t["code"])
                a = anchor(D, C)
                if a is None:
                    continue
                col_frames[c].append((D - a, C))
                pc = np.asarray(t["poly"])
                Dm = np.asarray(t["D"]) / U[film][c]
                coef = pc * np.array([U[film][c] ** 3, U[film][c] ** 2, U[film][c], 1.0])   # code(d) = poly(d·u_c)
                col_fits[c].append(cubic_frame(Dm, coef, a))
                col_raw[f"{film}/{fr}/{'RGB'[c]}"] = dict(d=np.round(D, 3).tolist(), code=np.round(C, 1).tolist(), d_118=round(a, 3))
    src51 = json.load(open(ROOT / "tools/eval/t51/sources.json"))["rpl"]
    out = dict(
        _note=("冲印店影调: 输出 8 位码值对底片密度 (相对锚点 u = D − D_118, D_118 = 该帧码值 118 处的密度; 冲印店逐帧定光只平移密度轴)。"
               "只存测量: 各帧 (密度, 码值) 分档中位 → 网格上的帧间中位与四分位。由 filmlab tools/export_silvergrain_labtone.py 生成 (T55b)。"),
        rpl_noritsu=dict(
            lab="Richard Photo Lab, Noritsu HS-1800 + EZ Controller, JPEG (645 2048 × 2796)",
            zip=src51["zip"], page=src51["page"],
            method=("扫描与同一底片的灯箱翻拍 (Neg Sequence, Nikon D5300) 配准; 翻拍按 sRGB 解码当线性, D = −log10(T/T_片基); "
                    "在 D 上分档取扫描码值中位。黑白: docs/tasks/T51 13.1 (tools/eval/t51b_bw_level.py); 彩色: docs/tasks/T54 2.2 "
                    "(tools/eval/t54_real.py), 翻拍密度按 u_c 换到模型染料量 (silvergrain 染料云路径的密度单位)。"),
            bw=dict(units="u = 银密度 (净, 漫射) − D_118", frames=len(bw_frames), films="Tri-X 400 (XTOL), HP5 Plus",
                    curve=pooled_by_code(bw_fits), u_grid_median=pooled(bw_frames)),
            colour=dict(units="u_c = 模型染料当量密度 (每层) − d_118,c; 通道 R G B", u_factors=U,
                        R=dict(curve=pooled_by_code(col_fits[0]), u_grid_median=pooled(col_frames[0])),
                        G=dict(curve=pooled_by_code(col_fits[1]), u_grid_median=pooled(col_frames[1])),
                        B=dict(curve=pooled_by_code(col_fits[2]), u_grid_median=pooled(col_frames[2])),
                        frames=[len(f) for f in col_frames], films="Portra 400 / 160 / 800, Ektar 100, Pro 400H"),
            per_frame=dict(bw=bw_raw, colour=col_raw)),
        frontier_epfl=dict(note="未导出: EPFL 的扫描配准对象是同场景数码 RAW (景物曝光), 不是底片密度 (T54 2.2), 含 800T 的特性曲线; "
                                "没有以底片密度为轴的 Frontier 影调测量。"),
    )
    dst = ROOT / "silvergrain" / "data" / "lab_tone.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for key, x in [("bw", out["rpl_noritsu"]["bw"]["curve"])] + [(ch, out["rpl_noritsu"]["colour"][ch]["curve"]) for ch in "RGB"]:
        print(key, "code", x["code"][0], "–", x["code"][-1], "u", x["u"][0], x["u"][-1])
        for c in (40, 68, 99, 118, 138, 160, 184, 210):
            if x["code"][0] <= c <= x["code"][-1]:
                i = int(np.argmin(np.abs(np.asarray(x["code"]) - c)))
                print(f"   code {c:4d} u {x['u'][i]:+.3f} slope {x['slope_median'][i]:6.1f} IQR {x['slope_q25'][i]:6.1f}–{x['slope_q75'][i]:6.1f} n {x['n_frames'][i]}")

if __name__ == "__main__":
    main()
