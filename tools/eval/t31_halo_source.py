"""T31 第 3/4 步 (docs/tasks/T31_HALO_COLOUR.md): 光晕源 G/R、B/G (曝光空间, 以中灰各层曝光为单位) 对遮罩放置、散射处理、返回角、光源的依赖。
用法: python3 t31_halo_source.py   (打印第 3、4 步的表; stack(place="phys") 与 film.MASK_PLACEMENT="layers" 逐位一致)"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import film, stocks, variation, lights, spectral
from filmlab.stocks import WAVELENGTHS as W
st = stocks.load_stock("kodak_vision3_500t")
v = variation.sample(None, 0, None)
S = 10.0 ** st.log_sensitivity
t = lambda d: 10.0 ** -d


def split_mask(edge=480.0):
    mask = np.clip(st.base_density - st.base_density.min(), 0, None)
    wm = 1.0 / (1.0 + np.exp((W - edge) / 10.0))      # 品红成色剂的黄色遮罩: 蓝区
    return mask * (1 - wm), mask * wm                   # Mc (红感层, 青成色剂的遮罩), Mm (绿感层)


def stack(place="model", scatter="extinct", mask_scale=1.0, mask_sub=0.0, path_back=1.0, path_down=1.0, turbidity=None, edge=480.0):
    A = np.asarray(v.layer_absorbance)[None, :] * S / S.max(0)
    Ay = v.yellow_filter / (1.0 + np.exp((W - 505.0) / 9.0))
    Mc, Mm = split_mask(edge)
    if mask_sub:      # 扣除假定的成色剂以外的 D-min 成分 (如品红灰雾), 只扣绿区形状: 按品红染料光谱
        Mc = np.clip(Mc - mask_sub * st.dye_density[:, 1] / st.dye_density[:, 1].max(), 0, None)
    Mc, Mm = Mc * mask_scale, Mm * mask_scale
    turb = (film.TURBIDITY if turbidity is None else turbidity) * (550.0 / W) ** 3
    if place == "none":
        Mc = Mm = 0 * W
    ext = turb if scatter == "extinct" else 0 * W
    a = [A[:, c] + ext for c in range(3)]
    ab = lambda al, ac: (1 - t(al)) * (ac / np.maximum(al, 1e-9))
    pd, pb = path_down, path_back
    mask = Mc + Mm
    if place in ("model", "none"):
        direct = np.stack([t(a[2] + Ay + a[1] + mask / 2) * ab(a[0], A[:, 0]), t(a[2] + Ay) * ab(a[1], A[:, 1]), ab(a[2], A[:, 2])], 1)
        Td = t(pd * (a[2] + Ay + a[1] + a[0] + mask))
        back = np.stack([ab(pb * a[0], pb * A[:, 0]), t(pb * (a[0] + mask / 2)) * ab(pb * a[1], pb * A[:, 1]),
                         t(pb * (a[0] + a[1] + Ay + mask)) * ab(pb * a[2], pb * A[:, 2])], 1)
    elif place == "phys":      # Mc 均匀在红感层, Mm 均匀在绿感层
        direct = np.stack([t(a[2] + Ay + a[1] + Mm + Mc / 2) * ab(a[0], A[:, 0]), t(a[2] + Ay + Mm / 2) * ab(a[1], A[:, 1]), ab(a[2], A[:, 2])], 1)
        Td = t(pd * (a[2] + Ay + a[1] + a[0] + mask))
        back = np.stack([t(pb * Mc / 2) * ab(pb * a[0], pb * A[:, 0]), t(pb * (a[0] + Mc + Mm / 2)) * ab(pb * a[1], pb * A[:, 1]),
                         t(pb * (a[0] + a[1] + Ay + mask)) * ab(pb * a[2], pb * A[:, 2])], 1)
    if scatter == "conserve":   # 散射不损失, 到达片基时已被散射 (可进入全反射角) 的比例 = 1 − 10^−Σ散射密度
        Td = Td * (1 - t(pd * 3 * turb)) / v.diffuse_fraction
    return Td, direct, back


def halo(E, **kw):
    Td, direct, back = stack(**kw)
    unit = (E @ S) / (E @ direct)
    h = (E * Td) @ back * unit * v.diffuse_fraction
    return h, E @ S / (E @ S)[1]


def check():
    Td, d, b = film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)
    Td2, d2, b2 = stack()
    assert np.allclose(Td, Td2) and np.allclose(d, d2) and np.allclose(b, b2)


LIGHTS = {"3200K": spectral.illuminant("BB3200"), "钨丝 2700K": lights._blackbody(2700),
          "高压钠 HPS": lights._sd("HPS"), "高压钠 C100S54": lights._sd("C100S54 (HPS)"),
          "白光 LED-B3": lights._sd("LED-B3"), "白光 YAG": lights._sd("Phosphor LED YAG"),
          "荧光 Cool White": lights._sd("Cool White FL"), "荧光 TL84": lights._sd("Philips TL-84"),
          "低压钠 589nm": np.exp(-0.5 * ((W - 589.0) / 3.0) ** 2)}


def gr(h):
    return h[1] / h[0], h[2] / max(h[1], 1e-12)


if __name__ == "__main__":
    check()
    print("emulsion_stack 复现: OK")
    mask = np.clip(st.base_density - st.base_density.min(), 0, None)
    print("遮罩 (base_density − min) @450/500/550/580/600/650:", [round(float(np.interp(x, W, mask)), 3) for x in (450, 500, 550, 580, 600, 650)])
    cases = [("① 现模型 (绿光过遮罩: 下行 1 次 + 返回 0.5 次)", dict()),
             ("② 物理放置 (Mc 在红感层, Mm 在绿感层; 绿光 1 + 1 次)", dict(place="phys")),
             ("③ 无遮罩", dict(place="none")),
             ("④ 物理放置, 遮罩扣品红灰雾 0.05 @550", dict(place="phys", mask_sub=0.05)),
             ("⑤ 物理放置, 遮罩扣 0.10", dict(place="phys", mask_sub=0.10)),
             ("⑥ 现模型, 散射守恒 (散射光进入漫射, 不作消光)", dict(scatter="conserve")),
             ("⑦ 物理放置 + 散射守恒", dict(place="phys", scatter="conserve")),
             ("⑧ 现模型, 返回斜光程 1/cos45°", dict(path_back=1 / np.cos(np.radians(45)))),
             ("⑨ 现模型, 返回斜光程 1/cos60°", dict(path_back=2.0)),
             ("⑩ 现模型, 下行 + 返回斜光程 1/cos45°", dict(path_back=1 / np.cos(np.radians(45)), path_down=1 / np.cos(np.radians(45)))),
             ("⑪ 现模型, 散射 0", dict(turbidity=0.0)),
             ]
    print("\n| 情形 | " + " | ".join(LIGHTS) + " |")
    for nm, kw in cases:
        cells = []
        for ln, E in LIGHTS.items():
            h, Hd = halo(E, **kw)
            g, b = gr(h)
            cells.append(f"{g:.3f} / {b:.2f}")
        print(f"| {nm} | " + " | ".join(cells) + " |")
    print("\n灯芯直接曝光 G/R (同单位):", {ln: round(float((E @ S)[1] / (E @ S)[0]), 3) for ln, E in LIGHTS.items()})
