"""包的独立性与数据库: 不导入 filmlab; 胶片 / 显影液 / 来源齐全; 推导确定。"""
import ast
from pathlib import Path

import numpy as np
import pytest

import silvergrain as sg
from silvergrain import database as db

ROOT = Path(sg.__file__).resolve().parent


def test_no_filmlab_imports():
    bad = []
    for p in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.level == 0 else []
            else:
                continue
            bad += [f"{p.name}: {n}" for n in names if n.split(".")[0] == "filmlab"]
    assert not bad, bad


def test_film_database():
    F = sg.films()
    bw = [k for k, f in F.items() if f.is_bw]
    assert len(bw) == 10 and "kodak_trix_400" in bw and "ilford_hp5_plus" in bw
    assert {"kodak_portra_400", "kodak_vision3_500t", "fujifilm_provia_100f"} <= set(F)
    for k, f in F.items():
        assert f.ei > 0 and f.source, k
        lam, ls = f.spectral()
        assert len(lam) > 10
    # 官方 RMS 与测量条件、来源 URL
    assert any(o["metric"] == "rms_net1" and o["value"] == "17" and o["source"].startswith("http") for o in F["kodak_trix_400"].official)
    assert any(o["metric"] == "rms_net1" for o in F["kodak_tmax_400"].official)
    assert all("http" in o["source"] for f in F.values() for o in f.official)


def test_developers():
    D = sg.developers()
    assert set(D) >= {"d76", "xtol", "rodinal", "fx39", "C-41", "ECN-2", "E-6"}
    assert D["rodinal"].cluster_mu == 2.0 and D["rodinal"].cluster_um == 1.0
    assert D["d76"].cluster_mu == 0.0
    assert D["xtol"].photons_r("kodak_trix_400") == pytest.approx(3.562)
    assert D["d76"].photons_r("kodak_trix_400") == 4.0
    assert sg.developer("Rodinal").key == "rodinal"


def test_derive_bw_and_colour():
    e = sg.derive("kodak_trix_400")
    assert e.path == "silver" and e.developer.key == "d76" and e.spec.thickness_um is None
    assert 3.5 < sum(e.spec.coat_g_m2) < 4.2                     # T47: 3.87 g/m²
    r = sg.derive("kodak_trix_400", "rodinal")
    assert r.spec.cluster_mode == "developed" and r.spec.cluster_mu == 2.0
    c = sg.derive("kodak_portra_400")
    assert c.path == "dye" and c.developer.key == "C-41" and c.spec.yield_cv == 0.3
    with pytest.raises(ValueError):
        sg.derive("kodak_trix_400", "C-41")
    with pytest.raises(ValueError):
        sg.derive("kodak_trix_400", grain_scale=4.0)


def test_universal_constants_from_data():
    U = sg.universal()
    t46 = db.universal()["t46"]
    assert U["eta"] == t46["eta"] == 0.0248 and U["beta"] == 0.875
    assert np.asarray(U["lam"]).shape == (3, 3)


def test_formats():
    assert sg.px_um_for((4000, 6000), "35mm") == pytest.approx(6.0)
    assert sg.px_um_for((2000, 3000), "half") == pytest.approx(8.0)
    assert sg.crop_to_format(np.zeros((100, 100)), "35mm").shape == (100, 67)
    assert sg.crop_to_format(np.zeros((100, 200)), "35mm").shape == (100, 150)
