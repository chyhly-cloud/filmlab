"""独立检验 (只用颗粒数据): 与 filmlab 的 T46 / T47 / T47b / T50 / T53 结果逐位相同, 门槛照旧。"""
import numpy as np
import pytest

from silvergrain.validation import refdata, t46_official, t47_schmidt, t50_developers, t53_scale


def test_t46_all_fit_identical():
    r = t46_official.run(full=False)
    assert r["max_abs_diff_vs_reference"] == 0.0
    assert r["beta"] == 0.875 and r["eta"] == pytest.approx(0.0248, abs=1e-4)


def test_t47_schmidt_four_apertures():
    r = t47_schmidt.schmidt_four_apertures()
    assert r["identical"] and r["passes"] and r["density_passes"]
    assert np.max(np.abs(r["closed_rel"])) < 0.03
    assert r["kodak48_x1000"] == pytest.approx(14.6, abs=0.05)


def test_t47b_tmax_and_schmidt():
    r = t47_schmidt.t47b()
    assert max(r["max_abs_diff_vs_reference"].values()) == 0.0
    off = {x["key"]: x for x in r["official"]}
    assert off["tmax100"]["t47_ratio"] == pytest.approx(1.00, abs=0.01)
    assert off["tmax400"]["t47_ratio"] == pytest.approx(1.11, abs=0.01)
    assert r["summary"]["holdout_median_abs_log"] == pytest.approx(0.10, abs=0.01)


def test_t50_d76_reference():
    q = t50_developers.quick()
    want = dict(trix=(13.6, 5.7), tmx=(9.0, 5.7), tmy=(11.6, 6.0), d100=(7.9, 5.7), d400=(9.9, 5.9), panf=(6.2, 5.5))
    for f, (rms, ak) in want.items():
        assert q["d76"][f]["rms_pred"] == pytest.approx(rms, abs=0.05)
        assert q["d76"][f]["ak_pred"] == pytest.approx(ak, abs=0.05)
    rod = q["developers"]["Rodinal"]["films"]["trix"]["pred"]
    assert rod["rms_ratio"] == pytest.approx(1.42, abs=0.01) and rod["rms"] == pytest.approx(19.2, abs=0.05)
    assert q["developers"]["XTOL"]["pass_ak"] == "6/6"


def test_t53_closed_form_identical():
    r = t53_scale.run()
    assert r["max_abs_diff_vs_reference"] == 0.0
    t = r["kodak_trix_400@d76"]["ks"]
    for k in ("0.5", "1.5", "2.0", "3.0"):
        assert t[k]["selwyn_ratio_over_k"]["96.0"] == pytest.approx(1.0, abs=0.02)
        assert t[k]["coat_ratio"] == pytest.approx(float(k), rel=1e-6)
    assert max(v["curve_max_abs_dD"] for v in r["kodak_tmax_400@d76"]["ks"].values()) < 0.01


def test_realscan_reference_is_measurement_only():
    ref = refdata.load("realscan_reference.json")
    assert ref["rpl"]["zip"].startswith("https://") and len(ref["rpl"]["sha256"]) == 64
    tx = ref["t51b_same_code"]["films"]["Tri-X 400 (RPL: XTOL)"]
    assert [b["real"]["s48"] for b in tx] == [4.45, 5.00, 5.02, 3.74]
    assert any(a["scale"] == "phone" for a in ref["t52_anchor"]["rows"])
    assert len(ref["t51_spectrum_rpl_bw"]["tri_x"]) == 16


@pytest.mark.slow
def test_t46_full_loo_identical():
    r = t46_official.run(full=True, log=lambda *a: None)
    assert r["compare"]["identical"]
    assert r["loo_median_abs_log"] == pytest.approx(0.170, abs=0.001)
