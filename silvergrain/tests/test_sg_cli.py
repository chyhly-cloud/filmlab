"""T55k: 命令行 python -m silvergrain (读写、RAW、并行数不影响结果)。"""
import sys
import types

import numpy as np
import pytest

import silvergrain as sg
from silvergrain import __main__ as cli


def _png(path, h=48, w=72):
    from PIL import Image
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([40 + 3 * x, 60 + 2 * y, 120 + x // 2], -1).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)
    return a


def test_cli_png_roundtrip_matches_api(tmp_path):
    src = tmp_path / "in.png"; _png(src)
    out = tmp_path / "out.png"
    assert cli.main([str(src), str(out), "--film", "kodak_trix_400", "--format", "1.5", "--seed", "3", "-q"]) == 0
    from PIL import Image
    got = np.asarray(Image.open(out))
    r = sg.apply_to_image(str(src), "kodak_trix_400", format=1.5, seed=3)
    want = np.round(sg.image.srgb_encode(np.clip(r.linear, 0, 1)) * 255).astype(np.uint8)
    assert got.shape == want.shape and np.array_equal(got, want)


def test_cli_16bit_and_workers_invariance(tmp_path):
    src = tmp_path / "in.png"; _png(src)
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    for o, wk in ((a, "1"), (b, "3")):
        assert cli.main([str(src), str(o), "--film", "kodak_portra_400", "--format", "1.5", "--bits", "16", "--workers", wk, "-q"]) == 0
    import tifffile
    x, y = tifffile.imread(a), tifffile.imread(b)
    assert x.dtype == np.uint16 and x.shape == (48, 72, 3) and np.array_equal(x, y)


def test_cli_list_and_bad_args(tmp_path, capsys):
    assert cli.main(["--list"]) == 0
    assert "kodak_trix_400" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "missing.png"), str(tmp_path / "o.png")])


def test_cli_raw_reader_scales_p99(monkeypatch, tmp_path):
    """RAW / DNG 经 rawpy 解成线性 sRGB, 乘常数使亮度第 99 百分位 = 0.85 (rawpy 用替身, 不需要第三方文件)。"""
    img16 = (np.random.default_rng(0).random((40, 60, 3)) * 30000).astype(np.uint16)

    class Raw:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def postprocess(self, **kw):
            assert kw["gamma"] == (1, 1) and kw["no_auto_bright"] and kw["output_bps"] == 16
            return img16

    fake = types.SimpleNamespace(imread=lambda p: Raw(), ColorSpace=types.SimpleNamespace(sRGB=1),
                                 HighlightMode=types.SimpleNamespace(Blend=2))
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    p = tmp_path / "x.dng"; p.write_bytes(b"0")
    lin = cli.read_input(p)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    assert np.percentile(Y, 99) == pytest.approx(0.85)
    assert np.allclose(lin / lin.max(), img16 / img16.max(), atol=1e-4)
