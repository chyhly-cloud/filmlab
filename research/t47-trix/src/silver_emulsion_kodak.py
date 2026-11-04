"""Read a separate manufacturer's characteristic-curve plate."""
import hashlib

import numpy as np
from PIL import Image

import silver_process as film
from silver_validation_reference import SOURCES, read_points

SRC = SOURCES / 'kodak-f4017-page8.png'
OUT = film.ROOT / 'silver-emulsion'
# Curve center lines, manually identified for 8 and 10 min, in the top-left
# 35 mm graph. Sample only where both dashed curves are actually drawn.
PICKS = [(319, 310.5, 290.5), (351, 285.5, 261.5), (382, 264, 239.5), (399, 254, 229)]


def main():
    a = np.asarray(Image.open(SRC).convert('L'))
    rows = []
    for x, *ys in PICKS:
        assert all(a[round(y)-1:round(y)+2, x].min() < 155 for y in ys)
        rows.append({'log_H_lux_s': (x-381.5)/61.6,
                     'gross_D_8min': (413.5-ys[0])/77,
                     'gross_D_10min': (413.5-ys[1])/77})
    film.write_json(dict(source='https://business.kodakmoments.com/sites/default/files/files/products/f4017_TriX.pdf#page=8',
                         sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
                         process='35 mm 400TX, D-76 large tank, 20 °C, agitation every minute, diffuse visual density',
                         rows=rows, approximate_plot_readout=True), OUT / 'kodak.json')


def developer_control():
    # The top series in Schmidt Fig. 4 is also Tri-X, processed in FX39.
    # Use the nearest ~25 µm dot to the D-76 dot already digitized.
    src = SOURCES / 'schmidt-aperture.png'
    img = np.asarray(Image.open(src).convert('RGB'))
    y, x = np.where(np.all(img[66:76, 682:691] == [94,129,181], axis=2))
    assert len(x) >= 25
    xpos, ypos = float(x.mean()+682), float(y.mean()+66)
    aperture = (xpos-91)*25/(682-91)
    rms = (419-ypos)*.8/(419-40)/aperture
    d76 = read_points()[1][-1]['published_rms_D']
    film.write_json(dict(aperture_um=aperture, gross_density=1, rms_FX39=rms, rms_D76=d76,
                         fx39_to_d76=rms/d76, point_xy=[xpos,ypos],
                         process='FX39 1+12, 10 min, 24 °C; D-76 stock, 9 min, 20 °C',
                         approximate_plot_readout=True), OUT / 'developer-control.json')


if __name__ == '__main__':
    main()
    developer_control()
