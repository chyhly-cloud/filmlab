"""Compare texture residuals against the corresponding mean at normal display size."""
import json

import numpy as np
from PIL import Image

import silver_process as film

OUT = film.ROOT / 'silver-emulsion'
PREV = film.ROOT / 'silver-validation'


def measure(folder, name):
    p, m = (Image.open(folder / f'{name}-{suffix}.png').convert('L')
            for suffix in ('positive', 'mean'))
    assert p.size == m.size and max(p.size) == film.LONG_EDGE
    shape = (round(p.width / p.height * 440), 440)
    x, y = (np.asarray(i.resize(shape, Image.Resampling.LANCZOS), dtype='f4') for i in (p,m))
    return dict(dimensions=p.size, display_rms_levels=float(np.std(x-y)),
                clipped_low_display_fraction=float(np.mean(x <= 1)),
                clipped_high_display_fraction=float(np.mean(x >= 254)))


def main():
    data = {}
    for name in film.PHOTOS:
        if not (OUT / f'{name}.json').exists():
            continue
        a,b = measure(PREV,name), measure(OUT,name)
        assert a['dimensions'] == b['dimensions']
        data[name] = dict(previous=a, experiment=b)
    print(json.dumps(data, indent=2))
    film.write_json(dict(photo_height=440, photos=data), OUT/'photo-audit.json')


if __name__ == '__main__':
    main()
