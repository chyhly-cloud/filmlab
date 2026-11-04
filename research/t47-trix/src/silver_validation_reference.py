"""Read published observations from a documented Tri-X/D-76 experiment."""
import hashlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

import silver_process as film

SOURCES = film.ROOT.parent / 'sources' / 'silver-validation'


def read_points():
    # From the published plot's axes and the purple D-76 markers.
    density_path = SOURCES / 'schmidt-density.png'
    pixels = np.asarray(Image.open(density_path).convert('RGB'))
    points = []
    for x, y in [(143,431), (179,425), (214,418), (249,409), (284,396), (319,382), (355,362), (389,340), (425,319),
                 (460,300), (495,284), (531,267), (566,251), (601,233), (635,215),
                 (671,199), (705,183), (741,167), (777,150)]:
        color = pixels[y-6:y+7, x-4:x+5]
        row, column = np.where(np.all(color == [135,120,179], axis=2))
        if len(row) < 15:
            raise ValueError('D-76 marker missing from the graph')
        # x=73 corresponds to E=0, x=773 to E=3; y=490 is D=0, y=42 is D=2.5.
        points.append(((x-4+column.mean()-73)*3/700, (490-y+6-row.mean())*2.5/448))

    # The second series is Tri-X D-76. These locations are blue measured dots,
    # not the author's dashed curve or a fitted granularity model.
    plot = SOURCES / 'schmidt-aperture.png'
    raw = np.asarray(Image.open(plot).convert('RGB'))
    data = []
    for nominal, (x, y) in zip((5,10,20,25), [(207.3,237.3), (324.4,190.9), (562.5,162.), (686.5,168.5)]):
        i, j = round(y), round(x)
        assert np.any(np.all(raw[i-2:i+3, j-2:j+3] == [94,129,181], axis=2))
        aperture = (x-91) * 25 / (682-91)
        product = (419-y)*.8 / (419-40)
        data.append({'aperture_um': aperture, 'nominal_aperture_um': nominal,
                     'published_rms_D': product / aperture, 'point_xy': [x,y]})
    return np.asarray(points), data, [hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (density_path,plot)]


def sampled_apertures(exposure, render, sizes):
    pitch, length, pad = .625, 500, 24
    trim = round(pad/pitch)
    shape = (round(length/pitch)+2*trim,) * 2
    uniform = np.full(shape, exposure, dtype=np.float32)
    deltas = [[] for _ in sizes]
    for seed in range(573, 585):
        t, _, _ = render(uniform, pitch, np.random.default_rng(seed))
        blurred = gaussian_filter(t, 1.5/pitch)[trim:-trim, trim:-trim]
        for size, samples in zip(sizes, deltas):
            b = round(size / pitch)
            d = -np.log10(film.aperture(blurred, b))
            samples.append(np.diff(d, axis=1).ravel() / np.sqrt(2))
    return [float(np.std(np.concatenate(samples))) for samples in deltas]


def compare(stats, mean_transmission, render=None):
    points, apertures, hashes = read_points()
    base = .318
    assert points[0,1] < base+.1 < points[-1,1]
    x_ref = float(np.interp(base+.1, points[:,1], points[:,0]))
    pitch = .3125
    model_base = -np.log10(mean_transmission(0., pitch))
    def density(exposure):
        return float(-np.log10(mean_transmission(exposure,pitch)))
    h_ref = brentq(lambda h: density(h)-model_base-.1, .01, 50)
    h_texture = brentq(lambda h: density(h)-1., .01, 50)
    rows = []
    for x, d in points:
        stops = float((x-x_ref)/np.log10(2))
        prediction = density(h_ref * 2**stops) - model_base
        rows.append({'stops_from_net_D_0p1': stops, 'observed_net_D': float(d-base),
                     'model_net_D': prediction, 'error_D': prediction-float(d-base)})
    for row in apertures:
        a = row['nominal_aperture_um']
        predictions = [stats(h_texture,pitch,optical_sigma=s,adjacent=True,apertures=(a,))
                       ['rms_D_linearized'][0] for s in (0.,1.5,3.)]
        row['model_rms_D_sigma_1p5um'] = predictions[1]
        row['model_optics_sensitivity'] = [min(predictions),max(predictions)]
    if render is not None:
        sizes = [row['nominal_aperture_um'] for row in apertures]
        for row, rms in zip(apertures, sampled_apertures(h_texture, render, sizes)):
            row['model_measured_rms_D_sigma_1p5um'] = rms
    return {
        'source_density': 'https://www.bernhard-w-schmidt.de/en/einzeln-trix',
        'source_apertures': 'https://www.bernhard-w-schmidt.de/en/methodgrain',
        'published_figures_sha256': hashes,
        'developer': 'Kodak D-76 stock, 9 min, 20 °C, EI 400',
        'density': rows,
        'apertures': apertures,
        'absolute_exposure_calibration_available': False,
        'density_anchor': 'Register relative exposure origins at net D=0.10; no curve shape parameter changed.',
        'aperture_notes': 'Same microscope image, neighboring square-cell difference / sqrt(2) at gross D=1.0. '
                          'Four digitized dots. No raw or independent scanner optical characterization.',
        'film_validation_passed': False,
    }
