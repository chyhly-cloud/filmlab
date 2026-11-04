"""Unfitted black-and-white negative experiment, in projected emulsion coordinates."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1] / 'results'
OUT = ROOT / 'silver-process'
FILM_EDGE = 36_000.0
LONG_EDGE = 1200
FINE_PITCH = 2.5
SCAN_SIGMA = 2.5
BASE_T = .85
DEV_TIME = 1.0
PHOTO_EXPOSURE = 10.0
Y = np.array([.2126, .7152, .0722], dtype=np.float32)
PHOTOS = ('portrait', 'night', 'still')


@dataclass(frozen=True)
class CrystalPopulation:
    per_um2: float
    capture: float
    threshold: int
    development_rate: float
    optical_area_um2: float
    footprint_sigma_um: float


# Illustrative projected populations. No manufacturer or scan measurements used.
COATING = (
    CrystalPopulation(.12, .64, 4, 2.7, 18., 1.1),
    CrystalPopulation(.045, 1.5, 3, 1.8, 42., 2.0),
)


def development(population, time, fog=True):
    return -np.expm1(-time * population.development_rate), -np.expm1(-time * .002) if fog else 0.


def probabilities(exposure, population):
    effective_events = exposure * population.capture
    # With independent effective captures and no decay during the exposure,
    # P(N >= k) for a Poisson count is the regularized lower gamma function.
    return -np.expm1(-effective_events), gammainc(population.threshold, effective_events)


def aperture(transmission, bin_size):
    height, width = transmission.shape
    return transmission.reshape(height // bin_size, bin_size, width // bin_size, bin_size).mean((1, 3))


def kernel(population, pitch):
    sigma = population.footprint_sigma_um / pitch
    radius = int(4 * sigma + .5)
    coordinate = np.arange(-radius, radius + 1, dtype=np.float32)
    axis = np.exp(-.5 * (coordinate / sigma) ** 2)
    axis /= axis.sum()
    return np.outer(axis, axis) * (population.optical_area_um2 / pitch**2)


def exposed_film(exposure, pitch, rng, time=DEV_TIME, fog=True, stages=False):
    depth = np.zeros(exposure.shape, dtype=np.float32)
    log_mean = np.zeros_like(depth)
    trace = {name: np.zeros(exposure.shape, dtype=np.int32)
             for name in ('coating', 'latent', 'eligible', 'silver', 'removed')} if stages else None
    for p in COATING:
        expected_sites = p.per_um2 * pitch**2
        p_latent, p_eligible = probabilities(exposure, p)
        q_dev, q_fog = development(p, time, fog)
        if trace is None:
            # Poisson thinning is the marginal distribution of the staged
            # crystal simulation below. Use it to render the same mechanism at scale.
            silver = rng.poisson(expected_sites * (p_eligible * q_dev + (1 - p_eligible) * q_fog))
        else:
            sites = rng.poisson(expected_sites, size=exposure.shape)
            latent = rng.binomial(sites, p_latent)
            conditional = np.divide(p_eligible, p_latent, out=np.zeros_like(p_latent), where=p_latent > 0)
            eligible = rng.binomial(latent, np.clip(conditional, 0, 1))
            silver = rng.binomial(eligible, q_dev) + rng.binomial(sites - eligible, q_fog)
            for name, count in zip(trace, (sites, latent, eligible, silver, sites - silver)):
                trace[name] += count

        # Project each developed grain's effective extinction footprint. Fixing
        # removes the remaining sites. Optical depths add before exponentiation.
        depth += gaussian_filter(silver.astype(np.float32), p.footprint_sigma_um / pitch,
                                 mode='constant') * (p.optical_area_um2 / pitch**2)
        expected_silver = expected_sites * (p_eligible * q_dev + (1 - p_eligible) * q_fog)
        # Exact ensemble transmission for independent Poisson site counts on
        # this grid: log E[exp(-kN)] = mean(N) * (exp(-k) - 1).
        log_mean += fftconvolve(expected_silver, np.expm1(-kernel(p, pitch)), mode='same')
    return BASE_T * np.exp(-depth), BASE_T * np.exp(log_mean), trace


def scanner(transmission, pitch, bin_size):
    light = gaussian_filter(transmission, SCAN_SIGMA / pitch, mode='reflect')
    return aperture(light, bin_size)


def homogeneous(exposure, pitch, time=DEV_TIME, fog=True):
    total = 0.
    for p in COATING:
        _, p_eligible = probabilities(exposure, p)
        q_dev, q_fog = development(p, time, fog)
        total += (p.per_um2 * pitch**2 * (p_eligible * q_dev + (1 - p_eligible) * q_fog)
                  * np.expm1(-kernel(p, pitch)).sum())
    return BASE_T * np.exp(total)


def positive(transmission, pitch):
    clear = homogeneous(0., pitch)
    saturated = homogeneous(np.inf, pitch)
    linear = np.clip((clear - transmission) / (clear - saturated), 0, 1)
    return linear ** (1 / 2.2)


def srgb_decode(values):
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def save_image(values, name, destination=OUT):
    path = destination / f'{name}.png'
    temporary = path.with_suffix('.part.png')
    Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(temporary)
    temporary.replace(path)


def write_json(obj, path):
    temporary = path.with_suffix('.part.json')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def experiment():
    # A single artificial negative with regions at known relative exposure.
    pitch = .5
    values = (0., .35, .7, 1.4, 2.8, 5.6, 11.2, 45.)
    width, height = 128, 768
    incident = np.repeat(np.asarray(values, dtype=np.float32), width)
    exposure = np.broadcast_to(incident, (height, incident.size))
    negative, expectation, trace = exposed_film(exposure, pitch, np.random.default_rng(151), stages=True)
    assert np.array_equal(trace['coating'], trace['silver'] + trace['removed'])
    assert (trace['eligible'] <= trace['latent']).all()
    assert (trace['latent'] <= trace['coating']).all()
    assert (trace['eligible'][:, :width] == 0).all()
    assert np.isclose(homogeneous(5., pitch, time=0), BASE_T)
    assert np.isclose(homogeneous(0., pitch, fog=False), BASE_T)
    assert np.allclose(negative, exposed_film(exposure, pitch, np.random.default_rng(151), stages=True)[0])

    scan_fine = scanner(negative, pitch, 4)
    scan_coarse = scanner(negative, pitch, 16)
    assert np.allclose(scan_coarse, aperture(scan_fine, 4), atol=2e-7)
    assert np.isclose(-np.log10(np.mean([1., .1])), .2596373105057561)
    show = {
        'exposure': np.log1p(exposure) / np.log1p(values[-1]),
        **{name: gaussian_filter((trace[name] > 0).astype(np.float32), 1., mode='constant') * 5
           for name in ('coating', 'latent', 'eligible', 'silver', 'removed')},
        'negative': negative,
        'scan-fine': scan_fine,
        'scan-coarse': scan_coarse,
        'positive': positive(scan_fine, pitch),
    }
    for name, pixels in show.items():
        save_image(pixels, name)

    rows = []
    for i, light in enumerate(values):
        region = slice(i * width + 24, (i + 1) * width - 24)
        native = (slice(32, -32), region)
        fine = -np.log10(scan_fine[8:-8, i * 32 + 6:(i + 1) * 32 - 6])
        coarse = -np.log10(scan_coarse[2:-2, i * 8 + 2:(i + 1) * 8 - 2])
        modeled = homogeneous(light, pitch)
        assert np.allclose(expectation[native], modeled, atol=2e-5)
        rows.append({
            'exposure': light,
            'developed_fraction': float(trace['silver'][native].sum() / trace['coating'][native].sum()),
            'predicted_transmission': float(modeled),
            'sample_transmission': float(negative[native].mean()),
            'scan_density': float(fine.mean()),
            'rms_density_2um': float(fine.std()),
            'rms_density_8um': float(coarse.std()),
        })
    assert np.all(np.diff([r['scan_density'] for r in rows]) > 0)
    assert max(abs(r['sample_transmission'] - r['predicted_transmission']) for r in rows) < .02
    assert all(r['rms_density_8um'] < r['rms_density_2um'] for r in rows)
    return {'coating': [asdict(p) for p in COATING], 'base_transmission': BASE_T,
            'scanner_sigma_um': SCAN_SIGMA, 'microscope_grid_um': pitch, 'exposure_steps': rows,
            'full_fine_grid_um': FINE_PITCH, 'photo_exposure_factor': PHOTO_EXPOSURE}


def render_photo(name, seed, *, destination=None, emulsion=exposed_film, display=positive, model_key=''):
    destination = OUT if destination is None else destination
    source = ROOT / 'full' / f'{name}-original.png'
    receipt = destination / f'{name}.json'
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key += model_key
    key += f':{source.stat().st_size}:{source.stat().st_mtime_ns}'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['key'] == key and all((destination / f'{name}-{suffix}.png').is_file()
                                        for suffix in ('negative', 'positive', 'mean')):
            return result

    bin_size = round(FILM_EDGE / LONG_EDGE / FINE_PITCH)
    with Image.open(source) as original:
        width, height = [round(n / max(original.size) * LONG_EDGE) for n in original.size]
        image = original.convert('RGB').resize((width * bin_size, height * bin_size), Image.Resampling.LANCZOS)
    drawn = np.empty((height, width), dtype=np.float32)
    expected = np.empty_like(drawn)
    tile = 64
    # Seed the coating on an output-aligned lattice. Render overlapping halos
    # with the same count draws by keying each fine row to its global position.
    pad = 2
    for y in range(0, height, tile):
        y1 = min(y + tile, height)
        top, bottom = max(0, y - pad), min(height, y1 + pad)
        crop = np.asarray(image.crop((0, top * bin_size, width * bin_size, bottom * bin_size)), dtype=np.float32)
        exposure = PHOTO_EXPOSURE * (srgb_decode(crop / 255) @ Y)

        class StripRng:
            def __init__(self):
                self.population = 0

            def poisson(self, intensity):
                counts = np.empty(intensity.shape, dtype=np.int32)
                for j, row in enumerate(intensity, start=top * bin_size):
                    generator = np.random.default_rng(np.random.SeedSequence([seed, self.population, j]))
                    counts[j - top * bin_size] = generator.poisson(row)
                self.population += 1
                return counts

        negative, mean, _ = emulsion(exposure, FINE_PITCH, StripRng())
        inside = slice(y - top, y1 - top)
        drawn[y:y1] = scanner(negative, FINE_PITCH, bin_size)[inside]
        expected[y:y1] = scanner(mean, FINE_PITCH, bin_size)[inside]
    # Exclude a sensor-width border for FFT zero padding at the emulsion edge.
    drawn = drawn[1:-1, 1:-1]
    expected = expected[1:-1, 1:-1]
    # Extend the last valid sample to keep the same full composition and size.
    drawn = np.pad(drawn, 1, mode='edge')
    expected = np.pad(expected, 1, mode='edge')
    shown = display(drawn, FINE_PITCH)
    shown_mean = display(expected, FINE_PITCH)
    save_image(drawn, f'{name}-negative', destination)
    save_image(shown, f'{name}-positive', destination)
    save_image(shown_mean, f'{name}-mean', destination)
    result = {'key': key, 'dimensions': [width, height], 'scan_pitch_um': bin_size * FINE_PITCH,
              'positive_rms_levels': float(np.std(shown - shown_mean) * 255)}
    write_json(result, receipt)
    print(name, result, flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    results = experiment()
    write_json(results, OUT / 'experiment.json')
    print('Exposure, fixing, reproducibility and rescanning checks passed.', flush=True)
    results['photos'] = {name: render_photo(name, 421 + i) for i, name in enumerate(PHOTOS)}
    write_json(results, OUT / 'experiment.json')


if __name__ == '__main__':
    main()
