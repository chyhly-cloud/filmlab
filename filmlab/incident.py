"""Opt-in incident spectral signals, independent of RGB illumination inference.

The camera receives outgoing radiance L = L_emitted + integral(f_r L_i cosθ dω).
RGB alone cannot identify the two terms. An explicit measurement can replace
the TOTAL incident signal on a mask, whether it came from a lamp or a reflector.
No brightness threshold classifies pixels here.

Spectra use the existing 81-bin engine signal convention (380..780 nm, 5 nm).
Their amplitude must share the RGB input's exposure normalization. These are
not absolute W/sr/m²/nm: stock sensitivity and log_k lack that calibration.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IncidentSpectrum:
    spectrum: np.ndarray  # (81,), measured/controlled nonnegative spectral shape
    amplitude: np.ndarray  # (h,w), nonnegative scale in engine input units
    mask: np.ndarray  # (h,w), explicit boolean coverage; never inferred by brightness
    provenance: str


def apply_incident(engine, H, halo_source, sources):
    """Replace explicit rays before spatial transport, retaining existing optics.

Return-unit conversion uses a fixed material-reference calibration, independent
of the source spectrum and lens filter. It is a linear proxy, not an independently
measured material absorption/quantum-efficiency response.
"""
    from . import film
    shape = H.shape[:2]
    covered = np.zeros(shape, bool)
    H = H.copy()
    halo_source = None if halo_source is None else halo_source.copy()
    sens, return_response = spectral_responses(engine)
    filt = film._filt(engine.p.lens_filter)
    for source in sources:
        spectrum = np.asarray(source.spectrum, float)
        amplitude = np.asarray(source.amplitude, float)
        mask = np.asarray(source.mask)
        if spectrum.shape != (81,) or amplitude.shape != shape or mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError('Incident spectrum needs 81 bins and matching amplitude/boolean mask')
        if not isinstance(source.provenance, str) or not source.provenance.strip():
            raise ValueError('Incident spectral signal requires provenance')
        if not np.isfinite(spectrum).all() or np.any(spectrum < 0) or spectrum.sum() <= 0:
            raise ValueError('Spectrum must be finite, nonnegative and nonzero')
        if not np.isfinite(amplitude).all() or np.any(amplitude < 0):
            raise ValueError('Amplitude must be finite and nonnegative')
        if np.any(covered & mask):
            raise ValueError('Overlapping incident signals would replace the same ray twice')
        covered |= mask
        effective = spectrum * filt
        direct_H = effective @ sens
        amount = amplitude[mask, None] * 2 ** engine.p.exposure_ev
        H[mask] = amount * direct_H
        if halo_source is not None:
            source_H = effective @ return_response
            halo_source[mask] = amount * source_H
    return H, halo_source


def spectral_responses(engine):
    """Fixed 81x3 forward responses in existing engine exposure units.

    Calibrate the optical proxy once against the stock's fixed reference
    illuminant, without a scene/lens filter. This preserves the old reference
    response but is additive for arbitrary known spectra and filters. It does
    not infer wavelength-wise quantum efficiency by dividing S(lambda)/A(lambda),
    and does not alter the historical RGB/default halation_response path.
    """
    from . import film
    stock = engine.e.stock
    sens = 10 ** stock.log_sensitivity
    down, direct, back = film.emulsion_stack(stock, engine.v.layer_absorbance, engine.v.yellow_filter)
    reference = film._spec(stock.reference_illuminant)
    denominator = reference @ direct
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError('Material-reference absorption must be positive and finite')
    unit = (reference @ sens) / denominator
    response = down[:,None] * back * unit * engine.v.diffuse_fraction * engine._back_reflect
    return sens, response
