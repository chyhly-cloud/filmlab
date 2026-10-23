"""Independent, opt-in input candidates preserving available CFA measurements.

These are censored-data diagnostics, not calibrated scene-radiance recovery or
800T material models. ``observed_floor`` leaves every camera observation alone;
``local_ratio`` estimates only partially saturated channels from a local colour
assumption. Both leave fully saturated RGB cells at their observed values.

The decoder matrix is measured with digital LibRaw probes, not fitted to the
photograph. Applying it in float avoids LibRaw's 16-bit XYZ output ceiling.
It is still the camera's decoder colour transform, not an independently measured
camera spectral sensitivity. Negative noise and out-of-gamut RGB are retained.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from filmlab.rawio import XYZ_TO_SRGB
from tools.eval.linear_raw_measurements import bayer_measurements, decoder_matrix


MODES = ("floor", "observed_floor", "local_ratio")


@dataclass
class SensorConstrainedInput:
    rgb: np.ndarray
    camera: np.ndarray
    saturation: np.ndarray
    estimated_mask: np.ndarray
    unknown_mask: np.ndarray
    matrix_xyz: np.ndarray
    metadata: dict


def constrain_camera(camera, saturation, *, mode="observed_floor"):
    """Return camera values and per-channel estimate flags, without mutation.

    For ``local_ratio``, a component's surrounding unsaturated bright pixels
    supply a colour vector. A nonnegative least-squares scalar is determined
    from the pixel's available channels. Only missing channels are increased,
    never below their observed clipped values. This is an explicit local-colour
    assumption: a different lamp spectrum or reflection can make it incorrect.
    No finite upper bound on missing radiance follows from this procedure.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cam = np.asarray(camera, dtype=float)
    sat = np.asarray(saturation, dtype=bool)
    if cam.ndim != 3 or cam.shape[-1] != 3 or sat.shape != cam.shape:
        raise ValueError("camera and saturation must have matching (H,W,3) shapes")
    if not np.isfinite(cam).all():
        raise ValueError("camera observations must be finite")
    out = cam.copy()
    estimated = np.zeros_like(sat)
    if mode in ("floor", "observed_floor"):
        return out, estimated
    any_sat = sat.any(-1)
    partial = any_sat & ~sat.all(-1)
    labels, _ = ndimage.label(any_sat)
    brightness = cam.mean(-1)
    for component, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        pad = max(4, int(np.ceil(.6 * np.sqrt((sl[0].stop-sl[0].start) * (sl[1].stop-sl[1].start)))))
        ys = slice(max(0, sl[0].start-pad), min(cam.shape[0], sl[0].stop+pad))
        xs = slice(max(0, sl[1].start-pad), min(cam.shape[1], sl[1].stop+pad))
        region = (ys, xs)
        blob = labels[region] == component
        use = blob & partial[region]
        if not use.any():
            continue
        ring = (ndimage.binary_dilation(blob, iterations=pad)
                & ~ndimage.binary_dilation(blob, iterations=2)
                & ~any_sat[region])
        if ring.sum() < 5:
            continue  # No independent neighbouring colour estimate: leave unknown.
        ring_values = brightness[region][ring]
        bright = ring & (brightness[region] >= np.percentile(ring_values, 70))
        colour = np.maximum(np.median(cam[region][bright], axis=0), 0)
        if not np.any(colour > 0):
            continue
        observations = cam[region][use]
        missing = sat[region][use]
        available = ~missing
        denominator = np.sum(np.where(available, colour**2, 0), axis=1)
        numerator = np.sum(np.where(available, observations*colour, 0), axis=1)
        scale = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        scale = np.maximum(scale, 0)
        proposal = scale[:, None] * colour
        changed = missing & (denominator[:, None] > 0) & (proposal > observations)
        values = np.where(changed, proposal, observations)
        out[region][use] = values
        estimated[region][use] = changed
    return out, estimated


def from_measurements(camera, saturation, matrix_xyz, *, mode="observed_floor", metadata=None):
    """Construct a candidate from already decoded observations and row matrix.

    ``matrix_xyz`` obeys ``xyz = camera @ matrix_xyz``. RGB uses the transpose
    of the repository's column-vector XYZ_TO_SRGB matrix, with no clipping.
    A second mode can reuse the same measurements/matrix via this function.
    """
    matrix = np.asarray(matrix_xyz, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix_xyz must be a finite (3,3) row-vector transform")
    original = np.asarray(camera, dtype=float)
    sat = np.asarray(saturation, dtype=bool)
    constrained, estimated = constrain_camera(original, sat, mode=mode)
    rgb = (constrained @ matrix) @ XYZ_TO_SRGB.T
    info = dict(metadata or {})
    info.update({
        "candidate": mode,
        "shape": list(rgb.shape),
        "camera_to_xyz_matrix": matrix.tolist(),
        "matrix_convention": "xyz = camera @ matrix_xyz; rgb = xyz @ XYZ_TO_SRGB.T",
        "counts": {
            "cells": int(sat.shape[0]*sat.shape[1]),
            "any_RGB_saturation_flag": int(sat.any(-1).sum()),
            "all_RGB_saturation_flags": int(sat.all(-1).sum()),
            "estimated_channels": int(estimated.sum()),
            "unknown_channels": int(sat.sum()),
            "observed_camera_channels_below_zero": int((original < 0).sum()),
            "output_RGB_channels_above_one": int((rgb > 1).sum()),
        },
        "unsaturated_camera_channels_max_change": float(np.max(np.abs(constrained[~sat]-original[~sat]), initial=0)),
        "limits": [
            "Saturation uses the existing pile-up heuristic, not independent sensor calibration.",
            "Each half-size Bayer cell averages two greens; green saturation flags combine them with ANY. All RGB flags do not prove every individual CFA sample saturated.",
            "observed_floor substitutes censored camera observations, not recovered radiance. With noise and uncertain ceilings it is not a rigorous bound on scene radiance.",
            "A camera-channel lower-bound interpretation does not imply componentwise RGB lower bounds: colour transforms can have negative coefficients.",
            "local_ratio is a neighbouring-colour assumption; estimated channels remain unknown measurements and can be spectrally wrong.",
            "No random brightness, denoising, exposure compensation, output clipping or colour fitting is applied.",
            "No actual 800T material, lens-flare or scanner response is measured by this candidate.",
        ],
    })
    return SensorConstrainedInput(rgb, constrained, sat.copy(), estimated, sat.copy(), matrix.copy(), info)


def load_sensor_constrained(path, *, mode="floor"):
    """Read native half-size Bayer observations, then build the explicit mode.

    This currently supports Bayer RAW files handled by ``bayer_measurements``.
    It does not demosaic a full-size image or resize the native half-size grid.
    """
    path = Path(path)
    import rawpy
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            raise ValueError("This candidate requires a 2x2 Bayer mosaic")
        desc = raw.color_desc.decode()
        names = [desc[int(v)] for v in pattern.ravel()]
        if sorted(names) != ["B", "G", "G", "R"]:
            raise ValueError("This candidate requires one red, two green and one blue sample per cell")
    cam, sat, sensor_info = bayer_measurements(path)
    matrix, probe_info = decoder_matrix(path)
    return from_measurements(cam, sat, matrix, mode=mode, metadata={
        "path": str(path), "sensor": sensor_info, "decoder_probe": probe_info,
    })
