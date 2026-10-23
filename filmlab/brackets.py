"""Opt-in replacement by validated, exposure-normalized RAW measurements.

This module neither detects lamps nor estimates spectra. A bright reflector
can be measured just as a lamp can. Caller-supplied trusted masks must encode
registration, saturation and temporal agreement checks. No random fallback is
created here; unverified input pixels retain their original values explicitly.
"""
from dataclasses import dataclass

import numpy as np

from .rawio import XYZ_TO_SRGB


@dataclass(frozen=True)
class MeasuredHighlightPatch:
    origin_xy: tuple
    camera: np.ndarray  # Already normalized to the primary shutter/aperture scale
    trusted: np.ndarray  # All three camera channels validated, boolean
    provenance: str


def replace_measured_highlights(rgb, primary_saturation, patches, camera_to_xyz):
    rgb = np.asarray(rgb)
    saturation = np.asarray(primary_saturation)
    matrix = np.asarray(camera_to_xyz, float)
    if saturation.shape != rgb.shape or saturation.dtype != np.bool_:
        raise ValueError('Saturation must be a per-camera-channel boolean array')
    if matrix.shape != (3,3) or not np.isfinite(matrix).all():
        raise ValueError('A finite decoder matrix is required')
    out=rgb.copy();replaced=np.zeros(rgb.shape[:2],bool)
    for patch in patches:
        a=np.asarray(patch.camera,float);trusted=np.asarray(patch.trusted)
        if a.ndim!=3 or a.shape[-1]!=3 or trusted.shape!=a.shape[:2] or trusted.dtype!=np.bool_:
            raise ValueError('Measured patch needs RGB samples and a boolean trusted mask')
        if not isinstance(patch.provenance,str) or not patch.provenance.strip():
            raise ValueError('Measurement provenance required')
        if not np.isfinite(a[trusted]).all() or np.any(a[trusted]<0):
            raise ValueError('Trusted RAW measurements must be finite and nonnegative')
        x,y=patch.origin_xy;h,w=a.shape[:2]
        if x<0 or y<0 or y+h>rgb.shape[0] or x+w>rgb.shape[1]:raise ValueError('Patch outside image')
        sl=np.s_[y:y+h,x:x+w]
        use=trusted&saturation[sl].any(-1)
        if np.any(replaced[sl]&use):raise ValueError('Choose one validated exposure per pixel before replacement')
        # Do not clamp XYZ/RGB: transformed highlights can legitimately exceed 1
        # or lie outside sRGB. Clamp only when presenting an output image.
        out[sl][use]=(a[use]@matrix)@XYZ_TO_SRGB.T
        replaced[sl]|=use
    return out,replaced
