"""Measure averaging-order bias on one fixed, blurred synthetic film field."""
from ctypes import CDLL, c_double, c_int
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter

import silver_field_check as field
import silver_process as film


def reference_grid_bias():
    # Exact covariance of continuous disks evaluated at the cell centers.
    # For grid-aligned boxes, compare it to the continuous aperture integral.
    rows = []
    for kind, radii in field.GRAIN.items():
        for tau in field.TAUS:
            for density in field.DENSITIES:
                rate, theory = field.expectation(density, tau, radii)
                support = int(2*max(radii)/field.PITCH)
                shifts = np.arange(-support, support+1)
                distance = np.hypot(shifts[:,None], shifts)*field.PITCH
                common = np.mean([field.overlap(r, distance) for r in radii],axis=0)
                attenuation = -np.expm1(-tau)
                covariance = theory['mean']**2 * np.expm1(rate*attenuation**2*common)
                errors = {}
                for aperture, continuous in zip(field.APERTURES[:4],theory['aperture_T_variance']):
                    bins = round(aperture/field.PITCH)
                    axis = (bins-np.abs(shifts))/bins**2
                    errors[str(aperture)] = float((axis @ covariance @ axis)/continuous-1)
                rows.append(dict(kind=kind,tau=f'{tau:g}',density=density,
                                 relative_variance_error=errors))
    return rows


def main():
    # Use a distinct seed and the same mixed disk case at the higher density.
    pitch, side, fov = .125, 416., field.FOV
    n = round(side/pitch)
    radii, tau, density = field.GRAIN['mixture'], 1., field.DENSITIES[-1]
    rate, _ = field.expectation(density, tau, radii)
    rng = np.random.default_rng(6910)
    count = rng.poisson(rate*side**2)
    particles = np.column_stack((rng.uniform(0,side,(count,2)),
                                 rng.choice(radii,count),rng.random(count)))
    low, high = (np.zeros((n,n),dtype='u2') for _ in range(2))
    source = Path(__file__).with_name('silver_field_counts.c')
    with TemporaryDirectory(dir=field.OUT) as tmp:
        library = Path(tmp)/'silver.dylib'
        subprocess.run(['/usr/bin/clang','-O3','-dynamiclib',str(source),'-o',str(library)],
                       check=True, timeout=30)
        handle = CDLL(str(library))
        stamp = handle.silver_counts
        stamp.argtypes = [np.ctypeslib.ndpointer('u2',flags='C_CONTIGUOUS')]*2 + [
            c_int,c_double,np.ctypeslib.ndpointer('f8',flags='C_CONTIGUOUS'),c_int]
        stamp.restype = None
        stamp(low, high, n, pitch, particles, count)
    t = gaussian_filter(np.exp(-np.float32(tau)*high),field.OPTICAL_SIGMA/pitch,mode='wrap')
    rows=[]
    for s in (pitch,)+field.SCAN+(203.2,):
        scan = field.boxes(t,pitch,s,int(np.ceil(fov/s)))
        macro = -np.log10(field.boxes(scan,s,fov,1)[0,0])
        pixel = field.boxes(-np.log10(scan),s,fov,1)[0,0]
        rows.append(dict(pitch_um=s, macro_D=float(macro), mean_pixel_D=float(pixel),
                         jensen_bias=float(pixel-macro)))
    assert all(row['jensen_bias'] >= 0 for row in rows)
    film.write_json(dict(nominal_density=density, rows=rows, grid_bias=reference_grid_bias()),field.OUT/'jensen-check.json')


if __name__ == '__main__':
    main()
