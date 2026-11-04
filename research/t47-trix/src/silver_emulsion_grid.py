"""Check coarse full-frame integration throughout the useful exposure range."""
import numpy as np
import silver_emulsion as study
import silver_process as film
import silver_sampling as sampling

coating = study.CASES['both']
rows = []
for h in (.6, 2., 4., 10., 30., 150.):
    a, b = (sampling.stats(h, pitch, apertures=(30,), coating=coating)
            for pitch in (2.5, study.PITCH))
    coarse_D, fine_D = a['D_from_mean_T'], b['D_from_mean_T']
    delta = abs(coarse_D-fine_D)
    rms_delta = a['rms_D_linearized'][0]/b['rms_D_linearized'][0]-1
    rows.append(dict(exposure=h, fine_D=float(fine_D), density_abs_error=float(delta),
                     rms_relative_error=rms_delta))
    if h <= film.PHOTO_EXPOSURE:
        assert delta < .001 and abs(rms_delta) < .08
    rows[-1]["passed"] = bool(delta < .001 and abs(rms_delta) < .08)
film.write_json(dict(aperture_um=30, coarse_um=2.5, fine_um=study.PITCH, rows=rows),
                study.OUT / 'grid.json')
