"""Experimental steady-state closure for the existing effective DIR model.

The historical single-pass predictor is a possible approximation, not a proven
chemistry bug. This candidate assumes released inhibitor tracks final developed
silver. It is not a time-resolved reaction/diffusion model or measured 800T law.
No default engine path imports this module; relaxation is numerical, not a
material fit. Failure to converge is an error, never an accepted render.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from filmlab import chemistry
_HISTORICAL=chemistry.develop_image


def solve(e,logH,tau,inhib_amount,blur_px,kind='color',*,tolerance=1e-8,max_iterations=300,relaxation=.5,diagnostics=None):
    if not 0 < relaxation <= 1:raise ValueError('relaxation must be in (0,1]')
    if tolerance<=0 or max_iterations<1:raise ValueError('positive tolerance and iteration limit required')
    x=e.log_exposure;S=chemistry.kinetic_curve(e.S_nom,tau,x,kind)
    if inhib_amount<=0:return chemistry._curves_at(logH,x,S)
    A=chemistry._dir_matrix(inhib_amount,e.design=='negative')
    if e.design in ('bw','paper'):A=np.diag(np.diag(A))
    C=chemistry._intrinsic_curve(e,S,A)
    def response(d):
        first=d/e.halide_cap
        if blur_px>.3 and first.ndim==3:first=gaussian_filter(first,(blur_px,blur_px,0))
        return np.minimum(chemistry._curves_at(logH-first@A,x,C),e.halide_cap)
    initial=np.minimum(_HISTORICAL(e,logH,tau,inhib_amount,blur_px,kind),e.halide_cap)
    # Save the historical function before research callers temporarily substitute solve.
    d=initial
    initial_residual=float(np.max(abs(response(d)-d)))
    for iteration in range(max_iterations):
        next_d=response(d);residual=float(np.max(abs(next_d-d)))
        if residual<tolerance:
            if diagnostics is not None:diagnostics.update(iterations=iteration,residual=residual,initial_residual=initial_residual)
            return d
        d=d+relaxation*(next_d-d)
    raise RuntimeError(f'DIR closure did not converge: residual={residual}')
