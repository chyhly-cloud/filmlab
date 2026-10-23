"""Linear, censored camera constraints for nonnegative spectral-library weights.

This evaluation-only module does not fit final RGB, a film model or a material.
For a K by 3 camera response matrix C, the predicted *linear camera* measurement
is ``coefficients @ C``. Coefficients are nonnegative; matrix entries and queried
linear-response vectors may be signed. All supplied numeric inputs must be finite.

Available channels constrain observation +/- (atol + rtol*abs(observation)).
Saturated channels supply only a hard lower bound equal to the observation; the
tolerances never weaken that bound. Bounds and observations must use the same
linear measurement units, before white balance, colour conversion or encoding.

An LP witness is an arbitrary feasible library combination, NOT a recovered true
spectrum, a minimum-norm estimate, or proof that the real lamp is in the library.
All-saturated observations supply no measured upper bound. For a feasible physical
nonnegative response the upper bound is generally infinite; zero or signed query
responses can still have finite extrema, so each bound is actually solved.
"""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import linprog


# This remains numerical feasibility, not an exact-arithmetic certificate.
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite numbers")
    return array


def _tolerance(value, name):
    try:
        array = np.broadcast_to(np.asarray(value, dtype=float), (3,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or length 3") from exc
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def observation_bounds(observed, saturated, *, relative_tolerance=0.,
                       absolute_tolerance=0.):
    """Return three-channel lower/upper bounds, using +inf only for censoring.

    Tolerances may be nonnegative scalars or length-three vectors. ``saturated``
    must be a length-three boolean mask; a saturated value is a lower limit,
    never an equality. Signed observations are allowed (e.g. black subtraction).
    """
    observed = _finite_array(observed, (3,), "observed")
    saturated = np.asarray(saturated)
    if saturated.shape != (3,) or saturated.dtype.kind != "b":
        raise ValueError("saturated must be a length-three boolean mask")
    rtol = _tolerance(relative_tolerance, "relative_tolerance")
    atol = _tolerance(absolute_tolerance, "absolute_tolerance")
    with np.errstate(over="ignore", invalid="ignore"):
        tolerance = atol + rtol*abs(observed)
        lower = observed - tolerance
        upper = observed + tolerance
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("observation bounds overflowed")
    lower = np.where(saturated, observed, lower)
    upper = np.where(saturated, np.inf, upper)
    return lower, upper


def _inputs(camera_response, observed, saturated, relative_tolerance, absolute_tolerance):
    camera = np.asarray(camera_response, dtype=float)
    if (camera.ndim != 2 or camera.shape[0] == 0 or camera.shape[1] != 3
            or not np.isfinite(camera).all()):
        raise ValueError("camera_response must be a finite, nonempty K by 3 matrix")
    lower, upper = observation_bounds(observed, saturated,
                                     relative_tolerance=relative_tolerance,
                                     absolute_tolerance=absolute_tolerance)
    # Row scaling preserves the feasible set and keeps dark measurements from
    # relying on a large absolute LP residual tolerance in camera units.
    scales = np.maximum(abs(np.asarray(observed, dtype=float)), .005)
    return camera, lower, upper, scales


def _inequalities(camera, lower, upper, scales):
    has_upper = np.isfinite(upper)
    a = np.concatenate([-camera.T/scales[:, None],
                        camera.T[has_upper]/scales[has_upper, None]])
    b = np.concatenate([-lower/scales, upper[has_upper]/scales[has_upper]])
    return a, b


def _solve(objective, a, b):
    result = linprog(objective, A_ub=a, b_ub=b, bounds=(0., None),
                     method="highs", options=_LP_OPTIONS)
    if result.status not in (0, 2, 3):
        raise RuntimeError(f"LP failed numerically (status {result.status}): {result.message}")
    return result


def solve_censored_constraints(camera_response, observed, saturated, *,
                               relative_tolerance=0., absolute_tolerance=0.,
                               responses=None):
    """Find feasibility and min/max of named linear direct/return responses.

    ``responses`` maps names to finite length-K vectors q. Each bound is for
    ``coefficients @ q``, with all original channel constraints retained. An
    unbounded minimum/maximum is -inf/+inf. Each finite endpoint includes its
    own minimum_coefficients/maximum_coefficients witness; unbounded endpoints
    have None. Direct and returned exposures must be computed from that SAME
    witness, not assembled from separately extremized layers. Infeasible inputs
    return no witness and an empty response_bounds dictionary. Other solver
    failures raise.

    The returned witness has no statistical or physical preference over other
    feasible combinations. No minimum norm or regularization is applied.
    """
    camera, lower, upper, scales = _inputs(camera_response, observed, saturated,
                                         relative_tolerance, absolute_tolerance)
    if responses is None:
        responses = {}
    if not isinstance(responses, Mapping):
        raise ValueError("responses must map names to length-K linear responses")
    queries = {name: _finite_array(value, (len(camera),), f"response {name!r}")
               for name, value in responses.items()}
    a, b = _inequalities(camera, lower, upper, scales)
    witness = _solve(np.zeros(len(camera)), a, b)
    report = {"feasible": witness.status == 0,
              "status": "feasible" if witness.status == 0 else "infeasible",
              "coefficients": None, "predicted_camera": None,
              "lower_bounds": lower, "upper_bounds": upper,
              "all_saturated": bool(np.all(saturated)),
              "has_measured_upper_bound": bool(np.isfinite(upper).any()),
              "response_bounds": {}}
    if witness.status == 2:
        return report
    if witness.status != 0:
        raise RuntimeError("Zero-objective feasibility LP unexpectedly reported unbounded")
    report["coefficients"] = witness.x
    report["predicted_camera"] = witness.x @ camera
    report["maximum_scaled_constraint_violation"] = float(max(0., np.max(a @ witness.x-b)))
    for name, query in queries.items():
        minimum = _solve(query, a, b)
        maximum = _solve(-query, a, b)
        if minimum.status == 2 or maximum.status == 2:
            raise RuntimeError("A response LP disagreed with the feasibility LP")
        report["response_bounds"][name] = {
            "minimum": -np.inf if minimum.status == 3 else float(minimum.fun),
            "maximum": np.inf if maximum.status == 3 else float(-maximum.fun),
            "minimum_coefficients": None if minimum.status == 3 else minimum.x,
            "maximum_coefficients": None if maximum.status == 3 else maximum.x,
        }
    return report


def fixed_spectrum_amplitude_interval(camera_rgb, observed, saturated, *,
                                      relative_tolerance=0., absolute_tolerance=0.):
    """Intersect all channel constraints for one fixed spectrum times a>=0.

    Returns the full feasible amplitude interval, not a selected brightness.
    Signed or zero camera responses are handled explicitly. A zero-response
    channel whose interval excludes zero makes the intersection infeasible.
    """
    response = _finite_array(camera_rgb, (3,), "camera_rgb")
    lower, upper = observation_bounds(observed, saturated,
                                     relative_tolerance=relative_tolerance,
                                     absolute_tolerance=absolute_tolerance)
    minimum, maximum = 0., np.inf
    feasible = True
    for value, lo, hi in zip(response, lower, upper):
        if value > 0:
            minimum, maximum = max(minimum, lo/value), min(maximum, hi/value)
        elif value < 0:
            minimum, maximum = max(minimum, hi/value), min(maximum, lo/value)
        elif not lo <= 0 <= hi:
            feasible = False
    feasible = feasible and minimum <= maximum
    return {"feasible": bool(feasible),
            "status": "feasible" if feasible else "infeasible",
            "minimum": float(minimum) if feasible else None,
            "maximum": float(maximum) if feasible else None,
            "all_saturated": bool(np.all(saturated)),
            "lower_bounds": lower, "upper_bounds": upper}


def minimum_relative_tolerance(camera_response, observed, saturated):
    """Minimum common normalized available-channel residual (a diagnostic).

    Minimize t>=0 with abs(prediction-observed)<=t*max(abs(observed), .005)
    for *available* channels. Saturated lower bounds stay hard, including when
    t grows. This is a Chebyshev consistency diagnostic, not a fitted spectrum
    estimate or a confidence interval. No user tolerance is added to this LP.
    If hard saturated constraints are infeasible, minimum t is +inf.
    """
    camera, lower, upper, scales = _inputs(camera_response, observed, saturated, 0., 0.)
    a, b = _inequalities(camera, lower, upper, scales)
    # Lower constraints come first for every channel; upper constraints exist
    # only for available channels. Both available sides receive -t.
    available = np.isfinite(upper)
    t_column = np.concatenate([-available.astype(float), -np.ones(available.sum())])
    a = np.column_stack([a, t_column])
    objective = np.zeros(len(camera)+1)
    objective[-1] = 1.
    result = _solve(objective, a, b)
    if result.status == 3:
        raise RuntimeError("Nonnegative tolerance objective unexpectedly reported unbounded")
    if result.status == 2:
        return {"feasible": False, "status": "infeasible_hard_saturation",
                "minimum_relative_tolerance": np.inf,
                "coefficients": None, "predicted_camera": None,
                "normalization": scales}
    return {"feasible": True, "status": "feasible",
            "minimum_relative_tolerance": float(result.x[-1]),
            "coefficients": result.x[:-1],
            "predicted_camera": result.x[:-1] @ camera,
            "normalization": scales,
            "maximum_scaled_constraint_violation": float(max(0., np.max(a @ result.x-b)))}
