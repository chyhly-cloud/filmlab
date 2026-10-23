"""Necessary, independent checks for conditional light-source inverse fits.

These checks do not turn a fitted spectrum into a measurement. In particular,
sensor consistency using a CIE/decoder proxy is only consistency with that
proxy, and a nonnegative spectrum does not validate undeveloped 800T optics.
No camera noise, clipping ceiling or energy calibration is invented here.
"""
import numpy as np


def censored_sensor_residual(predicted, observed, saturation, noise_sigma, *, lower_bound=None):
    """Noise-normalized equality residuals and one-sided clipping residuals.

    All inputs must use the same *linear camera-channel* normalization. For an
    available channel, (prediction-observation)/sigma is returned. A flagged
    channel supplies only a lower inequality; predictions above its supplied
    lower bound incur zero residual and are not claimed to be accurate.
    The default bound is its clipped observation, which is only appropriate if
    the caller has checked that sensor's black/white normalization and censoring.
    ``noise_sigma`` is supplied by an independent noise estimate, not fitted to
    hide violations. This function deliberately supplies no finite upper bound.
    """
    prediction = np.asarray(predicted, dtype=float)
    observation = np.asarray(observed, dtype=float)
    saturation = np.asarray(saturation, dtype=bool)
    if prediction.shape != observation.shape or saturation.shape != observation.shape:
        raise ValueError("prediction, observation and saturation shapes must match")
    if not np.isfinite(prediction).all() or not np.isfinite(observation).all():
        raise ValueError("camera values must be finite")
    sigma = np.broadcast_to(np.asarray(noise_sigma, dtype=float), observation.shape)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("independent noise_sigma must be finite and positive")
    bound = observation if lower_bound is None else np.broadcast_to(np.asarray(lower_bound, float), observation.shape)
    if not np.isfinite(bound).all():
        raise ValueError("camera censoring bounds must be finite")
    equality = (prediction - observation) / sigma
    inequality = np.minimum(prediction - bound, 0) / sigma
    return np.where(saturation, inequality, equality)


def sensor_consistency(predicted, observed, saturation, noise_sigma, *, lower_bound=None, tolerance_sigma=3.):
    """Report available-channel and censored-channel violations separately."""
    if not np.isfinite(tolerance_sigma) or tolerance_sigma <= 0:
        raise ValueError("tolerance_sigma must be positive")
    residual = censored_sensor_residual(predicted, observed, saturation, noise_sigma, lower_bound=lower_bound)
    saturation = np.asarray(saturation, dtype=bool)
    available, censored = residual[~saturation], residual[saturation]
    return {
        "available_channels": int(available.size),
        "censored_channels": int(censored.size),
        "available_rms_sigma": float(np.sqrt(np.mean(available**2))) if available.size else None,
        "available_max_abs_sigma": float(np.max(np.abs(available))) if available.size else None,
        "available_violation_fraction": float(np.mean(np.abs(available) > tolerance_sigma)) if available.size else None,
        "censored_lower_bound_violation_fraction": float(np.mean(censored < -tolerance_sigma)) if censored.size else None,
        "upper_radiance_bound_provided": False,
        "meaning": "Necessary consistency under the supplied camera operator/noise/ceiling; not unique radiance recovery",
    }


def fixed_spectrum_amplitude_bounds(camera_response, observed, saturation, noise_sigma,
                                    *, lower_bound=None, tolerance_sigma=3.):
    """Exact feasible amplitude interval for one fixed nonnegative camera ray.

    ``predicted_camera = amplitude * camera_response`` and amplitude >= 0.
    Available samples impose two-sided tolerance boxes; censored samples impose
    only lower bounds. Responses may include predeclared spatial footprints or
    known exposure ratios. No new amplitude may be fitted to a held-out sample.
    A ``None`` upper bound means infinity, not an estimated power limit.
    The tolerance interval is conditional on the operator/noise/ceiling and is
    not a statistical confidence interval for actual source radiance.
    """
    response = np.asarray(camera_response, float)
    observation = np.asarray(observed, float)
    saturated = np.asarray(saturation, bool)
    if response.shape != observation.shape or saturated.shape != observation.shape:
        raise ValueError("response, observation and saturation shapes must match")
    if not np.isfinite(response).all() or np.any(response < 0):
        raise ValueError("physical camera ray response must be finite and nonnegative")
    # Reuse independent input validation without relying on an optimizer.
    censored_sensor_residual(response, observation, saturated, noise_sigma, lower_bound=lower_bound)
    if not np.isfinite(tolerance_sigma) or tolerance_sigma <= 0:
        raise ValueError("tolerance_sigma must be positive")
    sigma = np.broadcast_to(np.asarray(noise_sigma, float), observation.shape)
    bound = observation if lower_bound is None else np.broadcast_to(np.asarray(lower_bound, float), observation.shape)
    lower = np.where(saturated, bound, observation) - tolerance_sigma*sigma
    upper = np.where(saturated, np.inf, observation + tolerance_sigma*sigma)
    positive = response > 0
    impossible_zero = (~positive) & ((lower > 0) | (upper < 0))
    lo = float(max(0., np.max(lower[positive]/response[positive], initial=0.)))
    hi = float(np.min(upper[positive]/response[positive], initial=np.inf))
    return {
        "feasible": bool(not impossible_zero.any() and hi >= lo),
        "amplitude_lower": lo,
        "amplitude_upper": hi if np.isfinite(hi) else None,
        "unbounded_above": bool(np.isposinf(hi)),
        "incompatible_zero_response_channels": int(impossible_zero.sum()),
        "meaning": "Exact interval under the supplied fixed ray and tolerance boxes, not actual-radiance identification",
    }


def spectral_rank_report(basis_spectra, observer, *, relative_tolerance=1e-10):
    """Report coefficient-space observational rank for fixed linear spectra.

    basis_spectra is (K,wavelengths), observer is (wavelengths,channels).
    Repeated exposures with the same channel responses scale this operator and
    do not create new spectral rank. Nonnegative coefficients constrain a cone,
    but positive coefficient solutions can still vary along null directions.
    """
    basis = np.asarray(basis_spectra, dtype=float)
    camera = np.asarray(observer, dtype=float)
    if basis.ndim != 2 or camera.ndim != 2 or basis.shape[1] != camera.shape[0]:
        raise ValueError("basis and observer wavelength dimensions must match")
    if not np.isfinite(basis).all() or not np.isfinite(camera).all() or np.any(basis < 0):
        raise ValueError("basis spectra must be finite and nonnegative; observer finite")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    matrix = (basis @ camera).T
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = relative_tolerance * singular[0] if singular.size else 0.
    rank = int(np.sum(singular > threshold))
    return {
        "basis_count": int(basis.shape[0]), "observed_channels": int(camera.shape[1]),
        "singular_values": singular.tolist(), "rank": rank,
        "coefficient_nullity": int(basis.shape[0] - rank),
        "observer_has_negative_coefficients": bool(np.any(camera < 0)),
        "meaning": "Linear observational rank; neither nonnegative uniqueness nor actual sensor calibration is certified",
    }


def heldout_residual_report(baseline, fitted, target, train_mask, test_mask):
    """Compare finite forward errors on disjoint, preselected spatial supports.

    Arrays are (...,channels); use matching linear RGB or matching Lab inputs,
    never a mixture of encoded and linear RGB. Euclidean RMSE is descriptive,
    not deltaE00, a significance test or a confidence bound. The caller must
    predeclare spatial/scene splits and not select a model by its held-out score.
    """
    baseline, fitted, target = [np.asarray(x, float) for x in (baseline, fitted, target)]
    train, test = np.asarray(train_mask, bool), np.asarray(test_mask, bool)
    if baseline.shape != fitted.shape or target.shape != fitted.shape or fitted.ndim < 2:
        raise ValueError("baseline, fitted and target must have matching channel arrays")
    if train.shape != fitted.shape[:-1] or test.shape != train.shape or np.any(train & test):
        raise ValueError("train/test supports must match the spatial grid and be disjoint")
    finite = np.isfinite(baseline).all(-1) & np.isfinite(fitted).all(-1) & np.isfinite(target).all(-1)
    result = {}
    for name, support in (("train", train), ("heldout", test)):
        mask = support & finite
        if not mask.any():
            raise ValueError(f"{name} support has no common finite samples")
        before = float(np.sqrt(np.mean((baseline[mask] - target[mask])**2)))
        after = float(np.sqrt(np.mean((fitted[mask] - target[mask])**2)))
        result[name] = {"samples": int(mask.sum()), "baseline_channel_RMSE": before,
                        "fitted_channel_RMSE": after, "RMSE_reduction": before-after}
    result["train_improves_but_holdout_worsens"] = bool(
        result["train"]["RMSE_reduction"] > 0 and result["heldout"]["RMSE_reduction"] < 0)
    result["meaning"] = "Descriptive disjoint-support prediction check; no physical uniqueness claim"
    return result


def near_equivalent_fit_report(coefficients, losses, *, accepted_loss_ceiling, basis_spectra=None):
    """Keep distinct near-equal fits visible instead of declaring one true SPD.

    The loss ceiling is supplied before interpreting radiance ranges. A narrow
    ensemble range is not a confidence interval or proof that other solutions
    do not exist; it only describes the finite trials actually performed.
    """
    coefficients, losses = np.asarray(coefficients, float), np.asarray(losses, float)
    if coefficients.ndim != 2 or losses.shape != (len(coefficients),):
        raise ValueError("coefficients and losses must be trials-by-parameter and trials")
    if (not np.isfinite(coefficients).all() or not np.isfinite(losses).all()
            or np.any(coefficients < 0) or not np.isfinite(accepted_loss_ceiling)):
        raise ValueError("trials must be finite and source coefficients nonnegative")
    accepted = coefficients[losses <= accepted_loss_ceiling]
    report = {"trials": int(len(coefficients)), "accepted_trials": int(len(accepted)),
              "loss_ceiling": float(accepted_loss_ceiling),
              "meaning": "Finite near-equivalent fit ensemble, not a posterior or uniqueness certificate"}
    if not len(accepted):
        return report
    report["coefficient_min"] = accepted.min(0).tolist()
    report["coefficient_max"] = accepted.max(0).tolist()
    if basis_spectra is not None:
        basis = np.asarray(basis_spectra, float)
        if basis.ndim != 2 or basis.shape[0] != coefficients.shape[1] or not np.isfinite(basis).all() or np.any(basis < 0):
            raise ValueError("basis must be finite nonnegative parameter-by-wavelength spectra")
        spectra = accepted @ basis
        report["spectrum_min"] = spectra.min(0).tolist()
        report["spectrum_max"] = spectra.max(0).tolist()
        report["discrete_spectral_sum_range"] = [float(spectra.sum(1).min()), float(spectra.sum(1).max())]
        report["spectral_sum_units"] = "Caller-supplied discrete spectral units, not SI radiance unless independently calibrated"
    return report
