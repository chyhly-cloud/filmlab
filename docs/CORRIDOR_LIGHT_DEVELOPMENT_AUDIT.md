# Corridor small-light development audit

The three manually selected R1-02-3 regions correspond to the user's marked
near ceiling light, middle ceiling light and orange reflection in the glass.
They must be checked independently of the previously studied large streetlights.

## What is actually modelled

`chemistry.develop_image` maps three-layer log exposure through stored characteristic
curves, with an approximate inverse DIR construction and spatial inhibition.
`chemistry.run` limits developed silver by available halide, forms dye limited by
coupler, and performs the subsequent process steps. This is a curve-based effective
model, not a first-principles chemical reaction simulation.

The colour development time response in `kinetic_curve` was fitted to Portra 800
push-process curves. Its transfer to Vision3 stock in C-41 is not independently
validated by those data. The ECN-2/C-41 relative activity remains an estimate.
The cross-process dye spectral shift includes cyan evidence for other couplers;
magenta and yellow shifts are estimates. These are hypotheses, not measured
CineStill 800T layer-by-layer responses. A neutral characteristic curve alone does
not establish the accuracy of mixed, strongly unequal layer exposure near lights.

## Controlled diagnostic

Run `tools/eval/corridor_light_audit.py` with FILMLAB_DATA set to the restored data
root. It uses native half-size RAW, fixed prior geometry and frame timing,
next_sensor input, grain disabled, and no output colour conditioning. It does not
fit exposure, colour or curves. Comparisons are real scan, exp_plain, paper_plain,
and an optical-halo-off diagnostic with the same development and scanner.
The two output candidates use the same negative. The halo-off path is not a
proposed physical film. All spatial distances assume a 36 mm scan width.

Saved output includes the additional layer exposure, the resulting dye difference,
sensor saturation flags, and linear RGB change in 0–100, 100–200 and 200–300 µm
rings about manually chosen centres. These ring averages include the scene and
other light contributions; they are not isolated measurements of the named lamp.
The glass feature may be reflected radiance. No source spectrum is inferred from
its apparent colour, and no unknown lamp core is relabelled a measurement.

Positive dye response proves that the implementation retains some additional
exposure through development. It does not prove that the magnitude, colour or
spatial shape is physically correct. A final scan alone cannot uniquely separate
optical transport, development and scanner error.

No production defaults, current/next generations or physical constants changed.

## Observed result

Native crop inspection shows a compact orange/red transition at the near ceiling
light in the real scan. Both output candidates produce a broader, whiter surround.
The no-optical-halo diagnostic visibly reduces that surround. Thus the optical
halo is enabled and contributes to the output; removing the exponential shoulder
alone does not restore the real transition.

In the 100–200 µm ring, mean added red-layer exposure is 0.256 EV for the near
ceiling light and 0.159 EV for the middle light; corresponding mean red-sensitive
layer dye changes are +0.0562 and +0.0353 model dye units. The glass feature also
shows an added exposure and dye response. This excludes total loss of secondary
exposure in development, not incorrect development gain, interlayer inhibition,
source colour, optical footprint or dye spectra. No response is calibrated to these
numbers: they are differences between model interventions, not physical measurements.

The near/middle 0–100 µm regions contain sensor-saturated samples (any channel:
61% and 52% respectively); there is no accepted short-exposure measurement for
this scene. Core radiance/spectrum therefore remains a confounder. Coordinates
are fixed manual hints rather than fitted source centres.

Artifacts: `/workspace/filmlab-audit/corridor-lights/`; `results.json` contains all
ring statistics and the PNG files preserve native scan crop resolution. Script
syntax check and git diff whitespace checks passed. Rendering ran successfully.

## Development-only follow-up

`tools/eval/development_response_audit.py` holds layer exposure fixed and evaluates
15,625 combinations across the stored log10 H range [-3, 4]. It records the local
CMY-by-RGB-layer-exposure Jacobian and native neutral reconstruction. Native
neutral reconstruction differs from the source nominal curves by at most 0.00034
model dye units; this is construction consistency, not independent evidence.

The grid has 73 very small negative red self derivatives, minimum -0.000141 model
dye units/log10 H, near the upper plateau; green and blue have none below -1e-6.
These small nonmonotonicities are recorded, not silently smoothed or used to
explain the large observed discrepancy. No physical repair follows from this grid.

All sampled model pixels within 300 µm of the three manual centres remain within
the stored exposure domain. None has zero local red-to-cyan dye gain. Median gains
are 0.740, 0.765 and 0.758 model dye units/log10 H. Thus curve-domain clamping or
complete local red response saturation does not explain these regions. Local
Jacobians use homogeneous samples (no spatial inhibitor diffusion); the actual
image ablation retains the spatial baseline and fixed input.

Two diagnostic full-chain removals were rendered with unchanged layer exposure,
scanner conversion, prior timing, geometry and output conditioning:

- Remove layer inhibition (DIR), retaining the same source curves. This changes
  contrast and scene colours but does not restore the compact orange/red ring.
- Remove the cross-process dye spectral shift, retaining the baseline developed
  dye amounts. This changes colours but does not restore the ring geometry.

Neither intervention is a proposed physical candidate. The unchanged baseline
crop reproduces the earlier exp_plain render to <6e-13 absolute linear RGB error
for all three regions. No new exposure/colour/curve fit was performed. Final PNGs,
arrays, Jacobians and JSON are in `/workspace/filmlab-audit/development-response/`.
The shader-free plot is a diagnostic response diagram, not measured film data.

These checks exclude simple failure hypotheses; they do not validate the absolute
layer response, the inverse-DIR model for unequal layer exposures, or the C-41 dye
spectra. A neutral curve alone does not identify the full three-variable response.
The single-layer gamma ratios cited in chemistry.py constrain the chosen model
using other C-41 materials, not direct CineStill 800T measurements. Attempts to
retrieve patents US6210870B1 and US5183729A through patents.google.com were blocked
by the network proxy (403); no newly retrieved source verification is claimed.

No production code was changed. Syntax compilation and diff whitespace checks
passed. The next evidence needed is film-specific separate-layer/known-spectrum
exposure response and material transport data, rather than a fit to these crops.

Later validity restriction: `CORRIDOR_DYE_RESPONSE.md` documents a visible local
correspondence failure for the glass reflection. Its cached fixed-coordinate
numbers remain reproducible but cannot validate matched-light film/scanner
behaviour. The two ceiling-light controls remain separate; no re-registration
was fitted to improve colour or appearance.
