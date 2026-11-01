# T14: Point-light halation (Claude; diagnostic, no production change)

Background: the user's CineStill 800T reference photos (gas-station canopy, highway, tunnel, Tokyo street) show the same thing.
Every bright point light (street lamps, canopy lights, tail lights, traffic signals) has a clear, saturated red/orange disc around it, several times the lamp's size.
The lamp core is white/yellow. The user's verdict: "the model has no halation at all."

## 1. Synthetic point lights: is the physics right? (`tools/eval/t14_point_source.py`, `docs/figures/t14/point_sources.jpg`)

- Setup: dark background with lamps 4–16 stops above mid-grey, rendered through `next_lab_halo`.
- Scale: true frame proportions, 36 mm = 1400 px (25.7 µm/px).

| Lamp (stops above mid-grey) | Small lamp (radius 40 µm) | Larger lamp (radius 130 µm) |
|---|---|---|
| +4 / +7 | no visible halation | no halation / faint dark-red ring |
| +10 | faint red ring at 200–300 µm, output R ≈ 0.04 | red ring out to ~500 µm |
| +13 | saturated red disc out to ~500 µm (R share 0.9+) | yellow → orange → red out to ~800 µm |
| +16 | white core → yellow → orange → red out to ~800 µm | red disc out to ~1.2 mm |

- Colour order (white/yellow core → orange → red outer edge) and the disc size of several times the lamp match the references.
- Halation size grows with lamp brightness, as on real film.
- **The halation physics (base reflection, layer depth, oblique attenuation) needs no change.**
- The question is how bright the lamp is in the input.

## 2. How bright lamps are after RAW reconstruction (`tools/eval/t14_lamp_stops.py`)

Run on 8 real night RAWs, `next_lab_halo`, following the `examples/night.py` workflow. Digital clip ≈ mid-grey +4.5 stops.

| Case | Prior used | Lamp peak (above mid-grey) |
|---|---|---|
| With EXIF (n29240, n40825, n40481, n25707, n30464, n51748) | luminance 2e3–3e4 cd/m² ÷ L_clip, almost always capped at MAX_STOPS = 8 | median +9.6 to +12.2, max +12.5 |
| Without EXIF (n27416, n20595, all EPFL frames) | clip +2 to +5 stops | median +6 to +7, max +9.5 |

By the table above, a lamp without EXIF lands at +6 to +9.5, which is **the no-halation range**.
Everything shown to the user before this came from the EPFL frames (no EXIF, dusk, mostly windows and façades), which is why "no halation".

## 3. Real night photos (`tools/eval/t14_render_night.py`)

- `docs/figures/t14/station.jpg` (n29240, has EXIF):
  - `next` shows white/orange haze;
  - `next_lab_halo` already gives every lamp a red disc, close to the references;
  - raising the cap 8 → 11 stops (diagnostic) makes the discs larger and redder, closer still to the references.
- `docs/figures/t14/street.jpg` (n27416, no EXIF):
  - default: almost no halation;
  - lamps +4 stops (`source_stops=4`, typical night lamp brightness): red halation appears.
  - Out-of-focus lamps spread their light over a larger image area, so their halation is weaker than for in-focus lamps; this is physically correct.

## 4. Why I did not change the default

- How far a lamp exceeds the clip is not in a single RAW (see T13).
- Existing evidence on the EXIF-path prior:
  - `docs/EXIF_PRIOR_HOLDOUT.md`: on two measured light apertures, EXIF + 2e3–3e4 cd/m² **overestimated by 2.8 / 4.0 EV**;
  - EPFL frames (no EXIF) have their best offsets around 0 to +3 under the current prior.
- Real street lamps at night (10^4–10^5 cd/m², with night mid-grey around 1 cd/m²) sit about +13 to +17 stops above mid-grey.
  Lit windows and light apertures are much dimmer, and a RAW cannot tell the two apart.
- Raising it uniformly would make windows and façades too red; keeping it uniform loses halation on night photos without EXIF.
  This is the missing input already identified in T13, not a halation-physics problem.

## Options (for the user to choose)

1. Make "night point-light brightness" the default for night scenes: a typical luminance of 10^4–10^5 cd/m², with the cap raised to match. This repeats an overestimate that existing holdout data has already shown (windows, apertures).
2. Keep the default and expose `source_stops` (lamp brightness) as a user control. Per-frame adjustment like this is effectively the same as how much a photographer over/underexposes.
3. Tell lamps from windows per clipped region (shape, colour, glow profile in the digital lens) and use different luminances. This needs new evidence; T13's flare index for this was uncorrelated (r = 0.13).
