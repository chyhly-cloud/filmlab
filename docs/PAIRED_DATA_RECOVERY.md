# RAW + CineStill 800T paired-source recovery

## Result

Recovered the previously missing SillyStill cafe pair from the author's public
repository, pinned to commit `b1ebdb971207b403b7126b759c1d30d30954a722`.
One Sony ARW + one published CineStill 800T JPEG scan were downloaded and parsed.
This restores a historical FilmLab scene; it is **not a new independent holdout**.
URLs, SHA256, size, metadata and eligibility are in
`tools/eval/sillystill_sources.json`. The data stays outside the checkout under
`$FILMLAB_DATA/pairs/sillystill/`.

- Source: https://github.com/mikasenghaas/sillystill
- Original attachments: https://github.com/mikasenghaas/sillystill/tree/master/imgs
- Acquisition/preprocessing description:
  https://github.com/mikasenghaas/sillystill/blob/master/report/sections/dataset.tex

The report describes 41 original image pairs and 38 processed pairs, using a Sony
Alpha 7 and Nikon F3 on the same tripod setup. This does **not** establish 41
publicly downloadable sensor RAW files. The README's Zenodo/Hugging Face download
links are empty placeholders; the releases page has no release. The checked
`imgs` directory contains only one ARW, its cafe scan and an example image.
Other shown pairs in `report/figures` have JPEG/PNG digital inputs, not sensor RAW.
No full-dataset download was recovered.

## Important limits

The report's processed film images undergo Lab luminance histogram matching to
the digital inputs. They must not be used as untouched scan responses. The cafe
attachment has sRGB metadata and a Photoshop save record (`changed=/`), but no
scanner model, film exposure or development record; its exact edit history is
unknown. It supports a conditional appearance comparison, not isolated film or
scanner calibration. Sunlight/shadow shapes and viewpoint/parallax differ.
The report's generic table lists ISO 100; the actual ARW EXIF is ISO 125, f/8,
1/60 s, 40 mm, 2024-03-20 12:41:10. No film-frame timestamp was recovered.

## Rendering performed

The restored ARW was rendered through next_sensor with the same fixed +0.53 EV
metering used in the recent EPFL gallery, frame automatic timing, roll 1, fixed
frame seed, grain off, and output conditioning off. The historical cafe-specific
+1.5 EV was deliberately not used for this comparison. No scan-based colour,
brightness, physical-parameter or tone-curve fit was performed. exp_plain and
paper_only share the same negative and automatic timing; only the additional
exponential shoulder differs.

Source-only ORB alignment has 365 inliers. This establishes geometric scene
correspondence, not identical illumination or a physically exact pixel match.
The comparison visibly retains a cooler/darker simulation than the published
scan. That is an observed whole-chain difference, not a reason to tune one stage.

Artifacts: `/workspace/filmlab-audit/restored-cafe/comparison_full.png`,
`settings.json`, `aligned.npz` and the local rendering script. Both files were
SHA256 recorded and parsed; rendering and diff whitespace checks passed.

## Other sources checked

| Source | Finding | Eligibility |
|---|---|---|
| EPFL cs-413-cinestill/800t-emulation | Existing 36 film files; 8 outdoor night scenes already assessed; other frames are charts, lightboxes or pending bulb pairings | Existing data, not additional independent night scenes |
| ValRat/Cinestill800TModel | `data/raw` points to `C:/Programming/FilmScraper/Data/full` | No published paired RAW data found |
| ns144/3D-LUT | README explicitly describes using unpaired analog/digital film data | Reject as same-scene RAW + film pairs |
| ahad-ali-khan/Hal-Net | 100 film scans from Flickr/Lomography, without corresponding digital RAWs | Reject as paired validation |
| LivDevGit/GradeStudio | Training targets synthetically constructed from digital photos | Reject as actual film pairs |

GitHub repository search and source pages were accessible. General search sites,
Zenodo API and Hugging Face API returned proxy CONNECT 403 in this environment;
blocked access does not prove data does not exist. Source snapshots/search results
are under `/workspace/filmlab-audit/paired-source-search/`. No author was contacted.
No additional independent downloadable night RAW + 800T pair was verified in this
search. Current/next and all model constants remain unchanged.
