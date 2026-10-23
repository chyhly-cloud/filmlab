"""Read-only audit of RAW/embedded-JPEG exposure metadata completeness.

This is an independent candidate merge, not a change to rawio._exif and not a
rerender. Nominal clipping luminance is an ISO/lens proxy, never source radiance.
Only exposure and camera identifiers are retained; GPS/preview pixels are not.
"""
import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys

import exifread
import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filmlab import rawio


FIELDS = {"time_s": ("EXIF ExposureTime",), "f_number": ("EXIF FNumber",),
          "iso": ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")}


def selected(tags):
    numeric, names, recorded = {}, {}, {}
    for field, keys in FIELDS.items():
        numeric[field] = None
        for key in keys:
            if key not in tags:
                continue
            try:
                value = float(Fraction(str(tags[key].values[0])))
                if not np.isfinite(value) or value <= 0:
                    continue
            except (ValueError, TypeError, IndexError, ZeroDivisionError):
                continue
            numeric[field], names[field], recorded[field] = value, key, str(tags[key].values[0])
            break
    text = {key: str(tags[key]) for key in ("Image Make", "Image Model", "EXIF DateTimeOriginal") if key in tags}
    return {"numeric": numeric, "tags_used": names, "recorded_rationals": recorded, "identity": text}


def inspect(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw_tags = exifread.process_file(stream, details=False)
    thumb_tags, error, fmt = {}, None, None
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        fmt = str(thumb.format)
        if thumb.format == rawpy.ThumbFormat.JPEG:
            thumb_tags = exifread.process_file(io.BytesIO(thumb.data), details=False)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    a, b = selected(raw_tags), selected(thumb_tags)
    common = {k: bool(np.isclose(a["numeric"][k], b["numeric"][k], rtol=1e-12, atol=0))
              for k in FIELDS if a["numeric"][k] is not None and b["numeric"][k] is not None}
    conflicts = [k for k, equal in common.items() if not equal]
    raw_complete = all(v is not None for v in a["numeric"].values())
    # A complete RAW record needs no merge. Otherwise differing common numeric
    # tags prevent copying missing metadata from another embedded record.
    # Exact common-tag differences are reported even when JPEG rational rounding
    # is the apparent explanation (Sony 1/30 versus33333/1000000, for example).
    merged, provenance = {}, {}
    if raw_complete or not conflicts:
        for field in FIELDS:
            source = "RAW EXIF" if a["numeric"][field] is not None else "embedded JPEG EXIF"
            value = a["numeric"][field] if source == "RAW EXIF" else b["numeric"][field]
            merged[field], provenance[field] = value, source if value is not None else "missing"
    complete = bool(merged and all(v is not None for v in merged.values()))
    nominal = 120*merged["f_number"]**2/(merged["time_s"]*merged["iso"]) if complete else None
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "raw": a, "embedded_jpeg": b,
            "thumbnail_format": fmt, "thumbnail_error": error,
            "common_numeric_agreement": common, "conflicting_fields": conflicts,
            "raw_metadata_complete": raw_complete,
            "common_numeric_relative_difference_JPEG_minus_RAW": {
                k: b["numeric"][k]/a["numeric"][k]-1 for k in common},
            "raw_has_exposure_time_triggers_early_return": "EXIF ExposureTime" in raw_tags,
            "current_exif_exposure": rawio.exif_exposure(path),
            "current_clip_luminance_cd_m2": rawio.clip_luminance(path),
            "candidate_merged_numeric": merged, "candidate_field_provenance": provenance,
            "candidate_complete": complete, "candidate_nominal_clip_luminance_cd_m2": nominal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"scope": __doc__, "files": [inspect(p) for p in args.paths],
              "limits": [
                  "120*N^2/(t*ISO) uses nominal saturation sensitivity and q about0.65, not a measured lens/camera radiometric calibration.",
                  "Reported ISO can be REI/SOS rather than independently measured saturation ISO; channel clipping and actual white levels also differ.",
                  "Nominal clipping luminance is not the unknown lamp luminance, SPD, unexposed source profile or an upper bound above clipping.",
                  "Embedded JPEG tags can fill missing recorded metadata when common exposure values agree; they are not new independent radiance measurements.",
                  "No production EXIF reading, prior, source fit baseline, current/next or rendered output changed.",
              ]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: f[k] for k in ("path", "current_clip_luminance_cd_m2", "candidate_nominal_clip_luminance_cd_m2", "common_numeric_agreement")} for f in report["files"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
