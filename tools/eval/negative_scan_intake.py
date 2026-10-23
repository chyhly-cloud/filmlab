"""Read-only TIFF intake. Bit depth and slide mode do not establish linearity.

No ICC conversion, inversion, gamma correction, density calculation or fitting.
"""
import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import tifffile
from PIL import ImageCms


def inspect_icc(blob):
    """Inspect a declared profile; do not claim it describes actual scan capture."""
    result = {'description': None, 'declared_transfer_curves': {},
              'actual_scan_encoding_verified': False}
    try:
        result['description'] = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(blob))).strip()
        if len(blob) < 132:
            raise ValueError('Truncated ICC header')
        count = struct.unpack_from('>I', blob, 128)[0]
        if 132 + 12 * count > len(blob):
            raise ValueError('Truncated ICC tag table')
        for j in range(count):
            sig, offset, size = struct.unpack_from('>4sII', blob, 132 + j * 12)
            if sig not in (b'rTRC', b'gTRC', b'bTRC'):
                continue
            if offset + size > len(blob) or size < 12:
                raise ValueError('Truncated ICC transfer curve')
            record = {'type': blob[offset:offset+4].decode('ascii', errors='replace')}
            if record['type'] == 'curv':
                n = struct.unpack_from('>I', blob, offset + 8)[0]
                if 12 + 2 * n > size:
                    raise ValueError('Truncated ICC curve entries')
                record['entries'] = n
                if n > 1:
                    curve = np.frombuffer(blob, dtype='>u2', count=n, offset=offset + 12) / 65535
                    record['value_at_encoded_half'] = float(np.interp(.5, np.linspace(0, 1, n), curve))
                elif n == 1:
                    record['gamma'] = struct.unpack_from('>H', blob, offset + 12)[0] / 256
                else:
                    record['identity_curve_declared'] = True
            result['declared_transfer_curves'][sig.decode()] = record
    except (ValueError, OSError, struct.error) as e:
        result['inspection_error'] = str(e)
    return result


def inspect_tiff(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    report = {'file_name': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest(),
              'linear_encoding_verified': False, 'negative_polarity_verified': False,
              'density_analysis_ready': False,
              'limits': ['TIFF metadata cannot verify scanner linearity, disabled corrections, dark offset or film identity.',
                         'Counts at code endpoints are observations, not proof of physical scanner/sensor saturation.',
                         'DPI does not establish optical resolution or measured negative-plane spatial calibration.']}
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        report.update(page_count=len(tf.pages), first_page_shape=list(p.shape), dtype=str(p.dtype))
        metadata = {}
        for name in ['BitsPerSample', 'SampleFormat', 'SamplesPerPixel', 'PhotometricInterpretation',
                     'PlanarConfiguration', 'Compression', 'Make', 'Model', 'Software',
                     'XResolution', 'YResolution', 'ResolutionUnit', 'Orientation', 'DateTime',
                     'WhitePoint', 'BlackLevel', 'WhiteLevel']:
            tag = p.tags.get(name)
            if tag is not None:
                value = tag.value
                metadata[name] = value if isinstance(value, (str, int, float, tuple, list)) else str(value)
        for name in ['InterColorProfile', 'TransferFunction']:
            tag = p.tags.get(name)
            if tag is not None:
                value = tag.value
                blob = value if isinstance(value, bytes) else np.asarray(value).tobytes()
                metadata[name] = {'present': True, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
                if name == 'InterColorProfile':
                    metadata[name]['declared_profile'] = inspect_icc(blob)
        report['metadata'] = metadata
        layout_ok = len(p.shape) == 3 and p.shape[-1] == 3 and int(p.photometric) == 2
        report['single_page_RGB_uint16_container'] = bool(len(tf.pages) == 1 and layout_ok and p.dtype == np.dtype('uint16'))
        if not layout_ok:
            report['statistics_skipped'] = 'Only contiguous three-channel RGB layout supported; no implicit channel reinterpretation.'
        elif np.prod(p.shape) * p.dtype.itemsize > 512 * 1024 * 1024:
            report['statistics_skipped'] = 'Decoded page exceeds 512 MiB intake budget.'
        else:
            a = p.asarray()
            report['raw_code_statistics'] = []
            for j, channel in enumerate('RGB'):
                v = a[..., j]
                r = {'channel': channel, 'min': float(v.min()), 'max': float(v.max()),
                     'percentile_0_1_50_99_100': np.percentile(v, [0, 1, 50, 99, 100]).tolist()}
                if np.issubdtype(v.dtype, np.integer):
                    info = np.iinfo(v.dtype)
                    r.update(code_min_fraction=float((v == info.min).mean()),
                             code_max_fraction=float((v == info.max).mean()),
                             distinct_codes=int(np.unique(v).size))
                report['raw_code_statistics'].append(r)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tiff', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.write_text(json.dumps(inspect_tiff(args.tiff), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
