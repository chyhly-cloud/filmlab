"""Read-only exposure/saturation triage of recovered EPFL alternate RAWs.

Counts are whole-frame Bayer-cell flags, not registered lamp measurements.
Do not merge exposures until geometry, movement and sensor linearity are checked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))


def inspect(path):
    import exifread,rawpy,io
    from filmlab.rawio import sensor_saturation
    with path.open('rb') as f:tags=exifread.process_file(f,details=False)
    def num(key):
        tag=tags.get('EXIF '+key)
        return float(tag.values[0]) if tag is not None else None
    with rawpy.imread(str(path)) as raw:
        sat=sensor_saturation(raw,per_channel=True)
        iso=num('ISOSpeedRatings') or num('PhotographicSensitivity')
        iso_source='RAW EXIF'
        if iso is None:
            thumb=raw.extract_thumb()
            if thumb.format==rawpy.ThumbFormat.JPEG:
                thumb_tags=exifread.process_file(io.BytesIO(thumb.data),details=False)
                tag=thumb_tags.get('EXIF ISOSpeedRatings') or thumb_tags.get('EXIF PhotographicSensitivity')
                iso=float(tag.values[0]) if tag is not None else None
                iso_source='embedded JPEG EXIF' if iso is not None else 'missing'
        info={'sensor_white_level':raw.white_level,'sensor_black_level':raw.black_level_per_channel,
              'raw_shape':list(raw.raw_image_visible.shape),'flip':raw.sizes.flip,
              'flagged_any':int(sat.any(-1).sum()),'flagged_all':int(sat.all(-1).sum()),
              'flagged_channels':sat.sum((0,1)).astype(int).tolist()}
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'file':path.name,'sha256':sha,'time_s':num('ExposureTime'),'f_number':num('FNumber'),
            'iso':iso,'iso_source':iso_source,'focal_mm':num('FocalLength'),
            'timestamp':str(tags.get('EXIF DateTimeOriginal','unknown')),**info}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--restore',action='store_true',help='Download the six recorded alternate files and verify acquisition digests')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((REPO/'tools/eval/epfl_night_brackets.json').read_text())
    acquired={r['file']:r for r in manifest}
    if args.restore:
        from tools.eval.download_epfl import download
        for r in manifest:
            if not re.fullmatch(r'P[0-9]+\.RW2',r['file']):raise ValueError('Invalid filename')
            if not r['url'].startswith('https://drive.usercontent.google.com/download?'):raise ValueError('Invalid download origin')
            print(download((r['url'],args.data/'epfl/data/digital'/r['file'],r['sha256'])),flush=True)
    rows=list(csv.DictReader((REPO/'tools/eval/epfl_800t_manifest.csv').open()))
    result=[]
    for row in rows:
        alt=[name for name in row['raw_alt_brackets'].split(',') if name and (args.data/'epfl/data/digital'/name).exists()]
        if not alt:continue
        primary=inspect(args.data/'epfl/data'/row['raw_file'])
        if primary['sha256']!=row['raw_sha256']:raise ValueError('Primary checksum mismatch')
        others=[inspect(args.data/'epfl/data/digital'/name) for name in alt]
        for r in others:
            if r['file'] in acquired and r['sha256']!=acquired[r['file']]['sha256']:raise ValueError('Acquisition checksum mismatch')
            r['optical_exposure_difference_stops']=float(np.log2((r['time_s']/r['f_number']**2)/(primary['time_s']/primary['f_number']**2)))
            r['same_iso']=(r['iso']==primary['iso']) if r['iso'] is not None and primary['iso'] is not None else None
        result.append({'scene':row['scene_id'],'primary':primary,'alternates':others})
    (args.out/'bracket-audit.json').write_text(json.dumps({'scope':__doc__,'scenes':result},indent=2,allow_nan=False))
    for r in result:
        print(r['scene'])
        for v in [r['primary']]+r['alternates']:
            print(v['file'],v['time_s'],v['f_number'],v['iso'],v.get('optical_exposure_difference_stops',0),
                  v['flagged_any'],v['flagged_all'],v['timestamp'])


if __name__=='__main__':main()
