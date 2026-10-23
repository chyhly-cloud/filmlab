"""Exposure records can be completed without inferring radiance or changing defaults."""
from fractions import Fraction
from types import SimpleNamespace

import exifread
import numpy as np
import pytest

from filmlab import rawio


class Tag:
    def __init__(self, value):
        self.values = [value]

    def __str__(self):
        return str(self.values[0])


def record(time='1/2', aperture='28/5', iso=None, model='DMC-G81'):
    tags = {'EXIF ExposureTime': Tag(time), 'EXIF FNumber': Tag(aperture),
            'Image Make': Tag('Panasonic'), 'Image Model': Tag(model)}
    if iso is not None:
        tags['EXIF ISOSpeedRatings'] = Tag(iso)
    return tags


def setup_records(monkeypatch, tmp_path, raw_tags, jpeg_tags, *, failure=False):
    path = tmp_path/'source.RW2';path.write_bytes(b'raw')
    calls = []

    def parse(stream, **kwargs):
        kind = stream.read();calls.append(kind)
        return raw_tags if kind == b'raw' else jpeg_tags

    class Raw:
        def __enter__(self):return self
        def __exit__(self, *_):pass
        def extract_thumb(self):
            if failure:raise RuntimeError('thumbnail unavailable')
            return SimpleNamespace(format=rawio.rawpy.ThumbFormat.JPEG, data=b'jpeg')

    monkeypatch.setattr(exifread, 'process_file', parse)
    monkeypatch.setattr(rawio.rawpy, 'imread', lambda _: Raw())
    return path, calls


def test_default_reading_keeps_legacy_early_return_and_one_argument_mock(monkeypatch, tmp_path):
    raw = record();path,calls = setup_records(monkeypatch,tmp_path,raw,record(iso=200))
    assert rawio.clip_luminance(path) is None
    assert calls == [b'raw']  # The explicit candidate does not alter default readers.
    monkeypatch.setattr(rawio,'_exif',lambda path: record(iso=200))
    assert rawio.exif_exposure(path) == (.5,5.6,200.)
    assert rawio.exif_iso(path) == 200
    assert rawio.clip_luminance(path) == pytest.approx(37.632)


def test_missing_iso_borrows_original_tag_only_after_exposure_and_identity_agree(monkeypatch,tmp_path):
    raw=record();jpeg=record(iso=200)
    path,_=setup_records(monkeypatch,tmp_path,raw,jpeg)
    trace={};merged=rawio._exif_complete(path,trace=trace)
    assert merged['EXIF ExposureTime'] is raw['EXIF ExposureTime']
    assert merged['EXIF ISOSpeedRatings'] is jpeg['EXIF ISOSpeedRatings']
    assert str(merged['EXIF FNumber'].values[0]) == '28/5'
    assert trace['field_sources']=={'time_s':'RAW EXIF','f_number':'RAW EXIF','iso':'embedded JPEG EXIF'}
    assert trace['recorded_values']['time_s']=='1/2'
    assert rawio.exif_exposure(path,complete_exif=True)==(.5,5.6,200.)
    assert rawio.exif_iso(path,complete_exif=True)==200.
    assert rawio.clip_luminance(path,complete_exif=True)==pytest.approx(37.632)
    assert 'EXIF ISOSpeedRatings' not in raw


@pytest.mark.parametrize('jpeg', [record(time='1/4',iso=200),record(aperture='4',iso=200),
                                 record(iso=200,model='different camera')])
def test_conflicting_embedded_record_is_not_used(monkeypatch,tmp_path,jpeg):
    raw=record();path,_=setup_records(monkeypatch,tmp_path,raw,jpeg)
    trace={};merged=rawio._exif_complete(path,trace=trace)
    assert merged is raw
    assert trace['conflicts']
    assert rawio.exif_exposure(path,complete_exif=True) is None
    assert rawio.clip_luminance(path,complete_exif=True) is None


def test_complete_raw_wins_without_reading_rounded_or_conflicting_thumbnail(monkeypatch,tmp_path):
    raw=record(time='1/30',iso=1600,model='ILCE-7C')
    path,calls=setup_records(monkeypatch,tmp_path,raw,record(time='33333/1000000',iso=200))
    trace={};merged=rawio._exif_complete(path,trace=trace)
    assert merged is raw and calls==[b'raw']
    assert trace['status']=='RAW complete'
    assert Fraction(trace['recorded_values']['time_s']) == Fraction(1,30)
    assert rawio.exif_iso(path,complete_exif=True)==1600


def test_thumbnail_failure_retains_all_available_raw_fields(monkeypatch,tmp_path):
    raw=record();path,_=setup_records(monkeypatch,tmp_path,raw,{},failure=True)
    trace={};merged=rawio._exif_complete(path,trace=trace)
    assert merged is raw
    assert trace['status']=='embedded exposure record unavailable'
    assert merged['EXIF FNumber'] is raw['EXIF FNumber']
    assert rawio.clip_luminance(path,complete_exif=True) is None


def test_valid_alternate_iso_does_not_use_invalid_primary_alias(monkeypatch,tmp_path):
    raw=record(iso=0);jpeg=record();jpeg['EXIF PhotographicSensitivity']=Tag(200)
    path,_=setup_records(monkeypatch,tmp_path,raw,jpeg)
    assert rawio.exif_iso(path,complete_exif=True)==200
    assert rawio.exif_exposure(path,complete_exif=True)==(.5,5.6,200.)
    assert rawio.clip_luminance(path,complete_exif=True)==pytest.approx(37.632)


def test_candidate_accepts_small_jpeg_rational_rounding_but_preserves_raw_fraction(monkeypatch,tmp_path):
    raw=record(time='1/30');path,_=setup_records(monkeypatch,tmp_path,raw,record(time='33333/1000000',iso=200))
    assert rawio.exif_exposure(path,complete_exif=True)[0]==1/30


def test_existing_iso_disagreement_rejects_borrowed_aperture(monkeypatch,tmp_path):
    raw=record(iso=200);del raw['EXIF FNumber']
    path,_=setup_records(monkeypatch,tmp_path,raw,record(iso=400))
    trace={};merged=rawio._exif_complete(path,trace=trace)
    assert merged is raw
    assert trace['conflicts']==['iso']
    assert rawio.exif_iso(path,complete_exif=True)==200.
    assert rawio.exif_exposure(path,complete_exif=True) is None
