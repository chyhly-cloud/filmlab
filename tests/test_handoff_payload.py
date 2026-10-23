import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/eval'))
from verify_handoff_payload import verify


def test_missing_and_same_size_corruption_are_rejected(tmp_path):
    p = tmp_path / 'data.bin'
    row = {'path': p.name, 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    m = {'included': [row]}
    assert verify(tmp_path, m)['errors']
    p.write_bytes(b'abc')
    assert verify(tmp_path, m) == {'expected': 1, 'verified': 1, 'errors': []}
    p.write_bytes(b'abd')
    assert 'SHA256 mismatch' in verify(tmp_path, m)['errors'][0]


def test_empty_and_duplicate_manifests_are_rejected(tmp_path):
    assert verify(tmp_path, {})['errors']
    p = tmp_path / 'a';p.write_bytes(b'x')
    row = {'path': 'a', 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
    assert verify(tmp_path, {'included': [row, row]})['errors']


def test_paths_and_symlinks_cannot_escape_import_root(tmp_path):
    root = tmp_path / 'import';root.mkdir()
    outside = tmp_path / 'outside';outside.write_bytes(b'x')
    (root / 'link').symlink_to(outside)
    for name in ['../outside', str(outside), 'link']:
        row = {'path': name, 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
        assert verify(root, {'included': [row]})['errors']
