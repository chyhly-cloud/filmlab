import hashlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/eval'))
from github_handoff_cache import verify_parts


def fixture_parts(tmp_path):
    parts = []
    for name, data in [('cache.part000', b'abc'), ('cache.part001', b'def')]:
        (tmp_path / name).write_bytes(data)
        parts.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return {'parts': parts, 'archive_bytes': 6, 'archive_sha256': hashlib.sha256(b'abcdef').hexdigest()}


def test_valid_chain_and_same_size_corruption(tmp_path):
    m = fixture_parts(tmp_path)
    assert verify_parts(tmp_path, m)['verified_parts'] == 2
    (tmp_path / 'cache.part000').write_bytes(b'abd')
    with pytest.raises(ValueError, match='SHA256'):
        verify_parts(tmp_path, m)


def test_missing_reordered_and_empty_manifest_fail(tmp_path):
    m = fixture_parts(tmp_path);m['parts'].reverse()
    with pytest.raises(ValueError, match='Concatenated'):
        verify_parts(tmp_path, m)
    (tmp_path / 'cache.part000').unlink()
    with pytest.raises(ValueError, match='Missing'):
        verify_parts(tmp_path, m)
    with pytest.raises(ValueError, match='Empty'):
        verify_parts(tmp_path, {'parts': []})


def test_duplicate_and_traversal_names_fail(tmp_path):
    m = fixture_parts(tmp_path);m['parts'].append(m['parts'][0])
    with pytest.raises(ValueError, match='duplicate'):
        verify_parts(tmp_path, m)
    m['parts'][0] = {**m['parts'][0], 'name': '../outside'}
    with pytest.raises(ValueError, match='Unsafe'):
        verify_parts(tmp_path, m)
