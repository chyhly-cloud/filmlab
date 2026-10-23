"""Verify a privately imported cache tree against the transfer manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, manifest):
    root = Path(root).resolve()
    rows = manifest.get('included', [])
    errors, seen, checked = [], set(), 0
    if not rows:
        errors.append('Empty payload manifest')
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            errors.append('Unsafe or duplicate path: ' + str(relative))
            continue
        seen.add(str(relative))
        p = root / relative
        try:
            p.resolve().relative_to(root)
        except ValueError:
            errors.append('Path escapes import root: ' + str(relative))
            continue
        if not p.is_file():
            errors.append('Missing file: ' + str(relative))
            continue
        if p.stat().st_size != row['bytes']:
            errors.append('Size mismatch: ' + str(relative))
            continue
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            errors.append('SHA256 mismatch: ' + str(relative))
            continue
        checked += 1
    return {'expected': len(rows), 'verified': checked, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, json.loads(args.manifest.read_text()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['errors'] else 0)


if __name__ == '__main__':
    main()
