"""Fetch SHA256-pinned research cache assets from a permitted GitHub Release.

Uses the target machine's normal gh authentication. Does not upload data,
print credentials, assemble a second 10GB copy or extract archives.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def verify_parts(directory, manifest):
    directory = Path(directory)
    parts = manifest.get('parts', [])
    if not parts:
        raise ValueError('Empty parts manifest')
    seen, total, whole = set(), 0, hashlib.sha256()
    for part in parts:
        name = part['name']
        if not name or Path(name).name != name or name in ('.', '..') or name in seen:
            raise ValueError('Unsafe or duplicate part name')
        seen.add(name)
        p = directory / name
        if not p.is_file() or p.is_symlink() or p.stat().st_size != part['bytes']:
            raise ValueError('Missing, invalid or wrong-size part: ' + name)
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block);whole.update(block);total += len(block)
        if digest.hexdigest() != part['sha256']:
            raise ValueError('SHA256 mismatch: ' + name)
    if total != manifest['archive_bytes'] or whole.hexdigest() != manifest['archive_sha256']:
        raise ValueError('Concatenated archive size/SHA256 mismatch')
    return {'verified_parts': len(parts), 'archive_bytes': total, 'archive_sha256': whole.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', help='Private GitHub data repository owner/name; do not use the public source repo')
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('handoff_release_parts.json'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not args.verify_only:
        if not args.repo or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
            parser.error('--repo owner/name is required for download')
        if manifest.get('publication_status') != 'PUBLISHED':
            parser.error('Assets are not published yet; consult the handoff status instead of retrying unavailable downloads')
        if args.repo != manifest.get('repository'):
            parser.error('Repository differs from the published manifest')
        args.directory.mkdir(parents=True, exist_ok=True)
        for part in manifest['parts']:
            if Path(part['name']).name != part['name'] or part['name'] in ('.', '..'):
                parser.error('Unsafe part name')
            if not (args.directory / part['name']).exists():
                subprocess.run(['gh', 'release', 'download', manifest['tag'], '--repo', args.repo,
                                '--pattern', part['name'], '--dir', str(args.directory)], check=True)
    print(json.dumps(verify_parts(args.directory, manifest), indent=2))


if __name__ == '__main__':
    main()
