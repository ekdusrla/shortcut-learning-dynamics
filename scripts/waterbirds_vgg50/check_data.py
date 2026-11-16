"""Read-only Waterbirds preflight. No extraction, dataset download, or marker writes."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

CATEGORIES = ('b/bamboo_forest', 'f/forest/broadleaf', 'o/ocean', 'l/lake/natural')


def inspect_root(root):
    root = Path(root)
    result = dict(root=str(root.resolve()), ready=False, errors=[], categories={})
    cub = root / 'CUB_200_2011'
    rows = []
    for name in ('images.txt', 'train_test_split.txt'):
        try:
            lines = (cub / name).read_text().splitlines()
            pairs = [line.split() for line in lines if line.strip()]
            if not pairs or any(len(pair) != 2 for pair in pairs):
                raise ValueError('Expected two columns')
            if name == 'images.txt':
                rows = pairs
            else:
                splits = dict(pairs)
                if len(splits) != len(pairs) or any(v not in ('0', '1') for v in splits.values()):
                    raise ValueError('Duplicate IDs or nonbinary split flags')
        except (OSError, ValueError) as error:
            result['errors'].append(f'{name}: {error}')
    if rows:
        if len(rows) != 11788:
            result['errors'].append(f'Expected 11788 birds, got {len(rows)}')
        if len({r[0] for r in rows}) != len(rows) or len({r[1] for r in rows}) != len(rows):
            result['errors'].append('Duplicate bird IDs or filenames')
        if 'splits' in locals() and set(splits) != {r[0] for r in rows}:
            result['errors'].append('Image IDs and split IDs differ')
        missing = 0
        for _, rel in rows:
            path = Path(rel)
            if path.is_absolute() or '..' in path.parts:
                result['errors'].append('Unsafe image relative path')
                continue
            missing += int(not (cub / 'images' / path).is_file())
            missing += int(not (cub / 'segmentations' / path.with_suffix('.png')).is_file())
        result['missing_image_or_mask_files'] = missing
        if missing:
            result['errors'].append(f'{missing} missing RGB/mask files')
    for category in CATEGORIES:
        folder = root / 'places365/data_256' / category
        try:
            count = sum(p.is_file() for p in folder.glob('*.jpg'))
            marker = (folder / '.complete').exists()
            result['categories'][category] = dict(jpeg_count=count, marker_present=marker)
            # Original fetcher only checks marker existence; check actual expected pool here.
            if count != 5000:
                result['errors'].append(f'{category}: expected 5000 JPEGs, got {count}')
        except OSError as error:
            result['errors'].append(f'{category}: {error}')
    result['ready'] = not result['errors']
    result['scope'] = 'Structure/counts only; run protocol.prepare for decode/shape/hash checks'
    return result


def probe_range(url, opener=None):
    """Optional 512-byte network probe. Reject ignored Range before reading body."""
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-511', 'Accept-Encoding': 'identity'})
    with opener(req, timeout=20) as response:
        if response.status != 206:
            raise ValueError(f'Range unsupported: status {response.status}; body not read')
        header = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes 0-511/(\d+)', header)
        if not match or int(match[1]) < 512:
            raise ValueError(f'Invalid Content-Range: {header}')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Encoded response cannot be used for tar byte offsets')
        if response.headers.get('Content-Length') not in (None, '512'):
            raise ValueError('Unexpected Content-Length')
        body = response.read(513)
        if len(body) != 512:
            raise ValueError('Truncated or oversized range body')
        return dict(range_supported=True, total_bytes=int(match[1]), probe_bytes=len(body),
                    scope='First range only; does not validate every fetch request or tar ordering')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--probe-url', help='Explicit optional 512-byte HTTP request; omitted means no network')
    args = parser.parse_args()
    result = inspect_root(args.root)
    if args.probe_url:
        result['range_probe'] = probe_range(args.probe_url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['ready'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
