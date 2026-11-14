"""Stage-specific selection and subset worker; comparison mechanics stay shared."""
from __future__ import annotations

import argparse
import json
import sys

from .config import SCRIPTS
from .paths import rel, resolve_video_source

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import enhancement_video as backend

METHODS = backend.METHODS


def output_for(source, values, profile='high'):
    return rel(backend.output_for(resolve_video_source(source), values, profile)) if source else ''


def validate_variants(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(METHODS):
        raise ValueError('Choose 1–6 enhancement methods.')
    methods = []
    for row in rows:
        if not isinstance(row, dict) or row.get('method') not in METHODS or row.get('preset') != 'settings':
            raise ValueError('Choose supported methods with current enhancement settings.')
        methods.append(row['method'])
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate enhancement methods.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--profile', default='high')
    args = parser.parse_args()
    rows, values = json.loads(args.variants), json.loads(args.settings)
    validate_variants(rows)
    if not isinstance(values, dict):
        raise ValueError('Expected enhancement settings.')
    for row in rows:
        backend.options({**values, 'method': row['method']})
    for index, row in enumerate(rows, 1):
        print(f'Enhancement batch {index}/{len(rows)}: {row["method"]}', flush=True)
        backend.run(resolve_video_source(args.source), {**values, 'method': row['method']}, args.profile)
        print(f'Enhancement batch complete {index}/{len(rows)}: {row["method"]}', flush=True)


if __name__ == '__main__':
    main()
