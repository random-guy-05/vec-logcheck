from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import analyze_matrix


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Heuristic VEC raw-count/log-scale checker.')
    p.add_argument('file', type=Path)
    p.add_argument('--max-cells', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', type=Path, dest='json_path')
    args = p.parse_args(argv)
    try:
        import anndata as ad
        a = ad.read_h5ad(args.file, backed='r')
        report = analyze_matrix(a.X, args.max_cells, args.seed)
        try:
            a.file.close()
        except Exception:
            pass
    except Exception as exc:
        print(f'ERROR: {exc}')
        return 2
    d = report.to_dict()
    print(f"{report.classification}  confidence={report.confidence}")
    for key in ['nonzero_integer_fraction', 'q99_nonzero', 'max_value', 'median_cell_sum', 'zero_fraction']:
        print(f'{key:34s}: {d[key]:.6g}')
    print('\nThis is a heuristic. VEC requires already log-normalized non-negative expression.')
    for reason in report.reasons:
        print(f'- {reason}')
    if args.json_path:
        args.json_path.write_text(json.dumps(d, indent=2) + '\n')
    return 4 if report.classification == 'LIKELY_RAW_COUNTS' else 3 if report.classification == 'POSSIBLY_RAW_COUNTS' else 0
