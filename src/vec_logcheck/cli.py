from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from .core import analyze_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Heuristic VEC raw-count/log-scale checker."
    )
    parser.add_argument("file", type=Path)
    parser.add_argument("--max-cells", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    try:
        import anndata as ad

        data = ad.read_h5ad(args.file, backed="r")
        try:
            report = analyze_matrix(data.X, args.max_cells, args.seed)
        finally:
            with suppress(AttributeError, OSError, ValueError):
                data.file.close()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    payload = report.to_dict()
    print(f"{report.classification}  confidence={report.confidence}")
    for key in [
        "nonzero_integer_fraction",
        "q99_nonzero",
        "max_value",
        "median_cell_sum",
        "zero_fraction",
    ]:
        print(f"{key:34s}: {payload[key]:.6g}")

    print(
        "\nThis is a heuristic. VEC requires already log-normalized "
        "non-negative expression."
    )
    for reason in report.reasons:
        print(f"- {reason}")

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if report.classification == "LIKELY_RAW_COUNTS":
        return 4
    if report.classification == "POSSIBLY_RAW_COUNTS":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
