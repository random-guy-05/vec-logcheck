# VEC LogCheck

A small pre-submission guardrail for a gap the official structural validator cannot close: **raw counts are finite and non-negative, so they can pass format validation and then be scored as if they were log-normalized**.

`vec-logcheck` inspects the value distribution in `.X` and reports whether a file looks like raw integer counts, a suspiciously count-like matrix, or a matrix that is not obviously raw. It is deliberately a **heuristic**, not a proof of normalization.

## Quick start

```bash
pip install -e .
vec-logcheck prediction.h5ad
vec-logcheck prediction.h5ad --json report.json
```

Example output:

```text
LIKELY_RAW_COUNTS  confidence=HIGH
integer-valued nonzero fraction : 0.993
q99 nonzero expression           : 87.0
max expression                   : 241.0
median per-cell sum              : 14526.0

Why this matters: VEC expects already log-normalized non-negative expression.
```

## What it checks

The heuristic uses a deterministic cell sample and measures:

- fraction of nonzero values that are almost exact integers;
- upper-tail expression values and maximum;
- median per-cell expression sum;
- fraction of values below 1;
- zeros, non-finite values, and negatives.

A high integer fraction plus large counts is strong evidence for raw counts. A log-normalized matrix can still contain many integer-looking zeros/ones, so LogCheck never labels a matrix "definitely normalized".

## Why it is distinct

The existing VEC validators correctly check shape, genes, finiteness, non-negativity, coordinates, and cell bounds. They cannot infer the semantic scale of `.X`. LogCheck targets that exact documented blind spot.

## Exit codes

- `0`: not obviously raw
- `3`: suspicious / possibly raw
- `4`: likely raw counts
- `2`: invalid input / read failure

See `docs/SOURCES.md` for the official statement that raw counts can pass validation and still score incorrectly.
