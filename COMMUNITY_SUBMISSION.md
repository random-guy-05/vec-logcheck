# Community Contribution submission text

## Title
VEC LogCheck — catch likely raw-count submissions before scoring

## Description
VEC LogCheck is a focused pre-submission heuristic for a failure mode the official structural validator explicitly cannot detect: a raw-count matrix is finite and non-negative, so it can pass format checks and then be scored as though it were log-normalized. The CLI samples `.X`, measures integer-valued nonzero fraction, upper-tail expression, maximum expression, median per-cell sum and related diagnostics, and returns a conservative classification (`LIKELY_RAW_COUNTS`, `POSSIBLY_RAW_COUNTS`, or `NOT_OBVIOUSLY_RAW`) with JSON output and CI-friendly exit codes. It clearly states that normalization cannot be proven from values alone. This saves entrants from spending a scored attempt on a file whose scale is wrong even though its structure is valid.
