from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ScaleReport:
    classification: str
    confidence: str
    score: int
    sampled_cells: int
    sampled_values: int
    nonzero_integer_fraction: float
    q50_nonzero: float
    q95_nonzero: float
    q99_nonzero: float
    max_value: float
    median_cell_sum: float
    fraction_below_one_nonzero: float
    zero_fraction: float
    negative_fraction: float
    nonfinite_fraction: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _dense_rows(matrix, rows: np.ndarray) -> np.ndarray:
    part = matrix[rows]
    if sparse.issparse(part):
        part = part.toarray()
    return np.asarray(part, dtype=np.float64)


def analyze_matrix(matrix, max_cells: int = 2048, seed: int = 0) -> ScaleReport:
    if len(matrix.shape) != 2:
        raise ValueError("expression matrix must be two-dimensional")
    n_cells = int(matrix.shape[0])
    if n_cells == 0:
        raise ValueError("matrix has zero cells")
    if max_cells <= 0:
        raise ValueError("max_cells must be positive")

    rng = np.random.default_rng(seed)
    rows = (
        np.arange(n_cells)
        if n_cells <= max_cells
        else np.sort(rng.choice(n_cells, max_cells, replace=False))
    )
    array = _dense_rows(matrix, rows)
    finite = np.isfinite(array)
    nonfinite_fraction = float(1.0 - finite.mean()) if array.size else 0.0
    safe = np.where(finite, array, 0.0)
    negative_fraction = float(np.mean(safe < 0)) if safe.size else 0.0
    zero_fraction = float(np.mean(safe == 0)) if safe.size else 1.0
    nonzero = safe[safe > 0]

    if nonzero.size:
        integer_fraction = float(
            np.mean(np.abs(nonzero - np.rint(nonzero)) <= 1e-6)
        )
        q50, q95, q99 = [
            float(value)
            for value in np.quantile(nonzero, [0.50, 0.95, 0.99])
        ]
        max_value = float(nonzero.max())
        below_one = float(np.mean(nonzero < 1.0))
    else:
        integer_fraction = q50 = q95 = q99 = max_value = below_one = 0.0

    cell_sums = safe.sum(axis=1)
    median_cell_sum = float(np.median(cell_sums))

    score = 0
    reasons: list[str] = []
    if integer_fraction >= 0.98:
        score += 4
        reasons.append("almost all nonzero values are integer-valued")
    elif integer_fraction >= 0.90:
        score += 2
        reasons.append("most nonzero values are integer-valued")
    elif integer_fraction >= 0.75:
        score += 1
        reasons.append("many nonzero values are integer-valued")

    if q99 >= 25:
        score += 2
        reasons.append(
            "99th percentile is large for a typical log-expression matrix"
        )
    elif q99 >= 12:
        score += 1
        reasons.append("upper expression tail is relatively large")

    if max_value >= 100:
        score += 2
        reasons.append("maximum expression is count-like")
    elif max_value >= 40:
        score += 1
        reasons.append("maximum expression is unusually large")

    if median_cell_sum >= 5000:
        score += 2
        reasons.append("median per-cell sum is strongly count-like")
    elif median_cell_sum >= 1500:
        score += 1
        reasons.append("median per-cell sum is high")

    if integer_fraction < 0.5 and q99 < 15 and max_value < 30:
        score -= 2
        reasons.append(
            "continuous low-range values argue against raw integer counts"
        )

    if score >= 4:
        classification, confidence = "LIKELY_RAW_COUNTS", "HIGH"
    elif score >= 3:
        classification, confidence = "POSSIBLY_RAW_COUNTS", "MEDIUM"
    else:
        classification, confidence = "NOT_OBVIOUSLY_RAW", "LOW"

    if nonfinite_fraction > 0 or negative_fraction > 0:
        reasons.append(
            "matrix also violates VEC finiteness/non-negativity requirements"
        )

    return ScaleReport(
        classification=classification,
        confidence=confidence,
        score=score,
        sampled_cells=len(rows),
        sampled_values=int(array.size),
        nonzero_integer_fraction=integer_fraction,
        q50_nonzero=q50,
        q95_nonzero=q95,
        q99_nonzero=q99,
        max_value=max_value,
        median_cell_sum=median_cell_sum,
        fraction_below_one_nonzero=below_one,
        zero_fraction=zero_fraction,
        negative_fraction=negative_fraction,
        nonfinite_fraction=nonfinite_fraction,
        reasons=reasons,
    )
