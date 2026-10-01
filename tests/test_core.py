import numpy as np
import pytest

from vec_logcheck.core import analyze_matrix


def test_raw_poisson_is_flagged():
    matrix = np.random.default_rng(1).poisson(5, size=(300, 100)).astype(float)
    report = analyze_matrix(matrix)
    assert report.classification == "LIKELY_RAW_COUNTS"
    assert report.nonzero_integer_fraction == 1.0


def test_continuous_log_range_is_not_high_confidence_raw():
    rng = np.random.default_rng(2)
    matrix = rng.uniform(0, 6, size=(300, 100))
    matrix[rng.random(matrix.shape) < 0.35] = 0
    report = analyze_matrix(matrix)
    assert report.classification == "NOT_OBVIOUSLY_RAW"


def test_negatives_and_nonfinite_values_reported():
    matrix = np.array([[0.0, -1.0, 0.5], [np.nan, 1.2, 2.3]])
    report = analyze_matrix(matrix)
    assert report.negative_fraction > 0
    assert report.nonfinite_fraction > 0


def test_empty_rejected():
    with pytest.raises(ValueError):
        analyze_matrix(np.empty((0, 10)))


def test_bad_max_cells_rejected():
    with pytest.raises(ValueError):
        analyze_matrix(np.ones((10, 3)), max_cells=0)
