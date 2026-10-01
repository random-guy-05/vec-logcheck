import numpy as np
from vec_logcheck.core import analyze_matrix


def test_raw_poisson_is_flagged():
    x = np.random.default_rng(1).poisson(5, size=(300, 100)).astype(float)
    r = analyze_matrix(x)
    assert r.classification == 'LIKELY_RAW_COUNTS'
    assert r.nonzero_integer_fraction == 1.0


def test_log1p_counts_are_not_high_confidence_raw():
    x = np.log1p(np.random.default_rng(2).poisson(3, size=(300, 100))).astype(float)
    r = analyze_matrix(x)
    assert r.classification != 'LIKELY_RAW_COUNTS'


def test_negatives_reported():
    x = np.array([[0.0, -1.0, 0.5], [0.0, 1.2, 2.3]])
    r = analyze_matrix(x)
    assert r.negative_fraction > 0


def test_empty_rejected():
    import pytest
    with pytest.raises(ValueError):
        analyze_matrix(np.empty((0, 10)))
