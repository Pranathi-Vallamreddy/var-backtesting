import numpy as np
import pytest

from varlib.stats_tests import basel_zone, christoffersen, conditional_coverage, kupiec


def series(n, hits):
    e = np.zeros(n, dtype=bool)
    e[list(hits)] = True
    return e


def test_kupiec_hand_computed():
    # N=250, x=10, p=0.01:
    # -2 * [240 ln(0.99) + 10 ln(0.01) - 240 ln(0.96) - 10 ln(0.04)] = 12.9555
    lr, p = kupiec(series(250, range(10)), 0.99)
    assert lr == pytest.approx(12.9555, abs=1e-4)
    assert p < 0.001


def test_kupiec_exact_rate_does_not_reject():
    lr, p = kupiec(series(1000, range(0, 1000, 100)), 0.99)
    assert lr == pytest.approx(0.0, abs=1e-10)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_miscalibrated_model():
    # 3% exceptions against a 1% target
    _, p = kupiec(series(1000, range(0, 1000, 33)), 0.99)
    assert p < 0.01


def test_kupiec_zero_exceptions_is_finite():
    # zero exceptions in 500 days is itself evidence of an over-conservative model
    lr, p = kupiec(np.zeros(500, dtype=bool), 0.99)
    assert lr == pytest.approx(-2 * 500 * np.log(0.99))
    assert 0 < p < 0.01


def test_christoffersen_rejects_clustered_exceptions():
    _, p = christoffersen(series(1000, range(500, 510)))
    assert p < 0.001


def test_christoffersen_accepts_evenly_spaced_exceptions():
    _, p = christoffersen(series(1000, range(0, 1000, 100)))
    assert p > 0.5


def test_christoffersen_no_consecutive_exceptions_is_finite():
    # n11 = 0 and no exceptions at all both leave a log(0) term to guard
    for e in (series(1000, range(5, 1000, 50)), np.zeros(1000, dtype=bool)):
        lr, p = christoffersen(e)
        assert np.isfinite(lr) and 0 <= p <= 1


def test_conditional_coverage_is_sum_of_parts():
    e = series(1000, [10, 11, 12, 400, 700])
    lr_cc, _ = conditional_coverage(e, 0.99)
    assert lr_cc == pytest.approx(kupiec(e, 0.99)[0] + christoffersen(e)[0])


@pytest.mark.parametrize("x, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zone_boundaries(x, zone):
    assert basel_zone(x) == zone
