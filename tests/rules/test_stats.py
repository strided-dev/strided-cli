"""
Tests for rules/_stats.py — the robust trend toolkit r06 relies on.

theil_sen_slope must recover a clean slope and shrug off a single wild outlier
(its whole reason for existing over OLS); mann_kendall must call a monotonic
trend significant and a flat/noisy series not, and both must degrade gracefully
on too-few points.
"""

from __future__ import annotations

import pytest

from rules._stats import mann_kendall, theil_sen_slope


class TestTheilSen:
    def test_recovers_clean_slope(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 12.0, 14.0, 16.0, 18.0]   # slope +2
        assert theil_sen_slope(xs, ys) == pytest.approx(2.0)

    def test_robust_to_single_outlier(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 12.0, 14.0, 16.0, 999.0]  # last point is a spike
        # Median of pairwise slopes stays near the true +2; OLS would be dragged up.
        assert theil_sen_slope(xs, ys) == pytest.approx(2.0, abs=0.5)

    def test_negative_slope(self) -> None:
        xs = [0.0, 10.0, 20.0, 30.0]
        ys = [100.0, 90.0, 80.0, 70.0]        # -1 per x-unit
        assert theil_sen_slope(xs, ys) == pytest.approx(-1.0)

    def test_none_when_too_few_points(self) -> None:
        assert theil_sen_slope([1.0], [1.0]) is None

    def test_none_when_all_x_equal(self) -> None:
        assert theil_sen_slope([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


class TestMannKendall:
    def test_monotonic_increasing_is_significant_up(self) -> None:
        t = mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert t is not None
        assert t.s > 0
        assert t.p_value < 0.05

    def test_monotonic_decreasing_is_significant_down(self) -> None:
        t = mann_kendall([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        assert t is not None
        assert t.s < 0
        assert t.p_value < 0.05

    def test_flat_series_is_not_significant(self) -> None:
        t = mann_kendall([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        assert t is not None
        assert t.s == 0
        assert t.p_value == pytest.approx(1.0)

    def test_noisy_nontrend_is_not_significant(self) -> None:
        t = mann_kendall([5.0, 6.0, 4.0, 7.0, 3.0, 6.0, 4.0, 5.0])
        assert t is not None
        assert t.p_value > 0.10

    def test_none_when_too_few_points(self) -> None:
        assert mann_kendall([1.0, 2.0]) is None
