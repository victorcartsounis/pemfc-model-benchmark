"""Metrics on synthetic data whose answers can be worked out by hand.

The point of testing metrics this way is that a metric has no reference
implementation to check against: it *is* the reference. So every expectation here
is a number derived on paper from the inputs, not a value captured from a
previous run of this code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pemfc1d_benchmark.cases import LossRegion
from pemfc1d_benchmark.metrics import (
    bias,
    compare_polarization,
    interpolate_curve,
    mae,
    mape,
    max_abs_error,
    max_relative_error,
    per_region_summaries,
    rmse,
    summarize,
)


class TestKnownErrors:
    """Deviations of 0, 1, 2 and 3 about observations of 10, so every metric
    comes out exact."""

    observed = np.array([10.0, 10.0, 10.0, 10.0])
    predicted = np.array([10.0, 11.0, 8.0, 13.0])   # errors: 0, +1, -2, +3

    def test_rmse(self):
        # sqrt((0 + 1 + 4 + 9)/4) = sqrt(3.5)
        assert rmse(self.predicted, self.observed) == pytest.approx(math.sqrt(3.5))

    def test_mae(self):
        # (0 + 1 + 2 + 3)/4
        assert mae(self.predicted, self.observed) == pytest.approx(1.5)

    def test_max_abs_error(self):
        assert max_abs_error(self.predicted, self.observed) == pytest.approx(3.0)

    def test_bias_keeps_the_sign(self):
        # (0 + 1 - 2 + 3)/4 = 0.5: high on average, though not everywhere
        assert bias(self.predicted, self.observed) == pytest.approx(0.5)

    def test_mape(self):
        # mean(0, 0.1, 0.2, 0.3) * 100
        assert mape(self.predicted, self.observed) == pytest.approx(15.0)

    def test_max_relative_error(self):
        assert max_relative_error(self.predicted, self.observed) == pytest.approx(0.3)

    def test_summarize_agrees_with_the_individual_metrics(self):
        summary = summarize(self.predicted, self.observed)
        assert summary.n_points == 4
        assert summary.rmse == pytest.approx(rmse(self.predicted, self.observed))
        assert summary.mae == pytest.approx(mae(self.predicted, self.observed))
        assert summary.bias == pytest.approx(bias(self.predicted, self.observed))
        assert summary.mape == pytest.approx(mape(self.predicted, self.observed))
        assert not summary.is_empty


class TestDegenerateInputs:
    def test_a_perfect_match_is_zero_everywhere(self):
        values = np.array([1.0, 2.0, 3.0])
        summary = summarize(values, values)
        assert summary.rmse == 0.0
        assert summary.mae == 0.0
        assert summary.max_abs_error == 0.0
        assert summary.bias == 0.0
        assert summary.mape == 0.0

    def test_bias_cancels_where_rmse_does_not(self):
        """The reason both are reported: +1 and -1 average to nothing."""
        predicted, observed = np.array([11.0, 9.0]), np.array([10.0, 10.0])
        assert bias(predicted, observed) == pytest.approx(0.0)
        assert rmse(predicted, observed) == pytest.approx(1.0)

    def test_nan_pairs_are_dropped_and_counted(self):
        """Profiles carry NaN where a quantity is not defined; those points vanish."""
        predicted = np.array([10.0, np.nan, 12.0])
        observed = np.array([10.0, 10.0, 10.0])
        summary = summarize(predicted, observed)
        assert summary.n_points == 2
        assert summary.rmse == pytest.approx(math.sqrt(2.0))  # errors 0 and +2

    def test_all_nan_gives_an_empty_summary_not_a_crash(self):
        summary = summarize(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
        assert summary.is_empty
        assert math.isnan(summary.rmse)

    def test_mape_skips_zero_observations(self):
        """A relative error against zero is undefined, so that point is dropped."""
        predicted = np.array([1.0, 11.0])
        observed = np.array([0.0, 10.0])
        assert mape(predicted, observed) == pytest.approx(10.0)
        assert rmse(predicted, observed) == pytest.approx(math.sqrt(1.0))

    def test_mape_is_nan_when_every_observation_is_zero(self):
        assert math.isnan(mape(np.array([1.0, 2.0]), np.zeros(2)))

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="interpolate onto a common grid"):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))


class TestInterpolateCurve:
    def test_linear_interpolation_at_a_midpoint(self):
        y = interpolate_curve([0.5], [0.0, 1.0], [0.0, 10.0])
        assert y[0] == pytest.approx(5.0)

    def test_source_order_does_not_matter(self):
        """A polarization curve is written in descending voltage, so it
        arrives here unsorted in current."""
        descending = interpolate_curve([0.5], [1.0, 0.0], [10.0, 0.0])
        ascending = interpolate_curve([0.5], [0.0, 1.0], [0.0, 10.0])
        assert descending[0] == pytest.approx(ascending[0])

    def test_outside_the_source_range_is_nan_by_default(self):
        y = interpolate_curve([-0.1, 0.5, 1.1], [0.0, 1.0], [0.0, 10.0])
        assert math.isnan(y[0])
        assert y[1] == pytest.approx(5.0)
        assert math.isnan(y[2])

    def test_extrapolation_is_opt_in_and_clamps(self):
        y = interpolate_curve([2.0], [0.0, 1.0], [0.0, 10.0], extrapolate=True)
        assert y[0] == pytest.approx(10.0)   # numpy.interp holds the end value

    def test_a_single_usable_point_cannot_be_interpolated(self):
        y = interpolate_curve([0.5], [0.0], [1.0])
        assert math.isnan(y[0])


class TestComparePolarization:
    def test_a_constant_offset_shows_up_as_bias_equal_to_rmse(self):
        """A model 10 mV high everywhere: the offset is systematic, not scatter."""
        reference_i = np.array([0.2, 0.4, 0.6])
        reference_u = np.array([0.80, 0.75, 0.70])
        summary = compare_polarization(reference_i, reference_u + 0.010,
                                       reference_i, reference_u)
        assert summary.n_points == 3
        assert summary.bias == pytest.approx(0.010)
        assert summary.rmse == pytest.approx(0.010)

    def test_the_model_is_interpolated_onto_the_reference_abscissa(self):
        """The reference's current densities are the ones compared at."""
        summary = compare_polarization(
            model_current_density=np.array([0.0, 1.0]),
            model_voltage=np.array([1.0, 0.0]),
            reference_current_density=np.array([0.25, 0.75]),
            reference_voltage=np.array([0.75, 0.25]),
        )
        assert summary.n_points == 2
        assert summary.rmse == pytest.approx(0.0)

    def test_reference_points_beyond_the_model_range_are_dropped(self):
        summary = compare_polarization(
            model_current_density=np.array([0.0, 0.5]),
            model_voltage=np.array([1.0, 0.5]),
            reference_current_density=np.array([0.25, 2.0]),
            reference_voltage=np.array([0.75, 0.10]),
        )
        assert summary.n_points == 1


class TestPerRegionSummaries:
    regions = {
        "activation": LossRegion(i_min=0.0, i_max=0.1),
        "ohmic": LossRegion(i_min=0.1, i_max=1.0),
        "mass_transport": LossRegion(i_min=1.0, i_max=None),
    }
    current = np.array([0.05, 0.5, 0.9, 1.5])
    observed = np.array([10.0, 10.0, 10.0, 10.0])
    predicted = np.array([10.0, 11.0, 9.0, 14.0])

    def test_each_window_sees_only_its_own_points(self):
        summaries = per_region_summaries(self.predicted, self.observed,
                                         self.current, self.regions)
        assert summaries["activation"].n_points == 1
        assert summaries["ohmic"].n_points == 2       # 0.5 and 0.9
        assert summaries["mass_transport"].n_points == 1

    def test_metrics_are_computed_within_the_window(self):
        summaries = per_region_summaries(self.predicted, self.observed,
                                         self.current, self.regions)
        # ohmic errors are +1 and -1: they cancel in bias, not in RMSE
        assert summaries["ohmic"].bias == pytest.approx(0.0)
        assert summaries["ohmic"].rmse == pytest.approx(1.0)
        assert summaries["mass_transport"].max_abs_error == pytest.approx(4.0)

    def test_bounds_are_lower_inclusive_and_upper_exclusive(self):
        """0.1 belongs to the ohmic region, not to activation."""
        summaries = per_region_summaries(
            np.array([10.0]), np.array([10.0]), np.array([0.1]), self.regions)
        assert summaries["activation"].n_points == 0
        assert summaries["ohmic"].n_points == 1

    def test_an_empty_region_yields_an_empty_summary_rather_than_disappearing(self):
        """A results table keeps the same rows, so a gap is visible as one."""
        summaries = per_region_summaries(
            np.array([10.0]), np.array([10.0]), np.array([0.05]), self.regions)
        assert set(summaries) == set(self.regions)
        assert summaries["mass_transport"].is_empty

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            per_region_summaries(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                 np.array([0.5]), self.regions)
