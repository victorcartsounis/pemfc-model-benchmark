"""Descriptive error metrics for comparing two curves or two profiles.

Nothing here decides whether a model is good. These functions describe a
discrepancy, and the interpretation belongs in the thesis text: an RMSE of
10 mV against an experimental curve digitised from a printed figure means
something quite different from the same number against a reference computation.

All metrics take ``predicted`` and ``observed`` in that order, treat ``observed``
as the thing being matched, and ignore pairs where either side is NaN -- profiles
carry NaN wherever a quantity is not physically defined, and a digitised
experimental curve can have gaps.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

#: Below this magnitude an observed value is treated as zero for MAPE, whose
#: relative error is undefined there. Such points are dropped and counted.
MAPE_FLOOR = 1e-12


def _aligned(predicted, observed) -> tuple[np.ndarray, np.ndarray]:
    """The two arrays as floats, same length, with non-finite pairs dropped."""
    prediction = np.asarray(predicted, dtype=float).ravel()
    observation = np.asarray(observed, dtype=float).ravel()
    if prediction.size != observation.size:
        raise ValueError(
            f"predicted has {prediction.size} points but observed has "
            f"{observation.size}; interpolate onto a common grid first"
        )
    usable = np.isfinite(prediction) & np.isfinite(observation)
    return prediction[usable], observation[usable]


def rmse(predicted, observed) -> float:
    """Root mean squared error, in the unit of the inputs."""
    prediction, observation = _aligned(predicted, observed)
    if prediction.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((prediction - observation) ** 2)))


def mae(predicted, observed) -> float:
    """Mean absolute error, in the unit of the inputs."""
    prediction, observation = _aligned(predicted, observed)
    if prediction.size == 0:
        return float("nan")
    return float(np.mean(np.abs(prediction - observation)))


def max_abs_error(predicted, observed) -> float:
    """Largest absolute deviation, in the unit of the inputs."""
    prediction, observation = _aligned(predicted, observed)
    if prediction.size == 0:
        return float("nan")
    return float(np.max(np.abs(prediction - observation)))


def bias(predicted, observed) -> float:
    """Mean signed error: positive when the model sits above the reference.

    Reported alongside RMSE because the two separate a systematic offset -- a
    model a steady 15 mV high everywhere -- from scatter about the right answer.
    """
    prediction, observation = _aligned(predicted, observed)
    if prediction.size == 0:
        return float("nan")
    return float(np.mean(prediction - observation))


def mape(predicted, observed) -> float:
    """Mean absolute percentage error [%], skipping near-zero observations."""
    prediction, observation = _aligned(predicted, observed)
    usable = np.abs(observation) > MAPE_FLOOR
    if not np.any(usable):
        return float("nan")
    relative = np.abs(
        (prediction[usable] - observation[usable]) / observation[usable]
    )
    return float(100.0 * np.mean(relative))


def max_relative_error(predicted, observed) -> float:
    """Largest relative deviation [-], skipping near-zero observations."""
    prediction, observation = _aligned(predicted, observed)
    usable = np.abs(observation) > MAPE_FLOOR
    if not np.any(usable):
        return float("nan")
    return float(np.max(np.abs(
        (prediction[usable] - observation[usable]) / observation[usable]
    )))


@dataclass(frozen=True)
class ErrorSummary:
    """Every metric of one comparison, and how many points it rests on.

    ``n_points`` is the count actually compared, after NaN pairs were dropped.
    It is part of the summary rather than an aside: an RMSE over three surviving
    points of a forty-point curve should not be read the same way as one over
    forty.
    """

    n_points: int
    rmse: float
    mae: float
    max_abs_error: float
    bias: float
    mape: float
    max_relative_error: float

    @property
    def is_empty(self) -> bool:
        """True when nothing could be compared."""
        return self.n_points == 0

    def to_dict(self) -> dict[str, float | int]:
        """A flat, JSON- and CSV-friendly view."""
        return {
            "n_points": self.n_points,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "bias": self.bias,
            "mape_percent": self.mape,
            "max_relative_error": self.max_relative_error,
        }


def summarize(predicted, observed) -> ErrorSummary:
    """Every metric of one comparison, in a single pass over the inputs."""
    prediction, observation = _aligned(predicted, observed)
    return ErrorSummary(
        n_points=int(prediction.size),
        rmse=rmse(prediction, observation),
        mae=mae(prediction, observation),
        max_abs_error=max_abs_error(prediction, observation),
        bias=bias(prediction, observation),
        mape=mape(prediction, observation),
        max_relative_error=max_relative_error(prediction, observation),
    )


def interpolate_curve(x_target, x_source, y_source,
                      extrapolate: bool = False) -> np.ndarray:
    """``y_source`` sampled at ``x_target``, by linear interpolation.

    Two curves are almost never measured or solved at the same current
    densities, so one has to be moved onto the other's abscissa before any
    metric means anything. ``x_source`` is sorted here rather than assumed
    sorted, because a polarization curve is naturally written in descending
    voltage, which is ascending current.

    Outside the source range the result is NaN unless ``extrapolate`` is set.
    NaN is the safer default: extrapolating a polarization curve past its
    limiting current invents exactly the behaviour a benchmark is trying to
    measure.
    """
    target = np.asarray(x_target, dtype=float).ravel()
    source_x = np.asarray(x_source, dtype=float).ravel()
    source_y = np.asarray(y_source, dtype=float).ravel()
    if source_x.size != source_y.size:
        raise ValueError(
            f"source curve has {source_x.size} abscissae but {source_y.size} values"
        )

    usable = np.isfinite(source_x) & np.isfinite(source_y)
    source_x, source_y = source_x[usable], source_y[usable]
    if source_x.size < 2:
        return np.full(target.shape, np.nan)

    order = np.argsort(source_x)
    source_x, source_y = source_x[order], source_y[order]

    interpolated = np.interp(target, source_x, source_y)
    if not extrapolate:
        outside = (target < source_x[0]) | (target > source_x[-1])
        interpolated = np.where(outside, np.nan, interpolated)
    return interpolated


def compare_polarization(model_current_density, model_voltage,
                         reference_current_density, reference_voltage,
                         extrapolate: bool = False) -> ErrorSummary:
    """Voltage error of a model curve against a reference curve.

    The comparison is made in voltage at the reference's current densities:
    current density is the controlled variable of a measurement, so it is the
    reference curve's abscissa that the model is asked to reproduce.
    """
    reference_i = np.asarray(reference_current_density, dtype=float).ravel()
    predicted = interpolate_curve(reference_i, model_current_density,
                                  model_voltage, extrapolate=extrapolate)
    return summarize(predicted, reference_voltage)


def per_region_summaries(
    predicted, observed, current_density,
    regions: Mapping[str, RegionBounds],
) -> dict[str, ErrorSummary]:
    """Metrics computed separately over each named current-density window.

    ``regions`` maps a name to anything with a ``mask(current_density)`` method
    returning a boolean selector -- :class:`pemfc1d_benchmark.cases.LossRegion`
    is what a case file supplies. A region with no points in range yields an
    empty summary rather than being dropped, so a results table always has the
    same rows and a gap is visible as one.
    """
    prediction = np.asarray(predicted, dtype=float).ravel()
    observation = np.asarray(observed, dtype=float).ravel()
    current = np.asarray(current_density, dtype=float).ravel()
    if not (prediction.size == observation.size == current.size):
        raise ValueError(
            f"predicted ({prediction.size}), observed ({observation.size}) and "
            f"current_density ({current.size}) must be the same length"
        )
    return {
        name: summarize(prediction[region.mask(current)],
                        observation[region.mask(current)])
        for name, region in regions.items()
    }


class RegionBounds:
    """Structural type of what :func:`per_region_summaries` accepts as a region.

    Declared for documentation only; the concrete implementation used in this
    repository is :class:`pemfc1d_benchmark.cases.LossRegion`, loaded from a
    case file.
    """

    def mask(self, current_density) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError
