"""Verification against the stored MATLAB MMM1D reference outputs.

Verification is not benchmarking. Here both codes solve the *same* equations
with the *same* parameters, so the only admissible difference is numerical: what
two different BVP solvers, on two different meshes, at their stated tolerances,
make of one problem. Any structural disagreement -- a sign, a unit, a boundary
condition, a constitutive relation -- is a bug in my implementation, and this
module exists to find it before a figure built on it reaches the thesis.

The comparison is made per ``(variable, layer, voltage)``, never over the MEA as
a whole. A profile averaged across a layer interface would smear a genuine
discontinuity into an apparent error, and an RMSE pooled over all eight
variables would be dominated by whichever one happens to have the largest
numerical magnitude.

Both codes choose their own meshes, so the model profile is interpolated onto
the reference positions within each layer. The reference is the abscissa: its
points are what was stored and what the thesis tabulates.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pemfc1d_benchmark.adapters.base import (
    LAYERS,
    VARIABLE_LAYERS,
    VARIABLE_UNITS,
    ModelResult,
    Profile,
)
from pemfc1d_benchmark.cases import Case, Tolerance
from pemfc1d_benchmark.metrics import (
    ErrorSummary,
    compare_polarization,
    interpolate_curve,
    summarize,
)

#: Columns a stored reference profile table must have. Documented, with the
#: MATLAB side of how they are produced, in ``reference/matlab/README.md``.
PROFILE_COLUMNS: tuple[str, ...] = (
    "layer", "variable", "x_um", "voltage", "value", "unit",
)

#: Columns a stored reference polarization curve must have.
CURVE_COLUMNS: tuple[str, ...] = ("current_density_A_cm2", "voltage_V")


def _require_columns(frame: pd.DataFrame, required: Sequence[str],
                     path: Path) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: reference table is missing required columns {missing}; "
            f"expected {list(required)}"
        )


def load_reference_profiles(path: str | Path) -> pd.DataFrame:
    """Load a long-format reference profile table.

    Rows whose variable or layer this repository does not know are an error
    rather than a warning: a reference file naming a quantity my model does not
    have means the two are not describing the same problem, which is precisely
    what verification is meant to notice.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    _require_columns(frame, PROFILE_COLUMNS, path)

    unknown_variables = sorted(set(frame["variable"]) - set(VARIABLE_UNITS))
    if unknown_variables:
        raise ValueError(
            f"{path}: unknown state variables {unknown_variables}; "
            f"known: {sorted(VARIABLE_UNITS)}"
        )
    unknown_layers = sorted(set(frame["layer"]) - set(LAYERS))
    if unknown_layers:
        raise ValueError(
            f"{path}: unknown layers {unknown_layers}; known: {list(LAYERS)}"
        )

    misplaced = [
        (variable, layer)
        for variable, layer in frame[["variable", "layer"]].drop_duplicates().values
        if layer not in VARIABLE_LAYERS[variable]
    ]
    if misplaced:
        raise ValueError(
            f"{path}: rows place variables in layers they are not defined on: "
            f"{misplaced}"
        )
    return frame.astype({"x_um": float, "voltage": float, "value": float})


def load_reference_curve(path: str | Path) -> pd.DataFrame:
    """Load a two-column reference or experimental polarization curve."""
    path = Path(path)
    frame = pd.read_csv(path, comment="#")
    _require_columns(frame, CURVE_COLUMNS, path)
    return frame.astype({name: float for name in CURVE_COLUMNS})


@dataclass(frozen=True)
class ProfileComparison:
    """Result of comparing one model profile against its reference profile."""

    variable: str
    layer: str
    voltage: float
    unit: str
    tolerance: Tolerance
    summary: ErrorSummary
    #: Largest ``|model - reference| - (atol + rtol*|reference|)`` over the
    #: profile. At or below zero every point passed; above it, this is the
    #: margin by which the worst point failed, in the variable's own unit.
    worst_exceedance: float
    #: The same worst point as a dimensionless ratio,
    #: ``|model - reference| / (atol + rtol*|reference|)``: 1.0 is exactly at
    #: tolerance, below passes, above fails. This is the number to compare
    #: *across* variables, since the exceedance above is in each variable's own
    #: unit and a gas pressure in Pa would otherwise dwarf a saturation in [-].
    worst_ratio: float
    #: [um] position of that worst point, for pointing at where it went wrong.
    worst_x_um: float
    n_failing: int

    @property
    def passed(self) -> bool:
        """True when every compared point met the tolerance."""
        return self.n_failing == 0 and not self.summary.is_empty

    def to_row(self) -> dict[str, object]:
        """One flat row, for the CSV summary and the terminal table."""
        return {
            "variable": self.variable,
            "layer": self.layer,
            "voltage_V": self.voltage,
            "unit": self.unit,
            "rtol": self.tolerance.rtol,
            "atol": self.tolerance.atol,
            "n_failing": self.n_failing,
            "worst_exceedance": self.worst_exceedance,
            "worst_ratio": self.worst_ratio,
            "worst_x_um": self.worst_x_um,
            "passed": self.passed,
            **self.summary.to_dict(),
        }


@dataclass(frozen=True)
class VerificationReport:
    """Every comparison made for one case, plus the polarization curve check."""

    case_name: str
    model: str
    library_version: str
    timestamp: str
    profiles: tuple[ProfileComparison, ...] = ()
    polarization: ErrorSummary | None = None
    #: Comparisons the case asked for that could not be made, with the reason.
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        """True when every comparison that was made passed.

        A report with nothing in it does not pass: an empty verification run is
        a missing reference file, not a verified model.
        """
        if not self.profiles:
            return False
        return all(comparison.passed for comparison in self.profiles)

    @property
    def failures(self) -> tuple[ProfileComparison, ...]:
        return tuple(c for c in self.profiles if not c.passed)

    def to_frame(self) -> pd.DataFrame:
        """The profile comparisons as a table, worst first.

        Ordered by the dimensionless ratio rather than the exceedance, so that
        the comparison needing attention is at the top whatever unit it is in.
        """
        if not self.profiles:
            return pd.DataFrame(columns=["variable", "layer", "voltage_V"])
        frame = pd.DataFrame([c.to_row() for c in self.profiles])
        return frame.sort_values("worst_ratio", ascending=False,
                                 ignore_index=True)


def compare_profile(model: Profile, reference_x_um, reference_values,
                    tolerance: Tolerance) -> ProfileComparison:
    """Compare one model profile against reference values at their own positions.

    The model is interpolated onto ``reference_x_um``; reference points outside
    the model's range for that layer become NaN and are dropped from the
    metrics, which is the honest outcome when the two do not cover the same
    interval.
    """
    reference_x = np.asarray(reference_x_um, dtype=float).ravel()
    reference = np.asarray(reference_values, dtype=float).ravel()
    interpolated = interpolate_curve(reference_x, model.x_um, model.values)

    deviation = np.abs(interpolated - reference)
    allowed = tolerance.atol + tolerance.rtol * np.abs(reference)
    exceedance = deviation - allowed

    # The ratio ranks points the same way the exceedance does whenever the
    # allowance is positive, which it is for any sane tolerance. A tolerance of
    # exactly zero is still guarded: the ratio is then infinite wherever the
    # profiles differ at all, and zero where they agree exactly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(allowed > 0.0, deviation / allowed,
                         np.where(deviation > 0.0, np.inf, 0.0))

    comparable = np.isfinite(exceedance)
    if np.any(comparable):
        worst = int(np.argmax(np.where(comparable, exceedance, -np.inf)))
        worst_exceedance = float(exceedance[worst])
        worst_ratio = float(ratio[worst])
        worst_x_um = float(reference_x[worst])
        n_failing = int(np.count_nonzero(exceedance[comparable] > 0.0))
    else:
        worst_exceedance = worst_ratio = worst_x_um = float("nan")
        n_failing = 0

    return ProfileComparison(
        variable=model.variable,
        layer=model.layer,
        voltage=model.voltage,
        unit=model.unit,
        tolerance=tolerance,
        summary=summarize(interpolated, reference),
        worst_exceedance=worst_exceedance,
        worst_ratio=worst_ratio,
        worst_x_um=worst_x_um,
        n_failing=n_failing,
    )


def _requested_keys(case: Case, reference: pd.DataFrame,
                    result: ModelResult) -> Iterator[tuple[str, str, float]]:
    """The ``(variable, layer, voltage)`` triples this case asks to compare.

    Restricted three ways: to what the reference file contains, to what the
    case selected, and to what the model actually produced.
    """
    selection = case.reference.matlab_profiles
    wanted_variables = set(selection.variables) if selection.variables else None
    wanted_layers = set(selection.layers) if selection.layers else None
    sampled = case.profile_voltages or list(result.sampled_voltages())

    available = reference[["variable", "layer", "voltage"]].drop_duplicates()
    for variable, layer, voltage in available.itertuples(index=False):
        if wanted_variables is not None and variable not in wanted_variables:
            continue
        if wanted_layers is not None and layer not in wanted_layers:
            continue
        if not any(abs(voltage - requested) <= 1e-9 for requested in sampled):
            continue
        yield variable, layer, float(voltage)


def verify_case(case: Case, result: ModelResult) -> VerificationReport:
    """Compare ``result`` against every reference the case names.

    Missing reference files and unsampled operating points are recorded in
    ``skipped`` rather than raised: a partially verified case still reports what
    it could check, and the gaps are then visible in one place.
    """
    metadata = result.metadata
    comparisons: list[ProfileComparison] = []
    skipped: list[tuple[str, str]] = []
    polarization: ErrorSummary | None = None

    selection = case.reference.matlab_profiles
    if selection is None:
        skipped.append(("profiles", "the case names no MATLAB profile reference"))
    else:
        path = case.resolve(selection.path)
        if not path.is_file():
            skipped.append(("profiles", f"reference file not found: {path}"))
        else:
            reference = load_reference_profiles(path)
            for variable, layer, voltage in _requested_keys(case, reference, result):
                model_profile = result.profile(variable, layer, voltage)
                if model_profile is None:
                    skipped.append((
                        f"{variable}/{layer} at {voltage:.3f} V",
                        "the model produced no profile here",
                    ))
                    continue
                rows = reference[
                    (reference["variable"] == variable)
                    & (reference["layer"] == layer)
                    & (np.abs(reference["voltage"] - voltage) <= 1e-9)
                ].sort_values("x_um")
                comparisons.append(compare_profile(
                    model_profile, rows["x_um"].to_numpy(),
                    rows["value"].to_numpy(), case.tolerance_for(variable),
                ))

    curve = case.reference.matlab_polarization
    if curve is None:
        skipped.append(("polarization", "the case names no MATLAB reference curve"))
    else:
        path = case.resolve(curve.path)
        if not path.is_file():
            skipped.append(("polarization", f"reference file not found: {path}"))
        else:
            reference_curve = load_reference_curve(path)
            polarization = compare_polarization(
                result.current_density, result.voltage,
                reference_curve["current_density_A_cm2"].to_numpy(),
                reference_curve["voltage_V"].to_numpy(),
            )

    return VerificationReport(
        case_name=case.name,
        model=metadata.model,
        library_version=metadata.library_version,
        timestamp=metadata.timestamp,
        profiles=tuple(comparisons),
        polarization=polarization,
        skipped=tuple(skipped),
    )
