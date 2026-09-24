"""Figures: polarization curves, verification residuals, profile overlays.

The figures are the argument. A verification table saying the largest deviation
is 4e-4 V is convincing only once the overlay shows the two curves lying on top
of each other and the residual plot shows the deviation is numerical scatter
rather than a systematic offset that happens to be small.

Matplotlib is configured for the non-interactive ``Agg`` backend at import, so
that the CLI draws the same files on my machine and over SSH.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow the backend choice
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pemfc1d_benchmark.adapters.base import VARIABLE_UNITS, ModelResult  # noqa: E402
from pemfc1d_benchmark.cases import Case  # noqa: E402
from pemfc1d_benchmark.metrics import interpolate_curve  # noqa: E402
from pemfc1d_benchmark.verification import (  # noqa: E402
    VerificationReport,
    load_reference_curve,
    load_reference_profiles,
)

#: Axis labels for the eight variables, in the unit each is reported in.
AXIS_LABELS: dict[str, str] = {
    "phi_e": r"$\phi_\mathrm{e}$ [V]",
    "phi_p": r"$\phi_\mathrm{p}$ [V]",
    "T": r"$T$ [K]",
    "lambda": r"$\lambda$ [-]",
    "wH2O": r"$w_\mathrm{H_2O}$ [-]",
    "wO2": r"$w_\mathrm{O_2}$ [-]",
    "s": r"$s$ [-]",
    "Pgas": r"$P_\mathrm{gas}$ [Pa]",
}

DPI = 200


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def _voltage_colors(n: int) -> list:
    """Dark blue at the first voltage through dark red at the last.

    The same colour convention my solver's own figures use, so a figure from
    this repository and one from the model repository can be read side by side.
    """
    colormap = plt.get_cmap("jet")
    return [colormap(index / max(1, n - 1)) for index in range(n)]


def plot_polarization(result: ModelResult, case: Case, path: Path,
                      show_power: bool = True) -> Path:
    """The model's polarization curve, with every reference curve the case names.

    The MATLAB reference is drawn as a line, because it is a computation on a
    continuum of the same equations; experimental curves are drawn as markers,
    because they are measurements at discrete operating points. The distinction
    is the verification/benchmarking one, made visible.
    """
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(result.current_density, result.voltage, "-o", markersize=3.5,
              color="C0", label=f"{result.metadata.model} (this work)")

    reference = case.reference.matlab_polarization
    if reference is not None:
        path_matlab = case.resolve(reference.path)
        if path_matlab.is_file():
            curve = load_reference_curve(path_matlab)
            axis.plot(curve["current_density_A_cm2"], curve["voltage_V"],
                      "--", color="k", linewidth=1.3,
                      label=reference.label or "MATLAB MMM1D reference")

    for index, experiment in enumerate(case.reference.experimental):
        path_experiment = case.resolve(experiment.path)
        if not path_experiment.is_file():
            continue
        curve = load_reference_curve(path_experiment)
        axis.plot(curve["current_density_A_cm2"], curve["voltage_V"], "s",
                  markersize=4.5, markerfacecolor="none",
                  color=f"C{index + 1}",
                  label=experiment.label or f"experiment {index + 1}")

    axis.set_xlabel(r"current density $i$ [A/cm$^2$]")
    axis.set_ylabel(r"cell voltage $U$ [V]")
    axis.set_title(f"{case.name} -- polarization curve")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    if show_power:
        power_axis = axis.twinx()
        power_axis.plot(result.current_density, result.power_density, ":",
                        color="C3", linewidth=1.2)
        power_axis.set_ylabel(r"power density $P$ [W/cm$^2$]", color="C3")
        power_axis.tick_params(axis="y", labelcolor="C3")
        power_axis.grid(False)

    return _save(figure, path)


def plot_polarization_residuals(result: ModelResult, case: Case,
                                path: Path) -> Path:
    """Model-minus-reference voltage against current density.

    Plotted on the reference's current densities, the same alignment the metrics
    use, so a point in this figure is a term in the reported RMSE.
    """
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    drawn = False

    reference = case.reference.matlab_polarization
    if reference is not None and case.resolve(reference.path).is_file():
        curve = load_reference_curve(case.resolve(reference.path))
        current = curve["current_density_A_cm2"].to_numpy()
        predicted = interpolate_curve(current, result.current_density,
                                      result.voltage)
        axis.plot(current, (predicted - curve["voltage_V"].to_numpy()) * 1e3,
                  "-o", markersize=3.5, color="C0",
                  label=reference.label or "vs MATLAB MMM1D")
        drawn = True

    for index, experiment in enumerate(case.reference.experimental):
        path_experiment = case.resolve(experiment.path)
        if not path_experiment.is_file():
            continue
        curve = load_reference_curve(path_experiment)
        current = curve["current_density_A_cm2"].to_numpy()
        predicted = interpolate_curve(current, result.current_density,
                                      result.voltage)
        axis.plot(current, (predicted - curve["voltage_V"].to_numpy()) * 1e3,
                  "s", markersize=4.5, markerfacecolor="none",
                  color=f"C{index + 1}",
                  label=experiment.label or f"vs experiment {index + 1}")
        drawn = True

    axis.axhline(0.0, color="k", linewidth=0.8)
    axis.set_xlabel(r"current density $i$ [A/cm$^2$]")
    axis.set_ylabel(r"$U_\mathrm{model} - U_\mathrm{reference}$ [mV]")
    axis.set_title(f"{case.name} -- polarization residuals")
    axis.grid(True, alpha=0.3)
    if drawn:
        axis.legend(loc="best", fontsize=9)
    else:
        axis.text(0.5, 0.5, "no reference curve available",
                  transform=axis.transAxes, ha="center", va="center")
    return _save(figure, path)


def plot_profile_overlay(result: ModelResult, case: Case, variable: str,
                         path: Path,
                         reference: pd.DataFrame | None = None) -> Path:
    """One variable across the MEA: my model as lines, the reference as markers.

    The layers are drawn into one axis with the interfaces marked, and each
    layer's curve is drawn separately -- the profile is genuinely discontinuous
    at an interface and joining across one would draw a line that no solution
    has.
    """
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    voltages = result.sampled_voltages()
    colors = _voltage_colors(len(voltages))
    layers = result.layers_of(variable)

    for color_index, voltage in enumerate(voltages):
        for layer in layers:
            profile = result.profile(variable, layer, voltage)
            if profile is None:
                continue
            label = f"{voltage:.2f} V" if layer == layers[0] else None
            axis.plot(profile.x_um, profile.values, "-",
                      color=colors[color_index], linewidth=1.4, label=label)
            if reference is None:
                continue
            rows = reference[
                (reference["variable"] == variable)
                & (reference["layer"] == layer)
                & (np.abs(reference["voltage"] - voltage) <= 1e-9)
            ].sort_values("x_um")
            if rows.empty:
                continue
            reference_label = (
                "MATLAB reference"
                if layer == layers[0] and voltage == voltages[0] else None
            )
            axis.plot(rows["x_um"], rows["value"], "k.", markersize=3.0,
                      label=reference_label)

    interfaces = _interface_positions(result, variable)
    for interface in interfaces:
        axis.axvline(interface, color="0.4", linewidth=0.8)

    axis.set_xlabel(r"position $x$ [$\mu$m]")
    axis.set_ylabel(AXIS_LABELS.get(
        variable, f"{variable} [{VARIABLE_UNITS[variable]}]"))
    axis.set_title(f"{case.name} -- {variable} across {', '.join(layers)}")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8, ncol=2)
    return _save(figure, path)


def _interface_positions(result: ModelResult, variable: str) -> list[float]:
    """Positions where one layer's profile of ``variable`` ends and the next begins."""
    voltages = result.sampled_voltages()
    if not voltages:
        return []
    edges: list[float] = []
    for layer in result.layers_of(variable)[:-1]:
        profile = result.profile(variable, layer, voltages[0])
        if profile is not None and profile.x_um.size:
            edges.append(float(profile.x_um[-1]))
    return edges


#: Ratios below this are drawn at this value, so that a profile matching the
#: reference exactly still has a visible bar on the logarithmic axis.
RATIO_FLOOR = 1e-4


def plot_verification_residuals(report: VerificationReport, path: Path) -> Path:
    """Worst tolerance ratio per compared profile, as a horizontal bar chart.

    The bars are ``|model - reference| / (atol + rtol*|reference|)`` at each
    profile's worst point: 1.0 is exactly at tolerance, left of the line passes,
    right of it fails. The ratio is plotted rather than the raw exceedance
    because the eight variables are in five different units, and a gas pressure
    deviation in Pa would otherwise set the axis scale and squash every other
    bar to nothing.

    The axis is logarithmic because a passing run spans orders of magnitude --
    that is the useful information in it, since a bar three decades clear of the
    line and one just under it are very different states of health.
    """
    frame = report.to_frame()
    figure, axis = plt.subplots(
        figsize=(7.5, max(3.0, 0.26 * max(len(frame), 1) + 1.2))
    )
    if frame.empty:
        axis.text(0.5, 0.5, "no profile comparisons were made",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return _save(figure, path)

    labels = [
        f"{row.variable}/{row.layer} @ {row.voltage_V:.2f} V"
        for row in frame.itertuples()
    ]
    ratios = np.clip(frame["worst_ratio"].to_numpy(dtype=float),
                     RATIO_FLOOR, None)
    positions = np.arange(len(frame))
    colors = ["C2" if passed else "C3" for passed in frame["passed"]]

    axis.barh(positions, ratios, color=colors, left=RATIO_FLOOR)
    axis.set_xscale("log")
    axis.set_xlim(RATIO_FLOOR, max(2.0, float(np.nanmax(ratios)) * 1.5))
    axis.axvline(1.0, color="k", linewidth=1.1)
    axis.text(1.0, -0.8, " tolerance", fontsize=8, va="center")
    axis.set_yticks(positions, labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel(
        "worst |model - reference| as a fraction of what the tolerance "
        "allowed  [-]"
    )
    axis.set_title(
        f"{report.case_name} -- verification against MATLAB MMM1D "
        f"(pemfc_1d {report.library_version})"
    )
    axis.grid(True, axis="x", alpha=0.3)
    return _save(figure, path)


def write_report(result: ModelResult, case: Case, directory: str | Path,
                 report: VerificationReport | None = None,
                 variables: Sequence[str] | None = None) -> list[Path]:
    """Draw every figure for one case into ``directory`` and return their paths."""
    directory = Path(directory)
    written = [
        plot_polarization(result, case, directory / "polarization.png"),
        plot_polarization_residuals(result, case,
                                    directory / "polarization_residuals.png"),
    ]

    reference_frame = None
    selection = case.reference.matlab_profiles
    if selection is not None and case.resolve(selection.path).is_file():
        reference_frame = load_reference_profiles(case.resolve(selection.path))

    chosen = list(variables) if variables is not None else list(result.variables)
    for variable in chosen:
        written.append(plot_profile_overlay(
            result, case, variable,
            directory / f"profile_{variable}.png", reference_frame,
        ))

    if report is not None:
        written.append(plot_verification_residuals(
            report, directory / "verification_residuals.png"))
    return written
