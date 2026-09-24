"""Adapter for my own solver, the ``pemfc_1d`` package.

Everything model-specific about my 1D solver is confined to this file: how a
parameter set is built, how the sweep is called, and how the stacked solution
vector is unpacked into per-layer profiles. The translation is mostly naming,
with two substantive conversions:

*Saturation.* The solved quantity in the two-phase layers is the liquid water
pressure ``P_liq``, not the saturation. ``pemfc_1d`` labels that slot
``Quantity.SATURATION`` because the saturation is what it is read off, and
exports the raw pressure in Pa. This repository reports the saturation ``s``
itself, because that is what the reference tables and the literature plot, so
the adapter inverts the capillary-pressure curve here.

*Position.* Profiles come back in metres from the solver and are reported in
micrometres, the unit the layer thicknesses are quoted in.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import numpy as np
import pemfc_1d
from pemfc_1d import (
    ACTIVE_REGIONS,
    Params,
    Quantity,
    Region,
    State,
    saturation_from_capillary_pressure,
    solve,
)

from pemfc1d_benchmark.adapters.base import (
    LAYERS,
    VARIABLE_UNITS,
    ModelAdapter,
    ModelResult,
    Profile,
    ProfileKey,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pemfc1d_benchmark.cases import Case

#: Row of a region block holding each variable this repository reports. ``s`` is
#: absent on purpose: it is derived from two rows rather than read from one, and
#: :func:`_saturation` computes it.
VALUE_ROWS: dict[str, State] = {
    "phi_e": Quantity.PHI_E.value_row,
    "phi_p": Quantity.PHI_P.value_row,
    "T": Quantity.T.value_row,
    "lambda": Quantity.LAMBDA.value_row,
    "wH2O": Quantity.W_H2O.value_row,
    "wO2": Quantity.W_O2.value_row,
    "Pgas": Quantity.P_GAS.value_row,
}

#: Quantity whose ``ACTIVE_REGIONS`` entry decides where each variable exists.
#: ``s`` shares ``Quantity.SATURATION``'s layers, which is the slot the liquid
#: pressure it is derived from lives in.
OWNING_QUANTITY: dict[str, Quantity] = {
    "phi_e": Quantity.PHI_E,
    "phi_p": Quantity.PHI_P,
    "T": Quantity.T,
    "lambda": Quantity.LAMBDA,
    "wH2O": Quantity.W_H2O,
    "wO2": Quantity.W_O2,
    "s": Quantity.SATURATION,
    "Pgas": Quantity.P_GAS,
}

#: Parameters given as sequences in a case file; converted to arrays because
#: ``Params`` stores them as such and the solver indexes them numerically.
ARRAY_PARAMS: frozenset[str] = frozenset({"L", "U_list"})


def settable_parameters() -> frozenset[str]:
    """Names a case file may override on ``Params``.

    The derived fields (``Lsum``, ``T_A``, the channel mass fractions) are
    ``init=False`` and recomputed in ``__post_init__`` from the others, so
    setting one is meaningless and is rejected rather than silently discarded.
    """
    return frozenset(field.name for field in fields(Params) if field.init)


def build_params(overrides: dict[str, object]) -> Params:
    """A ``Params`` with ``overrides`` applied, validating the names first."""
    settable = settable_parameters()
    unknown = sorted(set(overrides) - settable)
    if unknown:
        derived = sorted(
            field.name for field in fields(Params) if not field.init
        )
        hint = ""
        if set(unknown) & set(derived):
            hint = (
                " -- some of these are derived fields, computed from the others "
                "and not settable"
            )
        raise ValueError(
            f"case file sets parameters pemfc_1d.Params does not accept: "
            f"{unknown}{hint}"
        )
    prepared = {
        name: np.asarray(value, dtype=float) if name in ARRAY_PARAMS else value
        for name, value in overrides.items()
    }
    return replace(Params(), **prepared)


def _saturation(block: np.ndarray, params: Params) -> np.ndarray:
    """Liquid water saturation from the solved liquid and gas pressures.

    ``block`` is one region's 16 rows. The capillary pressure is
    ``P_liq - P_gas``, and the constitutive curve is inverted by the same
    routine the solver itself uses, so the saturation reported here and the
    saturation the model ran on are the same number.
    """
    capillary_pressure = block[State.P_LIQ] - block[State.P_GAS]
    return np.asarray(
        saturation_from_capillary_pressure(capillary_pressure, params.s_im),
        dtype=float,
    )


class Pemfc1dAdapter(ModelAdapter):
    """Runs the ``pemfc_1d`` solver over a case's voltage sweep."""

    name = "pemfc1d"

    @property
    def library_version(self) -> str:
        return pemfc_1d.__version__

    def run(self, case: Case) -> ModelResult:
        params = build_params(case.params)
        sweep = solve(voltages=np.asarray(case.voltages, dtype=float),
                      params=params, **self._solver_kwargs(case))

        unsolved = [
            voltage for voltage in case.voltages
            if not np.any(np.isclose(sweep.voltages, voltage))
        ]
        metadata = self._metadata(
            case,
            case.solver.as_metadata(),
            library_git_revision=pemfc_1d.git_revision(),
            converged=bool(sweep.converged),
            n_voltages_requested=len(case.voltages),
            n_voltages_solved=int(sweep.n_voltages),
            unsolved_voltages=unsolved,
        )
        return ModelResult(
            metadata=metadata,
            current_density=np.asarray(sweep.current_densities, dtype=float),
            voltage=np.asarray(sweep.voltages, dtype=float),
            profiles=self._profiles(case, sweep),
        )

    @staticmethod
    def _solver_kwargs(case: Case) -> dict[str, object]:
        """The solver settings to pass, leaving unset ones at the library default."""
        settings = case.solver
        kwargs: dict[str, object] = {"tol": settings.tol}
        if settings.n_per_region is not None:
            kwargs["n_per_region"] = settings.n_per_region
        if settings.max_nodes is not None:
            kwargs["max_nodes"] = settings.max_nodes
        return kwargs

    def _profiles(self, case: Case, sweep) -> dict[ProfileKey, tuple[Profile, ...]]:
        """Per-layer profiles at each of the case's profile voltages.

        A voltage the sweep did not reach is skipped rather than raised on: the
        solver stops a sweep early when continuation is poisoned, and the
        points it did solve are still worth comparing. Which voltages went
        missing is recorded in the run metadata.
        """
        if not case.profile_voltages:
            return {}

        collected: dict[ProfileKey, list[Profile]] = {}
        for voltage in case.profile_voltages:
            matches = np.flatnonzero(np.isclose(sweep.voltages, voltage))
            if matches.size == 0:
                continue
            index = int(matches[0])
            for profile in self._profiles_at(sweep, index, float(voltage),
                                             case.solver.n_dense):
                collected.setdefault((profile.variable, profile.layer), []).append(
                    profile
                )
        return {key: tuple(group) for key, group in collected.items()}

    @staticmethod
    def _profiles_at(sweep, index: int, voltage: float,
                     n_dense: int) -> list[Profile]:
        """Every (variable, layer) profile of one solved voltage."""
        # Imported here, not at module level: pemfc_1d keeps matplotlib behind
        # its postprocessing module so that a headless run does not pay for a
        # plotting stack, and importing it up here would defeat that.
        from pemfc_1d.postprocessing import extract_profiles

        params = sweep.params
        solution = sweep.solutions[index]
        s_dense = np.linspace(0.0, 1.0, n_dense)
        stacked = solution.sol(s_dense)

        # extract_profiles is called for its position grid and NaN masking, so
        # that this adapter and the library's own figures place a layer's points
        # identically; the raw stacked block is what the saturation needs.
        positions, _ = extract_profiles(solution, params, n_dense=n_dense)

        profiles: list[Profile] = []
        for region in Region:
            layer = LAYERS[int(region)]
            columns = slice(int(region) * n_dense, (int(region) + 1) * n_dense)
            x_um = positions[columns] * 1e6
            block = stacked[int(region) * len(State):(int(region) + 1) * len(State)]
            for variable, quantity in OWNING_QUANTITY.items():
                if region not in ACTIVE_REGIONS[quantity]:
                    continue
                values = (
                    _saturation(block, params) if variable == "s"
                    else np.asarray(block[VALUE_ROWS[variable]], dtype=float)
                )
                profiles.append(Profile(
                    variable=variable,
                    layer=layer,
                    voltage=voltage,
                    x_um=x_um,
                    values=values,
                    unit=VARIABLE_UNITS[variable],
                ))
        return profiles
