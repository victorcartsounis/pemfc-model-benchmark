"""Case files: what to run, and what to compare the answer against.

A case is one YAML file holding the operating conditions, the parameter set,
the voltage sweep, and the paths of whatever reference data the case is checked
against. It is the only place any of that is written down -- nothing in this
package carries a default operating point of its own, because a number silently
supplied by the harness is a number that never appears in the thesis.

The schema is validated with pydantic, so a typo in a case file is an error at
load time with the offending field named, rather than a plot that is quietly
wrong. Physical parameter names are *not* validated here: they belong to the
model being run, so the adapter checks them against its own parameter set.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pemfc1d_benchmark.adapters.base import LAYERS, VARIABLE_LAYERS, VARIABLE_UNITS

PositiveFloat = Annotated[float, Field(gt=0.0)]


class Tolerance(BaseModel):
    """Agreement my model is required to show against a reference profile.

    A point passes when ``|model - reference| <= atol + rtol * |reference|``,
    the usual mixed test: ``rtol`` carries most variables, and ``atol`` keeps a
    quantity that legitimately passes through zero -- an electron potential at
    the grounded anode, a flux at a sealed face -- from demanding infinite
    relative precision there.
    """

    model_config = ConfigDict(extra="forbid")

    rtol: float = Field(ge=0.0)
    atol: float = Field(ge=0.0)


class SolverSettings(BaseModel):
    """Numerical settings passed through to the model's solver.

    These are deliberately separate from the physics: two runs of the same case
    at different tolerances are the same case, and the mesh-convergence
    question is asked by varying this block alone.
    """

    model_config = ConfigDict(extra="forbid")

    tol: PositiveFloat = 1e-4
    #: Initial mesh points per layer. ``None`` leaves the model's own default.
    n_per_region: int | None = Field(default=None, gt=1)
    #: Ceiling on adaptive mesh refinement. ``None`` leaves the model's default.
    max_nodes: int | None = Field(default=None, gt=0)
    #: Points per layer at which profiles are sampled for comparison.
    n_dense: int = Field(default=201, gt=1)

    def as_metadata(self) -> dict[str, object]:
        """The settings as recorded in run metadata."""
        return self.model_dump()


class ReferenceProfiles(BaseModel):
    """A stored table of reference profiles, and which of it to compare.

    The file is expected in the long format documented in
    ``reference/matlab/README.md``: one row per point, with columns
    ``layer, variable, x_um, voltage, value, unit``.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    #: Source of the table, for the thesis's provenance trail. Free text.
    source: str
    #: Variables to compare; empty means every variable the file contains.
    variables: list[str] = Field(default_factory=list)
    #: Layers to compare; empty means every layer the file contains.
    layers: list[str] = Field(default_factory=list)

    @field_validator("variables")
    @classmethod
    def _known_variables(cls, names: list[str]) -> list[str]:
        unknown = sorted(set(names) - set(VARIABLE_UNITS))
        if unknown:
            raise ValueError(
                f"unknown state variables {unknown}; "
                f"known: {sorted(VARIABLE_UNITS)}"
            )
        return names

    @field_validator("layers")
    @classmethod
    def _known_layers(cls, names: list[str]) -> list[str]:
        unknown = sorted(set(names) - set(LAYERS))
        if unknown:
            raise ValueError(f"unknown layers {unknown}; known: {list(LAYERS)}")
        return names

    @model_validator(mode="after")
    def _variables_live_on_the_requested_layers(self) -> ReferenceProfiles:
        """A variable asked for only in layers where it does not exist is a typo."""
        if not (self.variables and self.layers):
            return self
        requested = set(self.layers)
        for variable in self.variables:
            if not requested & set(VARIABLE_LAYERS[variable]):
                raise ValueError(
                    f"{variable!r} is not defined on any of {sorted(requested)}; "
                    f"it lives on {list(VARIABLE_LAYERS[variable])}"
                )
        return self


class ReferenceCurve(BaseModel):
    """A stored polarization curve: MATLAB reference, or an experiment.

    Expected columns are ``current_density_A_cm2`` and ``voltage_V``; see the
    README in the directory the file lives in.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    #: Full citation for a published curve, or how a computed one was produced.
    #: Required, and enforced non-empty: an experimental curve with no source
    #: is not usable in a thesis.
    source: str = Field(min_length=1)
    label: str | None = None


class LossRegion(BaseModel):
    """A current-density window that errors are also reported over separately.

    The three regions of a polarization curve -- activation, ohmic, mass
    transport -- fail in different ways, and a single RMSE over the whole curve
    hides which one a model gets wrong. The bounds are per case because they
    depend on the cell and the conditions; this package does not guess them.
    """

    model_config = ConfigDict(extra="forbid")

    #: [A/cm^2] inclusive lower bound.
    i_min: float = Field(ge=0.0)
    #: [A/cm^2] exclusive upper bound; ``None`` means "to the end of the curve".
    i_max: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> LossRegion:
        if self.i_max is not None and self.i_max <= self.i_min:
            raise ValueError(
                f"i_max ({self.i_max}) must be above i_min ({self.i_min})"
            )
        return self

    def mask(self, current_density: Any) -> Any:
        """Boolean mask selecting the points of ``current_density`` in this window."""
        import numpy as np

        i = np.asarray(current_density, dtype=float)
        inside = i >= self.i_min
        if self.i_max is not None:
            inside &= i < self.i_max
        return inside


class Reference(BaseModel):
    """Everything a case is compared against."""

    model_config = ConfigDict(extra="forbid")

    #: MATLAB MMM1D profiles, for verification.
    matlab_profiles: ReferenceProfiles | None = None
    #: MATLAB MMM1D polarization curve, for verification.
    matlab_polarization: ReferenceCurve | None = None
    #: Published experimental polarization curves, for benchmarking.
    experimental: list[ReferenceCurve] = Field(default_factory=list)


class Case(BaseModel):
    """One verification or benchmarking case, as loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    #: Adapter to run, as registered by
    #: :func:`pemfc1d_benchmark.adapters.base.registry`.
    model: str = "pemfc1d"
    #: [V] cell voltages to solve, high to low -- the sweep runs by continuation
    #: from the previous solution, so the order matters.
    voltages: list[float] = Field(min_length=1)
    #: [V] voltages at which spatial profiles are compared. Must be a subset of
    #: ``voltages``; empty means profiles are not compared for this case.
    profile_voltages: list[float] = Field(default_factory=list)
    #: Overrides on the model's own parameter set, by parameter name. Names are
    #: checked by the adapter, which owns the parameter set they refer to.
    params: dict[str, Any] = Field(default_factory=dict)
    solver: SolverSettings = Field(default_factory=SolverSettings)
    reference: Reference = Field(default_factory=Reference)
    #: Profile tolerance applied to every variable without its own entry.
    tolerance: Tolerance = Tolerance(rtol=1e-3, atol=1e-8)
    #: Per-variable tolerance overrides, by variable name.
    tolerance_overrides: dict[str, Tolerance] = Field(default_factory=dict)
    #: Loss regions for per-region benchmarking metrics, by region name.
    loss_regions: dict[str, LossRegion] = Field(default_factory=dict)

    #: Where the file came from. Set by :func:`load_case`, not written in YAML.
    source_path: Path | None = Field(default=None, exclude=True)

    @field_validator("voltages")
    @classmethod
    def _descending_and_distinct(cls, voltages: list[float]) -> list[float]:
        if len(set(voltages)) != len(voltages):
            raise ValueError("voltages contains duplicates")
        if any(b >= a for a, b in zip(voltages, voltages[1:], strict=False)):
            raise ValueError(
                "voltages must be strictly descending: the sweep continues from "
                "each solution into the next, so it has to start at low current"
            )
        return voltages

    @field_validator("tolerance_overrides")
    @classmethod
    def _known_override_variables(
        cls, overrides: dict[str, Tolerance]
    ) -> dict[str, Tolerance]:
        unknown = sorted(set(overrides) - set(VARIABLE_UNITS))
        if unknown:
            raise ValueError(
                f"tolerance overrides name unknown variables {unknown}; "
                f"known: {sorted(VARIABLE_UNITS)}"
            )
        return overrides

    @model_validator(mode="after")
    def _profile_voltages_are_solved(self) -> Case:
        extra = sorted(set(self.profile_voltages) - set(self.voltages))
        if extra:
            raise ValueError(
                f"profile_voltages {extra} are not in voltages, so they would "
                f"never be solved"
            )
        return self

    def tolerance_for(self, variable: str) -> Tolerance:
        """The tolerance applying to ``variable``: its override, else the default."""
        return self.tolerance_overrides.get(variable, self.tolerance)

    def resolve(self, path: Path) -> Path:
        """``path`` as an absolute path, relative to the repository root.

        Reference paths are written in case files as repository-relative, so
        that a case reads the same on my machine and in the thesis appendix.
        They are resolved against the root -- the directory holding the case
        file's parent -- rather than the working directory.
        """
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    @property
    def root(self) -> Path:
        """The repository root this case's relative paths are resolved against."""
        if self.source_path is None:
            return Path.cwd()
        return self.source_path.resolve().parent.parent

    def reference_paths(self) -> Iterator[Path]:
        """Every reference file this case names, resolved."""
        reference = self.reference
        for entry in (reference.matlab_profiles, reference.matlab_polarization):
            if entry is not None:
                yield self.resolve(entry.path)
        for curve in reference.experimental:
            yield self.resolve(curve.path)

    def missing_reference_paths(self) -> list[Path]:
        """Reference files the case names that are not on disk yet.

        Reported rather than raised: a case whose MATLAB export has not been
        produced yet is still runnable, it just cannot be verified.
        """
        return [path for path in self.reference_paths() if not path.is_file()]


def load_case(path: str | Path) -> Case:
    """Load and validate one case file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"no case file at {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a case file must be a YAML mapping")
    return Case.model_validate({**raw, "source_path": path})


def load_cases(paths: Iterator[str | Path] | list[str | Path]) -> list[Case]:
    """Load several case files, in the order given."""
    return [load_case(path) for path in paths]


def discover_cases(directory: str | Path = "cases") -> list[Case]:
    """Every ``*.yaml`` case in ``directory``, by file name."""
    directory = Path(directory)
    return [load_case(path) for path in sorted(directory.glob("*.yaml"))]
