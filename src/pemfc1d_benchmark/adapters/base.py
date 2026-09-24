"""The standardized result object, and the adapter interface producing it.

Every model this repository compares -- my own 1D solver, and later a simpler
0D one -- is reached through a :class:`ModelAdapter`. The adapter's whole job is
to translate: it takes a :class:`~pemfc1d_benchmark.cases.Case` in this
repository's vocabulary, calls the model however that model wants to be called,
and returns a :class:`ModelResult` whose units and names are fixed here. The
comparison code downstream then never has to know which model produced a curve.

Units are fixed at this boundary, once, so that a metric computed on two
results is meaningful:

===================  ==========================================
current density      A/cm^2
cell voltage         V
profile position     um, measured from the anode GDL / gas channel face
profile values       stated per variable by :data:`VARIABLE_UNITS`
===================  ==========================================
"""
from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pemfc1d_benchmark.cases import Case

#: The five MEA layers, anode to cathode. These strings are the layer names
#: used in every case file, reference CSV and result CSV in this repository.
LAYERS: tuple[str, ...] = ("AGDL", "ACL", "PEM", "CCL", "CGDL")

#: The eight state variables, with the unit each one is reported in. The names
#: match the column values my solver's own CSV export writes, so a reference
#: table and a model table can be compared without a renaming step.
VARIABLE_UNITS: dict[str, str] = {
    "phi_e": "V",       # electron potential
    "phi_p": "V",       # proton potential
    "T": "K",           # temperature
    "lambda": "-",      # dissolved water content
    "wH2O": "-",        # water vapour mass fraction
    "wO2": "-",         # oxygen mass fraction
    "s": "-",           # liquid water saturation
    "Pgas": "Pa",       # gas pressure
}

#: Layers each variable is physically defined on. A comparison asked for a
#: variable outside these layers is a bug in a case file, not a disagreement
#: between two models, so the loaders check against this table.
VARIABLE_LAYERS: dict[str, tuple[str, ...]] = {
    "phi_e": ("AGDL", "ACL", "CCL", "CGDL"),
    "phi_p": ("ACL", "PEM", "CCL"),
    "T": LAYERS,
    "lambda": ("ACL", "PEM", "CCL"),
    "wH2O": ("AGDL", "ACL", "CCL", "CGDL"),
    "wO2": ("CCL", "CGDL"),
    "s": ("CCL", "CGDL"),
    "Pgas": ("AGDL", "ACL", "CCL", "CGDL"),
}


def utc_timestamp() -> str:
    """Now, as an ISO 8601 UTC string -- what goes into run metadata."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunMetadata:
    """Provenance of one model run.

    Recorded with every result and written beside every output file. The
    version and the timestamp are the point: a curve in the thesis has to be
    traceable to the solver revision that drew it.
    """

    model: str                     #: adapter name, e.g. ``"pemfc1d"``
    case_name: str                 #: ``Case.name`` the run came from
    library_version: str           #: version of the model library that ran
    timestamp: str = field(default_factory=utc_timestamp)  #: ISO 8601 UTC
    case_path: str | None = None   #: case file the run came from, if from disk
    solver_settings: Mapping[str, object] = field(default_factory=dict)
    #: Anything else worth recording, e.g. a git revision or convergence flags.
    extra: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable view, for the sidecar metadata file."""
        return {
            "model": self.model,
            "case_name": self.case_name,
            "library_version": self.library_version,
            "timestamp": self.timestamp,
            "case_path": self.case_path,
            "solver_settings": dict(self.solver_settings),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Profile:
    """One variable's spatial profile, in one layer, at one operating point.

    A profile is confined to a single layer on purpose. The eight quantities
    are discontinuous at the four internal interfaces -- each interface
    position is the last point of one layer and the first of the next, and the
    solution genuinely holds two different values there. Keeping layers apart
    means nothing has to guess which of the two a point belongs to.
    """

    variable: str          #: key of :data:`VARIABLE_UNITS`
    layer: str             #: member of :data:`LAYERS`
    voltage: float         #: [V] cell voltage of this operating point
    x_um: np.ndarray       #: [um] positions, increasing, measured from the AGDL face
    values: np.ndarray     #: the variable, in its :data:`VARIABLE_UNITS` unit
    unit: str

    def __post_init__(self) -> None:
        if self.x_um.shape != self.values.shape:
            raise ValueError(
                f"{self.variable}/{self.layer}: {self.x_um.size} positions but "
                f"{self.values.size} values"
            )


#: How profiles are keyed on a result: ``(variable, layer)``.
ProfileKey = tuple[str, str]


@dataclass(frozen=True)
class ModelResult:
    """What every adapter returns: a polarization curve, and optional profiles.

    ``profiles`` is keyed by ``(variable, layer)``; each entry holds one
    :class:`Profile` per operating point that was sampled. A 0D model has no
    profiles at all and leaves the mapping empty, which is why the verification
    code treats it as optional rather than required.
    """

    metadata: RunMetadata
    current_density: np.ndarray   #: [A/cm^2], one entry per solved voltage
    voltage: np.ndarray           #: [V], same length and order
    profiles: Mapping[ProfileKey, tuple[Profile, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_density.shape != self.voltage.shape:
            raise ValueError(
                f"polarization curve has {self.current_density.size} current "
                f"densities but {self.voltage.size} voltages"
            )

    @property
    def power_density(self) -> np.ndarray:
        """[W/cm^2] cell power density along the polarization curve."""
        return self.current_density * self.voltage

    @property
    def variables(self) -> tuple[str, ...]:
        """Variables that have at least one profile, in :data:`VARIABLE_UNITS` order."""
        present = {variable for variable, _ in self.profiles}
        return tuple(name for name in VARIABLE_UNITS if name in present)

    def layers_of(self, variable: str) -> tuple[str, ...]:
        """Layers this result carries ``variable`` on, anode to cathode."""
        present = {layer for name, layer in self.profiles if name == variable}
        return tuple(layer for layer in LAYERS if layer in present)

    def profile(self, variable: str, layer: str, voltage: float,
                atol: float = 1e-9) -> Profile | None:
        """The profile of ``variable`` in ``layer`` at ``voltage``, if sampled.

        ``voltage`` is matched within ``atol`` rather than exactly: it comes
        from a case file or a reference table and has been through a text
        round trip on the way.
        """
        for profile in self.profiles.get((variable, layer), ()):
            if abs(profile.voltage - voltage) <= atol:
                return profile
        return None

    def iter_profiles(self) -> Iterator[Profile]:
        """Every profile, grouped by key."""
        for group in self.profiles.values():
            yield from group

    def sampled_voltages(self) -> tuple[float, ...]:
        """Cell voltages that profiles were sampled at, high to low."""
        voltages = {profile.voltage for profile in self.iter_profiles()}
        return tuple(sorted(voltages, reverse=True))

    def with_metadata(self, **changes: object) -> ModelResult:
        """A copy whose metadata has ``changes`` applied."""
        return replace(self, metadata=replace(self.metadata, **changes))


class ModelAdapter(abc.ABC):
    """Runs one model on a case and reports the answer in standard form.

    Subclasses implement :meth:`run` and :attr:`library_version`. Adding a
    model to this repository should mean writing one of these and nothing else
    -- see the README section on adding an adapter.
    """

    #: Short name of this adapter; the value a case file's ``model:`` field
    #: takes, and what appears in metadata and output paths.
    name: str = ""

    @property
    @abc.abstractmethod
    def library_version(self) -> str:
        """Version of the library this adapter wraps, for run metadata."""

    @abc.abstractmethod
    def run(self, case: Case) -> ModelResult:
        """Solve ``case`` and return the result in standardized units."""

    def _metadata(self, case: Case, solver_settings: Mapping[str, object],
                  **extra: object) -> RunMetadata:
        """Metadata for a run of ``case``, with the version and timestamp filled in."""
        return RunMetadata(
            model=self.name,
            case_name=case.name,
            library_version=self.library_version,
            case_path=str(case.source_path) if case.source_path else None,
            solver_settings=dict(solver_settings),
            extra=dict(extra),
        )


def registry() -> dict[str, type[ModelAdapter]]:
    """Adapter classes by name.

    Kept as a function rather than a module-level dict so that importing an
    adapter -- and with it the model library it wraps -- is deferred until
    something actually asks for one.
    """
    from pemfc1d_benchmark.adapters.pemfc1d import Pemfc1dAdapter

    adapters: Sequence[type[ModelAdapter]] = (Pemfc1dAdapter,)
    # TODO: add the Springer-type 0D adapter here once it exists.
    return {adapter.name: adapter for adapter in adapters}


def get_adapter(name: str) -> ModelAdapter:
    """Instantiate the adapter registered under ``name``."""
    available = registry()
    if name not in available:
        known = ", ".join(sorted(available)) or "none"
        raise KeyError(f"unknown model adapter {name!r}; registered: {known}")
    return available[name]()
