"""Running cases through adapters, and writing the results to disk.

Every run writes a self-describing directory under ``results/``: the
polarization curve, the profiles, and a ``metadata.json`` recording which
version of the model library produced them and when. That metadata is the point
of this module. A figure in the thesis has to be traceable to the solver
revision that drew it, and the only reliable moment to record the version is the
moment of the run.

Nothing here is git-tracked. A results directory is reproducible from a case
file plus the recorded library version, and that pair is what the repository
keeps.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pemfc1d_benchmark.adapters.base import (
    LAYERS,
    VARIABLE_UNITS,
    ModelAdapter,
    ModelResult,
    Profile,
    RunMetadata,
    get_adapter,
)
from pemfc1d_benchmark.cases import Case

#: File names inside a run directory. Fixed, so that a reader -- or the report
#: command -- can find them without being told.
METADATA_FILE = "metadata.json"
POLARIZATION_FILE = "polarization.csv"
PROFILES_FILE = "profiles.csv"

#: Columns of ``profiles.csv``. Deliberately the same long format the stored
#: MATLAB reference tables use, so that a model export and a reference export
#: are the same shape and can be concatenated for plotting.
PROFILE_COLUMNS: tuple[str, ...] = (
    "layer", "variable", "x_um", "voltage", "value", "unit",
)


def run_directory_name(timestamp: str | None = None) -> str:
    """``run_20260924_143512`` -- a sortable directory name for one run."""
    moment = (
        datetime.fromisoformat(timestamp) if timestamp
        else datetime.now(UTC)
    )
    return f"run_{moment.strftime('%Y%m%d_%H%M%S')}"


def polarization_frame(result: ModelResult) -> pd.DataFrame:
    """The polarization curve as a table, in ascending current density."""
    frame = pd.DataFrame({
        "current_density_A_cm2": result.current_density,
        "voltage_V": result.voltage,
        "power_density_W_cm2": result.power_density,
    })
    return frame.sort_values("current_density_A_cm2", ignore_index=True)


def _profile_rows(profile: Profile) -> pd.DataFrame:
    return pd.DataFrame({
        "layer": profile.layer,
        "variable": profile.variable,
        "x_um": profile.x_um,
        "voltage": profile.voltage,
        "value": profile.values,
        "unit": profile.unit,
    })


def profiles_frame(result: ModelResult) -> pd.DataFrame:
    """Every profile as one long table, ordered anode to cathode.

    The ordering is by layer position, then by the canonical variable order,
    then by voltage and position -- so the file reads in the same order the
    figures are drawn in, and two runs of the same case produce diffable files.
    """
    if not result.profiles:
        return pd.DataFrame(columns=list(PROFILE_COLUMNS))

    frame = pd.concat(
        [_profile_rows(profile) for profile in result.iter_profiles()],
        ignore_index=True,
    )
    frame["_layer_order"] = frame["layer"].map(
        {layer: index for index, layer in enumerate(LAYERS)}
    )
    frame["_variable_order"] = frame["variable"].map(
        {name: index for index, name in enumerate(VARIABLE_UNITS)}
    )
    frame = frame.sort_values(
        ["_layer_order", "_variable_order", "voltage", "x_um"],
        ascending=[True, True, False, True], ignore_index=True,
    )
    return frame.drop(columns=["_layer_order", "_variable_order"])


@dataclass(frozen=True)
class StoredRun:
    """A run directory on disk, read back.

    Exists so that ``report`` and ``verify`` can work from a previous run
    instead of re-solving: a sweep is minutes of CPU, and re-running it to
    redraw a figure would also mean the figure no longer matches the numbers
    that were checked.
    """

    directory: Path
    metadata: dict[str, object]
    polarization: pd.DataFrame
    profiles: pd.DataFrame

    @property
    def library_version(self) -> str:
        return str(self.metadata.get("library_version", "unknown"))

    @property
    def timestamp(self) -> str:
        return str(self.metadata.get("timestamp", "unknown"))

    def as_result(self) -> ModelResult:
        """Rebuild a :class:`ModelResult` from the stored tables.

        The reconstruction is lossy in one harmless way: the profiles come back
        on the grid they were written on, which is the grid they were sampled
        on, so a comparison made from a stored run and one made in memory give
        the same numbers.
        """
        metadata = RunMetadata(
            model=str(self.metadata.get("model", "unknown")),
            case_name=str(self.metadata.get("case_name", "unknown")),
            library_version=self.library_version,
            timestamp=self.timestamp,
            case_path=self.metadata.get("case_path"),  # type: ignore[arg-type]
            solver_settings=self.metadata.get("solver_settings", {}),  # type: ignore[arg-type]
            extra=self.metadata.get("extra", {}),  # type: ignore[arg-type]
        )
        profiles: dict[tuple[str, str], list[Profile]] = {}
        if not self.profiles.empty:
            grouped = self.profiles.groupby(
                ["variable", "layer", "voltage"], sort=False
            )
            for (variable, layer, voltage), rows in grouped:
                rows = rows.sort_values("x_um")
                profiles.setdefault((str(variable), str(layer)), []).append(Profile(
                    variable=str(variable),
                    layer=str(layer),
                    voltage=float(voltage),
                    x_um=rows["x_um"].to_numpy(dtype=float),
                    values=rows["value"].to_numpy(dtype=float),
                    unit=str(rows["unit"].iloc[0]),
                ))
        return ModelResult(
            metadata=metadata,
            current_density=self.polarization["current_density_A_cm2"].to_numpy(
                dtype=float),
            voltage=self.polarization["voltage_V"].to_numpy(dtype=float),
            profiles={key: tuple(group) for key, group in profiles.items()},
        )


def save_result(result: ModelResult, directory: str | Path) -> Path:
    """Write one result into ``directory``, creating it if need be."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    polarization_frame(result).to_csv(directory / POLARIZATION_FILE, index=False)
    profiles_frame(result).to_csv(directory / PROFILES_FILE, index=False)
    (directory / METADATA_FILE).write_text(
        json.dumps(result.metadata.to_dict(), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return directory


def _json_default(value: object) -> object:
    """Make numpy scalars and arrays serialisable in the metadata file."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into metadata")


def load_run(directory: str | Path) -> StoredRun:
    """Read a run directory back off disk."""
    directory = Path(directory)
    metadata_path = directory / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{directory} has no {METADATA_FILE}")

    profiles_path = directory / PROFILES_FILE
    profiles = (
        pd.read_csv(profiles_path) if profiles_path.is_file()
        else pd.DataFrame(columns=list(PROFILE_COLUMNS))
    )
    return StoredRun(
        directory=directory,
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        polarization=pd.read_csv(directory / POLARIZATION_FILE),
        profiles=profiles,
    )


def latest_run(case: Case, results_root: str | Path = "results") -> StoredRun:
    """The most recent stored run of ``case``, by directory name.

    Directory names are UTC timestamps in a sortable format, so the newest is
    the last in lexicographic order and no file modification times are consulted.
    """
    base = Path(results_root) / case.name
    candidates = sorted(
        path for path in base.glob("run_*") if (path / METADATA_FILE).is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no completed runs of case {case.name!r} under {base}; run it first"
        )
    return load_run(candidates[-1])


def run_case(case: Case, adapter: ModelAdapter | None = None,
             results_root: str | Path | None = "results",
             ) -> tuple[ModelResult, Path | None]:
    """Solve one case and, unless ``results_root`` is ``None``, save it.

    Returns the result and the directory it was written to. The directory is
    ``<results_root>/<case name>/run_<UTC timestamp>/``: one directory per run
    rather than per case, because comparing two runs of the same case at
    different solver tolerances is the mesh-convergence check.
    """
    adapter = adapter if adapter is not None else get_adapter(case.model)
    result = adapter.run(case)
    if results_root is None:
        return result, None
    directory = (
        Path(results_root) / case.name / run_directory_name(result.metadata.timestamp)
    )
    return result, save_result(result, directory)


def run_cases(cases: Iterable[Case], results_root: str | Path | None = "results"
              ) -> list[tuple[Case, ModelResult, Path | None]]:
    """Run several cases in order, reusing one adapter instance per model."""
    adapters: dict[str, ModelAdapter] = {}
    finished: list[tuple[Case, ModelResult, Path | None]] = []
    for case in cases:
        if case.model not in adapters:
            adapters[case.model] = get_adapter(case.model)
        result, directory = run_case(case, adapters[case.model], results_root)
        finished.append((case, result, directory))
    return finished
