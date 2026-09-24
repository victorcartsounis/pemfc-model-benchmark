"""Case file loading and validation.

These tests are about the schema catching mistakes, not about physics. A case
file is the only place operating conditions are written down, so a typo in one
is a wrong figure in the thesis — and every check here corresponds to a mistake
that would otherwise be silent.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pemfc1d_benchmark.cases import (
    Case,
    LossRegion,
    Tolerance,
    discover_cases,
    load_case,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CASE = REPOSITORY_ROOT / "cases" / "example_placeholder.yaml"

MINIMAL = {
    "name": "unit_test_case",
    "voltages": [0.9, 0.8, 0.7],
}


def write_case(directory: Path, **overrides) -> Path:
    """Write a minimal valid case with ``overrides`` applied, and return its path."""
    # Nested a level down, so that `Case.root` — the case file's parent's parent —
    # is the temporary directory rather than something outside it.
    cases = directory / "cases"
    cases.mkdir(exist_ok=True)
    path = cases / "case.yaml"
    path.write_text(yaml.safe_dump({**MINIMAL, **overrides}), encoding="utf-8")
    return path


class TestTheExampleCaseFile:
    """The committed example must stay loadable: it is the schema's documentation."""

    def test_it_loads(self):
        case = load_case(EXAMPLE_CASE)
        assert case.name == "example_placeholder"
        assert case.model == "pemfc1d"

    def test_its_profile_voltages_are_a_subset_of_its_sweep(self):
        case = load_case(EXAMPLE_CASE)
        assert set(case.profile_voltages) <= set(case.voltages)

    def test_its_placeholder_reference_files_are_reported_as_missing(self):
        """The example names files that do not exist, and says so rather than
        crashing."""
        case = load_case(EXAMPLE_CASE)
        assert case.missing_reference_paths()

    def test_discover_cases_finds_it(self):
        names = [case.name for case in discover_cases(REPOSITORY_ROOT / "cases")]
        assert "example_placeholder" in names


class TestVoltageValidation:
    def test_a_descending_sweep_is_accepted(self, tmp_path):
        case = load_case(write_case(tmp_path))
        assert case.voltages == [0.9, 0.8, 0.7]

    def test_an_ascending_sweep_is_rejected(self, tmp_path):
        """Continuation starts from the previous solution, so the order is physical."""
        with pytest.raises(ValidationError, match="strictly descending"):
            load_case(write_case(tmp_path, voltages=[0.7, 0.8, 0.9]))

    def test_duplicate_voltages_are_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="duplicates"):
            load_case(write_case(tmp_path, voltages=[0.9, 0.9, 0.8]))

    def test_an_empty_sweep_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            load_case(write_case(tmp_path, voltages=[]))

    def test_profile_voltages_must_be_solved(self, tmp_path):
        """A profile voltage outside the sweep would never produce a profile."""
        with pytest.raises(ValidationError, match="not in voltages"):
            load_case(write_case(tmp_path, profile_voltages=[0.75]))

    def test_profile_voltages_inside_the_sweep_are_accepted(self, tmp_path):
        case = load_case(write_case(tmp_path, profile_voltages=[0.8]))
        assert case.profile_voltages == [0.8]


class TestUnknownFieldsAndNames:
    def test_an_unknown_top_level_field_is_rejected(self, tmp_path):
        """`extra="forbid"` turns a misspelled key into an error, not a default."""
        with pytest.raises(ValidationError):
            load_case(write_case(tmp_path, voltagess=[0.9]))

    def test_an_unknown_variable_in_a_tolerance_override_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="unknown variables"):
            load_case(write_case(
                tmp_path,
                tolerance_overrides={"phi_x": {"rtol": 1e-3, "atol": 1e-8}},
            ))

    def test_an_unknown_variable_in_a_profile_selection_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="unknown state variables"):
            load_case(write_case(tmp_path, reference={
                "matlab_profiles": {
                    "path": "reference/matlab/x.csv",
                    "source": "test",
                    "variables": ["not_a_variable"],
                },
            }))

    def test_an_unknown_layer_in_a_profile_selection_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="unknown layers"):
            load_case(write_case(tmp_path, reference={
                "matlab_profiles": {
                    "path": "reference/matlab/x.csv",
                    "source": "test",
                    "layers": ["MPL"],
                },
            }))

    def test_a_variable_asked_for_only_where_it_cannot_exist_is_rejected(
            self, tmp_path):
        """Oxygen has no meaning in the anode GDL; asking for it there is a typo."""
        with pytest.raises(ValidationError, match="not defined on any of"):
            load_case(write_case(tmp_path, reference={
                "matlab_profiles": {
                    "path": "reference/matlab/x.csv",
                    "source": "test",
                    "variables": ["wO2"],
                    "layers": ["AGDL"],
                },
            }))

    def test_a_variable_and_layer_pair_that_does_overlap_is_accepted(self, tmp_path):
        case = load_case(write_case(tmp_path, reference={
            "matlab_profiles": {
                "path": "reference/matlab/x.csv",
                "source": "test",
                "variables": ["wO2"],
                "layers": ["CCL", "CGDL"],
            },
        }))
        assert case.reference.matlab_profiles is not None

    def test_an_experimental_curve_needs_a_non_empty_source(self, tmp_path):
        """A measured curve with no citation cannot be used in the thesis."""
        with pytest.raises(ValidationError):
            load_case(write_case(tmp_path, reference={
                "experimental": [{"path": "reference/experimental/x.csv",
                                  "source": ""}],
            }))


class TestTolerances:
    def test_the_default_applies_where_there_is_no_override(self):
        case = Case(name="t", voltages=[0.9],
                    tolerance=Tolerance(rtol=1e-3, atol=1e-8))
        assert case.tolerance_for("lambda").rtol == pytest.approx(1e-3)

    def test_an_override_wins_for_its_own_variable_only(self):
        case = Case(name="t", voltages=[0.9],
                    tolerance=Tolerance(rtol=1e-3, atol=1e-8),
                    tolerance_overrides={"T": Tolerance(rtol=1e-4, atol=1e-3)})
        assert case.tolerance_for("T").atol == pytest.approx(1e-3)
        assert case.tolerance_for("wH2O").atol == pytest.approx(1e-8)

    def test_negative_tolerances_are_rejected(self):
        with pytest.raises(ValidationError):
            Tolerance(rtol=-1e-3, atol=0.0)


class TestLossRegions:
    def test_a_window_masks_lower_inclusive_and_upper_exclusive(self):
        region = LossRegion(i_min=0.1, i_max=1.0)
        mask = region.mask([0.05, 0.1, 0.5, 1.0, 1.5])
        assert list(mask) == [False, True, True, False, False]

    def test_an_open_ended_window_runs_to_the_end_of_the_curve(self):
        region = LossRegion(i_min=1.0)
        assert list(region.mask([0.9, 1.0, 5.0])) == [False, True, True]

    def test_an_inverted_window_is_rejected(self):
        with pytest.raises(ValidationError, match="must be above"):
            LossRegion(i_min=1.0, i_max=0.5)


class TestPathResolution:
    def test_relative_reference_paths_resolve_against_the_root(self, tmp_path):
        """Reference paths are repository-relative, not relative to the
        working directory a run happened to start in."""
        path = write_case(tmp_path, reference={
            "matlab_polarization": {"path": "reference/matlab/x.csv",
                                    "source": "test"},
        })
        case = load_case(path)
        assert case.root == tmp_path.resolve()
        assert list(case.reference_paths()) == [
            (tmp_path / "reference" / "matlab" / "x.csv").resolve()
        ]

    def test_an_existing_reference_file_is_not_reported_missing(self, tmp_path):
        reference = tmp_path / "reference" / "matlab"
        reference.mkdir(parents=True)
        (reference / "x.csv").write_text("current_density_A_cm2,voltage_V\n0.1,0.8\n")
        case = load_case(write_case(tmp_path, reference={
            "matlab_polarization": {"path": "reference/matlab/x.csv",
                                    "source": "test"},
        }))
        assert case.missing_reference_paths() == []

    def test_a_missing_case_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no case file at"):
            load_case(tmp_path / "nope.yaml")

    def test_a_case_file_that_is_not_a_mapping_is_rejected(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_case(path)
