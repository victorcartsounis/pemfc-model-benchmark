"""Verification machinery, exercised on synthetic profiles.

No solver runs here. The model results are constructed by hand so that the
answer is known in advance: a profile that is exactly right must pass, one
displaced by a known amount must fail by a known margin, and a reference file
that contradicts the model's structure must be rejected outright. Whether my
solver actually agrees with MATLAB is a question for `pemfc1d-bench verify`
against real reference data, not for a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest

from pemfc1d_benchmark.adapters.base import ModelResult, Profile, RunMetadata
from pemfc1d_benchmark.cases import Case, Tolerance
from pemfc1d_benchmark.verification import (
    compare_profile,
    load_reference_curve,
    load_reference_profiles,
    verify_case,
)

PROFILE_HEADER = "layer,variable,x_um,voltage,value,unit\n"


def metadata(**overrides) -> RunMetadata:
    defaults = {
        "model": "test",
        "case_name": "synthetic",
        "library_version": "0.0.0-test",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    return RunMetadata(**{**defaults, **overrides})


def linear_profile(variable: str, layer: str, voltage: float,
                   offset: float = 0.0, n: int = 11) -> Profile:
    """A straight line from 1 to 2 across the layer, optionally shifted."""
    x_um = np.linspace(0.0, 10.0, n)
    return Profile(variable=variable, layer=layer, voltage=voltage, x_um=x_um,
                   values=np.linspace(1.0, 2.0, n) + offset, unit="-")


def result_with(*profiles: Profile, **metadata_overrides) -> ModelResult:
    collected: dict[tuple[str, str], list[Profile]] = {}
    for profile in profiles:
        collected.setdefault((profile.variable, profile.layer), []).append(profile)
    return ModelResult(
        metadata=metadata(**metadata_overrides),
        current_density=np.array([0.1, 0.5]),
        voltage=np.array([0.9, 0.7]),
        profiles={key: tuple(group) for key, group in collected.items()},
    )


def write_reference_profiles(directory, rows: str) -> None:
    path = directory / "reference" / "matlab"
    path.mkdir(parents=True, exist_ok=True)
    (path / "profiles.csv").write_text(PROFILE_HEADER + rows, encoding="utf-8")


def case_for(directory, **overrides) -> Case:
    """A case rooted at ``directory``, so relative reference paths resolve there."""
    cases = directory / "cases"
    cases.mkdir(exist_ok=True)
    defaults = {
        "name": "synthetic",
        "voltages": [0.9, 0.7],
        "profile_voltages": [0.9],
        "source_path": cases / "case.yaml",
    }
    return Case(**{**defaults, **overrides})


class TestCompareProfile:
    tolerance = Tolerance(rtol=1e-3, atol=1e-8)

    def test_an_exact_match_passes(self):
        model = linear_profile("lambda", "PEM", 0.9)
        comparison = compare_profile(model, model.x_um, model.values, self.tolerance)
        assert comparison.passed
        assert comparison.n_failing == 0
        assert comparison.summary.rmse == pytest.approx(0.0)
        assert comparison.worst_exceedance < 0.0    # every point had margin to spare

    def test_a_deviation_inside_the_tolerance_passes(self):
        """Values near 1-2 allow about 1e-3 absolutely at rtol = 1e-3."""
        model = linear_profile("lambda", "PEM", 0.9, offset=5e-4)
        reference = linear_profile("lambda", "PEM", 0.9)
        comparison = compare_profile(model, reference.x_um, reference.values,
                                     self.tolerance)
        assert comparison.passed

    def test_a_deviation_outside_the_tolerance_fails_at_every_point(self):
        model = linear_profile("lambda", "PEM", 0.9, offset=0.1)
        reference = linear_profile("lambda", "PEM", 0.9)
        comparison = compare_profile(model, reference.x_um, reference.values,
                                     self.tolerance)
        assert not comparison.passed
        assert comparison.n_failing == reference.x_um.size
        assert comparison.summary.bias == pytest.approx(0.1)

    def test_the_exceedance_is_the_margin_by_which_the_worst_point_failed(self):
        """A uniform offset fails worst where the reference is *smallest*.

        The deviation is a constant 0.1 across the layer, but the allowance
        ``atol + rtol*|reference|`` grows with the reference, from 1e-3 at the
        left edge to 2e-3 at the right. So the worst point is the left edge,
        where the allowance is tightest, and the margin there is
        ``0.1 - 1e-3 - 1e-8``. This is the whole reason the exceedance is
        reported rather than the raw deviation: it says where the profile is
        worst relative to what was demanded of it, not merely where it is
        numerically farthest off.
        """
        model = linear_profile("lambda", "PEM", 0.9, offset=0.1)
        reference = linear_profile("lambda", "PEM", 0.9)
        comparison = compare_profile(model, reference.x_um, reference.values,
                                     self.tolerance)
        assert comparison.worst_exceedance == pytest.approx(0.1 - 1e-3 * 1.0 - 1e-8)
        assert comparison.worst_x_um == pytest.approx(0.0)

    def test_the_ratio_is_one_exactly_at_the_tolerance(self):
        """The ratio is what makes different variables comparable.

        A deviation sitting exactly on the allowance gives 1.0 whatever the
        variable's unit or magnitude, which is why the summary chart and the
        results table are ranked by it rather than by the raw exceedance.
        """
        x_um = np.array([0.0, 1.0])
        tolerance = Tolerance(rtol=1e-3, atol=0.0)
        reference = np.array([100.0, 100.0])
        allowed = tolerance.atol + tolerance.rtol * 100.0    # 0.1
        model = Profile("T", "PEM", 0.9, x_um, reference + allowed, "K")
        comparison = compare_profile(model, x_um, reference, tolerance)
        assert comparison.worst_ratio == pytest.approx(1.0)

    def test_the_ratio_ranks_across_variables_where_the_exceedance_cannot(self):
        """A large Pa deviation can be healthier than a small [-] deviation."""
        x_um = np.array([0.0, 1.0])
        pressure = Profile("Pgas", "AGDL", 0.9, x_um,
                           np.array([150010.0, 150010.0]), "Pa")
        pressure_comparison = compare_profile(
            pressure, x_um, np.array([150000.0, 150000.0]),
            Tolerance(rtol=1e-3, atol=0.0))          # allows 150 Pa
        saturation = Profile("s", "CCL", 0.9, x_um, np.array([0.13, 0.13]), "-")
        saturation_comparison = compare_profile(
            saturation, x_um, np.array([0.12, 0.12]),
            Tolerance(rtol=1e-3, atol=0.0))          # allows 1.2e-4

        # the pressure is off by 10 Pa and the saturation by only 0.01, yet it
        # is the saturation that fails
        assert (pressure_comparison.worst_exceedance
                < saturation_comparison.worst_exceedance)
        assert pressure_comparison.passed
        assert not saturation_comparison.passed
        assert pressure_comparison.worst_ratio < 1.0 < saturation_comparison.worst_ratio

    def test_an_exact_match_has_ratio_zero(self):
        model = linear_profile("lambda", "PEM", 0.9)
        comparison = compare_profile(model, model.x_um, model.values,
                                     self.tolerance)
        assert comparison.worst_ratio == pytest.approx(0.0)

    def test_a_zero_tolerance_is_not_a_division_by_zero(self):
        """A tolerance of exactly zero demands exactness; it must not raise."""
        x_um = np.array([0.0, 1.0])
        zero = Tolerance(rtol=0.0, atol=0.0)
        exact = Profile("T", "PEM", 0.9, x_um, np.array([1.0, 2.0]), "K")
        assert compare_profile(exact, x_um, np.array([1.0, 2.0]), zero).passed
        off = Profile("T", "PEM", 0.9, x_um, np.array([1.0, 2.1]), "K")
        comparison = compare_profile(off, x_um, np.array([1.0, 2.0]), zero)
        assert not comparison.passed
        assert np.isinf(comparison.worst_ratio)

    def test_the_absolute_floor_carries_a_variable_that_passes_through_zero(self):
        """At a reference of exactly zero, rtol allows nothing and atol must."""
        x_um = np.array([0.0, 1.0])
        model = Profile("phi_e", "AGDL", 0.9, x_um, np.array([1e-7, 1e-7]), "V")
        loose = compare_profile(model, x_um, np.zeros(2),
                                Tolerance(rtol=1e-3, atol=1e-6))
        strict = compare_profile(model, x_um, np.zeros(2),
                                 Tolerance(rtol=1e-3, atol=1e-9))
        assert loose.passed
        assert not strict.passed

    def test_the_model_is_interpolated_onto_the_reference_positions(self):
        """The two codes choose their own meshes, so the grids differ by design."""
        model = linear_profile("T", "PEM", 0.9, n=3)        # coarse
        reference = linear_profile("T", "PEM", 0.9, n=51)   # fine
        comparison = compare_profile(model, reference.x_um, reference.values,
                                     self.tolerance)
        assert comparison.summary.n_points == 51
        assert comparison.passed        # a straight line interpolates exactly

    def test_reference_points_outside_the_model_range_are_dropped(self):
        model = linear_profile("T", "PEM", 0.9)
        reference_x = np.array([-1.0, 5.0, 11.0])
        reference_values = np.interp(reference_x, model.x_um, model.values)
        comparison = compare_profile(model, reference_x, reference_values,
                                     self.tolerance)
        assert comparison.summary.n_points == 1

    def test_a_comparison_with_nothing_in_range_does_not_pass(self):
        model = linear_profile("T", "PEM", 0.9)
        comparison = compare_profile(model, np.array([100.0, 200.0]),
                                     np.array([1.0, 2.0]), self.tolerance)
        assert comparison.summary.is_empty
        assert not comparison.passed


class TestLoadReferenceProfiles:
    def test_a_well_formed_table_loads(self, tmp_path):
        write_reference_profiles(tmp_path, "PEM,lambda,0.0,0.9,5.0,-\n")
        frame = load_reference_profiles(
            tmp_path / "reference" / "matlab" / "profiles.csv")
        assert len(frame) == 1
        assert frame["value"].dtype == float

    def test_a_missing_column_is_named_in_the_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("layer,variable,x_um,value\nPEM,lambda,0.0,5.0\n")
        with pytest.raises(ValueError, match=r"missing required columns \['voltage'"):
            load_reference_profiles(path)

    def test_an_unknown_variable_is_rejected(self, tmp_path):
        write_reference_profiles(tmp_path, "PEM,phi_x,0.0,0.9,5.0,-\n")
        with pytest.raises(ValueError, match="unknown state variables"):
            load_reference_profiles(tmp_path / "reference" / "matlab" / "profiles.csv")

    def test_an_unknown_layer_is_rejected(self, tmp_path):
        write_reference_profiles(tmp_path, "MPL,lambda,0.0,0.9,5.0,-\n")
        with pytest.raises(ValueError, match="unknown layers"):
            load_reference_profiles(tmp_path / "reference" / "matlab" / "profiles.csv")

    def test_a_variable_in_a_layer_it_cannot_exist_in_is_rejected(self, tmp_path):
        """Oxygen in the anode GDL means the two codes solve different problems."""
        write_reference_profiles(tmp_path, "AGDL,wO2,0.0,0.9,0.2,-\n")
        with pytest.raises(ValueError, match="layers they are not defined on"):
            load_reference_profiles(tmp_path / "reference" / "matlab" / "profiles.csv")


class TestLoadReferenceCurve:
    def test_comment_lines_carry_provenance_and_are_skipped(self, tmp_path):
        path = tmp_path / "curve.csv"
        path.write_text("# MMM1D abc123, exported 2026-01-01\n"
                        "current_density_A_cm2,voltage_V\n0.1,0.85\n0.5,0.75\n")
        frame = load_reference_curve(path)
        assert len(frame) == 2
        assert frame["voltage_V"].iloc[0] == pytest.approx(0.85)

    def test_a_missing_column_is_rejected(self, tmp_path):
        path = tmp_path / "curve.csv"
        path.write_text("current_density_A_cm2\n0.1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_reference_curve(path)


class TestVerifyCase:
    def test_a_matching_model_passes_and_keeps_the_provenance(self, tmp_path):
        write_reference_profiles(tmp_path, "".join(
            f"PEM,lambda,{x:.1f},0.9,{value:.6f},-\n"
            for x, value in zip(np.linspace(0.0, 10.0, 11),
                                np.linspace(1.0, 2.0, 11), strict=True)
        ))
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data"},
        })
        report = verify_case(case, result_with(linear_profile("lambda", "PEM", 0.9)))
        assert report.passed
        assert len(report.profiles) == 1
        assert report.library_version == "0.0.0-test"

    def test_a_displaced_model_fails(self, tmp_path):
        write_reference_profiles(tmp_path, "".join(
            f"PEM,lambda,{x:.1f},0.9,{value:.6f},-\n"
            for x, value in zip(np.linspace(0.0, 10.0, 11),
                                np.linspace(1.0, 2.0, 11), strict=True)
        ))
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data"},
        })
        report = verify_case(
            case, result_with(linear_profile("lambda", "PEM", 0.9, offset=0.5)))
        assert not report.passed
        assert len(report.failures) == 1

    def test_only_the_selected_variables_and_layers_are_compared(self, tmp_path):
        write_reference_profiles(tmp_path,
                                 "PEM,lambda,0.0,0.9,1.0,-\n"
                                 "PEM,T,0.0,0.9,343.15,K\n")
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data",
                                "variables": ["lambda"]},
        })
        report = verify_case(case, result_with(
            linear_profile("lambda", "PEM", 0.9), linear_profile("T", "PEM", 0.9)))
        assert [c.variable for c in report.profiles] == ["lambda"]

    def test_a_voltage_outside_profile_voltages_is_not_compared(self, tmp_path):
        write_reference_profiles(tmp_path, "PEM,lambda,0.0,0.7,1.0,-\n")
        case = case_for(tmp_path, profile_voltages=[0.9], reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data"},
        })
        report = verify_case(case, result_with(linear_profile("lambda", "PEM", 0.9)))
        assert report.profiles == ()
        assert not report.passed

    def test_a_reference_the_model_cannot_match_is_recorded_as_skipped(
            self, tmp_path):
        write_reference_profiles(tmp_path, "CCL,wO2,0.0,0.9,0.15,-\n")
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data"},
        })
        report = verify_case(case, result_with(linear_profile("lambda", "PEM", 0.9)))
        assert any("wO2/CCL" in what for what, _ in report.skipped)
        assert not report.passed

    def test_a_missing_reference_file_is_skipped_rather_than_raised(self, tmp_path):
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/absent.csv",
                                "source": "not produced yet"},
        })
        report = verify_case(case, result_with(linear_profile("lambda", "PEM", 0.9)))
        assert any("not found" in why for _, why in report.skipped)
        assert not report.passed

    def test_a_case_with_no_reference_at_all_does_not_pass(self, tmp_path):
        """An empty verification is a missing reference, not a verified model."""
        report = verify_case(case_for(tmp_path),
                             result_with(linear_profile("lambda", "PEM", 0.9)))
        assert not report.passed
        assert len(report.skipped) == 2      # profiles and polarization

    def test_the_polarization_curve_is_compared_when_the_case_names_one(self, tmp_path):
        curves = tmp_path / "reference" / "matlab"
        curves.mkdir(parents=True, exist_ok=True)
        (curves / "curve.csv").write_text(
            "current_density_A_cm2,voltage_V\n0.2,0.85\n0.4,0.80\n")
        case = case_for(tmp_path, reference={
            "matlab_polarization": {"path": "reference/matlab/curve.csv",
                                    "source": "synthetic test data"},
        })
        result = ModelResult(metadata=metadata(),
                             current_density=np.array([0.1, 0.5]),
                             voltage=np.array([0.9, 0.75]))
        report = verify_case(case, result)
        assert report.polarization is not None
        assert report.polarization.n_points == 2

    def test_the_report_frame_puts_the_worst_comparison_first(self, tmp_path):
        write_reference_profiles(tmp_path,
                                 "PEM,lambda,0.0,0.9,1.0,-\n"
                                 "PEM,T,0.0,0.9,1.0,K\n")
        case = case_for(tmp_path, reference={
            "matlab_profiles": {"path": "reference/matlab/profiles.csv",
                                "source": "synthetic test data"},
        })
        report = verify_case(case, result_with(
            linear_profile("lambda", "PEM", 0.9, offset=1.0),   # badly wrong
            linear_profile("T", "PEM", 0.9),                    # exact
        ))
        frame = report.to_frame()
        assert frame["variable"].iloc[0] == "lambda"
