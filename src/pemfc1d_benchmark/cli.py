"""Command line interface: ``pemfc1d-bench run | verify | report``.

The three commands are the three things I do with this repository. ``run``
solves a case and stores the answer with its provenance; ``verify`` checks a
stored answer against the MATLAB reference; ``report`` draws the figures. They
are separate because a sweep costs minutes of CPU and a figure should be
redrawn, and re-checked, without re-solving -- and because a figure that was
redrawn from a fresh run no longer illustrates the numbers that were verified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pemfc1d_benchmark import __version__
from pemfc1d_benchmark.adapters.base import registry
from pemfc1d_benchmark.cases import Case, load_case
from pemfc1d_benchmark.runner import latest_run, load_run, run_case
from pemfc1d_benchmark.verification import verify_case

app = typer.Typer(
    add_completion=False,
    help="Verify and benchmark my 1D PEM fuel cell model.",
    no_args_is_help=True,
)

CasePath = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True,
                   help="Case file to use, e.g. cases/example_placeholder.yaml."),
]
ResultsRoot = Annotated[
    Path,
    typer.Option("--results", help="Root directory that run directories go under."),
]
RunDirectory = Annotated[
    Path | None,
    typer.Option("--run", exists=True, file_okay=False,
                 help="Use this stored run instead of the case's latest."),
]


def _echo_case(case: Case) -> None:
    typer.echo(f"case:    {case.name}   [{case.model}]")
    if case.description:
        typer.echo(f"         {case.description}")
    typer.echo(
        f"sweep:   {len(case.voltages)} voltages, "
        f"{max(case.voltages):.3f} V down to {min(case.voltages):.3f} V"
    )
    missing = case.missing_reference_paths()
    if missing:
        typer.secho(
            "warning: reference files named by this case are not on disk yet:",
            fg=typer.colors.YELLOW,
        )
        for path in missing:
            typer.secho(f"         {path}", fg=typer.colors.YELLOW)


def _resolve_run(case: Case, run: Path | None, results_root: Path):
    return load_run(run) if run is not None else latest_run(case, results_root)


@app.command()
def run(
    case_file: CasePath,
    results_root: ResultsRoot = Path("results"),
    no_save: Annotated[
        bool, typer.Option("--no-save", help="Solve but write nothing to disk.")
    ] = False,
) -> None:
    """Solve a case and store the result with its library version and timestamp."""
    case = load_case(case_file)
    _echo_case(case)

    result, directory = run_case(
        case, results_root=None if no_save else results_root
    )
    metadata = result.metadata

    typer.echo(
        f"solved:  {result.voltage.size} of {len(case.voltages)} voltages "
        f"with {metadata.model} (library {metadata.library_version})"
    )
    if result.current_density.size:
        typer.echo(
            f"current: {result.current_density.min():.4f} to "
            f"{result.current_density.max():.4f} A/cm^2"
        )
    if result.profiles:
        typer.echo(
            f"profiles: {len(result.profiles)} (variable, layer) pairs at "
            f"{len(result.sampled_voltages())} voltages"
        )
    unsolved = metadata.extra.get("unsolved_voltages") or []
    if unsolved:
        typer.secho(
            f"warning: the sweep did not reach {unsolved}", fg=typer.colors.YELLOW
        )
    if directory is not None:
        typer.secho(f"written: {directory}", fg=typer.colors.GREEN)


@app.command()
def verify(
    case_file: CasePath,
    results_root: ResultsRoot = Path("results"),
    run_dir: RunDirectory = None,
    solve: Annotated[
        bool,
        typer.Option("--solve", help="Solve now instead of reading a stored run."),
    ] = False,
) -> None:
    """Compare a case's result against the stored MATLAB MMM1D reference.

    Exits non-zero when a comparison exceeds its tolerance, or when nothing
    could be compared at all -- an empty verification is a missing reference
    file, not a verified model, and should fail a check the same way.
    """
    case = load_case(case_file)
    result = (
        run_case(case, results_root=None)[0] if solve
        else _resolve_run(case, run_dir, results_root).as_result()
    )

    report = verify_case(case, result)
    typer.echo(
        f"case:    {report.case_name}   [{report.model}, "
        f"library {report.library_version}]"
    )

    frame = report.to_frame()
    if not frame.empty:
        columns = ["variable", "layer", "voltage_V", "rmse", "max_abs_error",
                   "worst_ratio", "worst_x_um", "n_failing", "passed"]
        typer.echo(frame[columns].to_string(index=False,
                                            float_format=lambda v: f"{v:.3e}"))

    if report.polarization is not None:
        summary = report.polarization
        typer.echo(
            f"polarization vs reference: RMSE {summary.rmse * 1e3:.3f} mV, "
            f"max {summary.max_abs_error * 1e3:.3f} mV, "
            f"bias {summary.bias * 1e3:+.3f} mV, over {summary.n_points} points"
        )

    for what, why in report.skipped:
        typer.secho(f"skipped: {what} -- {why}", fg=typer.colors.YELLOW)

    if report.passed:
        typer.secho(
            f"PASS: {len(report.profiles)} profile comparisons within tolerance",
            fg=typer.colors.GREEN,
        )
        return
    typer.secho(
        f"FAIL: {len(report.failures)} of {len(report.profiles)} profile "
        f"comparisons exceed tolerance" if report.profiles
        else "FAIL: nothing could be compared; see the skipped entries above",
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1)


@app.command()
def report(
    case_file: CasePath,
    results_root: ResultsRoot = Path("results"),
    run_dir: RunDirectory = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Where the figures go; default is the run "
                                      "directory's figures/ subdirectory."),
    ] = None,
) -> None:
    """Draw the figures for a stored run: curves, residuals and profile overlays."""
    # Imported here so that `run` and `verify` do not pay for matplotlib.
    from pemfc1d_benchmark.report import write_report

    case = load_case(case_file)
    stored = _resolve_run(case, run_dir, results_root)
    result = stored.as_result()

    verification = verify_case(case, result) if case.reference.matlab_profiles else None
    directory = output if output is not None else stored.directory / "figures"
    written = write_report(result, case, directory, verification)

    typer.echo(f"run:     {stored.directory} (library {stored.library_version})")
    for path in written:
        typer.echo(f"figure:  {path}")
    typer.secho(f"written: {len(written)} figures into {directory}",
                fg=typer.colors.GREEN)


@app.command("list-models")
def list_models() -> None:
    """List the registered model adapters."""
    for name, adapter in sorted(registry().items()):
        typer.echo(f"{name}\t{adapter.__module__}.{adapter.__qualname__}")


@app.command("show-case")
def show_case(case_file: CasePath) -> None:
    """Validate a case file and print it back as resolved JSON."""
    case = load_case(case_file)
    typer.echo(json.dumps(case.model_dump(mode="json"), indent=2))
    missing = case.missing_reference_paths()
    if missing:
        typer.secho(f"warning: {len(missing)} reference file(s) missing",
                    fg=typer.colors.YELLOW)


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", help="Print this harness's version and exit.")
    ] = False,
) -> None:
    if version:
        typer.echo(f"pemfc1d-benchmark {__version__}")
        raise typer.Exit()


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
