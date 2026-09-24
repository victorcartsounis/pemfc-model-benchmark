# pemfc1d-benchmark

This repository verifies and benchmarks my 1D PEM fuel cell model. It is part of
my master's thesis at CEFET-MG, supervised by Sidney Nicodemos da Silva, and it
is my own authorial work.

The solver itself is not here. It lives in
[1D-PEMFC-Model](https://github.com/victorcartsounis/1D-PEMFC-Model) as the
`pemfc_1d` package and is consumed here as a dependency. I keep the two apart
deliberately: the model repository answers *what the model computes*, and this
one answers *whether I should believe it*. Mixing the two would let me tune a
solver until it matched the figure I wanted, with no record that I had.

## Verification is not benchmarking

The whole repository is organised around a distinction I want to keep sharp in
the thesis, because the two ask different questions and a good answer to one
looks like a bad answer to the other.

**Verification** asks whether my Python implementation reproduces the MATLAB
MMM1D reference implementation when both are given identical parameters and
identical operating conditions. Both codes solve the same equations, so the only
admissible difference is numerical — what two different BVP solvers, on two
different meshes, at their stated tolerances, make of one problem. **The expected
answer is agreement within solver tolerance.** A structural disagreement — a
sign, a unit, a boundary condition, a constitutive relation — is a bug in my
code, and it is the job of `pemfc1d-bench verify` to find it before a figure
built on it reaches the thesis. Verification compares full profiles of the eight
state variables across all five layers, plus the polarization curve, and it
passes or fails against a tolerance.

**Benchmarking** asks how closely a model reproduces a real cell, measured in a
laboratory and published. **The expected answer is a discrepancy.** No model of
this kind matches a measurement exactly, and a benchmark that did would mean the
parameters had been fitted to the data rather than measured independently. So
benchmarking produces *descriptive* metrics — RMSE, MAPE, bias, maximum error,
and the same metrics again within the activation, ohmic and mass-transport
regions of the curve — and no pass/fail verdict. The interpretation belongs in
the thesis text, not in the harness. The per-region split matters because the
three regions fail for different physical reasons and a single RMSE over the
whole curve hides which mechanism a model gets wrong.

Both comparisons are driven from the same case files and the same adapters, so
the second model I add — a simple Springer-type 0D model is the plan — is
benchmarked against exactly the same experimental curves, on exactly the same
metrics, as the 1D one.

## Repository structure

```
pemfc1d-benchmark/
├── cases/                      operating conditions and parameter sets (YAML), one file per case
├── reference/
│   ├── matlab/                 stored MATLAB MMM1D outputs (CSV) + how they were generated
│   └── experimental/           published polarization curves (CSV) + citation per file
├── results/                    generated outputs; git-ignored
├── src/pemfc1d_benchmark/
│   ├── cases.py                load and validate case files (pydantic)
│   ├── adapters/
│   │   ├── base.py             ModelAdapter, and the standardized result object
│   │   └── pemfc1d.py          adapter wrapping pemfc_1d
│   ├── metrics.py              RMSE, MAPE, bias, max error, per-region errors
│   ├── verification.py         profile-by-profile comparison against MATLAB, with tolerances
│   ├── runner.py               runs cases, saves results with the library version recorded
│   ├── report.py               plots: polarization curves, residuals, profile overlays
│   └── cli.py                  the pemfc1d-bench commands
└── tests/
```

Two directories are committed that might look like outputs but are not.
`reference/` holds **inputs**: the fixed standards my code is checked against. A
verification result means nothing if the reference can change silently
underneath it, so those files are tracked and documented. `results/` is the
opposite — everything in it is reproducible from a case file plus a library
version, so none of it is tracked.

## Installing and running

Poetry, Python 3.11 or newer:

```bash
poetry install
```

`pemfc_1d` is installed as a **path dependency** in develop mode, pointing at
`../1D-PEMFC-Model`, so the two repositories can be developed side by side and
an edit to the solver is picked up here without a reinstall. This assumes the
two are checked out as siblings. Once a thesis run has to be reproducible, the
comment in `pyproject.toml` shows how to switch to a git dependency pinned to a
tag instead.

The three commands are the three things I do with this repository:

```bash
# solve a case and store the answer, with its provenance
poetry run pemfc1d-bench run cases/example_placeholder.yaml

# check a stored answer against the MATLAB reference; exits non-zero on failure
poetry run pemfc1d-bench verify cases/example_placeholder.yaml

# draw the figures for a stored run
poetry run pemfc1d-bench report cases/example_placeholder.yaml
```

They are separate because a sweep costs minutes of CPU. A figure should be
redrawn, and re-checked, without re-solving — and a figure redrawn from a fresh
run no longer illustrates the numbers that were verified.

Two more, for inspecting things:

```bash
poetry run pemfc1d-bench list-models                              # registered adapters
poetry run pemfc1d-bench show-case cases/example_placeholder.yaml # validate a case file
```

Checks:

```bash
poetry run ruff check .
poetry run pytest
```

### Every run records its provenance

`run` writes a directory under `results/<case>/run_<UTC timestamp>/` containing
`polarization.csv`, `profiles.csv` and `metadata.json`. The metadata records the
`pemfc_1d` version and git revision, the timestamp, the solver settings, whether
the sweep converged, and any voltages it failed to reach. This is not
bookkeeping for its own sake: a figure in the thesis has to be traceable to the
solver revision that drew it, and the only reliable moment to record the version
is the moment of the run.

`profiles.csv` is written in the same long format the stored MATLAB references
use, so a model export and a reference export are the same shape.

### The example case

`cases/example_placeholder.yaml` documents the schema. **Every value in it marked
PLACEHOLDER is there to show the format, not because it is the value to use** —
nothing in it has been checked against the reference publication, and it names
reference files that do not exist yet. Copy it to start a real case rather than
editing it in place.

A verification case should normally override **no** parameters at all: the
`pemfc_1d` defaults already follow the reference publication, and the point of
the exercise is that both codes run on the same published parameter set.

## Adding a model adapter

An adapter is the only thing that needs writing to bring another model into the
comparison. Its job is translation, not physics:

1. Subclass `ModelAdapter` in a new module under `src/pemfc1d_benchmark/adapters/`.
2. Set `name` — the string a case file's `model:` field will use.
3. Implement `library_version`, returning the version of whatever library the
   adapter wraps. This ends up in the run metadata.
4. Implement `run(case) -> ModelResult`, converting the case's parameters into
   whatever form the model wants, calling it, and converting the answer back
   into the standard units fixed in `adapters/base.py`: current density in
   A/cm², voltage in V, profile positions in µm from the anode GDL face, and
   each variable in the unit `VARIABLE_UNITS` states.
5. Register the class in `registry()` in `adapters/base.py`.

`ModelResult` carries a polarization curve and, optionally, spatial profiles
keyed by `(variable, layer)`. Profiles are genuinely optional — a 0D model has
none, and the verification code treats their absence as a fact about the model
rather than an error. Keeping every unit conversion inside the adapter is what
lets the metrics and the plots stay model-agnostic: nothing downstream of
`adapters/` knows which model produced a curve.

Validate anything model-specific inside the adapter, not in `cases.py`. The
`pemfc1d` adapter checks that a case's parameter names exist on `pemfc_1d.Params`
and are actually settable, because those names belong to that model and a typo
in one would otherwise be silently ignored.

## Reference data must be documented and cited

Every file under `reference/` needs a recorded provenance, in the README of the
directory it lives in. This is a hard requirement, not a courtesy: a number I
cannot trace is a number I cannot defend in a viva.

For **MATLAB MMM1D outputs** (`reference/matlab/README.md`): the exact upstream
commit, the script used and any edits made to it, the parameters and operating
conditions, the `bvp4c` tolerance, and the date. The parameters must match the
case file that compares against the export — that identity is the entire basis
of verification. The verification tolerances in a case should be justified
against the tolerance the reference was computed at, not chosen for convenience.

For **experimental curves** (`reference/experimental/README.md`): the full
citation, the specific figure or table the points come from, the cell and its
operating conditions, and how the points were digitised, including the reading
uncertainty. A digitised curve carries an error of its own, and a benchmark RMSE
below that error is not a meaningful number. Anything the source does not state
must be written down as unknown rather than quietly assumed. A curve with no
recorded source cannot be used in the thesis.

## Physics reference and attribution

The model formulation, the constitutive relations and the parameter values
implemented in `pemfc_1d` follow:

> R. Vetter, J. O. Schumacher, *Free open reference implementation of a
> two-phase PEM fuel cell model*, Computer Physics Communications **234** (2019)
> 223–234. <https://doi.org/10.1016/j.cpc.2018.07.023>

Their MATLAB reference implementation
([PEMFC-1DMMM](https://github.com/Isomorph-Electrochemical-Cells/PEMFC-1DMMM))
is the standard I verify against, and the outputs stored under
`reference/matlab/` come from it.

That paper is cited here **as a reference, not as the owner of this code**. Cite
it for the physics; the Python implementation, the verification and benchmarking
work in this repository, and the extensions developed during the thesis are
mine.
