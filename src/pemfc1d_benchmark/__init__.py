"""Verification and benchmarking harness for my 1D PEM fuel cell model.

Two separate questions live in this package, and the distinction runs through
all of it:

*Verification* asks whether my Python solver reproduces the MATLAB MMM1D
reference implementation when both are given identical parameters and operating
conditions. The expected answer is agreement within solver tolerance, and a
disagreement is a bug in my code. See :mod:`pemfc1d_benchmark.verification`.

*Benchmarking* asks how closely a model matches measured cell behaviour. The
expected answer is a discrepancy, which is described rather than judged against
a tolerance. See :mod:`pemfc1d_benchmark.metrics`.

The solver itself is not here: it is the ``pemfc_1d`` package, consumed as a
dependency and reached through :mod:`pemfc1d_benchmark.adapters`.
"""
__version__ = "0.1.0"
