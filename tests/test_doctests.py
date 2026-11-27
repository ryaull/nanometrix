"""
Runs the doctest-style examples embedded in nanometrix's docstrings.

These examples live in the docstrings themselves (nanometrix/metrics.py and
nanometrix/calibration.py) so they stay next to the code they document and
double as runnable usage demonstrations. This file is what makes pytest
actually execute them and fail the suite if one goes stale -- without it,
a code change could silently make a docstring example wrong and nothing
would notice.

Only the two modules with curated examples are collected here, rather than
turning on pytest's blanket --doctest-modules, so a `>>>` appearing in an
unrelated module's docstring (as illustrative, non-runnable snippet) can
never be mistaken for a test.
"""

from __future__ import annotations

import doctest

import nanometrix.calibration as calibration
import nanometrix.metrics as metrics


def _run_doctests(module) -> doctest.TestResults:
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"{results.failed} doctest(s) failed in {module.__name__}"
    )
    return results


def test_metrics_doctests_pass():
    results = _run_doctests(metrics)
    # Guards against the examples silently stopping being collected (e.g. a
    # docstring edit that breaks the `>>>` syntax doctest looks for).
    assert results.attempted >= 10


def test_calibration_doctests_pass():
    results = _run_doctests(calibration)
    assert results.attempted >= 4
