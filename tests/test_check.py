"""check.py runs end to end on a single seed."""

from __future__ import annotations

import pytest

import check


@pytest.mark.slow
def test_check_loop_helps_runs_end_to_end():
    result = check.check_loop_helps(seeds=range(1))
    assert isinstance(result, bool)


@pytest.mark.slow
def test_check_null_runs_end_to_end():
    result = check.check_knows_when_not_learning(seeds=range(1))
    assert isinstance(result, bool)
