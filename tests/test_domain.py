"""Domain encode/decode round trip, active-factor pruning, constraint encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adoe.domain import (
    CategoricalFactor,
    ContinuousFactor,
    Domain,
    LinearConstraint,
    TargetSpec,
)


def _domain(constraints=None) -> Domain:
    return Domain(
        continuous=[
            ContinuousFactor(name="carbon", group="media", low=5.0, high=65.0, unit="g/L", step=1.0),
            ContinuousFactor(name="temp", group="process", low=20.0, high=40.0, unit="C", step=0.5),
        ],
        categorical=[CategoricalFactor(name="strain", group="strain", levels=["A", "B", "C"])],
        linear_constraints=constraints or [],
        target=TargetSpec(name="Titer", unit="g/L", direction="maximize", delta_practical_pct=10.0),
    )


@settings(max_examples=50, deadline=None)
@given(
    carbon=st.floats(min_value=5.0, max_value=65.0),
    temp=st.floats(min_value=20.0, max_value=40.0),
    strain=st.sampled_from(["A", "B", "C"]),
)
def test_decode_encode_is_identity_on_the_step_grid(carbon, temp, strain):
    domain = _domain()
    frame = pd.DataFrame([{"carbon": carbon, "temp": temp, "strain": strain}])
    decoded = domain.decode(domain.encode(frame))
    # decode snaps to the step grid; encoding and decoding again must be stable.
    twice = domain.decode(domain.encode(decoded))
    pd.testing.assert_frame_equal(decoded, twice)
    assert np.isclose((decoded["carbon"].iloc[0] - 5.0) % 1.0, 0.0, atol=1e-9)
    assert np.isclose((decoded["temp"].iloc[0] - 20.0) % 0.5, 0.0, atol=1e-9)


def test_active_factors_drops_single_level_and_zero_width():
    domain = Domain(
        continuous=[
            ContinuousFactor(name="carbon", group="media", low=5.0, high=65.0, unit="g/L", step=1.0),
            ContinuousFactor(name="fixed", group="media", low=7.0, high=7.0, unit="g/L"),
        ],
        categorical=[
            CategoricalFactor(name="strain", group="strain", levels=["A", "B"]),
            CategoricalFactor(name="single", group="media", levels=["only"]),
        ],
        target=TargetSpec(name="Titer", unit="g/L", direction="maximize", delta_practical_pct=10.0),
    )
    active = domain.active_factors()
    assert [f.name for f in active.continuous] == ["carbon"]
    assert [f.name for f in active.categorical] == ["strain"]


def test_constraint_encoding_round_trips_to_botorch_inequalities():
    constraint = LinearConstraint(columns=["carbon", "temp"], coefficients=[1.0, 1.0], sense="<=", rhs=80.0)
    domain = _domain([constraint])
    encoded = domain.encode_constraints()
    assert len(encoded) == 1
    feasible = pd.DataFrame([{"carbon": 10.0, "temp": 25.0, "strain": "A"}])
    infeasible = pd.DataFrame([{"carbon": 60.0, "temp": 39.0, "strain": "A"}])
    assert bool(domain.is_feasible(feasible).iloc[0])
    assert not bool(domain.is_feasible(infeasible).iloc[0])


def test_equality_sense_is_rejected():
    with pytest.raises(Exception):
        LinearConstraint(columns=["carbon"], coefficients=[1.0], sense="==", rhs=10.0)


def test_free_form_groups_and_percent_delta_in_describe():
    domain = _domain()
    text = domain.describe()
    assert "10%" in text
    assert "log scale" in text
