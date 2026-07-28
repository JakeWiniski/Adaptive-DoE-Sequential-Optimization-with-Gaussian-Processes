"""Full propose/save/record/propose cycle, immutability, and randomization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adoe.campaign import initial_design, propose, read_ledger, record, save_proposal
from adoe.simulate import FermentationSimulator


def _results_csv(run_set, sim, rng, path):
    frame = run_set[sim.domain.factor_names].copy()
    frame[sim.domain.block_column] = run_set["block"].to_numpy()
    outcome = sim.evaluate(frame, rng)
    results = pd.DataFrame(
        {
            "unit_id": run_set["unit_id"].to_numpy(),
            "y": np.where(outcome["failed"].to_numpy(), np.nan, outcome["y"].to_numpy()),
            "failed": outcome["failed"].to_numpy(),
        }
    )
    results.to_csv(path, index=False)
    return path


def test_full_propose_save_record_propose_cycle(tmp_path):
    sim = FermentationSimulator()
    domain = sim.domain
    rng = np.random.default_rng(0)
    ledger_path = tmp_path / "campaign.csv"

    design = initial_design(domain, n_conditions=12, seed=0)
    save_proposal(ledger_path, design)
    record(ledger_path, _results_csv(design, sim, rng, tmp_path / "r0.csv"))

    ledger = read_ledger(ledger_path)
    proposal = propose(ledger, domain, n_conditions=6, r=1, seed=0, num_restarts=4, raw_samples=64)
    assert int(proposal["round"].iloc[0]) == 1
    assert proposal["pred_median"].notna().all()
    save_proposal(ledger_path, proposal)
    record(ledger_path, _results_csv(proposal, sim, rng, tmp_path / "r1.csv"))

    final = read_ledger(ledger_path)
    assert len(final) == len(design) + len(proposal)
    assert final["y"].notna().sum() > 0


def test_record_raises_on_altered_identity_or_prediction(tmp_path):
    sim = FermentationSimulator()
    domain = sim.domain
    ledger_path = tmp_path / "campaign.csv"
    design = initial_design(domain, n_conditions=8, seed=1)
    save_proposal(ledger_path, design)

    results = pd.DataFrame(
        {
            "unit_id": design["unit_id"].to_numpy(),
            domain.factor_names[0]: design[domain.factor_names[0]].to_numpy(),
            "y": 20.0,
            "failed": False,
        }
    )
    # Alter one identity (factor) value.
    results.loc[0, domain.factor_names[0]] = results.loc[0, domain.factor_names[0]] + 100.0
    bad = tmp_path / "bad.csv"
    results.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="altered"):
        record(ledger_path, bad)


def test_record_raises_on_altered_prediction_column(tmp_path):
    sim = FermentationSimulator()
    domain = sim.domain
    rng = np.random.default_rng(0)
    ledger_path = tmp_path / "campaign.csv"
    design = initial_design(domain, n_conditions=8, seed=0)
    save_proposal(ledger_path, design)
    record(ledger_path, _results_csv(design, sim, rng, tmp_path / "r0.csv"))
    ledger = read_ledger(ledger_path)
    proposal = propose(ledger, domain, n_conditions=6, r=1, seed=0, num_restarts=4, raw_samples=64)
    save_proposal(ledger_path, proposal)

    results = pd.DataFrame(
        {
            "unit_id": proposal["unit_id"].to_numpy(),
            "pred_median": proposal["pred_median"].to_numpy() + 5.0,
            "y": 20.0,
            "failed": False,
        }
    )
    bad = tmp_path / "bad.csv"
    results.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="altered"):
        record(ledger_path, bad)


def test_run_order_is_reproducible_and_preserves_proposal_order():
    domain = FermentationSimulator().domain
    a = initial_design(domain, n_conditions=10, r=2, seed=7)
    b = initial_design(domain, n_conditions=10, r=2, seed=7)
    pd.testing.assert_series_equal(a["execution_order"], b["execution_order"])
    # Every replicate of a condition shares its proposal_order; orders cover 1..n.
    per_condition = a.groupby("condition_id")["proposal_order"].nunique()
    assert (per_condition == 1).all()
    assert sorted(a["execution_order"].unique()) == list(range(1, len(a) + 1))


def test_no_run_set_contains_duplicate_rounded_conditions():
    domain = FermentationSimulator().domain
    design = initial_design(domain, n_conditions=15, seed=3)
    condition_level = design.drop_duplicates(subset="condition_id")[domain.factor_names]
    assert not condition_level.duplicated().any()
