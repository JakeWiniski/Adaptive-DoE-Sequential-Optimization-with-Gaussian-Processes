"""Run-sheet rendering and status-page rendering, with and without replication."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adoe.campaign import compute_status, initial_design, read_ledger, record, save_proposal
from adoe.report import run_sheet, status_page
from adoe.simulate import FermentationSimulator


def _record_design(sim, ledger_path, results_path, *, r, seed):
    domain = sim.domain
    rng = np.random.default_rng(seed)
    design = initial_design(domain, n_conditions=12, r=r, seed=seed)
    save_proposal(ledger_path, design)
    frame = design[domain.factor_names].copy()
    frame[domain.block_column] = design["block"].to_numpy()
    outcome = sim.evaluate(frame, rng)
    pd.DataFrame(
        {
            "unit_id": design["unit_id"].to_numpy(),
            "y": np.where(outcome["failed"].to_numpy(), np.nan, outcome["y"].to_numpy()),
            "failed": outcome["failed"].to_numpy(),
        }
    ).to_csv(results_path, index=False)
    record(ledger_path, results_path)
    return read_ledger(ledger_path)


def test_run_sheet_renders_with_setpoints_on_the_step_grid(tmp_path):
    domain = FermentationSimulator().domain
    design = initial_design(domain, n_conditions=12, r=1, seed=0)
    assert len(design) == 12
    html_path, csv_path = run_sheet(design, domain, tmp_path / "round")
    assert html_path.exists() and csv_path.exists()
    assert "run-table" in html_path.read_text()
    template = pd.read_csv(csv_path)
    assert {"unit_id", "y", "failed"}.issubset(template.columns)


def test_status_page_renders_with_replication(tmp_path):
    sim = FermentationSimulator()
    ledger = _record_design(sim, tmp_path / "c.csv", tmp_path / "r.csv", r=3, seed=0)
    numbers = compute_status(ledger, sim.domain)
    assert numbers["noise_pct"] is not None
    figure = status_page(ledger, sim.domain, numbers)
    assert figure is not None
    assert len(figure.axes) >= 4


def test_status_page_renders_without_replication(tmp_path):
    sim = FermentationSimulator()
    ledger = _record_design(sim, tmp_path / "c.csv", tmp_path / "r.csv", r=1, seed=0)
    numbers = compute_status(ledger, sim.domain)
    assert numbers["noise_pct"] is None  # no replicated condition
    figure = status_page(ledger, sim.domain, numbers)
    assert figure is not None
