"""Log transform, non-positive handling, pooled noise, and prospective RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adoe.campaign import initial_design
from adoe.model import aggregate, fit_model, model_metadata, predict, to_model_scale
from adoe.simulate import FermentationConfig, FermentationSimulator


def _simulate(run_set, sim, rng):
    frame = run_set[sim.domain.factor_names].copy()
    frame[sim.domain.block_column] = run_set["block"].to_numpy()
    outcome = sim.evaluate(frame, rng)
    filled = run_set.copy()
    filled["y"] = np.where(outcome["failed"].to_numpy(), np.nan, outcome["y"].to_numpy())
    filled["failed"] = outcome["failed"].to_numpy()
    return filled


def test_log_transform_applied_and_inverted():
    sim = FermentationSimulator()
    values = np.array([10.0, 20.0, 40.0])
    modelled = to_model_scale(values, sim.domain.target)
    assert np.allclose(modelled, np.log(values))

    # A minimize target negates on the log scale.
    from adoe.domain import TargetSpec

    minimize = TargetSpec(name="days", unit="d", direction="minimize", delta_practical_pct=10.0)
    assert np.allclose(to_model_scale(values, minimize), -np.log(values))


def test_nonpositive_y_becomes_a_failure_and_is_dropped():
    domain = FermentationSimulator().domain
    rows = []
    # Build two conditions: one healthy, one with a dead (y<=0) bag.
    healthy = {f.name: f.low + (f.high - f.low) / 2 for f in domain.continuous}
    healthy.update({f.name: f.levels[0] for f in domain.categorical})
    dead = dict(healthy)
    dead[domain.continuous[0].name] = domain.continuous[0].high

    for replicate in range(3):
        rows.append({**healthy, "condition_id": "c-good", "block": 0, "y": 18.0 + replicate,
                     "failed": False})
    rows.append({**dead, "condition_id": "c-dead", "block": 0, "y": 0.0, "failed": False})
    ledger = pd.DataFrame(rows)

    result = aggregate(ledger, domain)
    assert "c-good" in set(result.data["condition_id"])
    assert "c-dead" not in set(result.data["condition_id"])
    assert result.dropped_failed_conditions == 1


def test_pooled_noise_recovers_the_simulator_cv_within_20_percent():
    # Isolate the multiplicative-noise regime the property assumes: no additive
    # noise floor and no block offset, so the simulator's noise is purely CV=0.12.
    # (Under the default floor/block, low-yield conditions carry extra additive
    # noise, so pooled log-noise legitimately differs from the CV.)
    sim = FermentationSimulator(FermentationConfig(minimum_noise_sd=1e-6, block_sd=0.0))
    domain = sim.domain
    estimates = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        ledger = _simulate(initial_design(domain, n_conditions=8, r=6, seed=seed), sim, rng)
        estimates.append(model_metadata(fit_model(ledger, domain, seed=seed)).s_log)
    # The simulator's multiplicative noise CV is 0.12; log-scale SD ≈ CV.
    s_log = float(np.median(estimates))
    assert 0.12 * 0.8 <= s_log <= 0.12 * 1.2


def test_prospective_rmse_beats_intercept_only_baseline_over_5_seeds():
    sim = FermentationSimulator()
    domain = sim.domain
    rmse_over_baseline = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        train = _simulate(initial_design(domain, n_conditions=24, seed=seed), sim, rng)
        test = _simulate(initial_design(domain, n_conditions=20, seed=seed + 500), sim, rng)
        model = fit_model(train, domain, seed=seed)
        prediction = predict(model, domain, test.assign(block=0))
        ok = test["failed"].eq(False) & test["y"].gt(0)
        log_y = np.log(test.loc[ok, "y"].to_numpy(dtype=float))
        pred_log = np.log(prediction.loc[ok, "median"].to_numpy(dtype=float))
        rmse = np.sqrt(np.mean((log_y - pred_log) ** 2))
        baseline = np.std(log_y, ddof=1)
        rmse_over_baseline.append(rmse / baseline)
    assert np.median(rmse_over_baseline) < 1.0
