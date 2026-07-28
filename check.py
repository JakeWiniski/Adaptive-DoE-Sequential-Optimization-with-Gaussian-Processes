"""The one honesty check for the workflow.

Answers two questions and no others:

1. Does the loop help?  Run the closed loop against the mechanistic simulator
   and compare final regret (at the model incumbent — the condition the
   workflow would actually recommend) against a matched space-filling design.
2. Does it know when it is not learning?  Run the same loop against a null
   simulator and confirm prospective error does not beat guessing the average.

Run with:  python check.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from adoe.campaign import _find_incumbent, compute_status, initial_design, propose
from adoe.domain import Domain
from adoe.model import fit_model
from adoe.simulate import FermentationSimulator, NullSimulator

warnings.simplefilter("ignore")

INITIAL_UNITS = 16
ROUND_UNITS = 6
ROUNDS = 5  # 16 + 4*6 = 40 units
NUM_RESTARTS = 3
RAW_SAMPLES = 48


def _evaluate_into(run_set: pd.DataFrame, sim, domain: Domain, rng: np.random.Generator) -> pd.DataFrame:
    frame = run_set[domain.factor_names].copy()
    frame[domain.block_column] = run_set["block"].to_numpy()
    outcome = sim.evaluate(frame, rng)
    filled = run_set.copy()
    filled["y"] = np.where(outcome["failed"].to_numpy(), np.nan, outcome["y"].to_numpy())
    filled["failed"] = outcome["failed"].to_numpy()
    return filled


def _run_adaptive(sim, domain: Domain, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ledger = _evaluate_into(initial_design(domain, n_conditions=INITIAL_UNITS, seed=seed), sim, domain, rng)
    for _ in range(ROUNDS - 1):
        proposal = propose(
            ledger, domain, n_conditions=ROUND_UNITS, r=1, seed=seed,
            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        ledger = pd.concat([ledger, _evaluate_into(proposal, sim, domain, rng)], ignore_index=True)
    return ledger


def _run_space_filling(sim, domain: Domain, seed: int, n_units: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 9973)
    design = initial_design(domain, n_conditions=n_units, seed=seed + 9973)
    return _evaluate_into(design, sim, domain, rng)


def _incumbent_regret(ledger: pd.DataFrame, sim, domain: Domain, seed: int) -> float:
    model = fit_model(ledger, domain, seed=seed)
    incumbent, _ = _find_incumbent(
        model, domain, block=0, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
    )
    frame = pd.DataFrame([incumbent])
    frame[domain.block_column] = 0
    true_here = float(sim.evaluate(frame, np.random.default_rng(0))["y_true"].iloc[0])
    _, optimum = sim.true_optimum()
    return optimum - true_here


def check_loop_helps(seeds: range = range(5)) -> bool:
    sim = FermentationSimulator()
    domain = sim.domain
    adaptive: list[float] = []
    space_filling: list[float] = []
    for seed in seeds:
        adaptive_ledger = _run_adaptive(sim, domain, seed)
        adaptive.append(_incumbent_regret(adaptive_ledger, sim, domain, seed))
        sf_ledger = _run_space_filling(sim, domain, seed, int(len(adaptive_ledger)))
        space_filling.append(_incumbent_regret(sf_ledger, sim, domain, seed))

    adaptive_arr = np.array(adaptive)
    sf_arr = np.array(space_filling)
    wins = int(np.sum(adaptive_arr < sf_arr))
    print("1) Does the loop help?  (regret at the model incumbent; lower is better)")
    print(f"   adaptive median regret       = {np.median(adaptive_arr):.3g}")
    print(f"   space-filling median regret  = {np.median(sf_arr):.3g}")
    print(f"   adaptive won {wins} of {len(list(seeds))} seeds")
    passed = bool(np.median(adaptive_arr) < np.median(sf_arr))
    print(f"   -> {'PASS' if passed else 'FAIL'}\n")
    return passed


def check_knows_when_not_learning(seeds: range = range(5)) -> bool:
    sim = NullSimulator()
    domain = sim.domain
    ratios: list[float] = []
    improvements: list[float] = []
    for seed in seeds:
        ledger = _run_adaptive(sim, domain, seed)
        numbers = compute_status(ledger, domain)
        if numbers["learning_rmse"] is not None and numbers["baseline_sd"]:
            ratios.append(float(numbers["learning_rmse"]) / float(numbers["baseline_sd"]))
        if numbers["improvement_pct"] is not None:
            improvements.append(abs(float(numbers["improvement_pct"])))

    ratio = float(np.median(ratios))
    print("2) Does it know when it is not learning?  (null response)")
    print(f"   median prospective RMSE / baseline SD = {ratio:.3g}  (≈1 means no better than guessing)")
    print(f"   median |apparent improvement vs control| = {np.median(improvements):.1f}%")
    passed = bool(ratio > 0.85)
    print(f"   -> {'PASS' if passed else 'FAIL'}\n")
    return passed


def main() -> int:
    print("=" * 68)
    ok_help = check_loop_helps()
    ok_null = check_knows_when_not_learning()
    print("=" * 68)
    all_ok = ok_help and ok_null
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
