"""The whole adaptive loop: initial design, propose, confirm, ledger, status.

The ledger is a single CSV, one row per experimental unit. Predictions are
written at proposal time and are never allowed to change afterwards — a
prediction that was not recorded before the run cannot be honestly scored. All
modelling happens on the log scale (see :mod:`adoe.model`); operator-facing
numbers here are physical units and percentages.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler

from adoe.domain import Domain
from adoe.model import (
    Model,
    aggregate,
    coerce_failed,
    encode_model_inputs,
    fit_model,
    model_metadata,
    predict,
    to_model_scale,
)

_INITIAL_RATIONALE = "Initial design — space-filling, before any model exists"
_CONTROL_RATIONALE = "Control — fixed reference condition, for drift and block comparison"
_CONFIRM_RATIONALE = "Confirm — replicated verification of a candidate best condition"
_CATEGORICAL_COMBINATION_LIMIT = 100
_MAX_COLLISION_RETRIES = 10

PRED_COLUMNS = ("pred_median", "pred_lo95", "pred_hi95", "pred_sd_log")
OUTCOME_COLUMNS = ("y", "failed", "failure_reason", "notes", "actual_execution_order")


# ---------------------------------------------------------------------------
# Ledger schema and IO
# ---------------------------------------------------------------------------


def _schema(domain: Domain) -> list[str]:
    return [
        "unit_id",
        "condition_id",
        "round",
        "replicate",
        "block",
        "proposal_order",
        "execution_order",
        "actual_execution_order",
        *domain.factor_names,
        "slot_type",
        "rationale",
        *PRED_COLUMNS,
        "y",
        "failed",
        "failure_reason",
        "notes",
    ]


def read_ledger(ledger_path: str | Path) -> pd.DataFrame:
    """Read the CSV ledger, returning an empty frame before the first round."""

    path = Path(ledger_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_proposal(ledger_path: str | Path, run_set: pd.DataFrame) -> None:
    """Append one proposed round to the CSV ledger, predictions filled in.

    Outcome columns stay blank until :func:`record` fills them. A round may not
    reuse an existing ``unit_id`` or ``round`` value.
    """

    path = Path(ledger_path)
    existing = read_ledger(path)
    if not existing.empty:
        if set(run_set["unit_id"]) & set(existing["unit_id"]):
            raise ValueError("proposal reuses an existing unit_id")
        if int(run_set["round"].iloc[0]) in set(existing["round"].astype(int)):
            raise ValueError(f"round {int(run_set['round'].iloc[0])} already exists")
        combined = pd.concat([existing, run_set], ignore_index=True, sort=False)
    else:
        combined = run_set.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def record(ledger_path: str | Path, results_csv: str | Path, domain: Domain | None = None) -> pd.DataFrame:
    """Fill outcome fields from a filled results CSV, then return the ledger.

    Every identity and prediction column supplied in the results file must
    match the stored ledger exactly — a plain column-by-column comparison. Under
    the default log transform a non-positive ``y`` is a failed replicate: it is
    recorded as a failure with reason ``"no yield"``. Pass ``domain`` with
    ``transform="none"`` to keep zero/negative values as genuine measurements.
    """

    treat_nonpositive_as_failure = domain is None or domain.target.transform == "log"
    path = Path(ledger_path)
    ledger = read_ledger(path)
    if ledger.empty:
        raise ValueError("cannot record results before a run set is proposed")
    results = pd.read_csv(results_csv)
    if "unit_id" not in results.columns:
        raise ValueError("results file must contain unit_id")
    if results["unit_id"].duplicated().any():
        raise ValueError("results contain duplicate unit_id values")

    indexed = ledger.set_index("unit_id")
    unknown = sorted(set(results["unit_id"]) - set(indexed.index))
    if unknown:
        raise ValueError(f"results contain unknown unit_id values: {', '.join(map(str, unknown))}")

    immutable = [
        column
        for column in results.columns
        if column not in OUTCOME_COLUMNS and column in ledger.columns and column != "unit_id"
    ]
    for _, row in results.iterrows():
        stored = indexed.loc[row["unit_id"]]
        for column in immutable:
            if not _same_value(stored[column], row[column]):
                raise ValueError(
                    f"column {column!r} was altered for unit {row['unit_id']!r}: "
                    f"ledger has {stored[column]!r}, results contain {row[column]!r}"
                )

    # Pre-cast the outcome columns so per-cell writes never fight the dtype.
    indexed["y"] = pd.to_numeric(indexed["y"], errors="coerce")
    indexed["actual_execution_order"] = pd.to_numeric(indexed["actual_execution_order"], errors="coerce")
    for column in ("failed", "failure_reason", "notes"):
        indexed[column] = indexed[column].astype("object")

    for _, row in results.iterrows():
        unit_id = row["unit_id"]
        y = pd.to_numeric(pd.Series([row.get("y")]), errors="coerce").iloc[0]
        failed = _parse_failed(row.get("failed"))
        reason = row.get("failure_reason")
        if treat_nonpositive_as_failure and not failed and pd.notna(y) and y <= 0:
            # A non-positive yield is a failed replicate, not a small number (see log contract).
            failed, y, reason = True, np.nan, "no yield"
        indexed.at[unit_id, "y"] = y if not failed else np.nan
        indexed.at[unit_id, "failed"] = bool(failed)
        indexed.at[unit_id, "failure_reason"] = reason if pd.notna(reason) else ""
        if "notes" in results.columns:
            indexed.at[unit_id, "notes"] = row.get("notes")
        if "actual_execution_order" in results.columns:
            indexed.at[unit_id, "actual_execution_order"] = pd.to_numeric(
                pd.Series([row.get("actual_execution_order")]), errors="coerce"
            ).iloc[0]

    updated = indexed.reset_index()
    updated.to_csv(path, index=False)
    return updated


# ---------------------------------------------------------------------------
# Initial design
# ---------------------------------------------------------------------------


def initial_design(
    domain: Domain,
    n_conditions: int | None = None,
    *,
    capacity: int | None = None,
    r: int = 1,
    control: dict[str, object] | None = None,
    block: object = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a space-filling cold-start run set, one row per unit.

    Provide either ``n_conditions`` (including the control) or ``capacity``
    (total unit budget). Categorical combinations are spread across the
    conditions; continuous factors are placed on a scrambled-Sobol grid.
    """

    n_conditions = _resolve_conditions(n_conditions, capacity, r)
    free_count = max(0, n_conditions - 1)
    _warn_if_small(domain, n_conditions)

    control_row = _resolve_control(domain, pd.DataFrame(), control)
    conditions = _space_filling_conditions(domain, free_count, seed=seed, exclude=control_row)
    conditions["slot_type"] = "initial"
    conditions["rationale"] = _INITIAL_RATIONALE

    control_frame = pd.DataFrame([{**control_row, "slot_type": "control", "rationale": _CONTROL_RATIONALE}])
    conditions = pd.concat([conditions, control_frame], ignore_index=True)
    if conditions.duplicated(subset=domain.factor_names).any():
        raise RuntimeError(
            "setpoint rounding produced duplicate initial conditions; "
            "reduce n_conditions or declare finer factor steps"
        )
    for column in PRED_COLUMNS:
        conditions[column] = np.nan
    return _finalize_run_set(conditions, domain, round_index=0, block=block, r=r, seed=seed)


# ---------------------------------------------------------------------------
# Adaptive proposal
# ---------------------------------------------------------------------------


def propose(
    ledger: pd.DataFrame,
    domain: Domain,
    n_conditions: int | None = None,
    *,
    capacity: int | None = None,
    r: int = 1,
    control: dict[str, object] | None = None,
    block: object | None = None,
    seed: int = 0,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> pd.DataFrame:
    """Propose one adaptive run set in physical units.

    An empty ledger, or one in which every unit failed, delegates to
    :func:`initial_design`. Otherwise the GP is fitted on the log scale and
    ``n_conditions − 1`` best-guess points are selected one at a time with
    ``qLogNEI`` (each appended to ``X_pending`` before the next), followed by
    the control.
    """

    round_index = _next_round(ledger)
    block = round_index if block is None else block
    if _is_cold_start(ledger):
        run_set = initial_design(
            domain, n_conditions, capacity=capacity, r=r, control=control, block=block, seed=seed
        )
        run_set["round"] = round_index
        run_set["condition_id"] = [
            cid.replace("round-000", f"round-{round_index:03d}") for cid in run_set["condition_id"]
        ]
        run_set["unit_id"] = [
            uid.replace("round-000", f"round-{round_index:03d}") for uid in run_set["unit_id"]
        ]
        return run_set

    n_conditions = _resolve_conditions(n_conditions, capacity, r)
    model = fit_model(ledger, domain, seed=_derived_seed(seed, round_index, 11))
    metadata = model_metadata(model)
    _check_categorical_limit(metadata.active_domain)

    control_row = _resolve_control(domain, ledger, control)
    pending = _pending_inputs(model, ledger)
    baseline = model.train_inputs[0].detach()

    selected_points: list[torch.Tensor] = []
    excluded = [control_row]
    rows: list[dict[str, object]] = []
    for slot_index in range(n_conditions - 1):
        acq_seed = _derived_seed(seed, round_index, 101 + slot_index)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=acq_seed)
        acquisition = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=baseline,
            X_pending=_combine_pending(pending, selected_points),
            sampler=sampler,
        )
        torch.manual_seed(acq_seed)
        condition, point = _select_unique_candidate(
            acquisition, model, domain, block=block, excluded=excluded,
            num_restarts=num_restarts, raw_samples=raw_samples, seed=acq_seed,
        )
        selected_points.append(point)
        excluded.append(condition)
        prediction = predict(model, domain, _condition_frame(domain, condition, block)).iloc[0]
        rows.append(
            {
                **condition,
                "slot_type": "explore",
                "rationale": (
                    f"Best-guess — predicted typical {prediction['median']:.3g} "
                    f"{domain.target.unit} (95% {prediction['lo95']:.3g}–{prediction['hi95']:.3g})"
                ),
                **_prediction_columns(prediction),
            }
        )

    control_prediction = predict(domain=domain, model=model, df=_condition_frame(domain, control_row, block)).iloc[0]
    rows.append(
        {
            **control_row,
            "slot_type": "control",
            "rationale": _CONTROL_RATIONALE,
            **_prediction_columns(control_prediction),
        }
    )
    conditions = pd.DataFrame(rows)
    if conditions.duplicated(subset=domain.factor_names).any():
        raise RuntimeError("proposal contains duplicate rounded conditions")
    return _finalize_run_set(conditions, domain, round_index=round_index, block=block, r=r, seed=seed)


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def confirm(
    ledger: pd.DataFrame,
    domain: Domain,
    r: int,
    *,
    block: object | None = None,
    seed: int = 0,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> pd.DataFrame:
    """Return a replicated run set for the incumbent, best observed, and control.

    ``r`` is chosen by the user; the status page prints a detectability guide to
    help. Each distinct condition is replicated ``r`` times, order randomized.
    """

    if _is_cold_start(ledger):
        raise ValueError("confirmation requires at least one successful outcome")
    round_index = _next_round(ledger)
    block = round_index if block is None else block
    model = fit_model(ledger, domain, seed=_derived_seed(seed, round_index, 53))
    _check_categorical_limit(model_metadata(model).active_domain)

    incumbent, _ = _find_incumbent(model, domain, block=block, num_restarts=num_restarts, raw_samples=raw_samples)
    best_observed = _best_observed_condition(ledger, domain)
    control_row = _resolve_control(domain, ledger, None)

    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for condition in (incumbent, best_observed, control_row):
        key = _condition_key(condition, domain)
        if key in seen:
            continue
        seen.add(key)
        prediction = predict(model, domain, _condition_frame(domain, condition, block)).iloc[0]
        rows.append(
            {
                **condition,
                "slot_type": "confirm",
                "rationale": _CONFIRM_RATIONALE,
                **_prediction_columns(prediction),
            }
        )
    conditions = pd.DataFrame(rows)
    return _finalize_run_set(conditions, domain, round_index=round_index, block=block, r=r, seed=seed)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Status:
    """Descriptive campaign status — no stop rule, no verdict."""

    best_recipe: dict[str, object]
    improvement_pct: float | None
    noise_pct: float | None
    delta_pct: float
    replicates_guide: int | None
    learning_rmse: float | None
    baseline_sd: float | None
    rms_z: float | None
    coverage: float | None
    failure_rate: float
    n_units: int
    figure: object


def status(ledger_path: str | Path, domain: Domain) -> Status:
    """Compute the two honesty checks, the best recipe, and the status figure."""

    from adoe.report import status_page

    ledger = read_ledger(ledger_path)
    if ledger.empty:
        raise ValueError("status requires at least one proposed experimental unit")

    numbers = compute_status(ledger, domain)
    numbers["figure"] = status_page(ledger, domain, numbers)
    return Status(**numbers)


def compute_status(ledger: pd.DataFrame, domain: Domain) -> dict[str, object]:
    """Compute the descriptive status numbers (no figure)."""

    target = domain.target
    completed = ledger[ledger["failed"].notna()]
    n_units = int(len(completed))
    failure_rate = float(completed["failed"].astype("boolean").fillna(False).mean()) if n_units else 0.0

    best_recipe: dict[str, object] = {}
    improvement_pct: float | None = None
    noise_pct: float | None = None
    replicates_guide: int | None = None
    learning_rmse: float | None = None
    baseline_sd: float | None = None
    rms_z: float | None = None
    coverage: float | None = None

    successful = _successful(ledger, domain)
    if not successful.empty:
        model = fit_model(ledger, domain, seed=0)
        metadata = model_metadata(model)
        incumbent, prediction = _find_incumbent(model, domain, block=0)
        best_recipe = {**incumbent, "predicted_median": float(prediction["median"]),
                       "lo95": float(prediction["lo95"]), "hi95": float(prediction["hi95"])}
        control_row = _resolve_control(domain, ledger, None)
        control_pred = predict(model, domain, _condition_frame(domain, control_row, 0)).iloc[0]
        if control_pred["median"] > 0:
            improvement_pct = float((prediction["median"] / control_pred["median"] - 1.0) * 100.0)

        if metadata.s_log is not None and metadata.has_replication:
            noise_pct = float(metadata.s_log * 100.0)
            step = math.log1p(target.delta_practical_pct / 100.0)
            replicates_guide = max(1, math.ceil((2.0 * metadata.s_log / step) ** 2))

        # Prospective, pre-recorded predictions only.
        scored = successful[successful["pred_median"].notna()].copy()
        baseline_sd = float(np.std(_observed_model_scale(successful, domain), ddof=1)) if len(successful) > 1 else None
        if not scored.empty:
            pred_mean_log = _pred_mean_log(scored, domain)
            residual = model_y_response(scored, domain) - pred_mean_log
            learning_rmse = float(np.sqrt(np.mean(residual**2)))
            z = residual / scored["pred_sd_log"].to_numpy(dtype=float)
            rms_z = float(np.sqrt(np.mean(z**2)))
            within = (scored["y"].to_numpy(dtype=float) >= scored["pred_lo95"].to_numpy(dtype=float)) & (
                scored["y"].to_numpy(dtype=float) <= scored["pred_hi95"].to_numpy(dtype=float)
            )
            coverage = float(np.mean(within))

    return {
        "best_recipe": best_recipe,
        "improvement_pct": improvement_pct,
        "noise_pct": noise_pct,
        "delta_pct": float(target.delta_practical_pct),
        "replicates_guide": replicates_guide,
        "learning_rmse": learning_rmse,
        "baseline_sd": baseline_sd,
        "rms_z": rms_z,
        "coverage": coverage,
        "failure_rate": failure_rate,
        "n_units": n_units,
    }


# ---------------------------------------------------------------------------
# Run-set assembly
# ---------------------------------------------------------------------------


def _finalize_run_set(
    conditions: pd.DataFrame,
    domain: Domain,
    *,
    round_index: int,
    block: object,
    r: int,
    seed: int,
) -> pd.DataFrame:
    conditions = conditions.reset_index(drop=True)
    conditions["condition_id"] = [
        f"round-{round_index:03d}-condition-{index + 1:04d}" for index in range(len(conditions))
    ]
    conditions["proposal_order"] = np.arange(1, len(conditions) + 1)

    units: list[dict[str, object]] = []
    for row in conditions.to_dict(orient="records"):
        for replicate in range(1, r + 1):
            unit = dict(row)
            unit["replicate"] = replicate
            unit["unit_id"] = f"{row['condition_id']}-replicate-{replicate:03d}"
            units.append(unit)
    run_set = pd.DataFrame(units)

    rng = np.random.default_rng(_derived_seed(seed, round_index, 41))
    run_set["execution_order"] = 0
    order = rng.permutation(len(run_set)) + 1
    run_set["execution_order"] = order

    run_set["round"] = round_index
    run_set["block"] = block
    run_set["actual_execution_order"] = np.nan
    run_set["y"] = np.nan
    run_set["failed"] = pd.array([pd.NA] * len(run_set), dtype="boolean")
    run_set["failure_reason"] = ""
    run_set["notes"] = ""
    run_set = run_set.sort_values("execution_order", kind="stable").reset_index(drop=True)
    return run_set[_schema(domain)]


def _prediction_columns(prediction: pd.Series) -> dict[str, float]:
    return {
        "pred_median": float(prediction["median"]),
        "pred_lo95": float(prediction["lo95"]),
        "pred_hi95": float(prediction["hi95"]),
        "pred_sd_log": float(prediction["sd"]),
    }


# ---------------------------------------------------------------------------
# Acquisition helpers
# ---------------------------------------------------------------------------


def _find_incumbent(
    model: Model,
    domain: Domain,
    *,
    block: object,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> tuple[dict[str, object], pd.Series]:
    acquisition = PosteriorMean(model=model, maximize=True)
    candidate, _ = _optimize(acquisition, model, num_restarts=num_restarts, raw_samples=raw_samples)
    condition, _ = _round_model_candidate(candidate.reshape(-1), model, domain, block=block)
    prediction = predict(model, domain, _condition_frame(domain, condition, block)).iloc[0]
    return condition, prediction


def _select_unique_candidate(
    acquisition,
    model: Model,
    domain: Domain,
    *,
    block: object,
    excluded: list[dict[str, object]],
    num_restarts: int,
    raw_samples: int,
    seed: int,
) -> tuple[dict[str, object], torch.Tensor]:
    excluded_keys = {_condition_key(condition, domain) for condition in excluded}
    excluded_points = [
        encode_model_inputs(model, _condition_frame(domain, condition, block))[0] for condition in excluded
    ]
    for _ in range(_MAX_COLLISION_RETRIES):
        optimized = acquisition
        if excluded_points:
            optimized = _RoundingExclusionAcquisition(
                acquisition, torch.stack(excluded_points), _setpoint_half_widths(model, domain)
            )
        candidate, _ = _optimize(optimized, model, num_restarts=num_restarts, raw_samples=raw_samples)
        condition, point = _round_model_candidate(candidate.reshape(-1), model, domain, block=block)
        if _condition_key(condition, domain) not in excluded_keys:
            return condition, point
        excluded_points.append(point)

    # Local restarts can be defeated by a coarse step grid; score a feasible pool.
    pool = _feasible_pool(model, domain, count=max(256, raw_samples), seed=seed, block=block)
    with torch.no_grad():
        scores = acquisition(pool.unsqueeze(-2)).reshape(-1)
    for index in torch.argsort(scores, descending=True).tolist():
        condition, point = _round_model_candidate(pool[index], model, domain, block=block)
        if _condition_key(condition, domain) not in excluded_keys:
            return condition, point
    raise RuntimeError(
        "no unique rounded condition remains; reduce the run-set size or declare finer factor steps"
    )


def _optimize(acquisition, model: Model, *, num_restarts: int, raw_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = model_metadata(model)
    domain = metadata.active_domain
    bounds = _model_bounds(model)
    inequality = domain.encode_constraints()
    fixed_list = _categorical_fixed_features(model)
    return optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        fixed_features_list=fixed_list,
        raw_samples=raw_samples,
        inequality_constraints=inequality or None,
        options={"batch_limit": min(5, num_restarts), "maxiter": 200},
    )


class _RoundingExclusionAcquisition(AcquisitionFunction):
    """Penalize acquisition on setpoint cells already occupied after rounding."""

    def __init__(self, base: AcquisitionFunction, excluded_points: torch.Tensor, half_widths: torch.Tensor) -> None:
        super().__init__(model=base.model)
        self.base = base
        self.register_buffer("excluded_points", excluded_points)
        self.register_buffer("half_widths", half_widths)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = self.base(X)
        points = X[..., 0, :]
        differences = torch.abs(points.unsqueeze(-2) - self.excluded_points)
        inside = (differences <= self.half_widths).all(dim=-1).any(dim=-1)
        return values - inside.to(dtype=values.dtype) * 1.0e6


def _model_bounds(model: Model) -> torch.Tensor:
    metadata = model_metadata(model)
    d = model.train_inputs[0].shape[-1]
    lower = torch.zeros(d, dtype=torch.double)
    upper = torch.ones(d, dtype=torch.double)
    active = metadata.active_domain
    for factor, dim in zip(active.categorical, active.cat_dims, strict=True):
        upper[dim] = len(factor.levels) - 1
    if metadata.block_dim is not None:
        upper[metadata.block_dim] = max(1, metadata.next_block_index or 1)
    return torch.stack([lower, upper])


def _categorical_fixed_features(model: Model) -> list[dict[int, float]]:
    metadata = model_metadata(model)
    domain = metadata.active_domain
    combinations = domain.categorical_combinations()
    fixed_list: list[dict[int, float]] = []
    for combination in combinations:
        fixed = dict(metadata.fixed_features)
        for factor, dim in zip(domain.categorical, domain.cat_dims, strict=True):
            fixed[dim] = float(factor.levels.index(combination[factor.name]))
        fixed_list.append(fixed)
    return fixed_list or [dict(metadata.fixed_features)]


def _setpoint_half_widths(model: Model, domain: Domain) -> torch.Tensor:
    metadata = model_metadata(model)
    active = metadata.active_domain
    widths = torch.full((model.train_inputs[0].shape[-1],), 1.0e-7, dtype=torch.double)
    for index, factor in enumerate(active.continuous):
        if factor.step is not None:
            widths[index] = min(0.5, factor.step / (2.0 * (factor.high - factor.low)))
    for dim in active.cat_dims:
        widths[dim] = 0.1
    if metadata.block_dim is not None:
        widths[metadata.block_dim] = 0.1
    if metadata.uses_dummy_dimension:
        widths[:] = 0.1
    return widths


def _round_model_candidate(
    candidate: torch.Tensor, model: Model, full_domain: Domain, *, block: object
) -> tuple[dict[str, object], torch.Tensor]:
    metadata = model_metadata(model)
    active = metadata.active_domain
    active_width = len(active.factor_names)
    active_frame = active.decode(candidate[:active_width].reshape(1, -1))
    condition = _expand_active_condition(full_domain, active_frame.iloc[0].to_dict())
    full_frame = full_domain.decode(full_domain.encode(pd.DataFrame([condition], columns=full_domain.factor_names)))
    if not full_domain.is_feasible(full_frame).iloc[0]:
        raise RuntimeError("setpoint rounding produced an infeasible acquisition candidate")
    condition = full_frame.iloc[0].to_dict()
    point = encode_model_inputs(model, _condition_frame(full_domain, condition, block))[0]
    return condition, point


def _expand_active_condition(full_domain: Domain, active_condition: dict[str, object]) -> dict[str, object]:
    condition: dict[str, object] = {}
    for factor in full_domain.continuous:
        condition[factor.name] = active_condition.get(factor.name, factor.low)
    for factor in full_domain.categorical:
        condition[factor.name] = active_condition.get(factor.name, factor.levels[0])
    return condition


def _feasible_pool(model: Model, domain: Domain, *, count: int, seed: int, block: object) -> torch.Tensor:
    active = model_metadata(model).active_domain
    dimension = max(1, len(active.continuous))
    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=seed)
    combinations = active.categorical_combinations()
    collected: list[torch.Tensor] = []
    got = 0
    attempts = 0
    while got < count:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("could not construct feasible pool; check constraints")
        batch = max(64, 3 * (count - got))
        draws = engine.draw(batch, dtype=torch.double).numpy()
        rows = pd.DataFrame(index=pd.RangeIndex(batch))
        for index, factor in enumerate(active.continuous):
            rows[factor.name] = factor.low + draws[:, index] * (factor.high - factor.low)
        for factor in active.categorical:
            rows[factor.name] = [combinations[(got + i) % len(combinations)][factor.name] for i in range(batch)]
        if active.linear_constraints:
            rows = active.decode(active.encode(rows))
            rows = rows.loc[active.is_feasible(rows)].reset_index(drop=True)
        if rows.empty:
            continue
        full_rows = pd.DataFrame([_expand_active_condition(domain, row) for row in rows.to_dict(orient="records")])
        if domain.block_column is not None:
            full_rows[domain.block_column] = block
        encoded = encode_model_inputs(model, full_rows)
        take = min(count - got, len(encoded))
        collected.append(encoded[:take])
        got += take
    return torch.cat(collected, dim=0)


# ---------------------------------------------------------------------------
# Space-filling initial conditions
# ---------------------------------------------------------------------------


def _space_filling_conditions(
    domain: Domain,
    count: int,
    *,
    seed: int,
    exclude: dict[str, object] | None = None,
) -> pd.DataFrame:
    if count == 0:
        return pd.DataFrame(columns=domain.factor_names)
    active = domain.active_factors()
    combinations = _balanced_categorical_schedule(active, count, seed=seed)

    continuous = [factor for factor in active.continuous]
    engine = (
        torch.quasirandom.SobolEngine(dimension=len(continuous), scramble=True, seed=seed)
        if continuous
        else None
    )
    # Seed the uniqueness set with the control cell so it is never sampled here —
    # the control is added separately, and a collision would raise downstream.
    seen: set[tuple[object, ...]] = set()
    if exclude is not None:
        seen.add(tuple(exclude[name] for name in domain.factor_names))
    collected: list[dict[str, object]] = []
    attempts = 0
    while len(collected) < count:
        attempts += 1
        if attempts > 200:
            raise RuntimeError("could not sample enough feasible initial conditions; check constraints")
        rows = pd.DataFrame(index=pd.RangeIndex(count - len(collected)))
        if engine is not None:
            draws = engine.draw(len(rows), dtype=torch.double).numpy()
            for index, factor in enumerate(continuous):
                rows[factor.name] = factor.low + draws[:, index] * (factor.high - factor.low)
        for factor in domain.continuous:
            if factor.low == factor.high:
                rows[factor.name] = factor.low
        # Assign a categorical combination per row from the balanced schedule, so
        # coverage spreads across combinations rather than collapsing onto one.
        base = len(collected)
        for factor in domain.categorical:
            rows[factor.name] = [
                combinations[(base + j) % len(combinations)][factor.name]
                for j in range(len(rows))
            ]
        rounded = domain.decode(domain.encode(rows))
        feasible = rounded.loc[domain.is_feasible(rounded)]
        for row in feasible.to_dict(orient="records"):
            key = tuple(row[name] for name in domain.factor_names)
            if key not in seen:
                seen.add(key)
                collected.append(row)
                if len(collected) == count:
                    break
    return pd.DataFrame(collected)[domain.factor_names]


def _balanced_categorical_schedule(active: Domain, count: int, *, seed: int) -> list[dict[str, str]]:
    combinations = active.categorical_combinations()
    if len(combinations) == 1 and not combinations[0]:
        return [{} for _ in range(count)]
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(combinations)))
    return [combinations[order[index % len(combinations)]] for index in range(count)]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _resolve_conditions(n_conditions: int | None, capacity: int | None, r: int) -> int:
    if not isinstance(r, int) or r < 1:
        raise ValueError("r must be a positive integer")
    if (n_conditions is None) == (capacity is None):
        raise ValueError("provide exactly one of n_conditions or capacity")
    if capacity is not None:
        if capacity < r:
            raise ValueError(f"capacity {capacity} cannot fit even one condition at r={r}")
        n_conditions = capacity // r
        remainder = capacity - n_conditions * r
        if remainder:
            warnings.warn(
                f"capacity {capacity} at r={r} leaves {remainder} unused unit(s)",
                UserWarning,
                stacklevel=2,
            )
    assert n_conditions is not None
    if n_conditions < 1:
        raise ValueError("n_conditions must be a positive integer")
    return n_conditions


def _resolve_control(domain: Domain, ledger: pd.DataFrame, supplied: dict[str, object] | None) -> dict[str, object]:
    if supplied is not None:
        point = dict(supplied)
    elif not ledger.empty and "slot_type" in ledger.columns and ledger["slot_type"].eq("control").any():
        point = ledger.loc[ledger["slot_type"].eq("control"), domain.factor_names].iloc[-1].to_dict()
    else:
        warnings.warn(
            "No control supplied; using the domain midpoint with the first level of each factor.",
            UserWarning,
            stacklevel=2,
        )
        point = {factor.name: factor.low + (factor.high - factor.low) / 2.0 for factor in domain.continuous}
        point.update({factor.name: factor.levels[0] for factor in domain.categorical})

    missing = [name for name in domain.factor_names if name not in point]
    if missing:
        raise ValueError(f"control condition is missing factor columns: {', '.join(missing)}")
    frame = domain.decode(domain.encode(pd.DataFrame([point], columns=domain.factor_names)))
    if not domain.is_feasible(frame).iloc[0]:
        raise ValueError("the control condition is infeasible after setpoint rounding")
    return frame.iloc[0].to_dict()


def _check_categorical_limit(active: Domain) -> None:
    count = len(active.categorical_combinations())
    if count > _CATEGORICAL_COMBINATION_LIMIT:
        raise ValueError(
            f"this campaign has {count} categorical combinations, above the limit of "
            f"{_CATEGORICAL_COMBINATION_LIMIT}; fix some categorical factors for this campaign"
        )


def _pending_inputs(model: Model, ledger: pd.DataFrame) -> torch.Tensor | None:
    if ledger.empty or "y" not in ledger.columns:
        return None
    failed = coerce_failed(ledger["failed"]).fillna(False).to_numpy(dtype=bool)
    pending = ledger.loc[pd.to_numeric(ledger["y"], errors="coerce").isna() & ~failed]
    if pending.empty:
        return None
    pending = pending.drop_duplicates(subset=["condition_id"])
    return encode_model_inputs(model, pending)


def _combine_pending(existing: torch.Tensor | None, selected: list[torch.Tensor]) -> torch.Tensor | None:
    pieces: list[torch.Tensor] = []
    if existing is not None and len(existing):
        pieces.append(existing)
    if selected:
        pieces.append(torch.stack(selected))
    return torch.cat(pieces, dim=0) if pieces else None


def _best_observed_condition(ledger: pd.DataFrame, domain: Domain) -> dict[str, object]:
    aggregated = aggregate(ledger, domain).data
    index = aggregated["model_y"].idxmax()  # model_y is already maximize-oriented
    return aggregated.loc[index, domain.factor_names].to_dict()


def _condition_frame(domain: Domain, condition: dict[str, object], block: object) -> pd.DataFrame:
    row = {name: condition[name] for name in domain.factor_names}
    if domain.block_column is not None:
        row[domain.block_column] = block
    return pd.DataFrame([row])


def _condition_key(condition: dict[str, object], domain: Domain) -> tuple[object, ...]:
    return tuple(condition[name] for name in domain.factor_names)


def _successful(ledger: pd.DataFrame, domain: Domain) -> pd.DataFrame:
    if "y" not in ledger.columns:
        return ledger.iloc[0:0]
    failed = coerce_failed(ledger["failed"]).fillna(True).to_numpy(dtype=bool)
    y = pd.to_numeric(ledger["y"], errors="coerce")
    positive = y > 0 if domain.target.transform == "log" else y.notna()
    return ledger.loc[~failed & y.notna() & positive]


def _observed_model_scale(frame: pd.DataFrame, domain: Domain) -> np.ndarray:
    return to_model_scale(pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float), domain.target)


def model_y_response(frame: pd.DataFrame, domain: Domain) -> np.ndarray:
    """Observed response on the (log) scale, in the response direction."""

    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    return np.log(y) if domain.target.transform == "log" else y


def _pred_mean_log(frame: pd.DataFrame, domain: Domain) -> np.ndarray:
    median = frame["pred_median"].to_numpy(dtype=float)
    return np.log(median) if domain.target.transform == "log" else median


def _next_round(ledger: pd.DataFrame) -> int:
    if ledger.empty or "round" not in ledger.columns:
        return 0
    values = pd.to_numeric(ledger["round"], errors="coerce")
    return 0 if values.dropna().empty else int(values.max()) + 1


def _is_cold_start(ledger: pd.DataFrame) -> bool:
    if ledger.empty or "y" not in ledger.columns:
        return True
    failed = coerce_failed(ledger["failed"]).fillna(False).to_numpy(dtype=bool)
    return not (pd.to_numeric(ledger["y"], errors="coerce").notna() & ~failed).any()


def _derived_seed(seed: int, round_index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([seed, round_index, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _warn_if_small(domain: Domain, n_conditions: int) -> None:
    active = domain.active_factors()
    categorical_minimum = 3 * len(active.categorical_combinations())
    if n_conditions < categorical_minimum:
        warnings.warn(
            f"Initial design has {n_conditions} conditions; at least {categorical_minimum} is advised "
            f"for {len(active.categorical_combinations())} categorical combinations.",
            UserWarning,
            stacklevel=3,
        )
    continuous_minimum = 2 * (len(active.continuous) + 1)
    if n_conditions < continuous_minimum:
        warnings.warn(
            f"Initial design has {n_conditions} conditions; at least {continuous_minimum} is advised "
            f"for {len(active.continuous)} continuous factors.",
            UserWarning,
            stacklevel=3,
        )


def _same_value(stored: object, incoming: object) -> bool:
    if pd.isna(stored) and pd.isna(incoming):
        return True
    if pd.isna(stored) or pd.isna(incoming):
        return False
    try:
        return math.isclose(float(stored), float(incoming), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(stored) == str(incoming)


def _parse_failed(value: object) -> bool:
    if pd.isna(value) or value == "":
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"failed contains an unrecognized value: {value!r}")
