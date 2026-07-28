"""Gaussian-process fit and prediction on the modelling scale.

By default the workflow models ``log(y)`` (``TargetSpec.transform == "log"``),
which makes proportional response variation approximately constant. Everything
here — aggregation, fitting, and the posterior — happens on that scale;
:func:`predict` back-transforms to physical target units for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from adoe.domain import Domain, TargetSpec

_INFERRED_NOISE_FLOOR = 1e-4

Model = SingleTaskGP | MixedSingleTaskGP


def coerce_failed(series: pd.Series) -> pd.Series:
    """Parse a ``failed`` column (bool, text, or blank) to a nullable boolean."""

    def parse(value: object) -> object:
        if pd.isna(value) or value == "":
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "1.0"}:
            return True
        if text in {"false", "0", "no", "n", "0.0"}:
            return False
        return pd.NA

    return series.map(parse).astype("boolean")


def to_model_scale(y: np.ndarray | pd.Series, target: TargetSpec) -> np.ndarray:
    """Map physical responses onto the internal modelling scale.

    Applies ``log`` for a ``"log"`` target, then negates for a minimize target
    so the acquisition can always maximize.
    """

    values = np.asarray(y, dtype=float)
    modelled = np.log(values) if target.transform == "log" else values
    return -modelled if target.direction == "minimize" else modelled


@dataclass(slots=True)
class ModelMetadata:
    """Encoding and noise state needed outside the fitted GPyTorch object."""

    active_domain: Domain
    target: TargetSpec
    block_column: str | None
    block_dim: int | None
    block_level_to_index: dict[object, int]
    next_block_index: int | None
    uses_dummy_dimension: bool
    has_replication: bool
    noise_var: float
    s_log: float | None
    noise_df: int
    training_condition_count: int
    dropped_failed_conditions: int

    @property
    def fixed_features(self) -> dict[int, float]:
        """Features held fixed during acquisition (the next, unseen block)."""

        if self.block_dim is None or self.next_block_index is None:
            return {}
        return {self.block_dim: float(self.next_block_index)}


@dataclass(slots=True)
class Aggregation:
    """Condition-level training rows and pooled repeat-to-repeat noise."""

    data: pd.DataFrame
    has_replication: bool
    s_log: float | None
    noise_df: int
    dropped_failed_conditions: int


def aggregate(ledger: pd.DataFrame, domain: Domain) -> Aggregation:
    """Aggregate successful units to one row per condition on the model scale.

    Non-positive ``y`` under a log transform is treated as a failure and
    dropped. The within-condition variance is pooled across every replicated
    condition; on the log scale a single constant-variance estimate is
    appropriate, so no per-condition or level-indexed noise model is needed.
    """

    required = {"condition_id", "y", "failed", *domain.factor_names}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"ledger is missing required columns: {', '.join(missing)}")

    prepared = ledger.copy()
    block_column = domain.block_column
    if block_column is not None and block_column not in prepared.columns:
        prepared[block_column] = 0
    prepared["y"] = pd.to_numeric(prepared["y"], errors="coerce")
    failed = coerce_failed(prepared["failed"]).fillna(True).to_numpy(dtype=bool)

    positive = prepared["y"] > 0 if domain.target.transform == "log" else prepared["y"].notna()
    successful = prepared.loc[~failed & prepared["y"].notna() & positive]
    completed_ids = set(prepared.loc[failed | prepared["y"].notna(), "condition_id"])
    dropped = len(completed_ids - set(successful["condition_id"]))
    if successful.empty:
        raise ValueError("no successful recorded outcomes are available for fitting")

    identity = [*domain.factor_names]
    if block_column is not None:
        identity.append(block_column)

    rows: list[dict[str, object]] = []
    pooled_numerator = 0.0
    noise_df = 0
    for condition_id, group in successful.groupby("condition_id", sort=False):
        modelled = to_model_scale(group["y"].to_numpy(dtype=float), domain.target)
        n = len(modelled)
        first = group.iloc[0]
        variance = float(np.var(modelled, ddof=1)) if n >= 2 else math.nan
        if n >= 2:
            pooled_numerator += (n - 1) * variance
            noise_df += n - 1
        rows.append(
            {
                "condition_id": condition_id,
                **{column: first[column] for column in identity},
                "n": n,
                "model_y": float(np.mean(modelled)),
                "variance": variance,
            }
        )

    data = pd.DataFrame(rows)
    has_replication = noise_df > 0
    if has_replication:
        pooled_variance = pooled_numerator / noise_df
        s_log = math.sqrt(pooled_variance)
        data["Yvar"] = pooled_variance / data["n"].to_numpy(dtype=float)
    else:
        s_log = None
        data["Yvar"] = np.nan
    return Aggregation(
        data=data,
        has_replication=has_replication,
        s_log=s_log,
        noise_df=noise_df,
        dropped_failed_conditions=dropped,
    )


def build_model(ledger: pd.DataFrame, domain: Domain) -> Model:
    """Construct, but do not fit, the surrogate GP on the model scale."""

    aggregation = aggregate(ledger, domain)
    active_domain = domain.active_factors()
    train_X, encoding = _encode_training_inputs(
        aggregation.data, active_domain, block_column=domain.block_column
    )
    train_Y = torch.tensor(
        aggregation.data["model_y"].to_numpy()[:, None], dtype=torch.double
    )

    train_Yvar: torch.Tensor | None = None
    likelihood: GaussianLikelihood | None = None
    if aggregation.has_replication:
        train_Yvar = torch.tensor(
            aggregation.data["Yvar"].to_numpy(dtype=float)[:, None], dtype=torch.double
        )
    else:
        noise_prior = LogNormalPrior(-2.5, 1.0)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                _INFERRED_NOISE_FLOOR, transform=None, initial_value=noise_prior.mode
            ),
        )

    categorical_dims = list(encoding["categorical_dims"])
    if categorical_dims:
        model: Model = MixedSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            cat_dims=categorical_dims,
            cont_kernel_factory=_matern_with_dimension_scaled_prior,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=_matern_with_dimension_scaled_prior(
                batch_shape=torch.Size(),
                ard_num_dims=train_X.shape[-1],
                active_dims=None,
            ),
            outcome_transform=Standardize(m=1),
        )

    noise_var = aggregation.s_log**2 if aggregation.s_log is not None else math.nan
    model.adoe_metadata = ModelMetadata(
        active_domain=active_domain,
        target=domain.target,
        block_column=domain.block_column,
        block_dim=encoding["block_dim"],
        block_level_to_index=encoding["block_level_to_index"],
        next_block_index=encoding["next_block_index"],
        uses_dummy_dimension=encoding["uses_dummy_dimension"],
        has_replication=aggregation.has_replication,
        noise_var=noise_var,
        s_log=aggregation.s_log,
        noise_df=aggregation.noise_df,
        training_condition_count=len(aggregation.data),
        dropped_failed_conditions=aggregation.dropped_failed_conditions,
    )
    return model


def fit_model(
    ledger: pd.DataFrame,
    domain: Domain,
    *,
    maxiter: int = 100,
    max_attempts: int = 2,
    seed: int = 0,
) -> Model:
    """Build and fit the exact GP deterministically from a recorded ledger."""

    torch.manual_seed(seed)
    model = build_model(ledger, domain)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": maxiter}},
        max_attempts=max_attempts,
    )
    model.eval()
    model.likelihood.eval()

    metadata = model_metadata(model)
    if not metadata.has_replication:
        # Recover the inferred unit-level noise variance on the model scale.
        standardized = float(model.likelihood.noise.detach().cpu().reshape(-1)[0])
        scale = float(model.outcome_transform.stdvs.detach().cpu().reshape(-1)[0])
        metadata.noise_var = standardized * scale**2
        metadata.s_log = math.sqrt(metadata.noise_var)
    return model


def predict(model: Model, domain: Domain, df: pd.DataFrame) -> pd.DataFrame:
    """Predict in physical target units, including unit-level response noise.

    ``median`` is ``exp`` of the log-scale posterior mean — a typical yield, not
    an average. ``sd`` is the total predictive SD on the model scale (latent
    surface plus repeat-to-repeat noise): what one fresh replicate will vary by.
    Intervals are ``exp(mean ± 1.96·sd)`` and are asymmetric in physical units.
    """

    metadata = model_metadata(model)
    target = metadata.target
    X = encode_model_inputs(model, df)
    with torch.no_grad():
        posterior = model.posterior(X, observation_noise=False)
        internal_mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        latent_var = posterior.variance.squeeze(-1).clamp_min(0.0).detach().cpu().numpy()

    # Undo the maximize-oriented sign; result is the mean on the (log) response scale.
    mean_scale = -internal_mean if target.direction == "minimize" else internal_mean
    noise_var = 0.0 if not math.isfinite(metadata.noise_var) else metadata.noise_var
    sd_latent = np.sqrt(latent_var)
    sd = np.sqrt(latent_var + noise_var)

    if target.transform == "log":
        median = np.exp(mean_scale)
        lo95 = np.exp(mean_scale - 1.96 * sd)
        hi95 = np.exp(mean_scale + 1.96 * sd)
    else:
        median = mean_scale
        lo95 = mean_scale - 1.96 * sd
        hi95 = mean_scale + 1.96 * sd

    return pd.DataFrame(
        {
            "mean_scale": mean_scale,
            "sd": sd,
            "sd_latent": sd_latent,
            "median": median,
            "lo95": lo95,
            "hi95": hi95,
        },
        index=df.index,
    )


def encode_model_inputs(model: Model, df: pd.DataFrame) -> torch.Tensor:
    """Encode physical factors and the modelled block for the fitted GP."""

    metadata = model_metadata(model)
    X = metadata.active_domain.encode(df)
    if metadata.block_dim is not None:
        codes = _prediction_block_codes(df, metadata)
        X = torch.cat([X, torch.tensor(codes[:, None], dtype=torch.double)], dim=1)
    if metadata.uses_dummy_dimension:
        X = torch.zeros((len(df), 1), dtype=torch.double)
    return X


def model_metadata(model: Model) -> ModelMetadata:
    """Return the workflow metadata attached during construction."""

    metadata = getattr(model, "adoe_metadata", None)
    if not isinstance(metadata, ModelMetadata):
        raise TypeError("model was not constructed by adoe.model.build_model")
    return metadata


def _encode_training_inputs(
    aggregated: pd.DataFrame,
    active_domain: Domain,
    *,
    block_column: str | None,
) -> tuple[torch.Tensor, dict[str, object]]:
    X = active_domain.encode(aggregated)
    categorical_dims = list(active_domain.cat_dims)
    block_dim: int | None = None
    block_level_to_index: dict[object, int] = {}
    next_block_index: int | None = None

    if block_column is not None:
        block_levels = list(pd.unique(aggregated[block_column]))
        block_level_to_index = {level: index for index, level in enumerate(block_levels)}
        codes = aggregated[block_column].map(block_level_to_index)
        block_dim = X.shape[1]
        next_block_index = len(block_levels)
        X = torch.cat(
            [X, torch.tensor(codes.to_numpy()[:, None], dtype=torch.double)], dim=1
        )
        categorical_dims.append(block_dim)

    uses_dummy_dimension = X.shape[1] == 0
    if uses_dummy_dimension:
        X = torch.zeros((len(aggregated), 1), dtype=torch.double)

    return X, {
        "categorical_dims": categorical_dims,
        "block_dim": block_dim,
        "block_level_to_index": block_level_to_index,
        "next_block_index": next_block_index,
        "uses_dummy_dimension": uses_dummy_dimension,
    }


def _prediction_block_codes(df: pd.DataFrame, metadata: ModelMetadata) -> np.ndarray:
    assert metadata.block_dim is not None and metadata.next_block_index is not None
    block_column = metadata.block_column
    if block_column is None or block_column not in df.columns:
        return np.full(len(df), metadata.next_block_index, dtype=float)

    mapping = dict(metadata.block_level_to_index)
    next_code = metadata.next_block_index
    codes: list[int] = []
    for value in df[block_column]:
        if value not in mapping:
            mapping[value] = next_code
            next_code += 1
        codes.append(mapping[value])
    return np.asarray(codes, dtype=float)


def _matern_with_dimension_scaled_prior(
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: list[int] | None,
) -> Kernel:
    """Matérn-5/2 with BoTorch's dimension-scaled LogNormal lengthscale prior."""

    return get_covar_module_with_dim_scaled_prior(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        use_rbf_kernel=False,
        active_dims=active_dims,
    )
