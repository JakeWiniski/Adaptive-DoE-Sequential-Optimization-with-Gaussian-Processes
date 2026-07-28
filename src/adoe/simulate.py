"""Synthetic response systems for development and validation.

The fermentation simulator is deliberately mechanistic enough to exercise the
mixed-domain workflow: it combines cardinal temperature / pH responses, Monod
growth with carbon inhibition, Luedeking–Piret product formation,
strain-specific optima, a deliberately poor category combination, block
effects, drift, heteroscedastic noise, and failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Protocol

import numpy as np
import pandas as pd

from adoe.domain import (
    CategoricalFactor,
    ContinuousFactor,
    Domain,
    TargetSpec,
)


class Simulator(Protocol):
    """Response-system contract used by tests and validation campaigns."""

    domain: Domain

    def evaluate(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Return ``y``, noise-free ``y_true``, and ``failed`` per input row."""
        ...

    def true_optimum(self) -> tuple[dict[str, object], float]:
        """Return a physically runnable optimum and its noise-free response."""
        ...


@dataclass(frozen=True, slots=True)
class FermentationConfig:
    """Configurable run-level effects for the default fermentation system."""

    duration_hours: float = 72.0
    inoculum_biomass: float = 0.15
    noise_cv: float = 0.12
    minimum_noise_sd: float = 0.10
    reference_titer: float = 20.0
    block_sd_fraction: float = 0.30
    block_sd: float | None = None
    drift_per_order: float = 0.0
    base_failure_probability: float = 0.005
    extreme_failure_probability: float = 0.65
    dud_product_multiplier: float = 0.035

    def __post_init__(self) -> None:
        positive = {
            "duration_hours": self.duration_hours,
            "inoculum_biomass": self.inoculum_biomass,
            "reference_titer": self.reference_titer,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

        nonnegative = {
            "noise_cv": self.noise_cv,
            "minimum_noise_sd": self.minimum_noise_sd,
            "block_sd_fraction": self.block_sd_fraction,
            "drift_per_order": abs(self.drift_per_order),
            "dud_product_multiplier": self.dud_product_multiplier,
        }
        if self.block_sd is not None:
            nonnegative["block_sd"] = self.block_sd
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")

        probabilities = {
            "base_failure_probability": self.base_failure_probability,
            "extreme_failure_probability": self.extreme_failure_probability,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie within [0, 1]")
        if self.extreme_failure_probability < self.base_failure_probability:
            raise ValueError(
                "extreme_failure_probability must be at least "
                "base_failure_probability"
            )

    @property
    def resolved_block_sd(self) -> float:
        """Block SD, defaulting to 0.3 times noise SD near the optimum."""

        if self.block_sd is not None:
            return self.block_sd
        process_sd = self.noise_cv * self.reference_titer
        return self.block_sd_fraction * process_sd


@dataclass(frozen=True, slots=True)
class _StrainParameters:
    mu_max: float
    alpha: float
    beta: float
    biomass_yield: float
    temperature_min: float
    temperature_opt: float
    temperature_max: float
    ph_min: float
    ph_opt: float
    ph_max: float


@dataclass(frozen=True, slots=True)
class _CarbonParameters:
    monod_ks: float
    inhibition_ki: float
    yield_multiplier: float
    product_multiplier: float


@dataclass(frozen=True, slots=True)
class _NitrogenParameters:
    monod_kn: float
    yield_multiplier: float
    product_multiplier: float


_STRAINS = {
    "A": _StrainParameters(
        mu_max=0.110,
        alpha=1.728,
        beta=0.0234,
        biomass_yield=0.50,
        temperature_min=18.0,
        temperature_opt=31.0,
        temperature_max=42.0,
        ph_min=3.8,
        ph_opt=6.1,
        ph_max=8.2,
    ),
    "B": _StrainParameters(
        mu_max=0.105,
        alpha=2.088,
        beta=0.0198,
        biomass_yield=0.54,
        temperature_min=19.0,
        temperature_opt=34.5,
        temperature_max=42.0,
        ph_min=4.0,
        ph_opt=6.8,
        ph_max=8.4,
    ),
}

_CARBON_SOURCES = {
    "glucose": _CarbonParameters(
        monod_ks=2.0,
        inhibition_ki=35.0,
        yield_multiplier=1.00,
        product_multiplier=1.00,
    ),
    "glycerol": _CarbonParameters(
        monod_ks=4.0,
        inhibition_ki=60.0,
        yield_multiplier=0.92,
        product_multiplier=1.12,
    ),
}

_NITROGEN_SOURCES = {
    "yeast_extract": _NitrogenParameters(
        monod_kn=0.35,
        yield_multiplier=1.00,
        product_multiplier=1.08,
    ),
    "ammonium": _NitrogenParameters(
        monod_kn=0.80,
        yield_multiplier=0.86,
        product_multiplier=0.95,
    ),
}


def default_fermentation_domain() -> Domain:
    """Return the physical domain for the mechanistic synthetic system."""

    return Domain(
        continuous=[
            ContinuousFactor(
                name="carbon_concentration",
                group="media",
                low=5.0,
                high=65.0,
                unit="g/L",
                step=1.0,
            ),
            ContinuousFactor(
                name="nitrogen_concentration",
                group="media",
                low=0.5,
                high=8.0,
                unit="g/L",
                step=0.25,
            ),
            ContinuousFactor(
                name="temperature",
                group="process",
                low=20.0,
                high=40.0,
                unit="°C",
                step=0.5,
            ),
            ContinuousFactor(
                name="pH",
                group="process",
                low=4.0,
                high=8.0,
                unit="pH",
                step=0.1,
            ),
        ],
        categorical=[
            CategoricalFactor(
                name="strain",
                group="strain",
                levels=list(_STRAINS),
            ),
            CategoricalFactor(
                name="carbon_source",
                group="media",
                levels=list(_CARBON_SOURCES),
            ),
            CategoricalFactor(
                name="nitrogen_source",
                group="media",
                levels=list(_NITROGEN_SOURCES),
            ),
        ],
        target=TargetSpec(
            name="Titer",
            unit="g/L",
            direction="maximize",
            transform="log",
            delta_practical_pct=10.0,
        ),
    )


class FermentationSimulator:
    """Mechanistic batch-fermentation simulator with realistic nuisance effects."""

    def __init__(self, config: FermentationConfig | None = None) -> None:
        self.config = config or FermentationConfig()
        self.domain = default_fermentation_domain()
        self._true_optimum_cache: tuple[dict[str, object], float] | None = None

    def evaluate(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Evaluate one row per experimental unit.

        Repeated condition rows receive independent noise and failure draws.
        ``y_true`` excludes noise, block, and drift effects so validation can
        compute regret against the underlying response surface.
        """

        _require_rng(rng)
        terms = self._mechanistic_terms(df)
        y_true = terms["y_true"].to_numpy(dtype=float)
        row_count = len(df)

        block_effect = self._block_effects(df, rng)
        drift_effect = self._drift_effects(df)
        noise_sd = np.maximum(
            self.config.minimum_noise_sd,
            self.config.noise_cv * np.abs(y_true),
        )
        noise = rng.normal(loc=0.0, scale=noise_sd, size=row_count)
        observed = y_true + block_effect + drift_effect + noise

        failure_probability = self._failure_probability(df)
        failed = rng.random(row_count) < failure_probability
        observed[failed] = np.nan

        return pd.DataFrame(
            {
                "y": observed,
                "y_true": y_true,
                "failed": failed,
            },
            index=df.index,
        )

    def true_optimum(self) -> tuple[dict[str, object], float]:
        """Find the global optimum by exhaustive search over the setpoint grid."""

        if self._true_optimum_cache is None:
            self._true_optimum_cache = self._dense_grid_optimum()
        condition, response = self._true_optimum_cache
        return dict(condition), response

    def _mechanistic_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        self.domain.encode(df)
        row_count = len(df)
        result = {
            "specific_growth_rate": np.empty(row_count, dtype=float),
            "final_biomass": np.empty(row_count, dtype=float),
            "growth_associated_product": np.empty(row_count, dtype=float),
            "non_growth_associated_product": np.empty(row_count, dtype=float),
            "y_true": np.empty(row_count, dtype=float),
        }

        category_columns = ["strain", "carbon_source", "nitrogen_source"]
        groups = df.groupby(category_columns, sort=False, dropna=False).indices
        for categories, positions in groups.items():
            strain, carbon_source, nitrogen_source = categories
            rows = df.iloc[positions]
            terms = _response_for_category(
                carbon=rows["carbon_concentration"].to_numpy(dtype=float),
                nitrogen=rows["nitrogen_concentration"].to_numpy(dtype=float),
                temperature=rows["temperature"].to_numpy(dtype=float),
                ph=rows["pH"].to_numpy(dtype=float),
                strain=strain,
                carbon_source=carbon_source,
                nitrogen_source=nitrogen_source,
                config=self.config,
            )
            for name, values in zip(result, terms, strict=True):
                result[name][positions] = values

        return pd.DataFrame(result, index=df.index)

    def _dense_grid_optimum(self) -> tuple[dict[str, object], float]:
        factors = {factor.name: factor for factor in self.domain.continuous}
        carbon_grid = _setpoint_grid(factors["carbon_concentration"])
        nitrogen_grid = _setpoint_grid(factors["nitrogen_concentration"])
        temperature_grid = _setpoint_grid(factors["temperature"])
        ph_grid = _setpoint_grid(factors["pH"])

        carbon = carbon_grid[:, None, None]
        nitrogen = nitrogen_grid[None, :, None]
        ph = ph_grid[None, None, :]
        best_response = -math.inf
        best_condition: dict[str, object] | None = None

        for strain, carbon_source, nitrogen_source in product(
            _STRAINS, _CARBON_SOURCES, _NITROGEN_SOURCES
        ):
            for temperature in temperature_grid:
                response = _response_for_category(
                    carbon=carbon,
                    nitrogen=nitrogen,
                    temperature=temperature,
                    ph=ph,
                    strain=strain,
                    carbon_source=carbon_source,
                    nitrogen_source=nitrogen_source,
                    config=self.config,
                )[-1]
                flat_index = int(np.argmax(response))
                candidate_response = float(response.flat[flat_index])
                if candidate_response <= best_response:
                    continue
                carbon_index, nitrogen_index, ph_index = np.unravel_index(
                    flat_index, response.shape
                )
                best_response = candidate_response
                best_condition = {
                    "carbon_concentration": float(carbon_grid[carbon_index]),
                    "nitrogen_concentration": float(
                        nitrogen_grid[nitrogen_index]
                    ),
                    "temperature": float(temperature),
                    "pH": float(ph_grid[ph_index]),
                    "strain": strain,
                    "carbon_source": carbon_source,
                    "nitrogen_source": nitrogen_source,
                }

        if best_condition is None:  # pragma: no cover - grids are validated nonempty
            raise RuntimeError("the fermentation domain contains no runnable points")
        return best_condition, best_response

    def _block_effects(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> np.ndarray:
        block_column = self.domain.block_column
        if block_column is None or block_column not in df.columns:
            return np.zeros(len(df), dtype=float)
        if df[block_column].isna().any():
            raise ValueError(f"{block_column!r} must not contain missing values")

        codes, unique_blocks = pd.factorize(df[block_column], sort=False)
        effects = rng.normal(
            loc=0.0,
            scale=self.config.resolved_block_sd,
            size=len(unique_blocks),
        )
        return effects[codes]

    def _drift_effects(self, df: pd.DataFrame) -> np.ndarray:
        if self.config.drift_per_order == 0:
            return np.zeros(len(df), dtype=float)
        if "execution_order" not in df.columns:
            raise ValueError(
                "execution_order is required when drift_per_order is nonzero"
            )
        order = pd.to_numeric(df["execution_order"], errors="coerce")
        if order.isna().any():
            raise ValueError("execution_order must contain finite numeric values")

        block_column = self.domain.block_column
        if block_column is not None and block_column in df.columns:
            centered = order - order.groupby(df[block_column]).transform("mean")
        else:
            centered = order - order.mean()
        return centered.to_numpy(dtype=float) * self.config.drift_per_order

    def _failure_probability(self, df: pd.DataFrame) -> np.ndarray:
        temperature = pd.to_numeric(
            df["temperature"], errors="coerce"
        ).to_numpy(dtype=float)
        ph = pd.to_numeric(df["pH"], errors="coerce").to_numpy(dtype=float)
        temperature_risk = _edge_risk(temperature, low=20.0, high=40.0)
        ph_risk = _edge_risk(ph, low=4.0, high=8.0)
        combined_risk = 1.0 - (1.0 - temperature_risk) * (1.0 - ph_risk)
        baseline = self.config.base_failure_probability
        extreme = self.config.extreme_failure_probability
        return baseline + (extreme - baseline) * combined_risk


class NullSimulator:
    """A response independent of every input, positive on the log scale.

    ``y = exp(N(log(reference), sigma))`` so ``log(y)`` is constant-mean noise
    unrelated to the factors. It verifies that the workflow does not claim to
    learn from a signal that is not there.
    """

    def __init__(
        self,
        domain: Domain | None = None,
        *,
        reference: float = 20.0,
        sigma: float = 0.12,
    ) -> None:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive number")
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("reference must be a finite positive number")
        self.domain = domain or default_fermentation_domain()
        self.reference = reference
        self.sigma = sigma

    def evaluate(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        _require_rng(rng)
        self.domain.encode(df)
        row_count = len(df)
        log_y = rng.normal(math.log(self.reference), self.sigma, row_count)
        return pd.DataFrame(
            {
                "y": np.exp(log_y),
                "y_true": np.full(row_count, self.reference, dtype=float),
                "failed": np.zeros(row_count, dtype=bool),
            },
            index=df.index,
        )

    def true_optimum(self) -> tuple[dict[str, object], float]:
        point: dict[str, object] = {
            factor.name: factor.low + (factor.high - factor.low) / 2.0
            for factor in self.domain.continuous
        }
        point.update({factor.name: factor.levels[0] for factor in self.domain.categorical})
        rounded = self.domain.decode(self.domain.encode(pd.DataFrame([point])))
        return rounded.iloc[0].to_dict(), self.reference


def _response_for_category(
    *,
    carbon: np.ndarray | float,
    nitrogen: np.ndarray | float,
    temperature: np.ndarray | float,
    ph: np.ndarray | float,
    strain: str,
    carbon_source: str,
    nitrogen_source: str,
    config: FermentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Monod growth and Luedeking–Piret product calculation."""

    carbon, nitrogen, temperature, ph = np.broadcast_arrays(
        np.asarray(carbon, dtype=float),
        np.asarray(nitrogen, dtype=float),
        np.asarray(temperature, dtype=float),
        np.asarray(ph, dtype=float),
    )
    strain_parameters = _STRAINS[strain]
    carbon_parameters = _CARBON_SOURCES[carbon_source]
    nitrogen_parameters = _NITROGEN_SOURCES[nitrogen_source]

    temperature_response = _cardinal_parameter_response(
        temperature,
        minimum=strain_parameters.temperature_min,
        optimum=strain_parameters.temperature_opt,
        maximum=strain_parameters.temperature_max,
    )
    ph_response = _cardinal_parameter_response(
        ph,
        minimum=strain_parameters.ph_min,
        optimum=strain_parameters.ph_opt,
        maximum=strain_parameters.ph_max,
    )

    # Haldane form: Monod saturation at low carbon and inhibition at high carbon.
    carbon_response = carbon / (
        carbon_parameters.monod_ks
        + carbon
        + carbon**2 / carbon_parameters.inhibition_ki
    )
    nitrogen_response = nitrogen / (
        nitrogen_parameters.monod_kn + nitrogen
    )
    specific_growth_rate = (
        strain_parameters.mu_max
        * temperature_response
        * ph_response
        * carbon_response
        * nitrogen_response
    )

    nitrogen_yield = nitrogen / (0.75 + nitrogen)
    carrying_biomass = (
        config.inoculum_biomass
        + strain_parameters.biomass_yield
        * carbon_parameters.yield_multiplier
        * nitrogen_parameters.yield_multiplier
        * carbon
        * nitrogen_yield
    )
    logistic_ratio = (
        carrying_biomass - config.inoculum_biomass
    ) / config.inoculum_biomass
    final_biomass = carrying_biomass / (
        1.0
        + logistic_ratio
        * np.exp(-specific_growth_rate * config.duration_hours)
    )
    biomass_formed = final_biomass - config.inoculum_biomass

    biomass_integral = np.full(
        specific_growth_rate.shape,
        config.inoculum_biomass * config.duration_hours,
        dtype=float,
    )
    growing = specific_growth_rate > 1e-12
    if growing.any():
        mu = specific_growth_rate[growing]
        capacity = carrying_biomass[growing]
        ratio = logistic_ratio[growing]
        biomass_integral[growing] = capacity / mu * (
            np.logaddexp(mu * config.duration_hours, np.log(ratio))
            - np.log1p(ratio)
        )

    category_multiplier = (
        carbon_parameters.product_multiplier
        * nitrogen_parameters.product_multiplier
    )
    if strain == "B" and nitrogen_source == "ammonium":
        category_multiplier *= config.dud_product_multiplier

    growth_associated = (
        strain_parameters.alpha * biomass_formed * category_multiplier
    )
    non_growth_associated = (
        strain_parameters.beta * biomass_integral * category_multiplier
    )
    y_true = growth_associated + non_growth_associated
    return (
        specific_growth_rate,
        final_biomass,
        growth_associated,
        non_growth_associated,
        y_true,
    )


def _cardinal_parameter_response(
    values: np.ndarray,
    *,
    minimum: float,
    optimum: float,
    maximum: float,
) -> np.ndarray:
    """Rosso-style cardinal parameter response, zero outside its limits."""

    values = np.asarray(values, dtype=float)
    response = np.zeros_like(values)
    inside = (values > minimum) & (values < maximum)
    selected = values[inside]
    numerator = (selected - maximum) * (selected - minimum) ** 2
    denominator = (optimum - minimum) * (
        (optimum - minimum) * (selected - optimum)
        - (optimum - maximum)
        * (optimum + minimum - 2.0 * selected)
    )
    response[inside] = numerator / denominator
    return np.clip(response, 0.0, 1.0)


def _setpoint_grid(factor: ContinuousFactor) -> np.ndarray:
    if factor.step is None:
        return np.linspace(factor.low, factor.high, 101)
    count = math.floor((factor.high - factor.low) / factor.step + 1e-10)
    grid = factor.low + factor.step * np.arange(count + 1)
    decimals = max(
        _decimal_places(factor.low),
        _decimal_places(factor.high),
        _decimal_places(factor.step),
    )
    return np.round(grid, decimals=decimals)


def _edge_risk(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    normalized_distance = np.abs(2.0 * (values - low) / (high - low) - 1.0)
    return np.clip((normalized_distance - 0.65) / 0.35, 0.0, 1.0)


def _require_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def _decimal_places(value: float) -> int:
    exponent = Decimal(str(value)).as_tuple().exponent
    return max(0, -exponent)
