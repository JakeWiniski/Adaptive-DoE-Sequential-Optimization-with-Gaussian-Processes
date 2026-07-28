"""Experimental domain definitions and physical-to-model-space translation.

The domain is the single source of truth for factor names, units, ranges,
categorical levels, and linear constraints. Operator-facing data stays in
physical units; model-facing continuous values are normalized to ``[0, 1]``
and categorical values are integer indices.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from itertools import product
from typing import Literal, Self, TypeAlias

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

ConstraintSense: TypeAlias = Literal["<=", ">="]
EncodedConstraint: TypeAlias = tuple[torch.Tensor, torch.Tensor, float]

_BOUND_TOLERANCE = 1e-10
_CONSTRAINT_TOLERANCE = 1e-9


class ContinuousFactor(BaseModel):
    """A numeric experimental factor expressed in physical units."""

    model_config = ConfigDict(extra="forbid")

    name: str
    group: str = "other"
    low: float
    high: float
    unit: str
    step: float | None = None

    @field_validator("name", "unit")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("low and high must be finite")
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        if self.step is not None and (
            not math.isfinite(self.step) or self.step <= 0
        ):
            raise ValueError("step must be a finite positive number")
        return self


class CategoricalFactor(BaseModel):
    """A categorical experimental factor with named levels."""

    model_config = ConfigDict(extra="forbid")

    name: str
    group: str = "other"
    levels: list[str]

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("levels")
    @classmethod
    def _validate_levels(cls, levels: list[str]) -> list[str]:
        cleaned = [level.strip() for level in levels]
        if not cleaned:
            raise ValueError("levels must contain at least one value")
        if any(not level for level in cleaned):
            raise ValueError("levels must not contain blank values")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("levels must be unique")
        return cleaned


class LinearConstraint(BaseModel):
    """A linear constraint on continuous factors in physical units."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    coefficients: list[float]
    sense: ConstraintSense
    rhs: float

    @model_validator(mode="after")
    def _validate_terms(self) -> Self:
        if not self.columns:
            raise ValueError("a linear constraint must contain at least one column")
        if len(self.columns) != len(self.coefficients):
            raise ValueError("columns and coefficients must have the same length")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("constraint columns must be unique")
        if any(not column.strip() for column in self.columns):
            raise ValueError("constraint columns must not be blank")
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("constraint coefficients must be finite")
        if all(value == 0 for value in self.coefficients):
            raise ValueError("at least one constraint coefficient must be nonzero")
        if not math.isfinite(self.rhs):
            raise ValueError("constraint rhs must be finite")
        return self


class TargetSpec(BaseModel):
    """The measured response and the smallest practically useful improvement.

    ``transform="log"`` is the default: the workflow models ``log(y)`` so that
    proportional variation becomes constant. Use ``"none"`` only for a response
    that is genuinely additive and can be zero or negative — a temperature, a
    pH, a signed difference. ``delta_practical_pct`` is the smallest gain worth
    pursuing expressed as a percentage of the current level.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    unit: str
    direction: Literal["maximize", "minimize"] = "maximize"
    transform: Literal["log", "none"] = "log"
    delta_practical_pct: float

    @field_validator("name", "unit")
    @classmethod
    def _target_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("delta_practical_pct")
    @classmethod
    def _delta_must_be_positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("delta_practical_pct must be a finite positive number")
        return value


class Domain(BaseModel):
    """The complete experimental search domain.

    Factor column order in model space is always all continuous factors followed
    by all categorical factors. This makes ``cat_dims`` stable and keeps
    categorical values in the integer-index representation required by
    ``MixedSingleTaskGP``.
    """

    model_config = ConfigDict(extra="forbid")

    continuous: list[ContinuousFactor]
    categorical: list[CategoricalFactor]
    block_column: str | None = "block"
    linear_constraints: list[LinearConstraint] = Field(default_factory=list)
    target: TargetSpec

    @model_validator(mode="after")
    def _validate_domain(self) -> Self:
        names = self.factor_names
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique across the domain")
        if self.block_column is not None:
            if not self.block_column.strip():
                raise ValueError("block_column must not be blank")
            if self.block_column in names:
                raise ValueError("block_column must not duplicate a factor name")

        continuous_names = {factor.name for factor in self.continuous}
        for constraint in self.linear_constraints:
            unknown = set(constraint.columns) - continuous_names
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(
                    "linear constraints may reference continuous factors only; "
                    f"unknown columns: {unknown_text}"
                )
        return self

    @property
    def factor_names(self) -> list[str]:
        """Factor names in their model-space column order."""

        return [
            *(factor.name for factor in self.continuous),
            *(factor.name for factor in self.categorical),
        ]

    @property
    def cat_dims(self) -> list[int]:
        """Indices of categorical columns in the encoded model space."""

        start = len(self.continuous)
        return list(range(start, start + len(self.categorical)))

    def encode(self, df: pd.DataFrame) -> torch.Tensor:
        """Translate physical-unit rows to the model's numeric representation."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("encode expects a pandas DataFrame")
        missing = [name for name in self.factor_names if name not in df.columns]
        if missing:
            raise ValueError(f"missing factor columns: {', '.join(missing)}")

        encoded_columns: list[np.ndarray] = []
        for factor in self.continuous:
            try:
                values = pd.to_numeric(df[factor.name], errors="raise").to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"continuous factor {factor.name!r} must contain numeric values"
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(
                    f"continuous factor {factor.name!r} contains a non-finite value"
                )
            if (
                (values < factor.low - _BOUND_TOLERANCE).any()
                or (values > factor.high + _BOUND_TOLERANCE).any()
            ):
                raise ValueError(
                    f"continuous factor {factor.name!r} must lie within "
                    f"[{factor.low:g}, {factor.high:g}] {factor.unit}"
                )

            width = factor.high - factor.low
            if width == 0:
                if not np.allclose(values, factor.low, atol=_BOUND_TOLERANCE, rtol=0):
                    raise ValueError(
                        f"fixed factor {factor.name!r} must equal {factor.low:g}"
                    )
                encoded_columns.append(np.zeros(len(df), dtype=float))
            else:
                encoded_columns.append((values - factor.low) / width)

        for factor in self.categorical:
            index_by_level = {level: index for index, level in enumerate(factor.levels)}
            encoded = df[factor.name].map(index_by_level)
            if encoded.isna().any():
                bad_levels = sorted(
                    {
                        repr(value)
                        for value in df.loc[encoded.isna(), factor.name].tolist()
                    }
                )
                raise ValueError(
                    f"categorical factor {factor.name!r} contains unknown levels: "
                    f"{', '.join(bad_levels)}"
                )
            encoded_columns.append(encoded.to_numpy(dtype=float))

        if not encoded_columns:
            return torch.empty((len(df), 0), dtype=torch.double)
        matrix = np.column_stack(encoded_columns)
        return torch.as_tensor(matrix, dtype=torch.double)

    def decode(self, X: torch.Tensor) -> pd.DataFrame:
        """Translate model-space rows back to achievable physical setpoints."""

        if not isinstance(X, torch.Tensor):
            raise TypeError("decode expects a torch Tensor")
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2:
            raise ValueError("encoded points must be a one- or two-dimensional tensor")
        if X.shape[1] != len(self.factor_names):
            raise ValueError(
                f"encoded points have {X.shape[1]} columns; "
                f"expected {len(self.factor_names)}"
            )
        if not torch.isfinite(X).all():
            raise ValueError("encoded points contain a non-finite value")

        values = X.detach().cpu().to(dtype=torch.double).numpy()
        decoded: dict[str, np.ndarray] = {}

        for index, factor in enumerate(self.continuous):
            encoded = values[:, index]
            if (
                (encoded < -_BOUND_TOLERANCE).any()
                or (encoded > 1 + _BOUND_TOLERANCE).any()
            ):
                raise ValueError(
                    f"encoded continuous factor {factor.name!r} must lie within [0, 1]"
                )
            encoded = np.clip(encoded, 0.0, 1.0)
            physical = factor.low + encoded * (factor.high - factor.low)
            decoded[factor.name] = _round_to_step(physical, factor)

        offset = len(self.continuous)
        for categorical_index, factor in enumerate(self.categorical):
            encoded = values[:, offset + categorical_index]
            integer_indices = np.rint(encoded).astype(int)
            if not np.allclose(
                encoded, integer_indices, atol=_BOUND_TOLERANCE, rtol=0
            ):
                raise ValueError(
                    f"encoded categorical factor {factor.name!r} must contain "
                    "integer indices"
                )
            if (
                (integer_indices < 0).any()
                or (integer_indices >= len(factor.levels)).any()
            ):
                raise ValueError(
                    f"encoded categorical factor {factor.name!r} contains an "
                    "out-of-range level index"
                )
            decoded[factor.name] = np.asarray(
                [factor.levels[index] for index in integer_indices], dtype=object
            )

        return pd.DataFrame(decoded, index=pd.RangeIndex(len(values)))

    def encode_constraints(self) -> list[EncodedConstraint]:
        """Encode physical inequality constraints for BoTorch optimization.

        BoTorch represents inequalities as ``sum(coefficients * X[indices]) >=
        rhs``. Physical ``<=`` constraints are therefore multiplied by ``-1``
        after translating to normalized continuous coordinates.
        """

        inequality_constraints: list[EncodedConstraint] = []
        index_by_name = {name: index for index, name in enumerate(self.factor_names)}
        factor_by_name = {factor.name: factor for factor in self.continuous}

        for constraint in self.linear_constraints:
            indices: list[int] = []
            coefficients: list[float] = []
            encoded_rhs = constraint.rhs

            for name, coefficient in zip(
                constraint.columns, constraint.coefficients, strict=True
            ):
                factor = factor_by_name[name]
                encoded_rhs -= coefficient * factor.low
                encoded_coefficient = coefficient * (factor.high - factor.low)
                if encoded_coefficient != 0.0:
                    indices.append(index_by_name[name])
                    coefficients.append(encoded_coefficient)

            if constraint.sense == "<=":
                coefficients = [-value for value in coefficients]
                encoded_rhs = -encoded_rhs

            if not indices:
                tolerance = _CONSTRAINT_TOLERANCE * max(1.0, abs(encoded_rhs))
                if 0.0 >= encoded_rhs - tolerance:
                    continue
                raise ValueError(
                    "a constraint becomes infeasible after fixed factors are removed: "
                    f"{_format_constraint(constraint)}"
                )

            inequality_constraints.append(
                (
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor(coefficients, dtype=torch.double),
                    float(encoded_rhs),
                )
            )

        return inequality_constraints

    def active_factors(self) -> Domain:
        """Return a domain with inert factors removed.

        A one-level categorical and a zero-width continuous factor carry no
        information for a surrogate. Fixed continuous terms are folded into
        the right-hand side of any affected physical-unit constraint.
        """

        active_continuous: list[ContinuousFactor] = []
        fixed_continuous: dict[str, ContinuousFactor] = {}
        for factor in self.continuous:
            if factor.low == factor.high:
                fixed_continuous[factor.name] = factor
                logger.info(
                    "Dropped continuous factor %s because low equals high (%g %s).",
                    factor.name,
                    factor.low,
                    factor.unit,
                )
            else:
                active_continuous.append(factor)

        active_categorical: list[CategoricalFactor] = []
        for factor in self.categorical:
            if len(factor.levels) == 1:
                logger.info(
                    "Dropped categorical factor %s because it has one level (%s).",
                    factor.name,
                    factor.levels[0],
                )
            else:
                active_categorical.append(factor)

        active_constraints: list[LinearConstraint] = []
        for constraint in self.linear_constraints:
            columns: list[str] = []
            coefficients: list[float] = []
            rhs = constraint.rhs
            for name, coefficient in zip(
                constraint.columns, constraint.coefficients, strict=True
            ):
                if name in fixed_continuous:
                    rhs -= coefficient * fixed_continuous[name].low
                elif coefficient != 0.0:
                    columns.append(name)
                    coefficients.append(coefficient)

            if not columns:
                if _physical_constant_constraint_is_satisfied(constraint.sense, rhs):
                    logger.info(
                        "Dropped constraint %s because fixed factors satisfy it.",
                        _format_constraint(constraint),
                    )
                    continue
                raise ValueError(
                    "fixed factors make a linear constraint infeasible: "
                    f"{_format_constraint(constraint)}"
                )

            active_constraints.append(
                LinearConstraint(
                    columns=columns,
                    coefficients=coefficients,
                    sense=constraint.sense,
                    rhs=rhs,
                )
            )

        return Domain(
            continuous=active_continuous,
            categorical=active_categorical,
            block_column=self.block_column,
            linear_constraints=active_constraints,
            target=self.target,
        )

    def categorical_combinations(self) -> list[dict[str, str]]:
        """Enumerate the Cartesian product over active categorical factors."""

        active = self.active_factors()
        if not active.categorical:
            return [{}]

        return [
            {
                factor.name: level
                for factor, level in zip(active.categorical, levels, strict=True)
            }
            for levels in product(
                *(factor.levels for factor in active.categorical)
            )
        ]

    def is_feasible(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate all declared linear constraints in physical units."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("is_feasible expects a pandas DataFrame")
        required = sorted(
            {
                column
                for constraint in self.linear_constraints
                for column in constraint.columns
            }
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"missing constraint columns: {', '.join(missing)}")

        feasible = pd.Series(True, index=df.index, dtype=bool)
        for constraint in self.linear_constraints:
            lhs = pd.Series(0.0, index=df.index)
            for name, coefficient in zip(
                constraint.columns, constraint.coefficients, strict=True
            ):
                values = pd.to_numeric(df[name], errors="coerce")
                lhs = lhs + coefficient * values

            tolerance = _CONSTRAINT_TOLERANCE * max(1.0, abs(constraint.rhs))
            if constraint.sense == "<=":
                feasible &= lhs <= constraint.rhs + tolerance
            else:
                feasible &= lhs >= constraint.rhs - tolerance
        return feasible

    def describe(self) -> str:
        """Return a bench-readable summary of the searched physical domain."""

        direction = "maximize" if self.target.direction == "maximize" else "minimize"
        scale = "log" if self.target.transform == "log" else "raw"
        lines = [
            f"Target: {direction} {self.target.name} ({self.target.unit}) [{scale} scale]",
            (
                "Smallest improvement worth pursuing: "
                f"{self.target.delta_practical_pct:g}%"
            ),
        ]

        all_factors = [*self.continuous, *self.categorical]
        seen_groups: list[str] = []
        for factor in all_factors:
            if factor.group not in seen_groups:
                seen_groups.append(factor.group)
        for group in seen_groups:
            lines.append(f"{group.title()}:")
            for factor in all_factors:
                if factor.group != group:
                    continue
                if isinstance(factor, ContinuousFactor):
                    detail = f"{factor.low:g} to {factor.high:g} {factor.unit}"
                    if factor.step is not None:
                        detail += f"; setpoint step {factor.step:g} {factor.unit}"
                else:
                    detail = ", ".join(factor.levels)
                lines.append(f"  - {factor.name}: {detail}")

        if self.block_column is not None:
            lines.append(
                f"Block: {self.block_column} (modeled as a nuisance; never optimized)"
            )
        if self.linear_constraints:
            lines.append("Physical constraints:")
            lines.extend(
                f"  - {_format_constraint(constraint)}"
                for constraint in self.linear_constraints
            )
        else:
            lines.append("Physical constraints: none")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize the complete provenance representation."""

        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Domain:
        """Restore a domain from its provenance representation."""

        return cls.model_validate_json(value)


def _round_to_step(
    values: np.ndarray, factor: ContinuousFactor
) -> np.ndarray:
    if factor.step is None or factor.low == factor.high:
        return np.full_like(values, factor.low) if factor.low == factor.high else values

    max_step_index = math.floor(
        (factor.high - factor.low) / factor.step + _BOUND_TOLERANCE
    )
    step_indices = np.rint((values - factor.low) / factor.step)
    step_indices = np.clip(step_indices, 0, max_step_index)
    rounded = factor.low + step_indices * factor.step
    decimal_places = max(
        _decimal_places(factor.low),
        _decimal_places(factor.high),
        _decimal_places(factor.step),
    )
    return np.round(rounded, decimals=decimal_places)


def _decimal_places(value: float) -> int:
    exponent = Decimal(str(value)).as_tuple().exponent
    return max(0, -exponent)


def _physical_constant_constraint_is_satisfied(
    sense: ConstraintSense, rhs: float
) -> bool:
    tolerance = _CONSTRAINT_TOLERANCE * max(1.0, abs(rhs))
    if sense == "<=":
        return 0.0 <= rhs + tolerance
    return 0.0 >= rhs - tolerance


def _format_constraint(constraint: LinearConstraint) -> str:
    terms: list[str] = []
    for index, (column, coefficient) in enumerate(
        zip(constraint.columns, constraint.coefficients, strict=True)
    ):
        magnitude = abs(coefficient)
        if math.isclose(magnitude, 1.0):
            term = column
        else:
            term = f"{magnitude:g} × {column}"
        if index == 0:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append(f"{'-' if coefficient < 0 else '+'} {term}")
    return f"{' '.join(terms)} {constraint.sense} {constraint.rhs:g}"
