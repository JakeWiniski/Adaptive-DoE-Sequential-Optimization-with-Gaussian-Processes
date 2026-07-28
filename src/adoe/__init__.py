"""Adaptive design of experiments for a small biology team."""

from adoe.campaign import (
    Status,
    compute_status,
    confirm,
    initial_design,
    propose,
    read_ledger,
    record,
    save_proposal,
    status,
)
from adoe.domain import (
    CategoricalFactor,
    ContinuousFactor,
    Domain,
    LinearConstraint,
    TargetSpec,
)
from adoe.model import (
    Aggregation,
    ModelMetadata,
    aggregate,
    build_model,
    encode_model_inputs,
    fit_model,
    model_metadata,
    predict,
    to_model_scale,
)
from adoe.report import run_sheet, status_page
from adoe.simulate import (
    FermentationConfig,
    FermentationSimulator,
    NullSimulator,
    Simulator,
    default_fermentation_domain,
)

__all__ = [
    "Aggregation",
    "CategoricalFactor",
    "ContinuousFactor",
    "Domain",
    "FermentationConfig",
    "FermentationSimulator",
    "LinearConstraint",
    "ModelMetadata",
    "NullSimulator",
    "Simulator",
    "Status",
    "TargetSpec",
    "aggregate",
    "build_model",
    "compute_status",
    "confirm",
    "default_fermentation_domain",
    "encode_model_inputs",
    "fit_model",
    "initial_design",
    "model_metadata",
    "predict",
    "propose",
    "read_ledger",
    "record",
    "run_sheet",
    "save_proposal",
    "status",
    "status_page",
    "to_model_scale",
]

__version__ = "0.2.0"
