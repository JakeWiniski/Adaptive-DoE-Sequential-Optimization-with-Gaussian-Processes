"""Printable run sheet and the single-page campaign status figure.

Every panel carries a plain-language caption and no panel presents a
conclusion. The run sheet is the artifact that gets the workflow used at the
bench, so its print layout is kept deliberately simple.
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from adoe.domain import Domain

_IDENTITY_COLUMNS = (
    "unit_id",
    "condition_id",
    "round",
    "replicate",
    "block",
    "execution_order",
    "slot_type",
)


def run_sheet(run_set: pd.DataFrame, domain: Domain, path: str | Path) -> tuple[Path, Path]:
    """Write a printable HTML run sheet and a fillable results CSV.

    One row per experimental unit, sorted by ``execution_order``, setpoints
    already on their declared ``step`` grid. Returns the two written paths.
    """

    if run_set.empty:
        raise ValueError("run_set must contain experimental units")
    _validate_setpoints(run_set, domain)

    base = Path(path)
    html_path = base.with_suffix(".html")
    csv_path = base.with_suffix(".results.csv")
    html_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = run_set.sort_values(["execution_order", "unit_id"], kind="stable").reset_index(drop=True)
    result_heading = f"{domain.target.name} ({domain.target.unit})"
    display_columns = ["execution_order", "unit_id", "replicate", *domain.factor_names, "rationale"]
    display = ordered[display_columns].copy()
    display[result_heading] = ""

    round_index = int(ordered["round"].iloc[0])
    blocks = ", ".join(map(str, pd.unique(ordered["block"].dropna())))
    composition = _composition(ordered)
    table_html = display.to_html(index=False, escape=True, border=0, classes="run-table")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Adaptive DoE — round {round_index}</title>
<style>
@page {{ size: landscape; margin: 0.35in; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Arial, sans-serif; margin: 0; color: #172B4D; font-size: 9pt; }}
h1 {{ margin: 0 0 2px; font-size: 17pt; }}
.meta {{ margin: 0 0 7px; color: #52606D; }}
.domain {{ white-space: pre-wrap; background: #F2F4F7; padding: 6px; margin: 0 0 7px;
           font-size: 8pt; line-height: 1.2; }}
table {{ border-collapse: collapse; width: 100%; page-break-inside: avoid; font-size: 7.6pt; }}
th, td {{ border: 1px solid #9FB3C8; padding: 3px 4px; text-align: left; vertical-align: top; }}
th {{ background: #DCE6F0; font-weight: 700; }}
tr {{ page-break-inside: avoid; }}
.footer {{ margin-top: 7px; page-break-inside: avoid; font-size: 8pt; }}
.notes {{ height: 34px; border: 1px solid #9FB3C8; margin-top: 3px; }}
@media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
</style>
</head>
<body>
<h1>Adaptive DoE run sheet</h1>
<p class="meta">Round {round_index} · Block {html.escape(blocks or "not modeled")}</p>
<div class="domain">{html.escape(domain.describe())}</div>
{table_html}
<div class="footer">
<p><strong>Composition:</strong> {html.escape(composition)}</p>
<p>Run in <strong>execution order</strong>. Record the result and note anything unusual.</p>
<p><strong>Notes</strong></p><div class="notes"></div>
</div>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")

    template = ordered[[*_IDENTITY_COLUMNS, *domain.factor_names]].copy()
    template["actual_execution_order"] = ""
    template["y"] = ""
    template["failed"] = ""
    template["failure_reason"] = ""
    template["notes"] = ""
    template.to_csv(csv_path, index=False)
    return html_path, csv_path


def status_page(ledger: pd.DataFrame, domain: Domain, numbers: dict[str, object]) -> Figure:
    """Render the single status figure from the descriptive status numbers."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    figure.suptitle("Campaign status — the workflow reports; you decide", fontsize=13, fontweight="bold")

    _draw_recipe(axes[0, 0], domain, numbers)
    _draw_progress(axes[0, 1], ledger, domain, numbers)
    _draw_learning(axes[1, 0], numbers)
    _draw_calibration(axes[1, 1], domain, numbers)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _draw_recipe(axis: Axes, domain: Domain, numbers: dict[str, object]) -> None:
    axis.axis("off")
    recipe = numbers.get("best_recipe") or {}
    lines = ["Best recipe (model incumbent)", ""]
    if recipe:
        for factor in domain.factor_names:
            lines.append(f"  {factor}: {_fmt(recipe.get(factor))}")
        lines.append("")
        lines.append(
            f"  predicted typical {domain.target.name}: {recipe['predicted_median']:.3g} {domain.target.unit}"
        )
        lines.append(f"  95% interval: {recipe['lo95']:.3g}–{recipe['hi95']:.3g} {domain.target.unit}")
        improvement = numbers.get("improvement_pct")
        if improvement is not None:
            lines.append(f"  vs control: {improvement:+.0f}%")
    else:
        lines.append("  (no successful results yet)")
    axis.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    _caption(axis, "exp(posterior mean) is a typical (median) yield, not an average.")


def _draw_progress(axis: Axes, ledger: pd.DataFrame, domain: Domain, numbers: dict[str, object]) -> None:
    from adoe.campaign import _successful

    successful = _successful(ledger, domain).copy()
    if successful.empty:
        _empty(axis, "Progress")
    else:
        successful["execution_order"] = pd.to_numeric(successful["execution_order"], errors="coerce")
        successful = successful.sort_values(["round", "execution_order"], kind="stable")
        y = pd.to_numeric(successful["y"], errors="coerce").to_numpy(dtype=float)
        cumulative = np.maximum.accumulate(y) if domain.target.direction == "maximize" else np.minimum.accumulate(y)
        units = np.arange(1, len(cumulative) + 1)
        axis.plot(units, cumulative, marker="o", ms=3, label="best observed")
        recipe = numbers.get("best_recipe") or {}
        if recipe:
            axis.axhline(recipe["predicted_median"], color="#B54708", ls="--", label="model incumbent")
        axis.set_xlabel("cumulative successful units")
        axis.set_ylabel(f"{domain.target.name} ({domain.target.unit})")
        axis.legend(fontsize=8, loc="best")
    axis.set_title("Progress")
    _caption(axis, "Best result seen so far against effort spent.")


def _draw_learning(axis: Axes, numbers: dict[str, object]) -> None:
    rmse = numbers.get("learning_rmse")
    baseline = numbers.get("baseline_sd")
    if rmse is None or baseline is None:
        _empty(axis, "Is it learning?")
    else:
        axis.bar(["model RMSE", "guess-the-mean"], [rmse, baseline], color=["#1D6F42", "#98A2B3"])
        axis.set_ylabel("log-scale error")
    axis.set_title("Honesty check A — is it learning?")
    _caption(axis, "If these are equal, the model is no better than guessing the average.")


def _draw_calibration(axis: Axes, domain: Domain, numbers: dict[str, object]) -> None:
    axis.axis("off")
    lines = ["Honesty check B — are the error bars honest?", ""]
    rms_z = numbers.get("rms_z")
    coverage = numbers.get("coverage")
    if rms_z is not None:
        flag = "" if rms_z <= 1.25 else "  (bars look too narrow)"
        lines.append(f"  rms_z = {rms_z:.2f}  (target ≈ 1.0){flag}")
        lines.append(f"  95% coverage = {coverage:.0%}  (target ≥ 90%)")
    else:
        lines.append("  (no pre-recorded predictions to score yet)")
    lines.append("")
    lines.append("Noise and replication")
    noise = numbers.get("noise_pct")
    if noise is not None:
        lines.append(f"  repeat-to-repeat noise ≈ ±{noise:.0f}% between identical replicates")
        lines.append(f"  worthwhile gain: {numbers['delta_pct']:g}%")
        lines.append(f"  rough replicates to detect it: {numbers['replicates_guide']}")
    else:
        lines.append("  (needs at least one replicated condition)")
    failure_rate = float(numbers.get("failure_rate") or 0.0)
    if failure_rate > 0:
        lines.append("")
        lines.append(f"Failure rate: {failure_rate:.0%} of units")
    axis.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5, family="monospace")
    _caption(axis, "rms_z uses RMS, not SD, so it catches systematic bias. Rough guide, not a power calc.")


def _composition(run_set: pd.DataFrame) -> str:
    counts = run_set["slot_type"].value_counts()
    parts = [f"{int(count)} {slot}" for slot, count in counts.items()]
    return f"{len(run_set)} units — " + ", ".join(parts)


def _validate_setpoints(run_set: pd.DataFrame, domain: Domain) -> None:
    reencoded = domain.decode(domain.encode(run_set[domain.factor_names]))
    for factor in domain.continuous:
        if not np.allclose(
            pd.to_numeric(run_set[factor.name]).to_numpy(dtype=float),
            reencoded[factor.name].to_numpy(dtype=float),
            atol=1e-9,
        ):
            raise ValueError(f"run sheet setpoint for {factor.name!r} is off its declared step grid")


def _fmt(value: object) -> str:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _caption(axis: Axes, text: str) -> None:
    axis.text(0.5, -0.16, text, transform=axis.transAxes, ha="center", va="top", fontsize=8, color="#52606D", wrap=True)


def _empty(axis: Axes, title: str) -> None:
    axis.text(0.5, 0.5, "not enough data yet", ha="center", va="center", fontsize=10, color="#98A2B3")
    axis.set_title(title)
