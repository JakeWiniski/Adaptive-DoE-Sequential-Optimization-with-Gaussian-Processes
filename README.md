# Adaptive DoE

**Find a better recipe in fewer experiments — one small round at a time.**

Adaptive DoE is a lightweight, operator‑facing workflow for **sequential design of experiments**.
You describe the settings you can change and the single result you want to improve; it proposes a
smart batch of conditions to run next, keeps an honest record of what happened, and — critically —
tells you *how much it has actually learned* so you know whether to trust its recommendation. It is
built for a small R&D team to run and maintain **without a statistician or a software engineer**.

Almost every number it reports is a physical measurement or a percentage, and every figure carries a
plain‑language caption. The two calibration diagnostics (`rms_z` and the log‑scale error bars) are
the deliberate exceptions, and each is explained on the page it appears on. If you can describe your
experiment in real units and fill in a spreadsheet of results, you can run it.

- **[What it is and why](#what-it-is-and-why)**
- **[How it works (the method)](#how-it-works-the-method)**
- **[The acquisition function (how recipes are chosen)](#the-acquisition-function-how-recipes-are-chosen)**
- **[Install](#install)**
- **[Quickstart](#quickstart)**
- **[Walking through a campaign](#walking-through-a-campaign)**
- **[The campaign ledger](#the-campaign-ledger)**
- **[Reading the status page](#reading-the-status-page)**
- **[Defining your experiment](#defining-your-experiment)**
- **[Reproducibility and provenance](#reproducibility-and-provenance)**
- **[Validation](#validation)**
- **[Limitations](#limitations-what-you-give-up)**
- **[Repository layout](#repository-layout)**

---

## What it is and why

### The goal
You want to find the combination of settings — temperature, moisture, medium composition, strain,
substrate, incubation time — that maximizes a result you care about: a **yield, biomass, titer,
growth rate, or efficiency**. The hard part is the setting: each run is typically **slow, noisy,
and expensive**, so your entire budget may be only a few dozen experiments.

### Traditional DoE, and where adaptive design fits
Classical design of experiments offers a rich, well‑proven toolkit — fractional factorials and
Plackett–Burman screening designs, response‑surface designs (central composite, Box–Behnken),
space‑filling designs, and optimal designs (D‑, I‑optimal). These are excellent when you can commit
the full plan up front and conditions are stable. What they have in common is that the **entire
experimental plan is fixed in advance**: you choose a design, run all of it, then analyze.

**Adaptive** (sequential) DoE makes a different trade — and that trade is its whole point. It runs
experiments in **small rounds** and uses each round's *actual* results to choose the next batch,
which gives a combination the one‑shot designs cannot:

- **Adaptability to real‑world conditions.** Each round is chosen *after* seeing the previous
  results, so the campaign can react to what actually happens — a surprising winner, a run of
  failures, batch‑to‑batch (block) shifts, drift over time — rather than committing every run before
  any of them are observed.
- **Maximum learning efficiency within those conditions.** With a limited budget it directs runs
  where they teach the most about the *optimum*: concentrating on the promising region while still
  probing where it is uncertain, instead of spreading effort evenly or fitting a fixed low‑order
  model across the whole space.

That makes it especially well suited when each experiment is costly and the response surface is
unknown and messy. It is a **complement to** classical DoE, not a replacement — in exchange for this
adaptability it gives up the closed‑form guarantees and simple analysis of a pre‑specified design
(see [Limitations](#limitations-what-you-give-up)).

### The practical use case
The motivating domain is applied biology — mycelial cultures, fermentation, media/substrate
optimization — where each run is slow and expensive, replicates are noisy, and you can only afford a
few dozen experiments total. But nothing is mushroom‑specific: it fits **any positive response with
proportional variation**, and a "replicate" is whatever your experimental unit is — a plate, flask,
tray, plot, or bag.

### The one idea that keeps everything simple
It models **`log(response)`** rather than the raw response. Biological results have noise that
**scales with their level** — a large flush varies more in absolute terms than a small one, but by
roughly the same *percentage*. Working on the log scale turns that proportional variation into
approximately **constant** noise, which is what lets the workflow report noise as a single
percentage (±X%), state the smallest worthwhile improvement as a percentage, and produce intervals
that are correct on the ratio scale. (For a genuinely additive response that can be zero or negative
— a temperature, a pH, a signed difference — set `transform="none"`, and read
[the warning about `record()`](#the-transformnone-warning) before you record anything.)

---

## How it works (the method)

The workflow is a **closed loop**:

> **propose** a round of recipes → **run** them → **record** the results → the model **learns** →
> **propose** the next round → …

Everything accumulates in a **single CSV** (one row per replicate), so the model always learns from
*all* results to date, and that file is your campaign record. See
[The campaign ledger](#the-campaign-ledger) for its schema and what you are expected to fill in.

Under the hood, a few deliberate choices do the work:

- **A Gaussian‑process surrogate on the log scale.** After each round the workflow fits a
  Gaussian process — a flexible model that predicts the response *and its uncertainty* across the
  whole search space, including combinations you have not run. Categorical factors (strain,
  substrate) are handled natively alongside continuous ones.
- **Balanced exploration and exploitation.** Recipes are chosen with *noisy expected improvement*,
  which trades pushing on the current best against reducing uncertainty where the model is unsure.
  Points are selected one at a time, each aware of the others already picked this round, so a batch
  is **diverse** rather than a dozen near‑duplicates. The very first round has nothing to fit, so it
  lays down an even **space‑filling** design.
- **Classical DoE discipline, kept.** Every round includes a fixed **control** (your reference
  recipe), so "how much better" is always anchored. Each round is a **block** (a batch/week/
  incubator), modeled as a nuisance so its shared quirks don't masquerade as factor effects. And
  run order is **randomized within the block**, so slow drift over the day can't bias the comparison.
- **Setpoints you can actually hit.** Each continuous factor can declare a `step` (the resolution of
  your equipment); every proposed setpoint is rounded to that grid. Omit `step` and the setpoint is
  returned at full floating‑point precision — usually not what you want at a bench.
- **Write‑once predictions.** The model's prediction for each recipe is written to the ledger
  **before** you run it, and `record()` refuses a results file in which any setpoint or stored
  prediction was altered. That makes the honesty checks below a fair, after‑the‑fact test the model
  cannot game. It is an honesty aid, not a cryptographic guarantee — the ledger is a plain CSV that
  anyone can edit outside the API.
- **Two honesty checks, and no stopping rule.** The status page reports (A) whether the model
  predicts better than just guessing the average, and (B) whether its error bars are honestly sized.
  Both stay blank until you have recorded at least one **adaptive** round — round 0 is a
  space‑filling design and carries no predictions to score. The page **never tells you to stop**; it
  reports what it knows, and *you* decide.

A `propose` call is not instantaneous: it fits the GP and runs a multi‑restart acquisition
optimization over the mixed continuous/categorical space. Expect seconds to a couple of minutes
depending on campaign size and the `num_restarts` / `raw_samples` settings.

---

## The acquisition function (how recipes are chosen)

Perhaps surprisingly, the workflow makes **no explicit explore‑vs‑exploit decision**. There is no
"exploration run" and "exploitation run", and no preset that allocates some slots to one and some to
the other. Every model‑chosen recipe in a round comes from the *same* acquisition function, which
balances exploration and exploitation **continuously and automatically** through the model's own
uncertainty.

### One function does both: noisy log expected improvement
For each model‑chosen point, the workflow builds a single acquisition —
`qLogNoisyExpectedImprovement` (qLogNEI) — and proposes the recipe that maximizes it. Expected
Improvement (EI) is where the tradeoff lives.

At any candidate recipe **x**, the fitted Gaussian process returns a predictive distribution: a
**mean** μ(x) (its best guess of the response there) and an **uncertainty** σ(x) (how unsure it is).
EI is the *expected amount by which the response at x would beat the best you have seen so far*,
integrated over that distribution. A candidate scores high on EI if **either**:

- **μ(x) is high** — it sits in a region the model already believes is good. This is **exploitation**.
- **σ(x) is large** — even with a middling mean, the wide uncertainty means there is a real chance it
  turns out much better than expected. This is **exploration**.

Points that are **both** promising and uncertain score highest. So a single candidate can be
exploitative, exploratory, or a blend, and EI simply picks whatever maximizes expected improvement —
the balance is baked into the formula, not chosen by a flag or a fraction. Early in a campaign, when
the model is uncertain almost everywhere, EI naturally spreads out (explores); as evidence
accumulates and uncertainty shrinks, it naturally tightens toward the current best (exploits). That
transition is automatic and gradual, not a switch that flips.

### Keeping a batch diverse
A round proposes several model‑chosen recipes, selected **one at a time**. Each recipe already chosen
this round is fed back to the acquisition as a *pending* point before the next is selected:

```python
# each recipe already picked this round is passed as X_pending before choosing the next
acquisition = qLogNoisyExpectedImprovement(
    model=model,
    X_baseline=observed_conditions,        # everything recorded so far
    X_pending=already_selected_this_round, # the batch-diversity mechanism
    sampler=SobolQMCNormalSampler(...),    # Monte-Carlo samples for the noisy posterior
)
```

qLogNEI treats those pending points as if they will be observed, which **lowers the acquisition value
near them**. The next selection is therefore pushed toward a different useful region. That is why a
batch comes out varied — some points near the incumbent, some in under‑sampled areas — without any
explicit "make this one an exploration point" rule. The diversity is an *emergent* property of
sequential (greedy) EI with pending points, not an allocation.

### The "Noisy" and "Log" parts
- **Noisy (NEI).** Because replicates are noisy, it does not anchor "improvement" to the single best
  *observed* value (which could be a lucky high reading). It integrates over the model's posterior
  about what the true best‑so‑far actually is, using Monte‑Carlo samples. This keeps it from
  over‑exploiting noise.
- **Log (qLogEI).** A numerically stable, log‑space reformulation of EI. Plain EI has vanishing
  gradients far from the incumbent, which starves the optimizer; the log form fixes that so the
  optimizer reliably finds strong candidates.

### The rest of the selection pipeline
For each point, the acquisition is maximized over the **mixed** space: the categorical combinations
(strain × substrate × …) are enumerated, and the continuous factors are optimized within each,
subject to your bounds and any `<=` / `>=` constraints, with the block held at the current round. The
winner is then **rounded to your `step` grid**, with a small collision guard so two proposals cannot
land on the same achievable setpoint. Finally the **control** is appended — the one condition *not*
chosen by the acquisition; it is a fixed reference included in every round.

### A note on the "explore" label in the ledger
Every model‑chosen row is written with `slot_type = "explore"` (and a rationale beginning
"Best‑guess — predicted typical …"), while the control is `slot_type = "control"`. Do not read a mode
into the word "explore": it only distinguishes model‑chosen recipes from the control. **There is no
`"exploit"` slot type**, and these rows are not "exploration‑mode" runs — they are all qLogNEI
selections whose explore/exploit character is internal to EI and varies from point to point.

### Why it is built this way
An earlier version of this project had an explicit explore/exploit portfolio: presets that split each
round between pure variance‑reduction (exploration) and expected improvement (exploitation), tuned by
an `explore_fraction`. That layer was removed during simplification, because qLogNEI with `X_pending`
is *already* batch‑aware and *already* trades exploration against exploitation through the posterior —
the extra machinery did not earn its keep. The honest description today is: **one acquisition
function, EI‑based, that handles the whole tradeoff on its own, per point, automatically.**

---

## Install

The pinned scientific stack (`torch==2.2.2`, `botorch==0.17.2`) is built and tested against
**Python 3.11**. `pyproject.toml` currently declares a wider window (`>=3.11,<3.14`), but 3.11 is
the version to use — other minors may fail to resolve wheels for the pinned `torch`.

```bash
git clone <your-repo-url> adaptive-doe && cd adaptive-doe

python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
```

`[dev]` brings the test dependencies; `[notebooks]` brings `ipykernel` and the notebook‑execution
libraries. **Neither extra installs Jupyter itself** — supply your own, then register this
environment as a kernel:

```bash
pip install jupyterlab          # skip if you already have Jupyter
python -m ipykernel install --user --name adaptive-doe
```

Select the **adaptive-doe** kernel when you open either notebook.

---

## Quickstart

Open **`notebooks/run_a_round.ipynb`** and run it top to bottom — it walks through one round with a
built‑in simulator standing in for the bench, and every step is explained in plain language.

The core API is a handful of calls:

```python
from pathlib import Path
from adoe import (read_ledger, propose, save_proposal, run_sheet, record,
                  status, confirm, default_fermentation_domain)

path = "my_campaign.csv"                 # one CSV holds the whole campaign
domain = default_fermentation_domain()   # the built-in demo domain; build your own instead
                                         # — see "Defining your experiment"

# Save the domain next to the ledger: the CSV records factor VALUES, not the domain itself.
Path("domain.json").write_text(domain.to_json())

# --- one round -------------------------------------------------------------
run = propose(                           # round 0 -> space-filling design
    read_ledger(path), domain,
    n_conditions=12,                     # recipes this round, INCLUDING the 1 control
    r=1,                                 # replicates per recipe
    seed=0,                              # same seed -> same plan
    control={                            # YOUR reference recipe — see "The control"
        "carbon_concentration": 30.0, "nitrogen_concentration": 3.0,
        "temperature": 30.0, "pH": 6.0,
        "strain": "A", "carbon_source": "glucose", "nitrogen_source": "yeast_extract",
    },
)
save_proposal(path, run)                 # append proposal + write-once predictions
html_path, results_path = run_sheet(run, domain, "round")   # round.html + round.results.csv

# ... run the experiment, type your measurements into round.results.csv ...

record(path, results_path, domain)       # fill outcomes; identity/predictions locked
                                         # PASS `domain` — see the transform="none" warning
report = status(path, domain)            # best recipe + honesty checks
report.figure.savefig("status.png", dpi=150)   # adoe.report forces matplotlib's Agg backend;
                                               # in a notebook run %matplotlib inline first

# --- next round: same call, the campaign keeps growing in the same CSV ------
run = propose(read_ledger(path), domain, n_conditions=8, r=1, seed=0)

# --- when a winner emerges, replicate it against the control to lock it in --
run = confirm(read_ledger(path), domain, r=3)
save_proposal(path, run)                 # confirm returns a run set like any other:
run_sheet(run, domain, "confirm")        # save it, print it, run it, record it
```

**A note on arguments.** `propose` and `confirm` take a **DataFrame** (`read_ledger(path)`);
`record` and `status` take the ledger **path**. That asymmetry is deliberate — the first two are
pure functions of the data, the second two write to or read the file — but it is easy to trip on.

`propose` auto‑detects a cold start — an empty campaign, *or* one in which nothing has succeeded
yet — and lays down the space‑filling design; on later rounds it fits the model and proposes
adaptively. You never call the model directly.

Instead of `n_conditions`, you can budget by total bench slots with `capacity=<total replicates>`;
the workflow divides by `r` and warns if the division leaves unused units.

---

## Walking through a campaign

A campaign is just this loop, repeated until you have a recipe you're happy with. In practice:

1. **Describe your experiment once** (the *domain*): your factors and their ranges, the response to
   maximize, and the smallest improvement worth chasing (as a percentage). See
   [Defining your experiment](#defining-your-experiment).
2. **Propose round 0.** With no data yet, you get an even spread of recipes across the whole space,
   balanced across your strains/substrates, plus the control. Remember `n_conditions` **includes**
   that control.
3. **Print the run sheet, run the round.** Take `round.html` to the bench and run the replicates
   **in the execution order shown** (it's randomized on purpose). This is where the real‑world time
   goes — for a living culture, days or weeks.
4. **Record the results.** Type each measured value into `round.results.csv` (and mark failures —
   see [The campaign ledger](#the-campaign-ledger)), then `record(...)` writes them into the
   campaign.
5. **Read the status page** (see [Reading the status page](#reading-the-status-page)) to see the best
   recipe so far and whether the model can be trusted yet. After round 0 the two honesty checks will
   be blank — that is expected, not a bug.
6. **Decide what to do next — the workflow won't decide for you:**
   - **Still improving and the model is learning** → run **another round** (call `propose` again;
     the campaign keeps growing in the same CSV).
   - **A clear front‑runner has emerged** → run a **confirmation round** (`confirm(..., r=3)`),
     which replicates the model's pick, the best result you actually measured, and the control side
     by side, so you can trust the comparison before committing or scaling.
   - **Nothing is improving after a few rounds** → revisit the domain: your ranges may be too narrow
     (the optimum fenced out) or too wide (budget spread thin), or the factors you chose may not be
     the ones that matter.

Because everything lives in one CSV, you can stop and resume anytime, open the campaign in any
spreadsheet, and hand it off without losing the thread — provided you hand over the saved
`domain.json` alongside it (see [Reproducibility and provenance](#reproducibility-and-provenance)).

---

## The campaign ledger

The ledger is a single CSV, **one row per experimental unit (replicate)**. `run_sheet` writes a
blank data‑entry copy of the current round (`round.results.csv`) containing the identity columns
already filled in; you type your results into it and `record()` merges them back.

### What you fill in

| Column | Who fills it | Notes |
|---|---|---|
| `y` | **you** | the measurement, in your target's units. Leave blank if the unit failed. |
| `failed` | **you** | `TRUE`/`yes`/`y`/`1` or `FALSE`/`no`/`n`/`0`. Blank means "not failed". |
| `failure_reason` | you (optional) | free text — contamination, dropped plate, lost sample |
| `notes` | you (optional) | free text |
| `actual_execution_order` | you (optional) | fill in only if you deviated from the printed order |
| everything else | **locked** | `record()` rejects the file if any of these changed |

The locked columns present in the results file are `unit_id`, `condition_id`, `round`, `replicate`,
`block`, `execution_order`, `slot_type`, and every factor column — change any of them and `record()`
refuses the file. The remaining ledger columns (`proposal_order`, `rationale`, and the four
prediction columns `pred_median`, `pred_lo95`, `pred_hi95`, `pred_sd_log`) are deliberately left out
of the results file altogether: they are written once at proposal time and are never writable.

### Two behaviours worth knowing before you type

- **`failed` accepts only the values listed above.** Writing something descriptive like
  `contaminated` into `failed` raises an error and aborts the whole `record()` call — put the
  description in `failure_reason` and a plain `TRUE` in `failed`.
- **Under the default `log` transform, a `y` of zero or less is recorded as a failure**, with
  reason `"no yield"`, not as a small number. That is the correct default for a yield (you cannot
  take the log of zero, and a dead replicate is genuinely a different kind of event), but it means a
  literal zero you meant to keep will not survive.

<a id="the-transformnone-warning"></a>

### The `transform="none"` warning

`record()` takes an optional third argument:

```python
record(ledger_path, results_csv, domain)
```

**If you omit `domain`, non‑positive `y` values are converted to failures regardless of your
transform.** So a campaign built with `transform="none"` — which this README recommends for
temperatures, pH, percentages that can hit zero, and signed differences — will silently discard
every zero and negative measurement unless you pass `domain`.

Pass `domain` every time. It costs nothing under the default `log` transform and prevents silent
data loss under `"none"`.

---

## Reading the status page

`status(path, domain)` returns a small `Status` object (with fields like `best_recipe`,
`improvement_pct`, `rms_z`, `coverage`, …) **and** a one‑page figure (`report.figure`) laid out as a
**2×2 grid of four panels**: the recipe card, progress, honesty check A, and honesty check B.
Noise/replication and the failure rate are text inside the honesty‑check‑B panel, not panels of
their own.

Two terms recur:

- **Model incumbent** — the recipe the *fitted model* currently believes is best: its top
  recommendation, found by searching the model's predicted surface. It may be a recipe you have
  **not actually run yet**.
- **Best observed** — the best result you have **actually measured**. As a campaign matures these two
  converge; a large gap means the model's pick still needs a real run to confirm it.

| Panel / metric | What it shows | How to read it |
|---|---|---|
| **Best recipe** (top‑left panel) | The recommended setpoints, the **predicted typical (median) response** with a 95% interval, and the **predicted % improvement over the control**. | "Typical" is a median (log scale), not an average. The interval is the range a single new replicate should fall in 95% of the time — *if check B passes*. The improvement is model‑prediction vs. model‑prediction, not a measured comparison — which is exactly why the confirmation round exists. |
| **Progress** (top‑right panel) | Best result observed vs. cumulative successful runs, with the model incumbent's *predicted* value drawn as a horizontal line. | Rises‑then‑flattens = converging; still climbing = keep going; flat = nothing improving. A big gap between the incumbent line and best observed → run a confirmation. |
| **Honesty check A — is it learning?** (bottom‑left panel) | The model's prediction error vs. the error of **just guessing the average** ("guess‑the‑mean"), lower is better. | Model bar **much lower** = it's learning real structure; **roughly equal** = it hasn't learned yet (be skeptical of recommendations). See the fine print below on how the two bars are computed. |
| **Honesty check B — honest error bars?** (bottom‑right panel) | **`rms_z`** (target ≈ 1.0) and **95% coverage** (target ≥ 90%). | `rms_z` ≈ 1 and coverage ≥ 90% → the recipe card's interval is trustworthy. `rms_z` above ~1.25 or coverage below 0.90 → the bars are **too narrow / overconfident**; treat intervals as rough and confirm before committing (common early on). |
| **Noise & replication** (text, in the check‑B panel) | Repeat‑to‑repeat noise as **±%**, your worthwhile‑gain threshold, and a rough count of replicates needed to detect a gain that size. | The noise is the bar an improvement must clear to be real. Use the replicate count to decide how hard to replicate a candidate. **Blank until at least one condition has been run more than once.** |
| **Failure rate** (text, shown only if nonzero) | Share of completed units that failed. | A rising rate often means you're pushing into a hostile region (extreme temperature/pH). |

The guiding principle: **the page reports, you decide.** There is deliberately no "stop" verdict —
`rms_z`, coverage, and the learning check give you the evidence to make that call yourself.

### Fine print on the numbers

Worth knowing if you are going to quote these figures to anyone:

- **Both honesty checks need an adaptive round.** They score only units that carry a *pre‑recorded*
  prediction. Round 0 is a space‑filling design with no predictions, so the checks read
  "(no pre‑recorded predictions to score yet)" until round 1 is recorded.
- **The two bars in check A are computed over different sets.** The model bar (RMSE) covers only the
  scored units — those with a stored prediction. The "guess‑the‑mean" bar is the spread of *all*
  successful results to date, including round 0. The comparison is still directionally informative
  and errs conservative, but it is not a strict like‑for‑like.
- **Noise as "±%" is an approximation.** The reported figure is the log‑scale SD read as a percent.
  That is accurate at ±10–15%; at larger values the true interval is asymmetric (an SD of 0.30 on
  the log scale is really about +35% / −26%, not ±30%).
- **The replicate count is a rough guide, not a power calculation** — the figure caption says so too.
- **The recipe card's predicted value and interval are conditioned on the campaign's first block.**
  The search that finds the incumbent runs against the *next, unseen* block; the number printed on
  the card is then evaluated at block 0. The recommended recipe is unaffected; the absolute
  predicted level carries that block's offset.

---

## Defining your experiment

The *domain* is the single source of truth — factor names, ranges, units, the response. You build
it once with plain constructors:

```python
from adoe import (Domain, ContinuousFactor, CategoricalFactor, TargetSpec, LinearConstraint)

domain = Domain(
    continuous=[
        # name, group (a run-sheet heading), low/high range, unit, and `step` = the finest
        # increment your equipment can set (proposals are rounded to this grid)
        ContinuousFactor(name="carbon",      group="media",   low=10, high=70, unit="%",  step=1.0),
        ContinuousFactor(name="nitrogen",    group="media",   low=5,  high=40, unit="%",  step=0.5),
        ContinuousFactor(name="temperature", group="process", low=20, high=28, unit="°C", step=0.5),
    ],
    categorical=[
        # unordered choices; 2-6 levels is comfortable, data-hungry beyond ~8.
        # A purely continuous campaign still needs an explicit `categorical=[]`.
        CategoricalFactor(name="strain",    group="strain",    levels=["A15", "B22", "C07"]),
        CategoricalFactor(name="substrate", group="substrate", levels=["straw", "sawdust"]),
    ],
    block_column="block",              # nuisance grouping; the workflow assigns one block per round
    linear_constraints=[],             # optional "<=" / ">=" limits linking continuous factors
    target=TargetSpec(
        name="yield", unit="g",
        direction="maximize",          # or "minimize" (days to colonize, contamination rate)
        transform="log",               # "log" for a positive/proportional response; "none" if additive
        delta_practical_pct=10.0,      # smallest improvement worth chasing, as a PERCENT
    ),
)
print(domain.describe())               # echoes the exact space that will be searched
```

Notes:

- **`delta_practical_pct`** is not a target to hit — it's the *resolution you care about*. It drives
  the "replicates to detect it" guide on the status page.
- **Constraints** support only `"<="` and `">="` (there is no equality). Use them for a physical or
  operational limit that couples factors, e.g. `carbon + nitrogen <= 100`.
- **Fixed‑total mixtures** (components that must sum to a total, e.g. a medium adding to 100%): there
  is no `sum == total` constraint, but you handle it by **varying `n − 1` components** as factors and
  computing the last as the remainder, bounding it with a `<=`/`>=` constraint so it stays in range.
  The operator notebook has a worked example.
- **Categorical combinations are capped at 100.** The Cartesian product of your active categorical
  factors must not exceed 100 combinations, or `propose` and `confirm` raise a `ValueError` and ask
  you to fix some factors for this campaign. Three factors with 5, 5, and 5 levels is 125 — over the
  line. Note that the cap is enforced once a model is being fitted, so an over‑sized campaign gets
  through round 0 and fails on round 1. Check before you design the campaign, not after.
- **Holding a factor fixed:** set `low == high` on a continuous factor, or give a categorical factor
  a single level. It is printed on every run sheet but not searched or optimized.
- **`transform="none"`** changes how you must call `record()` — see
  [the warning above](#the-transformnone-warning).

### The control

Every round includes a **control**: your fixed reference recipe. It anchors every "% improvement"
number in the campaign, so it is worth setting deliberately. `propose`, `initial_design`, and
`confirm` resolve it in this order:

1. **You pass `control={...}`** — a full dict of factor values. Use this on round 0.
2. **Otherwise the most recent control row already in the ledger is reused.** This is why you only
   need to pass it once, on the first round.
3. **Otherwise** the workflow emits a `UserWarning` and falls back to the **domain midpoint with the
   first level of every categorical factor** — which is almost certainly not your reference recipe.

If you see that warning on round 0, stop and supply a control. Every improvement figure for the rest
of the campaign is measured against whatever gets chosen here.

```python
run = propose(read_ledger(path), domain, n_conditions=12, r=1,
              control={"carbon": 40, "nitrogen": 20, "temperature": 24,
                       "strain": "A15", "substrate": "straw"})
```

The control must name **every** factor in the domain, and must remain feasible after setpoint
rounding, or you get a clear error.

---

## Reproducibility and provenance

- **Seeds.** `initial_design`, `propose`, and `confirm` all take `seed=`, and derive independent
  per‑round streams from it. Rerunning with the same seed, ledger, and domain reproduces the same
  plan — which is what makes the campaign auditable after the fact. Keep it fixed; change it only
  when you deliberately want a different draw.
- **The ledger does not contain the domain.** It stores factor *values*, not ranges, units, steps,
  levels, or the target spec. A campaign CSV on its own is not self‑describing, and both `status()`
  and `propose()` require you to reconstruct an identical `Domain`. Save it explicitly:

  ```python
  from pathlib import Path
  from adoe import Domain

  Path("domain.json").write_text(domain.to_json())          # once, when you define the campaign
  domain = Domain.from_json(Path("domain.json").read_text()) # every time you resume
  ```

  Keep `domain.json` next to `my_campaign.csv` and hand both over together.

---

## Validation

```bash
pytest            # fast unit + integration suite (~45s; the end-to-end checks are marked slow)
pytest -m slow    # the end-to-end checks (a few minutes)
python check.py   # the full honesty check (~5 minutes)
```

`check.py` runs the closed loop against a **mechanistic fermentation simulator** and confirms that
the adaptive loop beats a matched **space‑filling** design on **median final regret across 5 seeds**
(a median comparison, not a significance test), and that on a **null** response (pure noise) it does
**not** claim to learn.

For a **visual** and statistically robust version — the test summary, a 15‑seed
adaptive‑vs‑space‑filling comparison with a **paired Wilcoxon signed‑rank test**, a
regret‑vs‑effort trajectory, an observed‑performance trajectory, and the calibration checks, all
with plots — run **`notebooks/test_performance_report.ipynb`**.

> **Budget real time for it.** Section 2 of that notebook runs 15 seeds × 5 rounds at full
> acquisition settings — roughly **1–3 hours on a laptop**. For a quick smoke pass first, drop
> `SEEDS_COMPARISON` to `range(3)` in the setup cell. The remaining sections take minutes.

The copy of `test_performance_report.ipynb` in the repository is committed **with its outputs**, so
you can read the validation results without running anything. Those numbers are a recorded run on
one machine — rerun it yourself if you need results for your own environment.

---

## Limitations (what you give up)

Know the edges of the tool:

1. **One response at a time.** No multi‑objective. If you also care about, say, contamination, run
   yield as the response and watch contamination as the **failure rate**.
2. **No native mixture designs.** Components that must sum to a fixed total aren't a built‑in
   constraint; handle them by varying `n − 1` components and bounding the remainder (see above).
3. **The log transform assumes a positive response with proportional variation** (yield, biomass,
   efficiency). It does not fit a temperature or a signed difference — use `transform="none"`, and
   pass `domain` to `record()` when you do.
4. **Categorical factors are treated as unordered and equally different.** Fine for 2–6 strains or
   substrate types; data‑hungry beyond about 8, and hard‑capped at 100 total combinations.
5. **Batch selection is greedy, not jointly optimal.** Standard practice, and the honest description.
6. **The workflow never says stop.** It reports what it knows; you decide.
7. **The recipe card's interval is a true 95% interval only if the honesty checks pass.** If `rms_z`
   runs above ~1.25 or coverage below 0.90, the intervals are narrower than reality — treat them as
   rough. On a space like the example above (three continuous factors, two categoricals) with 30–60
   replicates, expect that early on.
8. **The honesty checks are diagnostics, not guarantees.** They compare a model bar and a baseline
   bar computed over slightly different sets of units, and they cannot score anything until an
   adaptive round has been recorded. See the fine print in
   [Reading the status page](#reading-the-status-page).
9. **Validated on a simulator, not a real process.** `check.py` shows the loop works on a synthetic
   mycelial‑growth model. It establishes nothing about your specific substrate — that's what your
   own campaign is for.

---

## Repository layout

```
src/adoe/
  __init__.py    the public API — every call in the Quickstart is exported here
  domain.py      factors, units, step, encode/decode, constraints, to_json/from_json
  model.py       Gaussian-process fit + predict, on the log scale
  campaign.py    initial design, propose, confirm, ledger, record, status  (the whole loop)
  report.py      printable run sheet + one-page status figure
  simulate.py    dev/test only: mechanistic + null simulators
                 (it ships inside the installed package, but nothing in the loop depends on it)
notebooks/
  run_a_round.ipynb             the operator notebook — run one round of the loop, fully explained
  test_performance_report.ipynb validation report: tests + adaptive-vs-space-filling comparison
                                (committed with outputs; ~1-3 hours to re-run in full)
tests/
  conftest.py                   puts the repo root on sys.path so `check` is importable
  test_domain.py, test_model.py, test_campaign.py, test_report.py, test_check.py
check.py                        the one honesty check
pyproject.toml                  package + pinned dependencies
```

The whole thing is intentionally small — five source modules, one operator notebook, five test
files, and one validation script — so a bench team can read, run, and maintain it end to end.

---

## AI Use Disclosure

This project was developed with assistance from generative AI and coding-agent tools. AI was used for coding, debugging, and development support; analytical design, validation, interpretation, and final decisions were performed by the author.
