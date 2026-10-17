# Architecture Decision Records

These records are short. Each one gives the context, the decision and its
consequences.

## ADR-001: Python 3.12 with pip-tools hashed lockfiles

- **Context:** The brief allows uv or pip-tools. uv is not installed on the
  development machine. The ML libraries planned for later phases (LightGBM, PyTorch,
  PyG) have mature wheels for 3.12 but not yet for 3.14.
- **Decision:** Target Python 3.12, with `requires-python >= 3.11`. Use pip-tools
  `--generate-hashes` lockfiles: `requirements/base.txt` for runtime and
  `requirements/dev.txt` for tooling. Installs use `--require-hashes`.
- **Consequences:** The supply chain is pinned and hash-verified. Dependencies are
  added per phase so the lock stays small.

## ADR-002: Account node ID = bank code + account ID

- **Context:** Account strings could repeat across banks. Bank codes carry
  meaningful leading zeros.
- **Decision:** Node ID is `"<bank>:<account>"`, and bank codes are always read as
  strings.
- **Consequences:** Two accounts can never merge by accident. IDs are slightly
  longer.

## ADR-003: Time split inside 2022-09-01..10; drop the tail

- **Context:** Both Small datasets have normal activity from Sept 1 to 10. After
  that, only a few hundred rows per day remain, about 60% of them laundering (an
  artefact of the simulator). A "last N days" test set would be tiny and
  dominated by positives.
- **Decision:** Use half-open splits. Train is everything before 09-07, validation
  is 09-07 to 09-09 (used only to fit fusion and calibration), and test is 09-09
  to 09-11. Rows from 09-11 on are `tail` and excluded from evaluation. The
  boundaries can be configured via env vars.
- **Consequences:** Test-set prevalence is realistic. Some laundering attempts cross
  split boundaries, so an account seen as a launderer in train may reappear in
  test. This matches deployment (known bad actors persist) but must be considered
  when results are interpreted. A per-attempt analysis is planned for Phase 8.

## ADR-004: FX rates estimated from data, fitted on train only

- **Context:** There are 15 currencies and no FX table. Each currency has direct
  USD cross-currency transactions in both directions.
- **Decision:** For each currency, the USD rate is the median of the implied rates
  `received / paid` over those transactions. Rates are fitted on the training
  partition and stored in the processed-data manifest.
- **Consequences:** Amounts can be compared. Fitting on train keeps test-period
  statistics out of preprocessing, and a leakage test enforces this.

## ADR-005: Aggregated DiGraph rather than MultiDiGraph

- **Context:** There are about 5 million transactions. A NetworkX multigraph with
  one edge per payment would need several GB.
- **Decision:** Use one directed edge per account pair per time window, carrying
  count, USD total, and first and last timestamps. Self-loops are excluded by
  default. Labels are never stored on the graph.
- **Consequences:** The graph fits in memory. Individual transactions in a case
  view come from the DataFrame, not the graph. Heavy graph features (Phase 3) may
  be computed with pandas/NumPy instead of NetworkX for speed.

## ADR-006: Typology mapping for Laya, deferred to Phase 4

- **Context:** IBM's 8 typologies don't match the brief's Laya label set
  {structuring, circular, mule_network, layering, benign}. In particular, IBM has
  no structuring pattern.
- **Decision:** Ask the project owner before Phase 4.

## ADR-007: Bundle a self-generated synthetic sample

- **Context:** The demo and CI must run without Kaggle credentials. IBM data must not
  be redistributed.
- **Decision:** A small seeded simulator (`aml_sentinel.data.synthetic`) writes about
  6.3k rows in the IBM file layout to `data/sample/`, and those files are committed.
  It plants all 8 IBM typologies, adds benign look-alikes (payroll, merchant,
  roommate cycles), and leaves about 20% of attempts out of the patterns file.
- **Consequences:** The demo works offline, and the loader, parser and FX code are
  exercised on known ground truth. Sample metrics are not evidence of model quality,
  because the sample's prevalence and pattern complexity differ widely from IBM's.
  Documentation must keep the two clearly separate.

## ADR-008: Synthetic "high-risk jurisdiction" list

- **Context:** The brief requires a jurisdiction rule, but IBM AMLworld has no
  country field. It has only bank codes and currencies.
- **Decision (owner's choice):** A seeded random 5% of the bank codes seen in the
  training split is declared "high-risk". Labels and volumes are ignored, because
  any smarter selection would be invented.
- **Consequences:** The rule keeps the familiar three-rule baseline shape, but it
  is **artificial by construction**. Measured test precision is about the base
  rate (HI-Small: 0.17% vs 0.11%). Every report labels it synthetic.

## ADR-009: Rule-baseline methodology

- **Velocity is entity-level.** The velocity threshold is a percentile of
  *sender-day* payment counts. The first implementation used a percentile over
  transaction rows. Hub accounts that send tens of thousands of payments a day
  dominated it, giving a threshold of 25,705/day, so the rule never fired.
  The sender-day percentile gives 24/day on HI-Small. The rule still counts only
  the current and earlier rows, which a brute-force property test checks.
- **There are two variants.** *Static* uses textbook thresholds: 10k USD, plus a
  train p99.9 sender-day velocity. *Tuned* is a 560-config grid search for train
  F1. All fitting uses only training rows.
- **The baseline to beat is chosen on validation F1.** Train-only tuning did not
  generalise on HI-Small: the train-best rule (velocity ≥ 100) caught 0
  documented-pattern transactions on test. Selecting on test would leak, so
  validation decides. Validation chose static for HI-Small and tuned for LI-Small;
  in both cases this is also the test winner. A test checks that test labels
  cannot change the choice.
- **Metrics are at transaction level** (one alert per transaction). Alert volume is
  reported per day, because analyst workload is what false positives cost.
