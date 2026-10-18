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

## ADR-010: System 1a features and the shortcut ablation

- **Features (35).** There are three kinds:
  - Row features: log USD amount, round amount, hour, weekday, same bank,
    self-transfer.
  - Past-only trailing windows: sender payments in 1h and 24h, sender USD in 24h,
    receiver inflow count and USD in 24h.
  - Daily graph snapshots, taken from all rows before midnight of the
    transaction's day: in/out degree and payment counts, USD flows, flow ratio,
    reciprocal counterparties, directed 3-cycles, history flag and account age.
    These cover both the sender and the receiver.

  A test checks that deleting every later row leaves earlier features
  bit-identical.
- **Shortcut ablation (owner's choice: report both).** `all` uses every feature.
  `no_shortcuts` drops `payment_format_code` and `cross_currency`.
- **Measured finding:** dropping the columns does not drop the shortcut. The
  `no_shortcuts` HI-Small model still ranks only ACH transactions in its top 100,
  by recovering ACH-ness from correlated amount and flow features. Its ranking
  quality also stays close to the full model (PR-AUC 0.37 vs 0.42).
  Simulator-specific signal remains in both variants, so the results must not be
  read as shortcut-free.
- **Known issue in the bundled sample:** the synthetic generator also pays every
  laundering transaction by ACH, so the sample copies the same shortcut. Fixing
  this is planned for Phase 4, where the generator is extended for structuring.

## ADR-011: Calibration is fitted on validation and does not survive drift

- **Decision:** Platt and isotonic calibrators are compared by time-ordered 2-fold
  cross-fitting inside validation. The winner, isotonic in all four runs, is
  refitted on the full validation split and stored as JSON breakpoints, never
  pickled. Early stopping uses the last training day, so validation is used only
  for calibration.
- **Measured finding:**
  - On held-out validation halves, calibration improves ECE 3–6×.
  - On test, ECE gets **worse** than the raw LightGBM scores on both datasets.
    For HI-Small `all` it goes from 0.00027 to 0.00036. Calibrated mean predictions
    overshoot the observed rate: 0.144% vs 0.111% on HI-Small, and 0.088% vs
    0.057% on LI-Small.
  - The relationship between score and label drifts between Sept 7–8 and
    Sept 9–10.
- **Consequences:** Phase 4 fusion must not assume validation-calibrated scores are
  calibrated on test. Phase 8 will evaluate recalibration on a rolling window as a
  mitigation. Fixing this with test data would be leakage, so the result is
  reported as measured.

## ADR-012: Laya label set, typology mapping and synthetic structuring

- **Labels (5):** structuring, circular, mule_network, layering, benign. Laya's
  README warns that it degrades with many labels, so the set stays small.
- **Mapping (owner-approved):**
  - CYCLE → circular.
  - FAN-IN and FAN-OUT → mule_network.
  - SCATTER-GATHER, GATHER-SCATTER, STACK, BIPARTITE and RANDOM → layering.
  - Label 0 → benign.
  - Laundering rows with no documented typology have no typology target. They still
    count as positives for the risk and escalation questions.
- **Structuring:** IBM AMLworld has none. The project's generator adds a STRUCTURING
  pattern (6–15 cash deposits at 80–99.5% of 10k USD into one account) and a benign
  look-alike: a car dealer receiving a few legitimate cash payments of 5–9.9k. On
  IBM test data the structuring class is therefore empty, and this is reported.
- **Generator fix:** laundering payment formats are now mixed (ACH 45%, Wire 20%,
  Cheque 15%, Card 10%, Cash 10%). Before, they were all ACH, which reproduced IBM's
  shortcut in the bundled sample (ADR-010).

## ADR-013: Laya in System 1, Jev benchmark-only, and how fine-tuning is done

- **Why Laya is in the pipeline:**
  - It is open-source (Apache-2.0) and runs locally.
  - It answers typed questions in one forward pass, so alert summaries never leave
    the machine.
  - Weights are safetensors and are SHA-256-verified before loading (both by us and
    by Laya's `expected_sha256`).
  - Its API was taken from the installed `laya` 0.4.9 source and the official README,
    not guessed.
- **Why Jev is benchmark-only:**
  - It is a closed, hosted API (TypeSafe; `POST https://api.typesafe.ai/v1/systemone`,
    per docs.typesafe.ai).
  - Every call sends data to a third party, and results may change between versions
    and cannot be reproduced offline.
  - The adapter is disabled unless `JEV_ENABLED=true` and a key is set. It refuses
    any batch not explicitly marked synthetic; `AlertBatch.is_synthetic` defaults to
    False. It sends only numeric summaries, with no identifiers. Nothing needs Jev.
- **Fine-tuning on this hardware:**
  - A full fine-tune of the 421M-parameter encoder does not fit the 4 GB RTX 3050
    (fp32 weights plus AdamW state is about 5 GB or more).
  - Owner's choice: run **head-only** fine-tuning locally
    (`TrainConfig(freeze_encoder=True)`, 3 epochs) and do the **full** fine-tune on
    Kaggle with `scripts/kaggle_laya_full_finetune.py`. Every result says which
    variant produced it.
  - PyPI's Windows torch wheel is CPU-only, so the CUDA build (2.14.1+cu126, which
    matches driver 566.07) is installed from PyTorch's index.
- **Where Laya runs:**
  - About 120 ms per alert on this GPU, so Laya cannot score every transaction.
    System 1a's top 1% forms the alert queue.
  - Laya scores a fixed seeded sample of 4,000 alerts per split. Every text model
    sees the same sample.
  - Fusion is fitted on validation alerts. The top 5% of alerts by `p_triage` go to
    System 2, with the threshold set on validation.
  - This interprets the brief's "escalate roughly 1–5%" as a share of **alerts**.

## ADR-014: Measured Laya outcome and what follows from it

- **Result (HI-Small, 4,000 test alerts from the top-1% queue):**
  - Laya alone is close to random: PR-AUC 0.09–0.10 against LightGBM's 0.65, and
    ROC-AUC 0.61 zero-shot and 0.54 head-only, against LightGBM's 0.88.
  - Head-only fine-tuning gives no significant improvement (bootstrap CI includes 0).
  - Stacking Laya with LightGBM adds +0.02 to +0.03 PR-AUC. That gain is significant
    on row resampling, but small, and it costs about 300–450 ms of GPU time per alert.
- **Interpretation:**
  - With the encoder frozen, Laya's representation of these numeric summaries is
    fixed, and the head cannot extract much signal from it.
  - The summaries also restate features LightGBM already uses, minus the shortcuts,
    so there is little new information for Laya to add.
  - A full fine-tune (Kaggle script) is the fair test of Laya's ceiling. It is
    prepared but not yet measured.
- **Decision:** Laya stays in the pipeline as an optional, ablatable `Triage` model.
  The default System 1 score is fusion with Laya when a verified checkpoint exists,
  and LightGBM alone otherwise. The README and model card state plainly that Laya
  underperforms LightGBM on its own.
- **Latency note:** the 304–449 ms per alert measured in the latest full run (an earlier run measured 296–442 ms; model outputs were identical) is higher than
  the 121 ms smoke-test figure. That smoke test used batches of 32 with only four
  distinct states. The full-run numbers are the ones reported.

## ADR-015: System 2 detectors run on past-only cases; findings are evidence

- **Case:**
  - The alert's two accounts, plus 2 hops of counterparties.
  - Payments in the 72 h before the alert, and never after it by default. This is
    the same no-lookahead rule as Phase 3.
  - Hubs are capped at 50 rows per account and 3,000 rows per case, and any
    truncation is recorded.
  - Labels never enter a case.
- **Detectors:** circular (time-respecting cycles with consistent amounts),
  structuring (at least 3 deposits at 80–100% of 10k from distinct senders),
  fan-in, fan-out, scatter-gather and gather-scatter. Each one has planted-pattern
  unit tests, including negative cases.
- **Performance fix:** the first scatter-gather took 2.3 s per dense case, because
  it scanned the case DataFrame for every candidate pair. A per-case pair index
  brought cases to about 20 ms median. Cycle enumeration has an explicit budget,
  recorded on the finding.
- **Measured (HI-Small, strict):**
  - "Any finding" gives a 1.8x lift over the base rate inside the queue.
    Scatter-gather is the sharpest (70% precision but low recall). Fan-in is weak
    (1.3x), because benign hubs look like fan-ins.
  - On the escalated subset the lift is only 1.1x, so the detectors are kept for
    **evidence and explanation**, not as a ranking signal.
- **Investigation horizon:** `next_day` (24 h) is reported as a separate, labelled
  scenario. It raises recall, but fan-in turns anti-informative.
- **Sample caveat:** the bundled sample's graph is much denser than IBM's, which
  makes the detectors anti-informative there. The generator's density is a
  documented limitation of the demo, not a property of the detectors.

## ADR-016: GNN is an experiment; message passing did not help here

- **Set-up:**
  - The unit is an account pair per period, labelled laundering if any payment on
    the pair in that period was laundering.
  - A 2-layer GraphSAGE embeds the accounts and an MLP scores each pair. Training
    is full batch, with early stopping on validation-pair AP.
  - Weights are saved as safetensors with a SHA-256 manifest.
  - The setting is end-of-period batch, which favours the GNN over the real-time
    per-transaction models.
- **Bug found and fixed:**
  - The first version computed node features from history *before* each period.
    The training period starts on the first day of data, so all training-graph
    node features were zero, while validation and test were not. The GNN scored
    PR-AUC 0.012.
  - The same-input LightGBM ablation (0.044) exposed the gap. Node features now come
    from each period's own structure, giving GNN PR-AUC 0.042, and a test asserts
    that training node features are non-constant.
- **Result:**
  - The GNN equals LightGBM on the same pair inputs (no significant difference),
    and both are far below per-transaction LightGBM aggregated to pairs (0.41).
  - Message passing over this graph adds no measurable ranking power beyond pair
    features. The main loss comes from aggregating payments into pairs.
- **Decision:** the GNN stays an optional experiment (`make gnn-setup gnn`) and is
  not part of the scoring pipeline. Validation on the Elliptic dataset was not done.
  It is optional in the brief and recorded as future work.

## ADR-017: Never let a library write into a verified model artifact

- **Finding:**
  - Re-running System 1b failed: the head-only Laya checkpoint was refused with
    `checksum mismatch for tokenizer_config.json`.
  - The file had been modified at the minute Phase 4 *loaded* it. On a scratch copy,
    `laya.load` (0.4.9) was reproduced rewriting `tokenizer/tokenizer_config.json` in
    the directory it loads from (`5bed9c3a…` became `891c0cde…`). Phase 4 passed only
    because verification ran *before* the load mutated the file.
- **Not done:** re-baselining the manifest to the new hash. That would have defeated
  the integrity control.
- **Done:**
  - The original file was restored from Laya's own `checkpoint_latest/` copy, whose
    hash equals the manifest's.
  - `load_agent` now copies the verified files into a fresh temporary directory,
    verifies the copy, loads from it, and deletes it. The stored artifact is never
    written to. Prediction was checked to work after the copy is deleted.
  - A regression test uses a deliberately mutating loader and asserts that the
    stored checkpoint still verifies after two loads.

## ADR-018: Final ranking = System 1 p_triage + System 2 evidence, fitted on validation

- **Inputs:**
  - The System 1b fused probability. On validation this is the **out-of-fold**
    score from `FusionModel.fit_oof`, never an in-sample prediction.
  - Past-only detector evidence: the finding count and the strongest finding per
    detector, log-transformed.
- **Model:** the same interpretable logistic stacker plus Platt/isotonic calibrator,
  fitted on validation alerts and evaluated once on test alerts. Its calibrated
  output is the system's final risk score. A leakage test checks that scrambling
  test labels leaves the fitted ranker unchanged.
- **Reporting:** the gain over System 1 alone uses a paired bootstrap and is
  reported whether it is positive or not. Phase 5 found that the detectors add
  little ranking power on escalated alerts.

## ADR-019: SAR drafting is grounded, pseudonymised, template-first and always a draft

- **Template-first:** the local template drafter is the default, and nothing leaves
  the machine. The external drafter (owner's choice: Anthropic Claude via the
  official SDK, `claude-opus-5-5` by default, configurable with `LLM_MODEL`) is
  opt-in through `LLM_PROVIDER=external` and `LLM_API_KEY`.
- **Server-side refusal fallbacks:** these are enabled by default on the external
  call (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), as Anthropic
  recommends for this model. Set `use_fallbacks=False` to turn them off.
- **Pseudonymisation:**
  - Account ids become `ACCT-nnn` before any rendering. The map stays in memory,
    and real ids are restored only after validation.
  - Amounts, dates and payment formats are kept, because the narrative must state
    them exactly. They are synthetic here, and SECURITY.md flags them for any
    real-data use.
- **Injection defence:**
  - Delimited JSON data, a system instruction to ignore instructions inside the data,
    and a strict reply schema (extra fields are rejected).
  - Unknown account tokens force a fallback to the template, and any fallback is
    recorded on the draft.
- **Grounding:** every `ACCT-nnn`, `$x,xxx.xx` amount and `YYYY-MM-DD` date must exist
  in the case facts (the 10,000 USD threshold is the only allowed constant).
  Mismatches are listed on the draft, never hidden.
- **Known limit:** grounding verifies entities, not characterisations. The
  "DRAFT - requires analyst review" stamp (always present, at both ends) and the
  analyst are the control for that.
- **Audit:** every generation appends a hash-chained record (who, what, when, case,
  source, grounded, number of issues) to the git-ignored `data/audit/audit.jsonl`.

## ADR-020: The analyst queue is ranked by System 1 p_triage

- **Context:** Phase 6 measured that adding System 2 evidence to the score made
  test ranking slightly worse: PR-AUC 0.651 for the final score vs 0.674 for System 1
  `p_triage`, with a 95% CI of [-0.050, +0.002] (ADR-018).
- **Decision (owner's choice):** the queue is ordered by System 1 `p_triage`. System 2
  findings are shown in each case as **evidence** and never change the order. The
  final combined score appears in case detail, labelled informational.
- **Where p_triage comes from:** the System 1b fused probability when
  `<dataset>.system1b_scores.parquet` exists (HI-Small:
  `fused[lgbm_all+laya[head_only]]`, over the 4,000 sampled test alerts Laya scored).
  Otherwise it is calibrated LightGBM over the System 1a queue (top 1% of the split,
  at least 200 alerts). The source is shown in the API response and the dashboard.
- **Escalation:** the same rule as System 1b, the 95th percentile of validation
  `p_triage`. Alerts below it are `auto_closed`. System 2 evidence is attached only
  to escalated alerts, matching the architecture.
- **Measured consequence on the bundled sample:** the isotonic calibrator's top bin
  is 0.5. The validation 95th percentile lands on that tie, so only 4 of 200 test
  alerts escalate (2%, not 5%), and they tie on score (ordered by tx id). On HI-Small the measured
  test share is 6.5% (261 of 4,000). This is
  the coarse calibration of a tiny dataset, reported as is.

## ADR-021: API and dashboard design

- **Authentication:** API keys rather than JWT. There is no identity provider to
  issue tokens, and keys are simpler to audit.
  - Keys are 256-bit random values from `aml-sentinel new-api-key`.
  - The server holds only SHA-256 digests (`API_KEYS=user:role:digest`). A fast hash
    is fine for high-entropy keys.
  - A presented key is compared with every digest using `hmac.compare_digest`, with
    no early exit.
  - Roles: `analyst` < `admin`. Only admins read the audit log.
- **Rate limiting:** an in-process token bucket per user. Failed authentication
  spends a per-client-IP budget, which slows key guessing. It is not shared across
  processes (see SECURITY.md).
- **Transport hardening:** pure-ASGI middleware adds security headers and enforces
  a body-size limit, including on streamed bodies. There is also a `Host` allowlist
  and an exact-origin CORS allowlist (wildcards are rejected at settings load).
  `/docs` and `/openapi.json` are disabled, because they would be unauthenticated.
  Validation errors never echo input.
- **Case store:** stdlib `sqlite3` with parameterised statements only, not an ORM
  dependency. Triggers make both tables append-only, so an edited SAR is a new
  version. The hash-chained audit log stays the tamper-evident record. Case views,
  SAR generations and edits, and decisions are audited. Notes are stored, and only
  their length is audited.
- **SAR review:** an edited draft is always re-stamped on the server. It is
  re-grounded after mapping real ids back to tokens, and unknown account-like ids
  are flagged. Approving is refused (409) unless the latest draft exists and is
  fully grounded.
- **POST /score (owner's choice: hybrid):** the scorer recomputes past-only features
  by appending the transaction to history at or before its timestamp, using the
  training code path. A test checks this reproduces the training score to 1e-12.
  It takes about 0.1 s on the 6.3k-row sample. Above `API_SCORE_MAX_HISTORY_ROWS`
  (100k by default) it returns 501; on HI-Small (5.1M rows) it would take minutes.
  Production needs an online feature store. Scored transactions are never added
  to history, so clients cannot poison other accounts' features. A test checks
  that fabricated future rows do not change a score.
- **Found while building:** the processed frame stores ids as pandas categoricals,
  so a new account id would have silently become NaN. The scorer converts ids to
  strings, and a new-account test covers it.
- **Dashboard:** Streamlit, as a pure API client with the analyst's own key, so roles
  and audit attribution are unchanged. Untrusted text appears only in plain-text
  widgets. Dynamic text in Markdown widgets is escaped, which blocks image-beacon
  exfiltration. Graph labels are HTML-escaped, because plotly renders some HTML.
  An AST test bans `unsafe_allow_html`, `st.markdown`, `st.write`, `st.html` and
  `st.cache*` in the dashboard. Streamlit telemetry is off, it binds to loopback,
  and error details are hidden.
- **Not done in Phase 7:** Docker. Docker is not installed on the development
  machine, so the image and compose file are deferred to Phase 9, where they can be
  tested.

## ADR-022: Smoothed isotonic is available; validation still chooses plain isotonic

- **Problem:** isotonic calibration gives a block that holds only positives a
  probability of exactly 1.0. On HI-Small, 82 test alerts display a final score of
  1.000, which overstates certainty (ADR-018).
- **Change:** a Beta-smoothed isotonic option. Each block is shrunk towards the base
  rate by 2 pseudo-observations (fixed in advance, not tuned), and monotonicity is
  re-imposed, so no output is ever exactly 0 or 1. It serialises as JSON breakpoints
  like the other calibrators.
- **Owner's choice (cheap layers only):** smoothing is a candidate for the final
  ranker, which refits from cached scores in seconds. The Laya-fused `p_triage` keeps
  its calibrator, because changing it means re-running Laya (about 2.5 h of GPU).
  System 1a/1b keep their two-candidate set, so `make reproduce` regenerates their
  reported numbers unchanged.
- **Measured selection (held-out validation ECE):**
  - Final ranker: isotonic 0.0158, Platt 0.0168, smoothed 0.0173. Plain isotonic
    stays, and the final scores are unchanged.
  - System 1a full split: isotonic 0.00009, smoothed 0.00010, Platt 0.00015.
  - On test, smoothed would have been slightly better (ECE 0.00034 vs 0.00036), and
    raw LightGBM better still (0.00027). That is not used to override the
    validation choice, which would be selecting on test.
- **Rolling recalibration (promised in ADR-011):** calibrators refitted on the last
  validation day only were **worse** on test (isotonic 0.00040, Platt 0.00043) than
  those fitted on both days. With two days of validation, a shorter window adds
  variance without tracking the drift.
- **What the user sees instead:** the API attaches each alert's *measured
  reliability*, the observed laundering rate of test alerts in the same `p_triage`
  band (from `reports/evaluation`). Bands are attached only when they were measured
  for the very score the queue uses.

## ADR-023: Phase 8 evaluation design and what it found

- **Two universes, never mixed:**
  - Rules vs LightGBM on every test transaction (862,792 on HI-Small).
  - The cumulative ablation on the 4,000 sampled test alerts that Laya scored.
    "Rules only" inside the queue is a ranking by rule score among alerts LightGBM
    already selected. It shows what each signal adds there, not the rules' standalone
    value.
- **Ablation method:** each step adds inputs to the previous one and refits one
  validation stacker plus calibrator, scored once on test. Δ is a paired bootstrap
  against the previous step (row resampling, so optimistic).
  - The "+ Laya" step uses only Laya's escalate probability, because its expected
    risk and benign probability were not persisted. That is why it differs slightly
    from the production fusion.
  - Jev was not run, and its row says so.
  - The GNN row is flagged as lookahead (end-of-period graphs).
- **Findings (HI-Small):**
  - LightGBM carries almost all the ranking power inside the queue: PR-AUC
    0.116 → 0.664.
  - Laya adds +0.010 [+0.006, +0.015], and graph evidence +0.011 [−0.008, +0.031],
    which is not significant. Graph evidence is positive here but was negative when
    stacked on the fused score (ADR-018), so its sign depends on how it is combined.
    The honest reading is "no reliable ranking gain".
  - The lookahead GNN adds +0.020.
  - False positives in the top 5% barely move after LightGBM (49–52). The later
    gains are deeper in the ranking, not in the escalation budget.
  - Attempt overlap: 39% of test positives continue attempts seen before the test
    period (PR-AUC 0.61), and 22% start new documented attempts (0.49). The 39% that
    are undocumented are mostly missed (queue recall 26%, PR-AUC 0.009). The model
    has learned documented simulator patterns much better than "laundering" in
    general.
- **Latency:** measured on the development machine and labelled with its source.
  Laya and detector timings are reused from their Phase 4 and 5 runs, and say so.
