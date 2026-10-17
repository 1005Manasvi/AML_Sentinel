# AML Sentinel: Project Brief for Claude Code

You are a senior ML engineer with a security mindset. Build "AML Sentinel", a
real-time Anti-Money Laundering (AML) transaction monitoring system, as a
complete, runnable, portfolio-quality project. It will be shown to recruiters
and hiring managers as a resume project, so correctness, honesty and polish
matter more than feature count.

WORKFLOW RULES
- Work ONE PHASE AT A TIME. After each phase: run ruff, mypy, pytest and the
  security scanners; summarize results, decisions and any problems; then STOP
  and wait for me to say "continue".
- If a requirement is ambiguous, conflicts with security, or depends on
  information you cannot verify, ask me before proceeding.
- Never invent numbers, API behavior or dataset details. Verify from source.

## 1. Purpose
Banks spend heavily on compliance, and the large majority of AML alerts are
false positives. Build a system that:
(a) detects laundering patterns on a transaction graph,
(b) ranks alerts by calibrated risk to reduce false positives versus a
    rule-based baseline,
(c) drafts Suspicious Activity Report (SAR) narratives for human review,
(d) provides an analyst case management dashboard.

## 2. Architecture: System 1 / System 2

System 1 is a fast, cheap decision layer that screens every transaction.
System 2 is slow, expensive, deep analysis that runs only on escalated alerts.

### System 1 (runs on everything)
1. Rule engine baseline (amount threshold, high-risk jurisdiction, velocity).
   This is the baseline I must beat and report against.
2. Graph feature extractor (degree, in/out flow ratio, velocity, short-cycle
   counts, counterparty diversity) feeding LightGBM -> p_tabular.
3. Alert summarizer: converts an alert's graph facts and transaction stats into
   a short plain-text summary (<= 512 tokens, to fit Laya's context limit).
4. Laya triage (MAIN PIPELINE, runs locally). Laya is an open-source
   (Apache 2.0) non-autoregressive "decision model" from Convai Innovations
   that answers closed-set typed questions in one forward pass. Reported repo:
   github.com/NandhaKishorM/laya. VERIFY the official repo, README, model card
   and license before using it, and use its real API. Never guess its
   interface. Questions to ask per alert:
     - Choice: typology in {structuring, circular, mule_network, layering,
       benign}. Keep label sets under 10; Laya degrades with many labels.
     - Score: risk 1-5.
     - Yes/No: escalate to System 2?
   Laya is weak zero-shot, so FINE-TUNE it on synthetic alert summaries
   derived from labeled data. Measure its accuracy, calibration and latency
   yourself; do not rely on published numbers. Consider the multilingual
   variant only if relevant and verified.
5. Jev triage (OPTIONAL, BENCHMARK-ONLY adapter, disabled by default). Jev is a
   closed, hosted decision-model API from TypeSafe AI with similar
   Choice/Score/Yes-No questions. Read its official docs first and never guess
   its API. Use it ONLY to benchmark against Laya, ONLY on synthetic data, ONLY
   when JEV_ENABLED=true and JEV_API_KEY is set. Enforce in code that the
   adapter refuses any input not flagged as synthetic. Document that it must
   never receive real customer data. Jev must not be required for anything to
   work.
6. Fusion and calibration: combine p_tabular and Laya's outputs (plus Jev's for
   benchmarking only) into p_triage with an interpretable stacker (logistic
   regression), then calibrate (Platt or isotonic). Report ECE and reliability
   diagrams. Alerts below threshold are auto-closed and logged; the rest
   escalate (target roughly 1-5%, but report what is actually measured).
7. Every System 1 model implements a common `Triage` protocol so models are
   swappable and individually ablatable.

### System 2 (runs only on escalated alerts)
1. NetworkX detectors: circular flows (A->B->C->A), structuring/smurfing (many
   sub-threshold deposits), mule fan-in/fan-out, scatter-gather.
2. GNN (PyTorch Geometric, GraphSAGE or GAT) for node/edge classification,
   clearly labeled as an experiment with its own evaluation and honest
   comparison against the NetworkX detectors. Optionally validate on the
   Elliptic Bitcoin dataset. Must run on CPU for small data; GPU optional.
3. SAR narrative drafting grounded ONLY in the case's own data, through a
   provider-agnostic LLM interface plus a fully local template fallback.

### Final ranking
Fuse System 1 and System 2 evidence into one calibrated risk score. Never
present raw model scores as true probabilities.

## 3. Data
- Primary: IBM AMLworld / AML-Sim synthetic transactions (Kaggle: "IBM
  Transactions for Anti Money Laundering (AML)"; paper: Altman et al.,
  NeurIPS 2023). Start with HI-Small and LI-Small. I will download them into
  data/raw/. Do not try to download from Kaggle without my credentials. Write
  data/README.md covering files, schema, license check and how to obtain them;
  verify current file names and schema from the data itself.
- Optional: Elliptic Bitcoin dataset for the GNN experiment.
- Generate synthetic alert summaries and labels for Laya fine-tuning from the
  labeled patterns. Include noise, ambiguous cases and benign look-alikes so
  the model cannot just memorize templates.
- Use TIME-BASED train/validation/test splits and fixed seeds. Write a test
  proving no future data leaks into training, fusion or calibration.
- Never use, request or commit real customer data. Raw data and large model
  artifacts stay out of git.
- Bundle a SMALL synthetic sample dataset (generated by this project, not
  copied from restricted sources) so the demo runs without Kaggle access.

## 4. Security requirements (non-negotiable)
- No secrets in code, config, logs or git. Use environment variables via
  pydantic-settings; provide .env.example; fail fast if a required secret is
  missing. Optional keys: LLM_API_KEY, JEV_API_KEY.
- API: authentication on all endpoints except /health (API keys or JWT with
  constant-time comparison); roles (analyst, admin); rate limiting; strict CORS
  allowlist; request size limits; secure headers.
- Validate all input with Pydantic v2. ORM or parameterized queries only. No
  eval/exec/pickle on untrusted input. Load models only from trusted local
  paths with checksum verification; prefer safetensors or native formats over
  pickle.
- Treat transaction fields, names and free text as untrusted. Escape all
  dashboard output. Defend the SAR step and the alert summarizer against
  prompt injection: delimit data, instruct the model to ignore instructions
  found inside data, validate output structure and field-level grounding.
- SAR drafting: pseudonymize identifiers before any external LLM call;
  document exactly what leaves the machine; keep the local template fallback;
  always stamp "DRAFT - requires analyst review". Verify every account, amount
  and date in a draft against source data and flag any mismatch.
- Append-only audit log of analyst actions and SAR generations (who, what,
  when, case ID). Never log sensitive field values.
- Pin dependencies (lockfile with hashes). Run pip-audit, bandit and gitleaks
  in CI and pre-commit.
- Docker: non-root user, minimal base image, no secrets baked in.
- Write SECURITY.md with a threat model: assets, trust boundaries, risks (data
  poisoning, model evasion, prompt injection, insider misuse, data
  exfiltration, third-party API leakage), mitigations and known limitations.

## 5. Engineering standards (idiomatic Python)
- Python 3.11+, src/ layout, pyproject.toml as the single source of config,
  managed with uv or pip-tools.
- Full type hints; mypy --strict passes on src/. ruff for lint and format.
- Pydantic v2 for schemas; dataclasses for internal value objects; pathlib, not
  os.path; structured logging, never print; explicit custom exceptions.
- Small, pure, testable functions; dependency injection for models and
  clients; no global mutable state.
- Tests (pytest): detector unit tests on planted patterns; hypothesis
  property-based tests for graph features; API auth matrix tests (no key,
  wrong key, wrong role); Laya and Jev adapter tests with mocks; leakage test;
  Jev synthetic-only enforcement test. Target 85%+ coverage on src/.
- GitHub Actions CI: ruff, mypy, pytest with coverage, bandit, pip-audit.
- Makefile targets: setup, test, run, train, demo, reproduce. docker-compose
  for the full stack.
- Conventional Commits, small meaningful commits.

## 6. Documentation requirements
- Google-style docstrings on every public module, class and function (purpose,
  args, returns, raises, security notes where relevant).
- README.md, docs/ARCHITECTURE.md, docs/DECISIONS.md (short ADRs, including why
  Laya is in the pipeline and Jev is benchmark-only), docs/MODEL_CARD.md
  (intended use, data, metrics, calibration, failure modes, fairness and bias
  considerations), SECURITY.md.
- Comments explain WHY, not what.

## 7. Portfolio requirements
Optimize for a reader who spends 60 seconds, then maybe 5 minutes running it.
- README top section: one-sentence pitch, demo GIF or screenshot, "Key results"
  table with MEASURED numbers only, badges (CI, coverage, license, Python
  version), and a 3-command quickstart. Put architecture and details below.
- Demo mode: `make demo` (or `docker compose up`) starts the API and dashboard
  with the bundled synthetic sample and small pre-trained models. No Kaggle
  credentials or GPU needed. Keep the repo small.
- docs/RESULTS.md: PR curve, precision@k, false-positive reduction versus the
  rule baseline, ablation table, calibration plot, latency per stage, and a
  "what did not work" section. Every number is reproducible with a documented
  command and fixed seed.
- `make reproduce` regenerates every figure and table in the README.
- Limitations section in README: synthetic data, no real-world validation,
  fairness and de-risking concerns, and what production would additionally
  require (model risk management, regulatory validation, streaming
  infrastructure, per-jurisdiction SAR formats).
- Save screenshots and GIFs in docs/assets/. If I must record them, tell me
  exactly what to capture.
- Repo hygiene: LICENSE (MIT or Apache-2.0), short CONTRIBUTING.md,
  .gitignore, tagged v1.0.0 release with release notes.
- docs/BLOG_POST.md: a 1,000-1,500 word engineering write-up (problem, design
  choices, results, lessons, honest limitations) I can publish.
- docs/RESUME_BULLETS.md: 3-5 bullet TEMPLATES with placeholders for my real
  measured numbers. Do NOT invent or estimate numbers; fill them only from
  results the repo produced, and tell me which results to plug in.
- docs/INTERVIEW_PREP.md: the 15 questions an interviewer is most likely to ask
  (leakage, calibration, class imbalance, fairness, security choices, Laya vs
  LightGBM, why Jev is benchmark-only, GNN vs NetworkX) with concise, accurate
  answers grounded in this repo.

## 8. Phases
1. Scaffold: repo layout, pyproject, tooling, CI, pre-commit, settings and
   secrets handling, SECURITY.md skeleton, synthetic data generator, data
   loader, transaction graph builder, EDA notebook, data/README.md.
2. Baseline: rule engine; measure alert volume, precision, recall and
   false-positive rate on the test period.
3. System 1a: graph features, LightGBM, calibration, Triage protocol.
4. System 1b: alert summarizer, synthetic alert dataset, Laya fine-tuning and
   evaluation (accuracy, ECE, latency) versus LightGBM alone; optional Jev
   adapter and synthetic-only comparison; fusion and calibration.
5. System 2: NetworkX detectors with planted-pattern tests; GNN experiment with
   honest comparison.
6. Ranking and SAR: final score fusion; SAR drafting with redaction, injection
   defenses, grounding checks, template fallback, review label.
7. API and dashboard: authenticated FastAPI (score, alerts, case detail, SAR
   draft, case actions); Streamlit dashboard with ranked queue, interactive
   transaction graph (pyvis or plotly), SAR editor, approve/dismiss/escalate
   with notes, audit log view.
8. Evaluation and hardening: PR curve, precision@k, false-positive reduction
   versus baseline, per-stage latency, ablation table (rules only / +LightGBM /
   +Laya / +Jev benchmark / +graph evidence / +GNN / full), security scan
   results, final docs.
9. Portfolio packaging: demo mode, RESULTS.md, `make reproduce`, README,
   BLOG_POST.md, RESUME_BULLETS.md, INTERVIEW_PREP.md, v1.0.0 release notes.
   Run a fresh-clone test: clone into a new directory, follow only the README,
   confirm it works, and report any friction.

## 9. Honesty rules
- Report real measured results, including where the new system does NOT beat
  the baseline or where Laya underperforms LightGBM. Do not claim a 60-80%
  false-positive reduction unless my measurements show it.
- Synthetic-data results do not prove real-world performance; say so in the
  README and model card.
- Jev comparisons must note that Jev is a hosted, closed model and that results
  may not be reproducible or equally conditioned.
- Be explicit about what is verified versus assumed in every phase summary.
