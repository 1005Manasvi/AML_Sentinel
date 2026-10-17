# Security

> **Status: skeleton (Phase 1).** Sections marked *planned* are filled in as the
> components they describe are built. The final version ships in Phase 8.

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
2. **API clients → FastAPI** *(planned, Phase 7)*. Authentication, roles, rate limits,
   size limits and Pydantic validation apply here.
3. **Local machine → external LLM / Jev** *(planned, Phases 4 and 6)*. Only
   pseudonymised, synthetic data may cross this boundary.
4. **Model artifacts on disk → process.** Artifacts load only from trusted local
   paths, after SHA-256 verification (`aml_sentinel.integrity`).
5. **Analyst → dashboard** *(planned, Phase 7)*. All rendered data is escaped.

### Risks and mitigations

| Risk | Mitigation | Status |
|---|---|---|
| Secrets in code or git | pydantic-settings and env vars; `SecretStr`; `.env` git-ignored; gitleaks in pre-commit and CI; validation errors never echo inputs | Done |
| Raw data or models committed | `.gitignore`; `check-added-large-files` hook | Done |
| Malformed or hostile input data | Header check, dtype coercion, value checks, alphanumeric-only IDs in the API schema | Done |
| Data leakage, which gives optimistic metrics | Time-based half-open splits; FX fit on train only; leakage tests | Done (extended in later phases) |
| Vulnerable dependencies | Hash-pinned lockfiles; pip-audit; bandit | Done |
| Malicious model files (pickle RCE) | Native/safetensors formats and checksum before load | Helper done, applied from Phase 3 |
| Data poisoning | *planned* | Phase 3+ |
| Model evasion (for example, splitting amounts under thresholds) | *planned*: graph features and System 2 detectors | Phase 3/5 |
| Prompt injection via transaction fields | Alert summaries are rendered only from numeric features through fixed templates, with no free text or IDs. A character allowlist, a length cap and an instruction to treat the state as data apply on top. Every model answer is schema-validated (`parse_answers`). The SAR step follows in Phase 6 | Done for System 1; SAR in Phase 6 |
| Data exfiltration to external LLMs | Summaries carry no identifiers. SAR facts are pseudonymised (`ACCT-nnn`) before any external call, and the map never leaves the machine. Outbound fields are documented below. The template drafter is the default | Done |
| Prompt injection into SAR drafting | Case data is sent as delimited JSON, with a system instruction to ignore instructions inside it. The reply must match a strict JSON schema (extra fields are rejected). Every account, amount and date is checked against the case facts, and unknown tokens make the draft fall back to the template. The "DRAFT - requires analyst review" stamp is always added | Done |
| Jev used with real data | The adapter is disabled by default and needs `JEV_ENABLED` plus a key. It raises `SyntheticDataRequiredError` unless the batch is explicitly `is_synthetic=True`, which defaults to False. A test proves nothing is sent | Done |
| Tampered model weights | Laya checkpoints are verified against a SHA-256 manifest and also through Laya's `expected_sha256` before parsing. Weights are safetensors only | Done |
| Insider misuse | Append-only, hash-chained audit log (`aml_sentinel.audit`): every SAR generation is recorded with who, what, when and the case. Details are restricted to an allowlist of non-sensitive keys. API roles arrive in Phase 7 | Partly done |

## What leaves the machine

| Component | Destination | Data sent | When |
|---|---|---|---|
| Laya | nothing (local) | n/a | always local |
| Laya base download | huggingface.co | none (download only), pinned commit | once |
| SAR drafter, external (opt-in) | api.anthropic.com over HTTPS (official `anthropic` SDK) | per case: the case id, `ACCT-nnn` tokens (never real account ids or bank codes), payment amounts, dates and formats, detector pattern names and sizes, and the calibrated risk score; plus the fixed system prompt and the API key | only if `LLM_PROVIDER=external` and `LLM_API_KEY` is set. The template default sends nothing |
| Jev (benchmark) | api.typesafe.ai over HTTPS | per alert: a numeric summary (amounts, counts, ratios, ages; **no** account IDs, bank codes or names), the fixed questions, the model name, and the API key in the `Authorization` header | only if `JEV_ENABLED=true`, only for synthetic data |

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

- CI actions are pinned to major-version tags rather than commit SHAs. Pinning to
  SHAs is planned before any non-portfolio use.
- Lockfiles are resolved on Windows. Platform-specific transitive dependencies may
  need a re-lock on Linux. See `docs/DECISIONS.md`.
