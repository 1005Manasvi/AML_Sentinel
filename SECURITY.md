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
| Prompt injection via transaction fields | *planned*: delimiting, instructions to ignore data-borne instructions, output schema and grounding validation | Phase 4/6 |
| Data exfiltration to external LLMs | *planned*: pseudonymisation, a documented outbound field list, template fallback | Phase 6 |
| Jev used with real data | *planned*: the adapter refuses inputs not flagged as synthetic | Phase 4 |
| Insider misuse | *planned*: roles, an append-only audit log, no sensitive values in logs | Phase 7 |

## Logging policy

- Logs are structured JSON. They hold counts, durations, file names and case IDs.
- Logs never contain account IDs, amounts, names, free text or secrets.
- As a backstop, fields whose names look like they hold secrets are masked.

## Known limitations

- CI actions are pinned to major-version tags rather than commit SHAs. Pinning to
  SHAs is planned before any non-portfolio use.
- Lockfiles are resolved on Windows. Platform-specific transitive dependencies may
  need a re-lock on Linux. See `docs/DECISIONS.md`.
