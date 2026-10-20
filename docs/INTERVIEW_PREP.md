# Interview prep

The 15 questions an interviewer is most likely to ask about this project, with short
answers grounded in the repo. Numbers are HI-Small test results from
[RESULTS.md](RESULTS.md); ADRs are in [DECISIONS.md](DECISIONS.md).

---

**1. How did you prevent data leakage?**

Three layers. First, the split is time-based and half-open: train before 09-07,
validation 09-07 to 09-09, test 09-09 to 09-11 (ADR-003). Second, every feature is
past-only: a feature at time *t* uses only rows before *t*, and a test deletes every
later row and checks that earlier features stay bit-identical. Third, the fitting is
separated: trees on train, then calibrators, stackers and every threshold on
validation, and test is touched once, for reporting. FX rates are estimated on train
only. Label-scrambling tests flip every test label and check that all trained,
fused and calibrated scores stay identical (`test_system1a.py`, `test_system1b.py`),
and `test_leakage.py` covers the splits and FX fitting. One leak I
could not remove is that some laundering attempts cross the split boundary, so part
of the test performance is recognising known actors. I measured it (39% of test
positives continue earlier attempts) instead of hiding it.

**2. With 0.11% positives, how did you handle class imbalance?**

I did not resample or reweight. Both change the score distribution and would make
calibration harder. Instead, LightGBM early-stops on average precision, and every
evaluation uses PR-AUC, precision@k and false positives at a fixed alert budget,
never accuracy (99.9% accuracy is free here). Operating thresholds are chosen on
validation. `min_child_samples=200` stops the trees from memorising the few
positives.

**3. Your headline is a 99.1% false-positive reduction. Is that real?**

It is measured, but it mostly reflects a weak baseline. At the rules' recall (52.1%),
LightGBM needs 2,830 alerts against the rules' 268,270, so false positives drop from
267,772 to 2,332. But the rules flag 31% of all transactions at 0.19% precision, so
almost any model looks great against them. That is why I also report PR-AUC (0.416 vs
0.0016) and compare models at matched alert volumes inside the queue, which is the
harder test.

**4. What does "calibrated" mean here, and did it work?**

It means a score of 0.3 should be laundering about 30% of the time. Platt and
isotonic were compared by time-ordered cross-fitting on validation; isotonic won.
On validation it improved ECE 3–6×. On test it made ECE *worse* (0.00027 → 0.00036),
because the score-to-label relationship drifts between the two periods (ADR-011).
I did not tune that away on test, because that would be leakage. Rolling
recalibration did not help either. Because the top isotonic bin is 1.0, the API
shows each score next to its measured test reliability instead of presenting it as a
true probability (ADR-022). A PSI drift report later showed the score distribution
itself was stable between validation and test (PSI 0.043), so what changed is the
score-to-label relationship, which input monitoring alone cannot catch (ADR-025).

**5. Why LightGBM rather than a deep model?**

The signal is in tabular, past-only graph aggregates, and gradient-boosted trees are
strong, fast (8.4 µs per transaction) and inspectable on that kind of data. The deep
alternatives I did try, Laya and a GNN, did not beat it. LightGBM is also saved in its
native text format with a checksum, so nothing needs pickle. Each alert is explained
with exact TreeSHAP values computed by LightGBM itself, and a test checks they add up
to the model's own log-odds.

**6. Laya vs LightGBM: why keep Laya when it lost?**

Alone, Laya is close to random inside the queue (PR-AUC 0.096 vs LightGBM's 0.649).
Its zero-shot "94% typology accuracy" is just a majority-class artefact. In fusion it
adds a small but significant +0.010 [+0.006, +0.015] PR-AUC, at about 300 ms of GPU
time per alert. It stays because it is local (open source, Apache-2.0, nothing leaves
the machine), it is behind the same `Triage` protocol, so it can be removed in
ablation, and its full fine-tune is still untested. I only fine-tuned the head,
because the 421M encoder does not fit a 4 GB GPU. The README says plainly that it
underperforms (ADR-014).

**7. Why is Jev benchmark-only?**

It is a closed, hosted API. Every call sends data to a third party, results can
change between model versions and cannot be reproduced offline. So the adapter is off
by default, needs `JEV_ENABLED=true` and a key, sends only numeric summaries without
identifiers, and refuses any batch not explicitly marked synthetic. There is a test
for that refusal. Nothing depends on it, and it was never run: there is no Jev number
anywhere in the repo (ADR-013).

**8. GNN vs NetworkX detectors: which is better?**

They answer different questions. The detectors (cycles, structuring, fan-in/out,
scatter-gather) are strictly past-only and produce named, checkable evidence for an
analyst and a SAR. As a ranking signal they add nothing significant, and fan-in is
anti-informative because benign hubs collect payments too. The GraphSAGE experiment
matched LightGBM on the same pair inputs (−0.002 PR-AUC), so message passing added
nothing, and its ablation gain uses end-of-period graphs, which a real-time system
cannot see. The same-input ablation also caught a bug: all training node features
were zero (ADR-016).

**9. How do you know the model learned laundering and not the simulator?**

Largely, it learned the simulator. Dropping the obvious shortcut features (payment
format, cross-currency) leaves the top 100 alerts 100% ACH, because amount and flow
features recover them. And the 39% of test positives that belong to no documented
pattern get 26% queue recall, against 85–87% for documented attempts. That is the
most important limitation, and the README says so.

**10. Fairness: what did you check?**

Nothing could be measured: the data has no customer attributes. I documented the
risk instead. Monitoring systems tend to concentrate alerts on certain segments,
corridors and cash-heavy businesses, which leads to de-risking. The synthetic
"high-risk jurisdiction" rule is an example of geography-based flagging that would
need legal and fairness review. Scores only prioritise; a human makes every decision.

**11. How is the API secured?**

API keys of 256 bits, stored only as SHA-256 digests and compared with
`hmac.compare_digest` against every entry without early exit. Analyst and admin
roles are checked per endpoint, with an auth-matrix test over every endpoint × no
key / wrong key / wrong role. Per-user rate limits, plus a per-IP budget for failed
authentication. Body-size limit, Host and exact-origin CORS allowlists, strict
security headers, and docs/OpenAPI disabled because they would be unauthenticated.
Validation errors never echo input. Parameterised SQL only.

**12. How do you defend SAR drafting against prompt injection and hallucination?**

The local template is the default, so by default nothing reaches an LLM. With the
opt-in external drafter, the model gets pseudonymised facts (`ACCT-nnn` tokens) as
delimited JSON with an instruction to ignore instructions inside the data, and its
reply must match a strict schema. Then every account, amount and date in the draft is
checked against the case's own data, and mismatches are listed. An edited draft is
re-stamped *DRAFT - requires analyst review* and re-grounded, and approval is refused
unless it is fully grounded. Limitation: grounding checks entities, not claims, so a
wrong characterisation still needs the human reviewer.

**13. What is in the audit log, and can it be tampered with?**

Who, what, when and case ID for every case view, SAR generation, edit and decision.
It stores no sensitive values; notes are stored in the case store and only their
length is audited. Each record hashes the previous one, so edits are evident.
Tamper-evident is not tamper-proof: someone with write access could rebuild the
chain. Production would anchor the head hash in WORM storage.

**14. What would it take to put this in production?**

Real data and independent model validation (SR 11-7 style model risk management),
streaming ingestion with an online feature store (`/score` rebuilds features from
history and is capped at 100k rows), OIDC instead of static keys, TLS, signed model
artifacts, a four-eyes approval rule, per-jurisdiction SAR formats, drift
monitoring, and adversarial testing, since fixed thresholds can be learned and
avoided.

**15. What would you do differently?**

Build the evaluation harness first. The attempt-overlap and same-input ablations
found the most important truths (memorised patterns, the GNN bug), and they arrived
late. I would also spend the GPU budget on a full Laya fine-tune before anything else,
evaluate per attempt rather than per row (the bootstrap intervals are optimistic),
and make the synthetic sample's graph sparser, so the demo's detectors behave like
they do on IBM data.
