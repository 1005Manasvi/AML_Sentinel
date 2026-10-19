# What I learned building an AML monitor that tells you where it fails

Banks screen every payment for money laundering, and most of the alerts their
monitoring systems raise turn out to be false positives. Analysts spend their days
closing them. I wanted to see how far a careful, honest pipeline could cut that
noise, so I built **AML Sentinel**: a transaction monitor that ranks alerts by
calibrated risk, explains them with graph evidence and drafts suspicious-activity
reports (SARs) for a human to review.

This post covers the design, the results and, mostly, the parts that did not work.
Everything here was measured on synthetic data, IBM's AMLworld simulator (Altman et
al., NeurIPS 2023). That is the most important caveat, and I come back to it below.

## Two systems: fast for everything, slow for the few

The architecture borrows the fast/slow framing from psychology.

**System 1 screens every transaction, cheaply.** It has three parts:

- A rule engine (amount thresholds, a high-risk-jurisdiction list and velocity) that
  serves as the baseline everything is reported against.
- LightGBM over 35 features: the transaction's own amount, time and format, plus
  degree, flow ratios, short cycles, trailing velocity and counterparty diversity,
  all computed from rows strictly before the transaction.
- Laya, an open-source "decision model" that answers typed questions (typology,
  1–5 risk, escalate yes/no) about a short text summary of each alert, in one
  forward pass.

A logistic stacker fuses these scores, and an isotonic or Platt calibrator chosen on
validation turns them into `p_triage`. About 5% of alerts escalate.

**System 2 runs only on escalated alerts.** NetworkX detectors look for circular
flows, structuring, fan-in/fan-out and scatter-gather in a strictly past-only case
subgraph. A GraphSAGE model is there as a labelled experiment. A SAR drafter writes a
narrative from the case's own facts, then checks every account, amount and date in
it against the source data.

Every System 1 model implements one `Triage` protocol, so each one can be switched
off and measured. That turned out to be the most useful design decision in the
project.

## Getting the evaluation right first

AML data has two properties that make it easy to fool yourself.

**Extreme imbalance.** 0.11% of HI-Small test transactions are laundering, so a
model that says "no" to everything is 99.9% accurate. I reported PR-AUC, precision@k
and false positives at a fixed alert budget, and never accuracy. I did not resample
or reweight, because both distort the score distribution that calibration depends
on.

**Time.** Laundering is a process, and features computed with future information
look brilliant offline and fail in production. Splits are half-open on time (train,
then validation, then test). Every feature uses only earlier rows, and a test
deletes all later rows and checks that earlier features stay bit-identical. Trees fit
on train; calibrators, stackers and thresholds fit on validation; test is used once.
Label-scrambling tests flip every test label and assert that no score changes.

## The headline number, and why I distrust it

At the rule baseline's recall (52.1%), LightGBM needs 2,830 alerts where the rules
need 268,270, so false positives fall from 267,772 to 2,332. That is a **99.1%
reduction**, and the number is real.

It is also mostly a statement about the baseline. The rules flag 31% of all
transactions at 0.19% precision, so nearly any ranking model beats them. The fairer
comparisons are PR-AUC (0.416 vs 0.0016 on the full test split) and, harder,
comparisons *inside* the alert queue at matched volumes. In the queue's top 5%,
adding LightGBM to the rules cuts false positives from 170 to 49. After that, nothing
I added moved that number by more than three.

## What did not work

**The decision model was near random on its own.** On 4,000 test alerts from the
queue (6.8% laundering), Laya scored a PR-AUC of 0.096 against LightGBM's 0.649. Its
zero-shot typology accuracy of 94% was a majority-class artefact: it labelled
everything benign. A 4 GB laptop GPU only allowed head-only fine-tuning, with the
421M-parameter encoder frozen, and that did not measurably improve it. In fusion it
adds a small but significant +0.010 PR-AUC, at about 300 ms of GPU time per alert
against 8.4 µs per transaction for LightGBM. I kept it because it is local and
cleanly removable, and because the full fine-tune is still the fair test. But I say
in the README that it underperforms.

**Calibration did not survive time.** On held-out validation, isotonic calibration
improved ECE 3–6×. On test it made ECE worse (0.00027 → 0.00036), because the
relationship between score and label drifts between the two periods. Rolling
recalibration on the last validation day did not fix it. Tuning on test would have
been leakage, so I reported it. Because the top isotonic bin is 1.0, the API shows
each score next to its measured test reliability, instead of a 1.000 that overstates
certainty.

**Graph evidence explains alerts but does not rank them.** The detectors fire on
52% of laundering alerts, but also on 27% of benign ones. Benign hubs naturally
collect payments, so fan-in gets negative weight in the stacker. Stacked on top of
System 1, detector evidence changed PR-AUC by −0.023 in one configuration and
+0.011 in another; neither is significant. Their real value is the named accounts
and patterns they put into a case and a SAR.

**The GNN learned nothing LightGBM did not.** On identical pair-level inputs,
GraphSAGE and LightGBM differ by −0.002 PR-AUC (95% CI −0.013 to +0.011). Its
apparent gain in the ablation comes from end-of-period graphs, which a real-time
system never has. The same-input ablation also caught a bug in the first run:
every training node feature was zero. Without that ablation I would have reported a
GNN result that was really just a bias term.

**The model learned the simulator.** Removing the obvious shortcuts (payment format
and cross-currency) still leaves the top 100 alerts 100% ACH, because amount and
flow features recover them. More telling: 39% of test positives belong to no
documented laundering pattern, and the model's queue recall on them is 26%, against
85–87% for documented attempts. It recognises the simulator's typologies, not
laundering in general.

## Security as a design input

Transaction fields, names and analyst notes are untrusted input, and an LLM in the
loop is an exfiltration channel. So:

- The SAR drafter defaults to a local template. The optional external model gets
  pseudonymised facts as delimited JSON, with an instruction to ignore instructions
  in the data, and must return a strict schema. Every entity in the draft is then
  grounded against the case, and approval is refused unless the draft is fully
  grounded and stamped *DRAFT - requires analyst review*.
- The hosted benchmark adapter (Jev) refuses any batch not flagged synthetic, sends
  only numeric summaries, and is off by default. It was never run, so there is no
  Jev number to report.
- API keys are stored as digests and compared in constant time; there are roles,
  rate limits, strict CORS and a hash-chained audit log. Models load only after
  checksum verification, from native or safetensors formats. Nothing is pickled.

## Lessons

1. **Build the ablation harness before the models.** A swappable protocol plus
   paired bootstrap comparisons found the two most important truths in the
   project: the GNN bug and the memorised patterns.
2. **Report against the strongest comparison you have, not the weakest.** "99%
   fewer false positives" is true and nearly meaningless without the in-queue,
   matched-volume numbers next to it.
3. **Calibration is a property of a time period, not of a model.** Validate it
   where it will be used, and show reliability instead of pretending.
4. **A negative result is still a result.** Most of the table in this post is
   things that did not help. They were cheap to measure and expensive to ignore.

## Honest limitations

All of this is synthetic. Nothing has been validated on real transactions, and the
simulator's artefacts help the model. The test window is two days, and the bootstrap
intervals resample rows, not attempts, so they are optimistic. The data has no
customer attributes, so fairness could not be measured, even though monitoring
systems are known to concentrate alerts on particular segments and corridors and
can drive "de-risking". Production would need independent model validation,
streaming features, identity integration and per-jurisdiction SAR formats.

The code, the reports behind every number and a one-command demo are in the
repository. Run `make demo` and look for yourself.
