# Data

AML Sentinel uses **synthetic** transaction data only. Never place real customer data
in this directory.

## Bundled synthetic sample (`data/sample/`, no download needed)

`Sample_Trans.csv` and `Sample_Patterns.txt` are produced by this project's own
seeded simulator (`aml_sentinel.data.synthetic`). Nothing in them is copied from
IBM. They use the IBM file layout, so the same loader and tests apply. Regenerate
them with `make sample`; seed 42 reproduces the committed files exactly, and a test
enforces this.

| | Value (seed 42) |
|---|---|
| Rows | 6,306 (Sept 1–10, 2022) |
| Laundering rows | 207 (3.3%), from 24 attempts covering all 8 IBM typologies |
| In the patterns file | 172 rows. About 20% of attempts are deliberately left undocumented, as in IBM |
| Benign look-alikes | Payroll fan-outs, merchant fan-ins and roommate rent cycles, all labelled 0 |
| Currencies | 6, at fixed illustrative rates (`SAMPLE_FX`), which the FX estimator recovers |

Use the sample for demos and CI. **Don't report sample results as model
performance.** Its prevalence (3.3%) is about 30 times IBM HI-Small's, and its
patterns are far simpler.

## Source

**IBM Transactions for Anti Money Laundering (AML)** (also called IBM AMLworld), on Kaggle:
<https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>

The data comes from a multi-agent simulator described in Altman et al., *Realistic
Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023
Datasets and Benchmarks. Every transaction is simulated, and the ground-truth
laundering labels come from the simulator. They are not investigator decisions.

### License check

The Kaggle page lists the dataset under the **Community Data License Agreement –
Sharing 1.0 (CDLA-Sharing-1.0)**. Before you redistribute anything derived from it,
re-check the license shown on the Kaggle page, because licenses can change. This
repository never redistributes the data. It ships code only, and `data/raw/` is
git-ignored.

## Download (manual)

The project does not download from Kaggle, because that needs your Kaggle
credentials. To get the files:

1. Sign in to Kaggle and open the dataset page above.
2. Download these four files:
   `HI-Small_Trans.csv`, `HI-Small_Patterns.txt`, `LI-Small_Trans.csv`, `LI-Small_Patterns.txt`.
3. Put them in `data/raw/`.
4. Run `make data`. This step validates the files, fits FX rates on the training
   window, and writes `data/processed/<name>.parquet` plus `<name>.manifest.json`.

HI means a higher illicit ratio and LI a lower one. The Medium and Large variants
have the same schema and can be added later.

### Expected checksums (SHA-256)

These are the hashes of the files used during development. If your downloads have
different hashes, they are a different release, and the numbers in this project
will not reproduce exactly.

| File | SHA-256 |
|---|---|
| HI-Small_Trans.csv | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| HI-Small_Patterns.txt | `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b` |
| LI-Small_Trans.csv | `f7a9940339c78b5d1476071505b5867c0f5319640b9c4279d13f531bf965319d` |
| LI-Small_Patterns.txt | `734c15b7ff4be1a04a3529bd1524b92c1e8b62289d3ce4881423596ee47a208f` |

## Measured contents

| File | Rows | Laundering | Rate | Timestamps |
|---|---|---|---|---|
| HI-Small_Trans.csv | 5,078,345 | 5,177 | 0.102% | 2022-09-01 to 2022-09-18 |
| LI-Small_Trans.csv | 6,924,049 | 3,565 | 0.051% | 2022-09-01 to 2022-09-17 |

| Patterns file | Attempts | Transactions in attempts |
|---|---|---|
| HI-Small_Patterns.txt | 370 | 3,209 (about 62% of positives) |
| LI-Small_Patterns.txt | 117 | 1,023 (about 29% of positives) |

## Schema: `*_Trans.csv`

| Raw header | Canonical name | Type | Notes |
|---|---|---|---|
| Timestamp | `timestamp` | datetime | Format `YYYY/MM/DD HH:MM`, minute resolution, no timezone |
| From Bank | `from_bank` | string | Numeric code with **leading zeros that matter** (`010` is not `10`) |
| Account | `from_account` | string | Hex-like ID. The header name is duplicated, so columns are mapped by position |
| To Bank | `to_bank` | string | Same as From Bank |
| Account | `to_account` | string | Second `Account` column |
| Amount Received | `amount_received` | float | In Receiving Currency |
| Receiving Currency | `receiving_currency` | category | 15 values, including Bitcoin |
| Amount Paid | `amount_paid` | float | In Payment Currency |
| Payment Currency | `payment_currency` | category | |
| Payment Format | `payment_format` | category | Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin |
| Is Laundering | `is_laundering` | int8 | 0 or 1 |

The loader adds these columns:

- `tx_id`: the row number in the source file, used as a stable key.
- `from_id` / `to_id`: account node IDs of the form `"<bank>:<account>"`.
- `amount_*_usd`: amounts converted to USD.
- `typology` / `attempt_id`: taken from the patterns file. `NaN` means the
  transaction is in no documented attempt. It does not mean the transaction is benign.
- `split`: one of train, val, test or tail.

## Schema: `*_Patterns.txt`

The file is a series of blocks:

```
BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>: <description>
<rows in the _Trans.csv layout, no header>
END LAUNDERING ATTEMPT - <TYPOLOGY>
```

There are eight typologies: FAN-IN, FAN-OUT, CYCLE, GATHER-SCATTER, SCATTER-GATHER,
BIPARTITE, STACK and RANDOM.

## Quirks that affect modelling

1. **Mixed currencies.** No FX table is shipped. USD rates are estimated as the
   median implied rate of direct USD cross-currency transactions, and only on the
   training window (`aml_sentinel.data.fx`).
2. **Self-transfers.** About 10% of rows are `Reinvestment` from an account to
   itself. They are excluded from graph edges by default.
3. **Collapsing tail.** After 2022-09-10, normal activity stops but laundering
   continues, leaving a few hundred rows per day that are about 60% positive. These
   rows go into a separate `tail` split and are excluded from evaluation. See
   `docs/DECISIONS.md`, ADR-003.
4. **Incomplete typology coverage.** Typology labels exist for only part of the
   positives (see the table above).
5. **Simulator shortcuts (HI-Small).** These were measured in `notebooks/01_eda.ipynb`:
   - ACH carries 87% of positives.
   - Wire and Reinvestment carry no positives at all.
   - Cross-currency transfers are never laundering (0 of 72,170).

   Models can exploit these shortcuts in ways that would not carry over to real
   data, so evaluation will report an ablation without them.
