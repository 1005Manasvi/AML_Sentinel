# System 2 detectors: Sample

Detectors produce evidence, not verdicts. `strict` cases see nothing after the alert (headline). `next_day` cases may see 24 h past it, modelling a later investigation (a different scenario).

## Horizon `strict`: all sampled test alerts

633 alerts, 211 laundering (base rate 33.3%). Median 39 ms per case (p95 352 ms); 80.6% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 495 | 72.0% | 81.3% | 30.7% | 0.92x |
| circular | 209 | 9.0% | 45.0% | 9.1% | 0.27x |
| structuring | 25 | 11.4% | 0.2% | 96.0% | 2.88x |
| fan_in | 404 | 51.7% | 69.9% | 27.0% | 0.81x |
| fan_out | 378 | 35.1% | 72.0% | 19.6% | 0.59x |
| scatter_gather | 252 | 15.6% | 51.9% | 13.1% | 0.39x |
| gather_scatter | 399 | 39.3% | 74.9% | 20.8% | 0.62x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 16 | n/a | 68.8% |
| CYCLE | 14 | 28.6% | 57.1% |
| FAN-IN | 19 | 84.2% | 84.2% |
| FAN-OUT | 13 | 76.9% | 84.6% |
| GATHER-SCATTER | 26 | 50.0% | 80.8% |
| RANDOM | 14 | n/a | 50.0% |
| SCATTER-GATHER | 20 | 55.0% | 85.0% |
| STACK | 22 | n/a | 54.5% |
| STRUCTURING | 29 | 79.3% | 93.1% |

## Horizon `next_day`: all sampled test alerts

633 alerts, 211 laundering (base rate 33.3%). Median 75 ms per case (p95 410 ms); 90.7% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 556 | 89.1% | 87.2% | 33.8% | 1.01x |
| circular | 226 | 15.2% | 46.0% | 14.2% | 0.42x |
| structuring | 33 | 14.2% | 0.7% | 90.9% | 2.73x |
| fan_in | 472 | 68.2% | 77.7% | 30.5% | 0.92x |
| fan_out | 445 | 52.1% | 79.4% | 24.7% | 0.74x |
| scatter_gather | 350 | 33.2% | 66.4% | 20.0% | 0.60x |
| gather_scatter | 467 | 55.5% | 82.9% | 25.1% | 0.75x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 16 | n/a | 87.5% |
| CYCLE | 14 | 50.0% | 71.4% |
| FAN-IN | 19 | 100.0% | 100.0% |
| FAN-OUT | 13 | 92.3% | 100.0% |
| GATHER-SCATTER | 26 | 69.2% | 92.3% |
| RANDOM | 14 | n/a | 57.1% |
| SCATTER-GATHER | 20 | 70.0% | 95.0% |
| STACK | 22 | n/a | 86.4% |
| STRUCTURING | 29 | 100.0% | 100.0% |
