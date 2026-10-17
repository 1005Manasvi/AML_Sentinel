# System 2 detectors: HI-Small

Detectors produce evidence, not verdicts. `strict` cases see nothing after the alert (headline). `next_day` cases may see 24 h past it, modelling a later investigation (a different scenario).

## Horizon `strict`: all sampled test alerts

4,000 alerts, 271 laundering (base rate 6.8%). Median 17 ms per case (p95 36 ms); 18.6% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 1,156 | 51.7% | 27.2% | 12.1% | 1.79x |
| circular | 121 | 7.7% | 2.7% | 17.4% | 2.56x |
| structuring | 4 | 0.7% | 0.1% | 50.0% | 7.38x |
| fan_in | 691 | 22.5% | 16.9% | 8.8% | 1.30x |
| fan_out | 363 | 26.9% | 7.8% | 20.1% | 2.97x |
| scatter_gather | 37 | 9.6% | 0.3% | 70.3% | 10.37x |
| gather_scatter | 178 | 17.0% | 3.5% | 25.8% | 3.81x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 18 | n/a | 44.4% |
| CYCLE | 20 | 25.0% | 50.0% |
| FAN-IN | 14 | 42.9% | 64.3% |
| FAN-OUT | 34 | 70.6% | 73.5% |
| GATHER-SCATTER | 52 | 28.8% | 69.2% |
| RANDOM | 14 | n/a | 14.3% |
| SCATTER-GATHER | 40 | 52.5% | 75.0% |
| STACK | 37 | n/a | 32.4% |

## Horizon `strict`: escalated subset

200 alerts, 148 laundering (base rate 74.0%). Median 20 ms per case (p95 38 ms); 33.5% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 104 | 57.4% | 36.5% | 81.7% | 1.10x |
| circular | 13 | 7.4% | 3.8% | 84.6% | 1.14x |
| structuring | 1 | 0.7% | 0.0% | 100.0% | 1.35x |
| fan_in | 46 | 23.6% | 21.2% | 76.1% | 1.03x |
| fan_out | 53 | 32.4% | 9.6% | 90.6% | 1.22x |
| scatter_gather | 20 | 12.2% | 3.8% | 90.0% | 1.22x |
| gather_scatter | 32 | 18.9% | 7.7% | 87.5% | 1.18x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 13 | n/a | 38.5% |
| CYCLE | 8 | 25.0% | 37.5% |
| FAN-IN | 7 | 42.9% | 57.1% |
| FAN-OUT | 27 | 70.4% | 74.1% |
| GATHER-SCATTER | 32 | 34.4% | 71.9% |
| RANDOM | 10 | n/a | 0.0% |
| SCATTER-GATHER | 27 | 59.3% | 81.5% |
| STACK | 24 | n/a | 33.3% |

## Horizon `next_day`: all sampled test alerts

4,000 alerts, 271 laundering (base rate 6.8%). Median 20 ms per case (p95 46 ms); 25.4% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 1,947 | 64.6% | 47.5% | 9.0% | 1.33x |
| circular | 212 | 11.1% | 4.9% | 14.2% | 2.09x |
| structuring | 11 | 0.7% | 0.2% | 18.2% | 2.68x |
| fan_in | 1,438 | 32.1% | 36.2% | 6.1% | 0.89x |
| fan_out | 403 | 32.8% | 8.4% | 22.1% | 3.26x |
| scatter_gather | 50 | 12.5% | 0.4% | 68.0% | 10.04x |
| gather_scatter | 207 | 21.8% | 4.0% | 28.5% | 4.21x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 18 | n/a | 55.6% |
| CYCLE | 20 | 35.0% | 60.0% |
| FAN-IN | 14 | 71.4% | 92.9% |
| FAN-OUT | 34 | 91.2% | 94.1% |
| GATHER-SCATTER | 52 | 38.5% | 78.8% |
| RANDOM | 14 | n/a | 21.4% |
| SCATTER-GATHER | 40 | 67.5% | 92.5% |
| STACK | 37 | n/a | 35.1% |

## Horizon `next_day`: escalated subset

200 alerts, 148 laundering (base rate 74.0%). Median 22 ms per case (p95 48 ms); 39.5% of cases hit a hub cap.

| Detector | Fires | On laundering | On benign | Precision | Lift |
|---|---:|---:|---:|---:|---:|
| any | 128 | 69.6% | 48.1% | 80.5% | 1.09x |
| circular | 20 | 11.5% | 5.8% | 85.0% | 1.15x |
| structuring | 1 | 0.7% | 0.0% | 100.0% | 1.35x |
| fan_in | 60 | 31.8% | 25.0% | 78.3% | 1.06x |
| fan_out | 65 | 38.5% | 15.4% | 87.7% | 1.19x |
| scatter_gather | 25 | 15.5% | 3.8% | 92.0% | 1.24x |
| gather_scatter | 41 | 23.6% | 11.5% | 85.4% | 1.15x |

| Typology | Alerts | Matching detector fired | Any detector fired |
|---|---:|---:|---:|
| BIPARTITE | 13 | n/a | 38.5% |
| CYCLE | 8 | 37.5% | 50.0% |
| FAN-IN | 7 | 71.4% | 100.0% |
| FAN-OUT | 27 | 92.6% | 96.3% |
| GATHER-SCATTER | 32 | 46.9% | 81.2% |
| RANDOM | 10 | n/a | 0.0% |
| SCATTER-GATHER | 27 | 74.1% | 96.3% |
| STACK | 24 | n/a | 37.5% |
