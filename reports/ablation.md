# Ablation

Baseline: `baseline`. Deltas are mAP50 against it.

**Noise floor: 0.0056 mAP50.** That is the spread measured across all seven ExDark variants on a single seed, with identical data and hyper-parameters. A delta smaller than it is not evidence of anything, and is marked *(within noise)* below.

| variant | seeds | mAP50 | min-max | mAP50-95 | Δ mAP50 | +params % | GFLOPs | ms/img |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 1 | 0.5672 | 0.5672-0.5672 | 0.2662 | +0.0000 | 0.00 | 3.97 | 24.35 |
| `der` | 1 | 0.5688 | 0.5688-0.5688 | 0.2718 | +0.0016 *(within noise)* | 7.20 | 4.15 | 30.23 |
| `der_enh` | 1 | 0.5657 | 0.5657-0.5657 | 0.2724 | -0.0015 *(within noise)* | 9.52 | 4.14 | 34.38 |
| `der_enh_entreg` | 1 | 0.5721 | 0.5721-0.5721 | 0.2706 | +0.0049 *(within noise)* | 9.52 | 4.14 | 34.97 |
| `der_enh_sevcond` | 1 | 0.5756 | 0.5756-0.5756 | 0.2677 | +0.0084 | 9.53 | 4.14 | 35.92 |

## mAP50 by bin

| variant | test_q1 | test_q2 | test_q3 | test_q4 |
|---|---|---|---|---|
| `baseline` | 0.5610 | 0.6359 | 0.5882 | 0.5710 |
| `der` | 0.5618 | 0.6731 | 0.5438 | 0.5701 |
| `der_enh` | 0.5554 | 0.6587 | 0.5586 | 0.5615 |
| `der_enh_entreg` | 0.5757 | 0.6572 | 0.5687 | 0.5654 |
| `der_enh_sevcond` | 0.5973 | 0.6550 | 0.5554 | 0.5774 |
