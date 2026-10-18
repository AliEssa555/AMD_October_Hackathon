# Ablation

**Noise floor: 0.0126 mAP50** (measured across der_enh_sevcond x3 seeds). A delta smaller than this is not evidence and is marked *(within noise)*.

Each dataset is compared against **its own** baseline; deltas are never taken across datasets, since BDD100K's absolute mAP is far lower (ten classes including traffic lights and signs, a few pixels across at 448 px).

Excluded as smoke tests rather than seeds: bdd_baseline-s99 (3 epochs).

## bdd100k

| variant | seeds | epochs | mAP50 | min-max | mAP50-95 | Δ mAP50 | +params % | GFLOPs | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| `bdd_baseline` | 1 | 50 | 0.3696 | 0.3696-0.3696 | 0.2184 | +0.0000 | 0.00 | 0.00 | - |
| `bdd_der` | 1 | 50 | 0.3731 | 0.3731-0.3731 | 0.2195 | +0.0035 *(within noise)* | 7.20 | 0.00 | - |
| `bdd_der_enh` | 1 | 48 | 0.3720 | 0.3720-0.3720 | 0.2188 | +0.0024 *(within noise)* | 9.52 | 0.00 | - |
| `bdd_enh` | 1 | 50 | 0.3695 | 0.3695-0.3695 | 0.2178 | -0.0001 *(within noise)* | 2.32 | 0.00 | - |

## exdark

| variant | seeds | epochs | mAP50 | min-max | mAP50-95 | Δ mAP50 | +params % | GFLOPs | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 3 | 100 | 0.5661 | 0.5619-0.5694 | 0.2677 | +0.0000 | 0.00 | 3.97 | 24.35 |
| `der` | 1 | 100 | 0.5688 | 0.5688-0.5688 | 0.2718 | +0.0027 *(within noise)* | 7.20 | 0.00 | - |
| `der_enh` | 1 | 100 | 0.5657 | 0.5657-0.5657 | 0.2724 | -0.0004 *(within noise)* | 9.52 | 0.00 | - |
| `der_enh_deronly10` | 1 | 100 | 0.5700 | 0.5700-0.5700 | 0.2749 | +0.0039 *(within noise)* | 9.52 | - | - |
| `der_enh_deronly50` | 1 | 100 | 0.3163 | 0.3163-0.3163 | 0.1416 | -0.2498 | 9.52 | - | - |
| `der_enh_ent01` | 1 | 100 | 0.5605 | 0.5605-0.5605 | 0.2678 | -0.0056 *(within noise)* | 9.52 | - | - |
| `der_enh_ent03` | 1 | 100 | 0.5672 | 0.5672-0.5672 | 0.2766 | +0.0010 *(within noise)* | 9.52 | - | - |
| `der_enh_ent05` | 1 | 100 | 0.5646 | 0.5646-0.5646 | 0.2686 | -0.0016 *(within noise)* | 9.52 | - | - |
| `der_enh_ent10` | 1 | 100 | 0.5638 | 0.5638-0.5638 | 0.2702 | -0.0023 *(within noise)* | 9.52 | 0.00 | - |
| `der_enh_ent20` | 1 | 100 | 0.5755 | 0.5755-0.5755 | 0.2798 | +0.0094 *(within noise)* | 9.52 | - | - |
| `der_enh_ent50` | 1 | 100 | 0.4147 | 0.4147-0.4147 | 0.1926 | -0.1515 | 9.52 | 0.00 | - |
| `der_enh_entreg` | 1 | 100 | 0.5721 | 0.5721-0.5721 | 0.2706 | +0.0060 *(within noise)* | 9.52 | 4.14 | 34.97 |
| `der_enh_sevcond` | 3 | 100 | 0.5699 | 0.5630-0.5756 | 0.2696 | +0.0037 *(within noise)* | 9.53 | 0.00 | - |
| `der_routes_enh` | 1 | 100 | 0.5676 | 0.5676-0.5676 | 0.2747 | +0.0015 *(within noise)* | 2.32 | 0.00 | - |
| `enh` | 1 | 100 | 0.5608 | 0.5608-0.5608 | 0.2697 | -0.0053 *(within noise)* | 2.32 | 0.00 | - |

### exdark: mAP50 by bin

| variant | test_q1 | test_q2 | test_q3 | test_q4 |
|---|---|---|---|---|
| `baseline` | 0.5610 | 0.6359 | 0.5882 | 0.5710 |
| `der` | - | - | - | - |
| `der_enh` | - | - | - | - |
| `der_enh_deronly10` | - | - | - | - |
| `der_enh_deronly50` | - | - | - | - |
| `der_enh_ent01` | - | - | - | - |
| `der_enh_ent03` | - | - | - | - |
| `der_enh_ent05` | - | - | - | - |
| `der_enh_ent10` | - | - | - | - |
| `der_enh_ent20` | - | - | - | - |
| `der_enh_ent50` | - | - | - | - |
| `der_enh_entreg` | 0.5757 | 0.6572 | 0.5687 | 0.5654 |
| `der_enh_sevcond` | - | - | - | - |
| `der_routes_enh` | - | - | - | - |
| `enh` | - | - | - | - |

## Measurements that failed

- `der_enh_deronly10`: RuntimeError: Invalid device string: '0'
- `der_enh_deronly10`: get_flops returned 0 (thop missing?)
- `der_enh_deronly50`: RuntimeError: Invalid device string: '0'
- `der_enh_deronly50`: get_flops returned 0 (thop missing?)
- `der_enh_ent01`: RuntimeError: Invalid device string: '0'
- `der_enh_ent01`: get_flops returned 0 (thop missing?)
- `der_enh_ent03`: RuntimeError: Invalid device string: '0'
- `der_enh_ent03`: get_flops returned 0 (thop missing?)
- `der_enh_ent05`: RuntimeError: Invalid device string: '0'
- `der_enh_ent05`: get_flops returned 0 (thop missing?)

