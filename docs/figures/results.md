| Technique | Std error at N = 10⁶ | Variance reduction | Time (ms) | Efficiency gain |
|---|---|---|---|---|
| none | 0.01473 | ×1.0 | 72 | ×1.0 |
| antithetic | 0.01042 | ×2.0 | 71 | ×2.0 |
| control_variate | 0.00562 | ×6.9 | 102 | ×4.9 |
| importance_sampling | 0.01473 | ×1.0 | 82 | ×0.9 |
| stratification | 0.00117 | ×159.1 | 113 | ×101.5 |
| conditioning | 0.00969 | ×2.3 | 175 | ×0.9 |

European call closed form: 10.450584
Brennan–Schwartz: errors 1.3e-03, 4.2e-04, 1.4e-04, 5.0e-05, 1.7e-05; times (ms) 2, 4, 11, 30, 94
PSOR: errors 1.3e-03, 4.2e-04, 1.4e-04, 5.0e-05, 1.9e-05; times (ms) 4, 14, 63, 391, 2651

American put reference (S0=36): 4.486671
LSM 2 dates: 4.2057 ± 0.0165 (bias -0.2810)
LSM 5 dates: 4.3819 ± 0.0141 (bias -0.1048)
LSM 10 dates: 4.4377 ± 0.0132 (bias -0.0490)
LSM 25 dates: 4.4785 ± 0.0128 (bias -0.0082)
LSM 50 dates: 4.4741 ± 0.0128 (bias -0.0126)
LSM 100 dates: 4.4720 ± 0.0128 (bias -0.0147)
