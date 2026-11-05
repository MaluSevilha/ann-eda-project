| Modelo | Configuração | MAE (US$) | RMSE (US$) | R² |
|---|---|---:|---:|---:|
| MLP (média ± desvio) | `low_lr_regularized`; 3 seeds | 6.573,47 ± 18,90 | 11.152,28 ± 86,80 | 0,846778 ± 0,002380 |
| MLP (seed 42) | `low_lr_regularized` | 6.590,14 | 11.215,08 | 0,845054 |
| MLP (seed 43) | `low_lr_regularized` | 6.552,94 | 11.188,54 | 0,845786 |
| MLP (seed 44) | `low_lr_regularized` | 6.577,33 | 11.053,23 | 0,849494 |
| Ridge | `alpha=1` | 8.035,06 | 12.938,42 | 0,793776 |
| DummyRegressor | `strategy=mean` | 20.329,93 | 28.530,06 | -0,002726 |
