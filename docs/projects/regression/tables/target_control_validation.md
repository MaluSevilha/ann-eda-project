| Modelo | Alvo no treino | Configuração | MAE (US$) | RMSE (US$) | R² |
|---|---|---|---:|---:|---:|
| Ridge oficial | dólares | `alpha=1` | 7.813,89 | 11.795,41 | 0,816399 |
| Ridge de controle | `log1p` | `alpha=10` | 7.658,60 | 12.021,95 | 0,809278 |
| MLP final | `log1p` | `low_lr_regularized` | 6.265,05 | 9.821,37 | 0,872710 |
