| condição                         |   quantidade | classificação                            | decisão                                                  |
|:---------------------------------|-------------:|:-----------------------------------------|:---------------------------------------------------------|
| pct_above_hs_threshold_5yr > 100 |          427 | Inconsistência matemática                | Excluir: variável de 5 anos, posterior ao alvo           |
| pct_working_in_state_5yr > 100   |          145 | Inconsistência matemática                | Excluir: variável de 5 anos, posterior ao alvo           |
| crescimento 1º–5º ano > 200%     |          215 | Valor extremo possível                   | Excluir: usa informação posterior ao alvo                |
| credential_level = 99            |         1700 | Categoria válida: Non-Credential Program | Usar credential_name; excluir apenas o código redundante |
| tuition in-state igual a zero    |           89 | Valor plausível                          | Manter a feature; não corrigir como erro                 |
