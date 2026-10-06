# Week 03

## Objetivo

1. Comparar **holdout vs CV** e decidir em qual número confiar.
2. Testar se trocar o **scaler** (none/standard/minmax/robust) e a **imputação** (median vs KNN) melhora o resultado.

## Comparação — Holdout vs Cross-Validation


| Modelo              | Holdout (1 split) | Holdout (30 seeds) | CV (mean ± std) | Gap treino−val |
| ------------------- | ----------------- | ------------------ | --------------- | -------------- |
| Dummy (maioria)     | 0.550             | 0.550 – 0.550      | 0.549 ± 0.000   | 0.000          |
| Logistic regression | 0.672             | 0.658 – 0.697      | 0.672 ± 0.013   | +0.003         |
| Decision tree       | 0.602             | 0.558 – 0.624      | 0.611 ± 0.015   | +0.084         |
| Random forest       | 0.655             | 0.623 – 0.674      | 0.652 ± 0.017   | +0.080         |




## Comparação — Scaler (regressão logística, target encoder)


| Scaler                | CV (mean ± std) | Gap    |
| --------------------- | --------------- | ------ |
| none (sem normalizar) | 0.6720 ± 0.0133 | +0.003 |
| standard (z-score)    | 0.6722 ± 0.0131 | +0.003 |
| minmax ([0,1])        | 0.6720 ± 0.0136 | +0.002 |
| robust (atual)        | 0.6723 ± 0.0132 | +0.003 |


