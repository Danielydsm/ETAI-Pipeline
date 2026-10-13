# Week 05

## Objetivo

1. Adicionar os ensembles ao pipeline (`src/model.py`): funções `oblique_random_forest()` e `adaboost()`, registradas no `_MODELS`.
2. Adicionar os search spaces de tuning no `config.yaml` para `random_forest`, `oblique_random_forest` e `adaboost`.
3. Comparar **default vs tuned (nested CV)** de todos os modelos, nos mesmos 5 folds.

## Comparação — default vs tuned (validação, média ± std dos 5 folds)


| Modelo                | Default: CV   | Default: gap | Tuned: nested CV | Tuned: gap |
| --------------------- | ------------- | ------------ | ---------------- | ---------- |
| Logistic regression   | 0.672 ± 0.013 | +0.003       | —                | —          |
| Decision tree         | 0.611 ± 0.015 | +0.084       | 0.672 ± 0.017    | +0.013     |
| Random forest         | 0.652 ± 0.019 | +0.081       | 0.681 ± 0.015    | +0.009     |
| Oblique random forest | 0.650 ± 0.009 | +0.113       | 0.679 ± 0.013    | +0.013     |
| AdaBoost              | 0.676 ± 0.012 | +0.006       | 0.682 ± 0.017    | +0.004     |


## Conclusões

- **Tuning importa mais onde o default overfita.** As duas florestas crescem árvores inteiras por padrão (gaps de +0.08 e +0.11) e o tuning as eleva ~3 pontos (→ ~0.68). O AdaBoost com stumps já era bem-comportado (+0.006).
- **Tunados, os 3 ensembles empatam.** O melhor (AdaBoost) vence o random forest por só **+0.001** — escolher "vencedor" é escolher ruído.
- **A floresta oblíqua não vence a comum.** As features do COMPAS (contagens + categorias codificadas) são bem expressas por splits de 1 feature.
- **Tudo converge para ~0.68** → o teto é a informação nas features, não o modelo.

