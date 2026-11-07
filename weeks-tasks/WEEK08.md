# Week 08

## Objetivo

Trazer o pipeline para a versão desta semana (notebook `08_feature_selection.ipynb`, Part A):

1. **A1 — XGBoost:** registrar `XGBClassifier` no `_MODELS` e o search space (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`, …) no `config.yaml`.
2. **A2 — Feature selection:** adicionar o `VoteSelector` (random forest + gradient boosting + regressão logística L1 votam; mantém features com ≥ `min_votes` votos) como passo `select` dentro do `Pipeline`, ligado por `config.yaml` (`feature_selection.enabled`), com cache por run para o tuning não refazer `prep`/`select` a cada trial.
3. **A3 — A seleção compensa?:** rodar XGBoost **tunado** com e sem seleção, nos mesmos folds, e comparar com o teste t pareado corrigido.

## Resultados — XGBoost tunado, com vs sem feature selection (nested CV, F1)

| Config      | F1 (mean ± std) | Untuned | Features mantidos               |
| ----------- | --------------- | ------- | ------------------------------- |
| Com seleção | 0.622 ± 0.015   | 0.590   | 2 (`age`, `priors_count`)       |
| Sem seleção | 0.624 ± 0.012   | 0.604   | 10 (todas)                      |

- Com seleção, o nested CV escolheu `select__min_votes = 3` (manter só os unânimes): sobraram `age` e `priors_count` — exatamente os dois que o `VoteSelector` vota com 3/3 (o `age_cat` fica com 2).
- Sem seleção, o tuner converge para `max_depth = 1` (um booster raso de ~527 árvores), com ROC-AUC levemente acima (0.727 vs 0.724).

## Teste t pareado corrigido (com − sem seleção, mesmos 5 folds)

| mean_diff | wins | t     | p_value |
| --------- | ---- | ----- | ------- |
| −0.002    | 3/5  | −0.27 | 0.80    |

## Conclusões

- **A seleção não compensa no COMPAS.** A diferença é de 0.002 pontos — dentro do ruído (p = 0.80). O dataset já é limpo (~10 features, quase nenhuma coluna inútil), então não há ruído para o seletor descartar e recuperar score, ao contrário do notebook (onde 20 colunas de ruído eram recuperadas).
- **O modelo menor não perde nada.** Com só `age` e `priors_count`, o F1 fica idêntico ao modelo completo, só que mais rápido e muito mais fácil de explicar. Para COMPAS, deixar a seleção ligada custa tempo de treino sem ganho; vale ligar quando houver muitas features ruidosas (muitas colunas one-hot ou variáveis irrelevantes).
- **XGBoost não desbanca o HGB da week 6/7.** Tunado, fica em ~0.62 (abaixo do HistGradientBoosting 0.644 e do random forest 0.639). No COMPAS, os freios extras do XGBoost só passam a importar com mais features/ruído.
