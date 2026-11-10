# Week 09

## Objetivo

Fazer o pipeline funcionar **além do COMPAS** (notebook `09_beyond_compas.ipynb`), **Option A**: treinar e avaliar o mesmo pipeline num dataset novo — **Adult income** (1994 US Census / UCI), ainda uma pergunta binária (`income_over_50k`) — e depois prever os 2.000 casos novos de `adult_new.csv`.

## O que mudou

- **`config_adult.yaml`** — novo config (target, `sensitive_attr`, diagnóstico, preprocessing, output) mantendo o `src/` geral.
- **`main_week08.py`** — novo entry point: roda o pipeline em `config_adult.yaml` e prevê `adult_new.csv` (mesma limpeza, **sem** de-dup).
- **`main.py`** — extraí `run_pipeline(config_path)` (retorna `config, run, pipeline` refitado); `python main.py` (COMPAS) continua idêntico.
- **`src/evaluate.py`** — `fairness_report` agora só adiciona a comparação "COMPAS's own score" se `score_text` existir (o único ponto COMPAS-specific que quebrava).
- **`data/`** — `adult_train.csv`, `adult_new.csv`, `ADULT_README.md`.

## O que quebrou (e como corrigi)

| Quebrou | Por quê | Correção |
|---|---|---|
| `fairness_report` → `KeyError: 'score_text'` | o Adult não tem a coluna `score_text` do COMPAS | comparar com o COMPAS só se `score_text` estiver presente em `extras` |
| xgboost default (depth 6, 100 árvores, lr 0.3) ficava em accuracy 0.848, gap +0.12 | overfit no dev set pequeno (4.800 linhas) | modelo regularizado: `max_depth: 3, n_estimators: 300, learning_rate: 0.1` |
| tuning piorava F1 (0.669 → 0.655) | 15 trials overfitavam os folds internos | deixar tuning off; o default regularizado já bate os alvos |

## Resultados (alvos atingidos)

5-fold CV no development set (4.800 linhas), os 3 alvos **no mesmo run** (`python main_week08.py`):

| Métrica | Alvo | Floor (sempre "≤50K") | Atingido |
|---|---|---|---|
| ROC-AUC | ≥ 0.90 | 0.50 | **0.911 ± 0.016** ✓ |
| Accuracy | ≥ 0.855 | 0.76 | **0.862 ± 0.010** ✓ |
| F1 (classe 1, threshold 0.5) | ≥ 0.66 | 0.00 | **0.685 ± 0.030** ✓ |

- Gap train–validação: **+0.048** (bem regularizado; o default tinha +0.12).
- **Features mantidas (8):** `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `marital_status`, `occupation`, `relationship` — o `VoteSelector` **descartou todos os 25 `v`-cols anônimos** + `workclass`, `native_country`, `race`.
- **Previsão:** `results_adult/adult_new_predictions.csv` (2.000 linhas, colunas `id, probability, prediction`; rows out = rows in).

## Conclusões

- O pipeline **generalizou bem**: quase nada COMPAS-specific precisou mudar — só o `fairness_report` tinha uma dependência silenciosa de `score_text`.
- No Adult (36 features, 25 anônimas), a **feature selection compensa** (F1 0.685 vs 0.646 sem seleção) — o oposto do COMPAS, onde não compensava. É exatamente a situação que o notebook da week 8 diz que justifica a seleção: muitas features ruidosas.
- O alvo difícil foi a **accuracy**: o default overfitava e ficava em 0.848. Um modelo mais raso e de aprendizado mais lento generaliza melhor e sobe a accuracy para 0.862.
- **Tuning não ajudou aqui**: com 4.800 linhas de dev, 15 trials overfitam os folds internos e o nested score saiu *pior* que um bom default.
