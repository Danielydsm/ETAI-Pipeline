# Week 06

## Objetivo

Trazer o pipeline para a versão desta semana: **Adicionar métricas e dois novos modelos ao pipeline. Como desafio extra, implementar feature selection.** 

---

## Resultados — Scoreboard (F1, nested CV)


| Modelo                | Tuned | F1 (mean ± std) | Untuned |
| --------------------- | ----- | --------------- | ------- |
| HistGradientBoosting  | ✓     | 0.644 ± 0.018   | 0.630   |
| Random forest         | ✓     | 0.639 ± 0.018   | 0.606   |
| Logistic regression   | ✓     | 0.633 ± 0.012   | 0.634   |
| Decision tree         | ✓     | 0.633 ± 0.019   | 0.567   |
| Gradient boosting     | ✓     | 0.620 ± 0.021   | 0.607   |
| AdaBoost              | ✓     | 0.619 ± 0.016   | 0.599   |
| Oblique random forest | ✓     | 0.616 ± 0.011   | 0.596   |
| Naive Bayes           | ✓     | 0.404 ± 0.013   | 0.403   |
| Dummy (floor)         | ✗     | 0.000 ± 0.000   | 0.000   |




## Teste t pareado corrigido (melhor modelo = HistGradientBoosting, contra cada um)


| Contra                | mean_diff | wins | t     | p_value |
| --------------------- | --------- | ---- | ----- | ------- |
| random_forest         | +0.0051   | 4/5  | 0.63  | 0.563   |
| logistic_regression   | +0.0104   | 4/5  | 1.44  | 0.224   |
| decision_tree         | +0.0108   | 4/5  | 1.26  | 0.278   |
| gradient_boosting     | +0.0235   | 5/5  | 2.92  | 0.043   |
| adaboost              | +0.0242   | 5/5  | 3.69  | 0.021   |
| oblique_random_forest | +0.0279   | 5/5  | 3.72  | 0.021   |
| naive_bayes           | +0.2395   | 5/5  | 14.52 | 0.0001  |




## Conclusões

- **O topo ficou empatado:** O melhor modelo acabou sendo o **HistGradientBoosting (HGB)** com F1 de 0.644, mas na prática não teve diferença estatística entre ele, o Random Forest, a Regressão Logística e a Decision Tree. 
- **A vantagem real apareceu contra o meio da tabela:** Apesar do empate lá em cima, o HGB conseguiu provar que é realmente superior ao Gradient Boosting tradicional, ao AdaBoost e ao Oblique. Contra estes modelos, a diferença estatística realmente existe.

