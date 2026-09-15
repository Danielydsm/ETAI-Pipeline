# Week 01

## Objetivo

1. Fazer o setup do repositório git para cada integrante.
2. Fazer uma alteração no repositório.
3. Alterar o modelo do pipeline (`logistic_regression` → `decision_tree`).
4. Comparar o resultado.
5. Subir atualizações no git. 

## Comparação


| Métrica        | logistic_regression | decision_tree |
| -------------- | ------------------- | ------------- |
| Train accuracy | 0.679               | 0.829         |
| Test accuracy  | 0.676               | 0.630         |
| Gap            | +0.004              | +0.199        |


A árvore de decisão (`max_depth=1000`) superajustou (gap de +0.199): melhor no
treino, pior no teste.

## Troubleshooting

Foi necessário criar um token (Personal Access Token) para realizar o `push`,
pois o GitHub não aceita mais senha via HTTPS.