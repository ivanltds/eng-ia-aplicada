gere um resumo curto da aula22:11Claude respondeu: Sprint 1 — Resumo da AulaSprint 1 — Resumo da Aula
O que é treinar um modelo
Treinar = encontrar o peso certo numa função desconhecida. O modelo começa com um chute e vai ajustando até errar o mínimo possível.
Pesos
São os parâmetros que o modelo ajusta. No nosso exemplo: preço = área × peso. O modelo descobriu que peso ≈ 5.000 sozinho.
Gradient Descent
O algoritmo que guia o ajuste. O gradiente aponta a direção que aumenta o erro — o modelo vai na direção oposta. Sempre descendo em direção ao mínimo, como descer uma montanha no escuro a passos.
Learning Rate
O tamanho do passo. Muito grande → pula o mínimo. Muito pequeno → demora demais.
Loss Functions
A "nota de erro" do modelo. Duas principais:

MSE → para saídas numéricas (preço, temperatura). Eleva ao quadrado para evitar que erros opostos se cancelem.
Cross-Entropy → para categorias (positivo/negativo). Pune confiança errada com mais força.

Deliverable
regressao_linear.py — regressão linear do zero com NumPy, sem nenhuma biblioteca de ML pronta.