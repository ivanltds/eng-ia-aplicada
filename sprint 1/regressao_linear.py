import numpy as np

# ============================================================
# MBA EM ENGENHARIA DE IA APLICADA
# Épico 1 — Sprint 1 | Deliverable: Regressão Linear do Zero
# Aluno: Ivan Souza
# ============================================================
#
# O QUE ESTE ARQUIVO FAZ:
# Treina um modelo do zero para prever preços de imóveis
# usando apenas NumPy — sem bibliotecas de ML prontas.
# Você vai ver o modelo aprendendo passo a passo.
# ============================================================


# ============================================================
# PARTE 1 — OS DADOS
# ============================================================

# Nossos dados de treinamento: área do imóvel e preço real
# O modelo vai tentar descobrir a relação entre os dois
area_m2       = np.array([50,  80,  100, 120, 60,  90])   # metros quadrados
preco_real    = np.array([250, 400, 500, 600, 300, 450])   # em R$ mil

# A relação "escondida" nos dados é: preço ≈ área × 5
# O modelo vai descobrir isso sozinho, sem que a gente diga

print("=" * 55)
print("DADOS DE TREINAMENTO")
print("=" * 55)
for area, preco in zip(area_m2, preco_real):
    print(f"  Área: {area:3d} m²  →  Preço real: R$ {preco}k")


# ============================================================
# PARTE 2 — INICIALIZAÇÃO
# ============================================================

# O "peso" é o parâmetro que o modelo vai ajustar.
# No nosso caso: preço = área × peso
# Começa com um chute ruim de propósito — o modelo vai corrigir

peso = 1.0   # chute inicial (o certo seria ~5.0)

# Learning rate = tamanho do passo em cada ajuste
# Muito grande → pula o mínimo
# Muito pequeno → demora demais para convergir
learning_rate = 0.0001

# Número de rodadas de treinamento
epocas = 200

print("\n" + "=" * 55)
print("INÍCIO DO TREINAMENTO")
print(f"  Peso inicial:   {peso}")
print(f"  Learning rate:  {learning_rate}")
print(f"  Épocas:         {epocas}")
print("=" * 55)


# ============================================================
# PARTE 3 — O LOOP DE TREINAMENTO
# ============================================================
# Esse loop é o coração do Machine Learning.
# A cada época:
#   1. O modelo faz uma previsão com o peso atual
#   2. Calcula o erro (o quanto errou)
#   3. Calcula o gradiente (direção do ajuste)
#   4. Atualiza o peso na direção que reduz o erro

historico_erros = []  # vamos guardar o erro de cada época

for epoca in range(epocas):

    # ---------------------------------------------------------
    # PASSO 1: PREVISÃO
    # O modelo usa o peso atual para prever os preços
    # ---------------------------------------------------------
    previsao = area_m2 * peso
    # Ex: se peso=1.0 e área=100 → previsão = 100k (mas real é 500k)

    # ---------------------------------------------------------
    # PASSO 2: ERRO (Loss Function — MSE)
    # MSE = Média dos erros ao quadrado
    # Usamos quadrado para:
    #   → tornar todos os erros positivos
    #   → punir erros grandes mais do que pequenos
    # ---------------------------------------------------------
    erros_individuais = (previsao - preco_real) ** 2
    mse = np.mean(erros_individuais)
    historico_erros.append(mse)

    # ---------------------------------------------------------
    # PASSO 3: GRADIENTE
    # O gradiente nos diz: "em qual direção o peso deve ir?"
    # Fórmula: média de (erro × área)
    # Multiplicamos pela área porque o peso é responsável
    # por quanto da área ele transforma em preço
    # ---------------------------------------------------------
    gradiente = np.mean((previsao - preco_real) * area_m2)

    # ---------------------------------------------------------
    # PASSO 4: ATUALIZAÇÃO DO PESO
    # Subtraímos o gradiente porque queremos ir na direção
    # que REDUZ o erro (direção oposta ao gradiente)
    # ---------------------------------------------------------
    peso = peso - learning_rate * gradiente

    # Mostra o progresso a cada 40 épocas
    if epoca % 40 == 0:
        print(f"  Época {epoca:3d}  |  Peso: {peso:6.2f}  |  Erro (MSE): {mse:,.0f}")

print(f"  Época {epocas-1:3d}  |  Peso: {peso:6.2f}  |  Erro (MSE): {historico_erros[-1]:,.0f}")


# ============================================================
# PARTE 4 — RESULTADO FINAL
# ============================================================

print("\n" + "=" * 55)
print("RESULTADO DO TREINAMENTO")
print("=" * 55)
print(f"  Peso encontrado pelo modelo: {peso:.4f}")
print(f"  Peso ideal (real):           5.0000")
print(f"  Diferença:                   {abs(peso - 5.0):.4f}")

print("\nPREVISÕES APÓS TREINAMENTO:")
print("-" * 45)
for area, preco in zip(area_m2, preco_real):
    previsto = area * peso
    erro_rs = abs(previsto - preco)
    print(f"  {area:3d} m²  →  Previsto: R$ {previsto:6.1f}k  |  Real: R$ {preco}k  |  Erro: R$ {erro_rs:.1f}k")


# ============================================================
# PARTE 5 — USANDO O MODELO TREINADO
# ============================================================

print("\n" + "=" * 55)
print("USANDO O MODELO PARA NOVOS IMÓVEIS")
print("=" * 55)

novos_imoveis = [75, 110, 200]

for area in novos_imoveis:
    preco_previsto = area * peso
    print(f"  Imóvel de {area} m²  →  Preço estimado: R$ {preco_previsto:.1f}k")

print("\n✅ Sprint 1 concluído! O modelo aprendeu a relação entre área e preço.")
print("   Próximo passo: /tutor-marcos Sprint 2 — Redes Neurais com PyTorch")
