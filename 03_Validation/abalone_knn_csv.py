#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Atividade 03 - Avaliação de classificadores
Algoritmo utilizado: k-NN (k-Nearest Neighbors)
Dataset: Abalone — classificar o tipo (1, 2 ou 3) do molusco
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTAÇÕES
# Cada linha traz uma ferramenta específica para o código usar
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
# sklearn é a biblioteca de machine learning
# KNeighborsClassifier é o algoritmo k-NN já pronto para usar

from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler é o normalizador — coloca todas as colunas na escala 0 a 1
# sem isso, colunas com valores maiores dominariam o cálculo de distância do k-NN

from sklearn.model_selection import cross_val_score
# cross_val_score é a função que faz o cross-validation automaticamente
# ela divide os dados em folds e testa o modelo em cada um

import requests
import os



# ─────────────────────────────────────────────────────────────────
# CAMINHO DOS ARQUIVOS
# Garante que o Python procure os CSVs na mesma pasta deste script
# sem isso, daria FileNotFoundError se o terminal estiver em outra pasta
# ─────────────────────────────────────────────────────────────────

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.path.abspath(__file__)  → caminho completo deste arquivo .py
# os.path.dirname(...)       → pega só a pasta onde ele está
# os.chdir(...)              → muda o diretório de trabalho para essa pasta


# ─────────────────────────────────────────────────────────────────
# 1. LEITURA DO DATASET
# ─────────────────────────────────────────────────────────────────

print('\n--- Lendo o dataset ---')
data = pd.read_csv('abalone_dataset.csv')
# lê o arquivo CSV e armazena como uma tabela (DataFrame) na variável data

print(f'Tamanho do dataset: {data.shape[0]} linhas e {data.shape[1]} colunas')
print('Distribuição das classes:\n', data['type'].value_counts())
# verifica se as 3 classes estão equilibradas — importante para o modelo não ficar viciado


# ─────────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# Limpa e prepara os dados antes de entrar no modelo
# ─────────────────────────────────────────────────────────────────

print('\n--- Pré-processamento ---')

# 2.1 Converter sex de texto para número
# o k-NN só trabalha com números — texto precisa ser convertido
# M (macho) → 0 | F (fêmea) → 1 | I (infantil) → 2
data['sex'] = data['sex'].map({'M': 0, 'F': 1, 'I': 2})
# .map() substitui cada valor de acordo com o dicionário fornecido
print('sex convertido: M=0, F=1, I=2')


# 2.2 Tratar os 2 zeros em height
# height=0 é biologicamente impossível (molusco com altura zero não existe)
# esses zeros são dados não coletados — tratamos como ausentes
data['height'] = data['height'].replace(0, np.nan)
# .replace(0, np.nan) troca os zeros por NaN (célula vazia)

data['height'] = data['height'].fillna(data['height'].median())
# .fillna() preenche os NaN com a mediana da coluna
# usamos mediana (e não média) pois é mais resistente a valores extremos
print(f'Zeros em height tratados com a mediana: {data["height"].median():.4f}')


# 2.3 Tratar outliers com o método IQR
# outliers são valores extremos que distorcem o cálculo de distância do k-NN
# o método IQR "apara" esses valores até o limite aceitável
# sem deletar nenhuma linha do dataset

numeric_cols = ['length', 'diameter', 'height',
                'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']
# lista das colunas numéricas onde verificaremos outliers
# sex não entra pois já virou 0, 1, 2 — não tem outlier possível

Q1 = data[numeric_cols].quantile(0.25)
# Q1 é o valor que 25% dos dados estão abaixo — o "início da faixa central"

Q3 = data[numeric_cols].quantile(0.75)
# Q3 é o valor que 75% dos dados estão abaixo — o "fim da faixa central"

IQR = Q3 - Q1
# IQR é a distância entre Q1 e Q3 — representa a "largura" da faixa normal

for col in numeric_cols:
    limite_inferior = Q1[col] - 1.5 * IQR[col]
    limite_superior = Q3[col] + 1.5 * IQR[col]
    data[col] = data[col].clip(lower=limite_inferior, upper=limite_superior)
    # .clip() apara os valores: qualquer coisa abaixo do limite inferior vira
    # o limite inferior, e qualquer coisa acima do superior vira o superior

print('Outliers tratados com IQR nas colunas numéricas')
print('Dados após pré-processamento:\n', data.describe().round(3))


# ─────────────────────────────────────────────────────────────────
# 3. SEPARANDO X e y
# X = as colunas que o modelo usa para aprender (entradas)
# y = a coluna que o modelo vai prever (saída)
# ─────────────────────────────────────────────────────────────────

feature_cols = ['sex', 'length', 'diameter', 'height',
                'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']
# todas as colunas de entrada — exceto 'type' que é o que queremos prever

X = data[feature_cols]
# X contém as 8 colunas de características de cada abalone

y = data['type']
# y contém o tipo correto de cada abalone (1, 2 ou 3)
# é com y que o modelo aprende o que é "certo"


# ─────────────────────────────────────────────────────────────────
# 4. NORMALIZAÇÃO
# Coloca todas as colunas na mesma escala (0 a 1)
# sem isso, colunas como whole_weight dominariam o cálculo de distância
# e colunas menores como sex (0, 1, 2) quase não influenciariam
# ─────────────────────────────────────────────────────────────────

scaler = MinMaxScaler()
# cria o normalizador — ainda está vazio, não aprendeu nada

X_scaled = scaler.fit_transform(X)
# fit   → aprende o mínimo e máximo de cada coluna nos dados de treino
# transform → aplica a fórmula: (valor - mínimo) / (máximo - mínimo)
# resultado: todos os valores ficam entre 0 e 1

X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
# converte de volta para DataFrame para manter os nomes das colunas
print('\nNormalização aplicada — todas as colunas agora entre 0 e 1')


# ─────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION
# Testa o modelo em 10 partes diferentes dos dados para medir
# a acurácia real — sem usar os dados de teste durante o treino
#
# Por que 10 folds? Com 3132 linhas, 10 folds cria grupos de ~313
# linhas de teste por rodada — representativo o suficiente
# ─────────────────────────────────────────────────────────────────

print('\n--- Cross-Validation (10 folds) ---')
print('Testando diferentes valores de k para encontrar o melhor:\n')

resultados = []
for k in [3, 5, 7, 9, 11, 15, 21]:
    modelo_teste = KNeighborsClassifier(n_neighbors=k)
    # cria um modelo k-NN com k vizinhos — ainda não treinou

    scores = cross_val_score(modelo_teste, X_scaled, y, cv=10)
    # divide X_scaled e y em 10 folds
    # em cada rodada: treina com 9 folds e testa no 1 restante
    # retorna um array com 10 acurácias (uma por rodada)

    media = scores.mean()
    desvio = scores.std()
    resultados.append((k, media, desvio))

    print(f'k={k:2d} | acurácia média: {media:.4f} ({media*100:.1f}%) '
          f'| desvio: {desvio:.4f}')
    # média alta + desvio baixo = modelo estável e confiável


# Escolhendo o melhor k automaticamente
melhor = max(resultados, key=lambda x: x[1])
melhor_k = melhor[0]
print(f'\nMelhor resultado: k={melhor_k} com média {melhor[1]*100:.1f}% '
      f'e desvio {melhor[2]:.4f}')
print('Decisão: usar k=21 — melhor acurácia média com desvio baixo e estável')


# ─────────────────────────────────────────────────────────────────
# 6. TREINAR O MODELO FINAL
# Após validar, treina com 100% dos dados de treino
# quanto mais dados o k-NN vê, mais vizinhos ele conhece — melhor
# ─────────────────────────────────────────────────────────────────

print('\n--- Treinando modelo final com k=21 ---')
melhor_k = 21
# fixamos k=21 pois foi o que deu melhor resultado no cross-validation

modelo_final = KNeighborsClassifier(n_neighbors=melhor_k)
# cria o modelo com o k escolhido

modelo_final.fit(X_scaled, y)
# .fit() treina o modelo com TODOS os dados de treino
# o k-NN "memoriza" todos os pontos para calcular distâncias depois
print('Modelo treinado com sucesso')


# ─────────────────────────────────────────────────────────────────
# 7. PRÉ-PROCESSAMENTO E PREVISÃO DO ARQUIVO DE APLICAÇÃO
# O arquivo app contém abalones novos sem type — o modelo vai prever
# IMPORTANTE: aplicar exatamente os mesmos passos do treino
# ─────────────────────────────────────────────────────────────────

print('\n--- Processando arquivo de aplicação ---')
data_app = pd.read_csv('abalone_app.csv')
# lê os abalones novos que precisam ser classificados

# aplicando o mesmo pré-processamento
data_app['sex'] = data_app['sex'].map({'M': 0, 'F': 1, 'I': 2})
# converte sex igual ao treino

data_app['height'] = data_app['height'].replace(0, np.nan)
data_app['height'] = data_app['height'].fillna(data['height'].median())
# usa a mediana do treino (não do app) para manter consistência

for col in numeric_cols:
    limite_inferior = Q1[col] - 1.5 * IQR[col]
    limite_superior = Q3[col] + 1.5 * IQR[col]
    data_app[col] = data_app[col].clip(lower=limite_inferior, upper=limite_superior)
# aplica os mesmos limites de outliers calculados nos dados de treino

X_app = data_app[feature_cols]
# seleciona apenas as colunas de entrada — igual ao treino

X_app_scaled = scaler.transform(X_app)
# .transform() (SEM fit) — usa os min/max aprendidos no treino
# se fizéssemos fit_transform aqui, a escala seria diferente e o modelo erraria
print('Arquivo de aplicação pré-processado com a mesma escala do treino')

y_pred = modelo_final.predict(X_app_scaled)
# .predict() classifica cada abalone do app em tipo 1, 2 ou 3
# para cada abalone, encontra os 21 mais próximos no treino e vota
print(f'Previsões geradas: {len(y_pred)} abalones classificados')
print(f'Distribuição das previsões: {pd.Series(y_pred).value_counts().to_dict()}')


# ─────────────────────────────────────────────────────────────────
# 8. ENVIO AO SERVIDOR
# Envia as previsões para o servidor registrar a acurácia da equipe
# ATENÇÃO: só 1 envio a cada 12h — valide bem antes de rodar essa parte
# ─────────────────────────────────────────────────────────────────

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
# endereço do servidor — diferente da atividade 01

DEV_KEY = "grupo central"
# chave de identificação da equipe — substitua se necessário

data_envio = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}
# cria o pacote de dados no formato que o servidor espera
# .to_json(orient='values') transforma o array de previsões em texto JSON

r = requests.post(url=URL, data=data_envio)
# envia as previsões via requisição POST (como um formulário web)

print('\n--- Resposta do servidor ---')
print(r.text)
# imprime o resultado — acurácia registrada pelo servidor
