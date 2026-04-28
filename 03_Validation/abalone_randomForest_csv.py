#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Atividade 03 - Avaliação de classificadores
Algoritmo utilizado: Random Forest (Substituindo o k-NN)
Dataset: Abalone — classificar o tipo (1, 2 ou 3) do molusco
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import requests

# Importando o Random Forest e as ferramentas de validação
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

# ─────────────────────────────────────────────────────────────────
# CAMINHO DOS ARQUIVOS
# ─────────────────────────────────────────────────────────────────

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────
# 1. LEITURA DO DATASET
# ─────────────────────────────────────────────────────────────────

print('\n--- Lendo o dataset ---')
data = pd.read_csv('abalone_dataset.csv')
print(f'Tamanho do dataset: {data.shape[0]} linhas e {data.shape[1]} colunas')

# ─────────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────

print('\n--- Pré-processamento ---')

# 2.1 Converter sex de texto para número
data['sex'] = data['sex'].map({'M': 0, 'F': 1, 'I': 2})

# 2.2 Tratar os 2 zeros em height
data['height'] = data['height'].replace(0, np.nan)
data['height'] = data['height'].fillna(data['height'].median())

# 2.3 Tratar outliers com o método IQR
# Embora o Random Forest seja resistente a outliers, mantê-los sob controle
# ajuda a estabilizar as árvores em datasets ruidosos.
numeric_cols = ['length', 'diameter', 'height',
                'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']

Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

for col in numeric_cols:
    limite_inferior = Q1[col] - 1.5 * IQR[col]
    limite_superior = Q3[col] + 1.5 * IQR[col]
    data[col] = data[col].clip(lower=limite_inferior, upper=limite_superior)

print('Dados pré-processados. Escalonamento (MinMaxScaler) ignorado pois Random Forest não precisa disso.')

# ─────────────────────────────────────────────────────────────────
# 3. SEPARANDO X e y
# ─────────────────────────────────────────────────────────────────

feature_cols = ['sex', 'length', 'diameter', 'height',
                'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']

X = data[feature_cols]
y = data['type']

# ─────────────────────────────────────────────────────────────────
# 4. NESTED CROSS-VALIDATION + GRID SEARCH
# ─────────────────────────────────────────────────────────────────

print('\n--- Nested Cross-Validation e Otimização do Random Forest ---')

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grade de parâmetros para o Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],       # Número de árvores na floresta
    'max_depth': [None, 10, 15],           # Profundidade máxima (None = cresce até o fim)
    'min_samples_leaf': [1, 2, 4]          # Mínimo de amostras no final do galho (ajuda contra overfitting)
}

# random_state=42 garante que os resultados sejam reproduzíveis
modelo_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=modelo_base, param_grid=param_grid, 
                           cv=inner_cv, scoring='accuracy', n_jobs=-1)

print('Executando Nested CV (isso vai exigir mais do processador)...')
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')

media_nested = nested_scores.mean()
desvio_nested = nested_scores.std()
print(f'Acurácia real esperada (Nested CV): {media_nested:.4f} ({media_nested*100:.1f}%) | desvio: {desvio_nested:.4f}')

# ─────────────────────────────────────────────────────────────────
# 5. TREINAR O MODELO FINAL
# ─────────────────────────────────────────────────────────────────

print('\n--- Treinando modelo final com GridSearchCV ---')

grid_search.fit(X, y)
modelo_final = grid_search.best_estimator_

print(f'Melhores hiperparâmetros escolhidos automaticamente:\n{grid_search.best_params_}')

# ─────────────────────────────────────────────────────────────────
# 6. PRÉ-PROCESSAMENTO E PREVISÃO DO ARQUIVO DE APLICAÇÃO
# ─────────────────────────────────────────────────────────────────

print('\n--- Processando arquivo de aplicação ---')
data_app = pd.read_csv('abalone_app.csv')

# Aplicando o mesmo pré-processamento
data_app['sex'] = data_app['sex'].map({'M': 0, 'F': 1, 'I': 2})
data_app['height'] = data_app['height'].replace(0, np.nan)
data_app['height'] = data_app['height'].fillna(data['height'].median())

for col in numeric_cols:
    limite_inferior = Q1[col] - 1.5 * IQR[col]
    limite_superior = Q3[col] + 1.5 * IQR[col]
    data_app[col] = data_app[col].clip(lower=limite_inferior, upper=limite_superior)

X_app = data_app[feature_cols]

# AQUI ESTÁ A GRANDE DIFERENÇA: não usamos scaler.transform(). 
# Passamos os dados direto para o modelo prever.
y_pred = modelo_final.predict(X_app)

print(f'Previsões geradas: {len(y_pred)} abalones classificados')
print(f'Distribuição das previsões: {pd.Series(y_pred).value_counts().to_dict()}')

# ─────────────────────────────────────────────────────────────────
# 7. ENVIO AO SERVIDOR
# ─────────────────────────────────────────────────────────────────

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "grupo central"

data_envio = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}

print('\n--- Enviando para o servidor... ---')
r = requests.post(url=URL, data=data_envio)

print('\n--- Resposta do servidor ---')
print(r.text)