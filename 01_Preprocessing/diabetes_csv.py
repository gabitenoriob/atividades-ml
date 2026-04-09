#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
##importando o normalizador MinMaxScaler para colocar todas as colunas na mesma escala
from sklearn.preprocessing import MinMaxScaler
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
#Pré processamento = limpeza e enriquecimento dos dados
print(' - Realizando pré-processamento dos dados')
print("nulos por coluna:\n", data.isnull().sum(), "\n")
print("dados estatísticos:\n", data.describe(), "\n")

#LIdando com zeros pois são dados faltantes para a maioria das colunas, então vou substituir por nulos para facilitar o preenchimento depois.
data['Glucose'] = data['Glucose'].replace(0, pd.NA)
data['BloodPressure'] = data['BloodPressure'].replace(0, pd.NA)
data['SkinThickness'] = data['SkinThickness'].replace(0, pd.NA)
data['Insulin'] = data['Insulin'].replace(0, pd.NA)
data['BMI'] = data['BMI'].replace(0, pd.NA)
#valores zeros nisso sao incompativeis com a vida

#Como insulina que é um dado muito importante tem muitos nulos, vou substituiir pela media de acordo com a faixa etária. para isso, vou tornar a idade um dado categórico, criando faixas etárias.
data['Age_cat'] = pd.cut(data.Age, bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
data['Insulin'] = data.groupby('Age_cat')['Insulin'].transform(lambda x: x.fillna(x.mean()))
print("media insulina por faixa etária:\n", data.groupby('Age_cat')['Insulin'].mean(), "\n")
#Valore entre 60 e 80 ficaram vazios, então vou preencher com a média geral.
data['Insulin'] = data['Insulin'].fillna(data['Insulin'].mean())
print("Insulina por faixa etária após preenchimento:\n", data.groupby('Age_cat')['Insulin'].mean(), "\n")

#Tambem preenchi os valores de pressão sanguínea com a média por faixa etária, pois tem poucos nulos e é um dado importante para o modelo.
data['BloodPressure'] = data.groupby('Age_cat')['BloodPressure'].transform(lambda x: x.fillna(x.mean()))
print("Pressão sanguínea por faixa etária:\n", data.groupby('Age_cat')['BloodPressure'].mean(), "\n")
#70 a 80 ficou vazio, então preencho com a média geral.
data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].mean())
print("Pressão sanguínea por faixa etária após preenchimento:\n", data.groupby('Age_cat')['BloodPressure'].mean(), "\n")

#E tambem BMI, media por idade
data['BMI'] = data.groupby('Age_cat')['BMI'].transform(lambda x: x.fillna(x.mean()))
print("BMI por faixa etária:\n", data.groupby('Age_cat')['BMI'].mean(), "\n")

#SkinThickness tem muitos nulos mas acho que isso o BMI pode ajudar a preencher, então vou preencher o bmi agrupando por idade e BMI e fazendo a media dos que estao naquela faixa idade e faixa de bmi, e depois preencher os valores nulos de skin thickness com a media do grupo.
data['BMI_cat'] = pd.cut(data.BMI, bins=[0, 18.5, 25, 30, 35, 40, 60], labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese', 'Morbidly Obese'])
data['SkinThickness'] = data.groupby(['Age_cat', 'BMI_cat'])['SkinThickness'].transform(lambda x: x.fillna(x.mean()))
print("SkinThickness por faixa etária e BMI:\n", data.groupby(['Age_cat', 'BMI_cat'])['SkinThickness'].mean(), "\n")
#Preencho os valores nulos de skin thickness com a média geral, pois tem muitos nulos e a média por faixa etária e BMI ficou muito baixa.
data['SkinThickness'] = data['SkinThickness'].fillna(data['SkinThickness'].mean())
print("SkinThickness por faixa etária e BMI após preenchimento:\n", data.groupby(['Age_cat', 'BMI_cat'])['SkinThickness'].mean(), "\n")
print("SkinThickness apos preenchimento:\n", data['SkinThickness'].describe(), "\n")

print("nulos por coluna após preenchimento:\n", data.isnull().sum(), "\n")
#os outros nulos tem poucos e não são tão importantes, então vou preencher com a média geral.
data['Glucose'] = data['Glucose'].fillna(data['Glucose'].mean())
data['BMI_cat'] = data['BMI_cat'].fillna(data['BMI_cat'].mode()[0])
data['Age_cat'] = data['Age_cat'].fillna(data['Age_cat'].mode()[0])
data['BMI'] = data['BMI'].fillna(data['BMI'].mean())    
print("nulos por coluna após preenchimento geral:\n", data.isnull().sum(), "\n")

#ajustar valores anomalos - detectando outliers com o método do IQR e substituindo por valores limites
numeric_cols = data.select_dtypes(include='number').columns.drop(['Outcome', 'Age'], errors='ignore')
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
print("IQR:\n", IQR, "\n")
# Substituindo os outliers por valores limites apenas nas colunas numéricas
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
for col in numeric_cols:
        data[col] = data[col].clip(lower=lower[col], upper=upper[col])
print("dados estatísticos após tratamento de outliers:\n", data.describe(), "\n")


# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI']
X = data[feature_cols]
y = data.Outcome

#normalizando os dados de treino com MinMaxScaler, colocando todas as colunas na escala de 0 a 1
scaler = MinMaxScaler()
#aprendendo os valores mínimo e máximo de cada coluna e aplicando a normalização nos dados de treino
X_scaled = scaler.fit_transform(X)
#convertendo o resultado de volta para DataFrame, recolocando os nomes das colunas
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_scaled, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
#aplicando a mesma normalização do treino nos dados de aplicação, sem reaprender os valores mínimo e máximo
X_app_scaled = scaler.transform(data_app)
#Mudando data_app por X_app_scaled e realizando as previsões com os dados de aplicação normalizados
y_pred = neigh.predict(X_app_scaled)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "grupo central"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")