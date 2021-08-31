from sklearn.model_selection import cross_val_score
from collections import Counter
import pandas as pd
import numpy as np

df = pd.read_csv('situacao_do_cliente.csv')

X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]



def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)

    print(f'Taxa de acerto do {nome}: {taxa_de_acerto}')

    return taxa_de_acerto


resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modelo_one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=10000))
resultado_one_vs_rest = fit_and_predict('OneVsRestClassifier', modelo_one_vs_rest, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_rest] = modelo_one_vs_rest

from sklearn.multiclass import OneVsOneClassifier
modelo_one_vs_one = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
resultado_one_vs_one = fit_and_predict('OneVsOneClassifier', modelo_one_vs_one, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_one] = modelo_one_vs_one

from sklearn.naive_bayes import MultinomialNB
modelo_multinomial = MultinomialNB()
resultado_multinomial = fit_and_predict('MultinomialNB', modelo_multinomial, treino_dados, treino_marcacoes)
resultados[resultado_multinomial] = modelo_multinomial

from sklearn.ensemble import AdaBoostClassifier
modelo_ada_boost = AdaBoostClassifier()
resultado_ada_boost = fit_and_predict('AdaBoostClassifier', modelo_ada_boost, treino_dados, treino_marcacoes)
resultados[resultado_ada_boost] = modelo_ada_boost

print(resultados)



maximo = max(resultados)
vencedor = resultados[maximo]

print(f'Vencedor: {vencedor}')

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_dados)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100

print(f'Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {taxa_de_acerto}')



# a eficacia do algoritmo que chuta tudo um unico valor
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = acerto_base / len(validacao_marcacoes) * 100
print(f'Taxa de acerto base: {taxa_de_acerto_base}')

print(f'Total de testes: {len(validacao_dados)}')
