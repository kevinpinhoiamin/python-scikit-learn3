from sklearn.model_selection import cross_val_score
from collections import Counter
import pandas as pd
import numpy as np

texto1 = 'Se eu comprar cinco anos antecipados, eu ganho algum desconto?'
texto2 = 'O exercício 15 do curso de Java 1 está com a resposta errada. Pode conferir pf?'
texto3 = 'Existe algum curso para cuidar do marketing da minha empresa?'

classificacoes = pd.read_csv('emails.csv')
textos_puros = classificacoes['email']
textos_quebrados = textos_puros.str.lower().str.split(' ')

dicionario = set()
for lista in textos_quebrados:
    dicionario.update(lista)

total_de_palavras = len(dicionario)
tuplas = zip(dicionario, range(total_de_palavras))
tradutor = {palavra: indice for palavra, indice in tuplas}
print(total_de_palavras)

def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)

    for palavra in texto:
        if palavra in tradutor:
            posicao = tradutor[palavra]
            vetor[posicao] += 1

    return vetor


vetores_de_texto = [vetorizar_texto(texto, tradutor) for texto in textos_quebrados]
marcas = classificacoes['classificacao']

X = vetores_de_texto
Y = marcas

porcentagem_de_treino = 0.8
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[:tamanho_do_treino]
treino_marcacoes = Y[:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]



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
modelo_ada_boost = AdaBoostClassifier(random_state=0)
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
