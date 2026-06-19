"""
Curva de aprendizado: TF-IDF + Regressão Logística treinado com frações
crescentes do conjunto de treino (10% a 100%), avaliado no mesmo teste.

Uso:
    python3 curva_aprendizado.py
"""

import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from avaliacao_alergenos import (
    split, micro_f1_pares, macro_f1_pares,
    limiares_por_rotulo, preditos_por_rotulo,
)

SEED = 42
FRACOES = [0.10, 0.25, 0.50, 0.75, 1.00]


def treinar_avaliar_tfidf(treino_sub, validacao, teste, golds):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_treino = vec.fit_transform([p["norm"] for p in treino_sub])
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_treino, np.array([p["y"] for p in treino_sub]))
    probs_val = clf.predict_proba(vec.transform([p["norm"] for p in validacao]))
    probs_teste = clf.predict_proba(vec.transform([p["norm"] for p in teste]))
    limiares = limiares_por_rotulo(probs_val, validacao)
    pares = list(zip(preditos_por_rotulo(probs_teste, limiares), golds))
    return micro_f1_pares(pares), macro_f1_pares(pares)


def main():
    treino, validacao, teste = split()
    golds = [p["gold"] for p in teste]
    embaralhado = list(treino)
    random.Random(SEED).shuffle(embaralhado)

    print(f"{'fração':>7} {'n_treino':>9} {'micro-F1':>9} {'macro-F1':>9}")
    for fracao in FRACOES:
        n = int(len(embaralhado) * fracao)
        micro, macro = treinar_avaliar_tfidf(embaralhado[:n], validacao, teste, golds)
        print(f"{fracao:7.2f} {n:9d} {micro:9.3f} {macro:9.3f}", flush=True)


if __name__ == "__main__":
    main()
