"""
Comparação dos paradigmas de detecção de alérgenos no mesmo conjunto de teste.

  Simbólico
    1) BASE, substring da palavra-base
    2) ONTOLOGIA, derivações via OWL/SPARQL
    3) ONTOLOGIA + REGRAS, negação/precaução/idioma
  Híbrido
    4) ONTOLOGIA + BERT, ambiguidade resolvida por contexto (WSD)
  Aprendizado supervisionado
    5) TF-IDF + RegLog
    6) BERTimbau multirrótulo

Cada sistema é reportado com IC95% (bootstrap) e testes de significância
pareados contra o sistema 3. HIBRIDO=0 pula a abordagem 4 (lenta em CPU).

Uso:
    python3 comparacao.py
"""

import os

import numpy as np

from conhecimento_alergenos import ALERGENOS
from avaliacao_alergenos import (
    relatorio, detectar, FORMAS_BASE, split,
    limiares_por_rotulo, preditos_por_rotulo, teste_pareado,
)
from deteccao_ontologia import carregar_formas_da_ontologia
from desambiguacao import detectar_preciso

MODELO_NEURAL = "modelo_bert_alergenos"


def pares_simbolico(teste, fn):
    return [(fn(p["norm"]), p["gold"]) for p in teste]


def pares_tfidf(treino, validacao, teste):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_treino = vec.fit_transform([p["norm"] for p in treino])
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_treino, np.array([p["y"] for p in treino]))

    probs_val = clf.predict_proba(vec.transform([p["norm"] for p in validacao]))
    probs_teste = clf.predict_proba(vec.transform([p["norm"] for p in teste]))
    limiares = limiares_por_rotulo(probs_val, validacao)
    preds = preditos_por_rotulo(probs_teste, limiares)
    return [(pred, p["gold"]) for pred, p in zip(preds, teste)]


def pares_neural(teste):
    probs = np.load(f"{MODELO_NEURAL}/probs_teste.npy")
    limiares = np.load(f"{MODELO_NEURAL}/limiares.npy")
    preds = preditos_por_rotulo(probs, limiares)
    return [(pred, p["gold"]) for pred, p in zip(preds, teste)]


def main():
    treino, validacao, teste = split()
    mapa = carregar_formas_da_ontologia()

    sistemas = {}
    sistemas["1) SIMBÓLICO, BASE"] = pares_simbolico(teste, lambda t: detectar(t, FORMAS_BASE))
    sistemas["2) SIMBÓLICO, ONTOLOGIA"] = pares_simbolico(teste, lambda t: detectar(t, mapa))
    sistemas["3) SIMBÓLICO, ONTOLOGIA + REGRAS"] = pares_simbolico(teste, lambda t: detectar_preciso(t, mapa))

    if os.environ.get("HIBRIDO", "1") == "1":
        from desambiguacao_bert import detectar_hibrido
        sistemas["4) HÍBRIDO, ONTOLOGIA + BERT (WSD)"] = [
            (detectar_hibrido(p, mapa), p["gold"]) for p in teste]

    sistemas["5) NEURAL, TF-IDF + RegLog"] = pares_tfidf(treino, validacao, teste)

    if os.path.exists(f"{MODELO_NEURAL}/probs_teste.npy"):
        sistemas["6) NEURAL, BERTimbau multirrótulo"] = pares_neural(teste)
    else:
        print("(6) BERTimbau não treinado, rode `python3 bert_classificador.py`")

    for titulo, pares in sistemas.items():
        relatorio(pares, titulo, ic=True)

    referencia = "3) SIMBÓLICO, ONTOLOGIA + REGRAS"
    print(f"\n### Significância (bootstrap pareado de micro-F1) vs «{referencia}»")
    for titulo, pares in sistemas.items():
        if titulo == referencia:
            continue
        dif, p = teste_pareado(pares, sistemas[referencia])
        sinal = "significativo" if p < 0.05 else "n.s."
        print(f"  {titulo:42s} Δ={dif:+.3f}  p={p:.3f}  ({sinal})")


if __name__ == "__main__":
    main()
