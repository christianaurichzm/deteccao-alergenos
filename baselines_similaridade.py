"""
Baselines por similaridade (Levenshtein, Jaccard, spaCy, fastText, BERT-cosseno).

Para cada métrica, o limiar é ajustado na validação e o desempenho é medido
no conjunto de teste com IC95%. Métricas que exigem dependências externas
(spaCy, fastText) são opcionais: se o modelo não estiver instalado, o baseline
correspondente é pulado com aviso.

Uso:
    python3 baselines_similaridade.py
"""

import difflib
import functools
import unicodedata

from avaliacao_alergenos import split, relatorio, micro_f1_pares, normalizar
from conhecimento_alergenos import FORMAS_BASE

BASE_NORM = {a: [normalizar(f) for f in formas] for a, formas in FORMAS_BASE.items()}


def ingredientes(produto):
    partes = [s.strip() for s in produto["norm"].split(",") if s.strip()]
    return partes or [produto["norm"]]


def sim_levenshtein(a, b):
    """Razão de similaridade por edição (≈ fuzz.ratio, via difflib)."""
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def sim_jaccard(a, b):
    """Jaccard de caracteres."""
    ca, cb = set(a), set(b)
    uniao = ca | cb
    return (len(ca & cb) / len(uniao) * 100) if uniao else 0.0


@functools.lru_cache(maxsize=1)
def _bert():
    import torch
    from transformers import AutoTokenizer, AutoModel
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(disp).eval()
    return tok, model, disp, torch


@functools.lru_cache(maxsize=None)
def _emb_bert(texto):
    tok, model, disp, torch = _bert()
    enc = tok(texto, return_tensors="pt", truncation=True, max_length=32).to(disp)
    with torch.no_grad():
        vetor = model(**enc).last_hidden_state[0].mean(0)
    return torch.nn.functional.normalize(vetor, dim=0)


def sim_bert_cosseno(a, b):
    """Cosseno entre embeddings BERTimbau (palavra vs. frase)."""
    _, _, _, torch = _bert()
    return float(_emb_bert(a) @ _emb_bert(b)) * 100


@functools.lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.load("pt_core_news_md")


@functools.lru_cache(maxsize=None)
def _doc(texto):
    return _spacy()(texto)


def sim_spacy(a, b):
    da, db = _doc(a), _doc(b)
    return da.similarity(db) * 100 if da.vector_norm and db.vector_norm else 0.0


@functools.lru_cache(maxsize=1)
def _fasttext():
    import fasttext
    return fasttext.load_model("cc.pt.300.bin")


@functools.lru_cache(maxsize=None)
def _vec_fasttext(texto):
    import numpy as np
    modelo = _fasttext()
    vetores = [modelo.get_word_vector(t) for t in texto.split()] or [
        np.zeros(modelo.get_dimension())]
    media = np.mean(vetores, axis=0)
    norma = np.linalg.norm(media)
    return media / norma if norma else media


def sim_fasttext(a, b):
    import numpy as np
    return float(np.dot(_vec_fasttext(a), _vec_fasttext(b))) * 100


def escores(conjunto, sim):
    linhas = []
    for p in conjunto:
        ings = ingredientes(p)
        linha = {}
        for alergeno, formas in BASE_NORM.items():
            linha[alergeno] = max((sim(i, f) for i in ings for f in formas), default=0.0)
        linhas.append(linha)
    return linhas


def preditos(linhas, limiar):
    return [{a for a, s in linha.items() if s > limiar} for linha in linhas]


def avaliar_metrica(nome, sim, validacao, teste):
    esc_val = escores(validacao, sim)
    esc_teste = escores(teste, sim)
    golds_val = [p["gold"] for p in validacao]
    melhor_t, melhor_f1 = 0, -1.0
    for t in range(0, 100, 2):
        f1 = micro_f1_pares(list(zip(preditos(esc_val, t), golds_val)))
        if f1 > melhor_f1:
            melhor_f1, melhor_t = f1, t
    pares = list(zip(preditos(esc_teste, melhor_t), [p["gold"] for p in teste]))
    relatorio(pares, f"similaridade, {nome} (limiar {melhor_t}, ajustado na validação)", ic=True)


def main():
    _, validacao, teste = split()
    metricas = [
        ("Levenshtein", sim_levenshtein),
        ("Jaccard (char)", sim_jaccard),
        ("spaCy (vetores)", sim_spacy),
        ("fastText (cosseno)", sim_fasttext),
        ("BERTimbau (cosseno)", sim_bert_cosseno),
    ]
    for nome, sim in metricas:
        try:
            avaliar_metrica(nome, sim, validacao, teste)
        except (ImportError, OSError, ValueError) as erro:
            print(f"\n### similaridade, {nome}: pulado ({type(erro).__name__}: dependência/modelo ausente)")


if __name__ == "__main__":
    main()
