"""
Experimento de robustez a ruído de OCR/digitação (tarefa motivada pelos dados).

O dataset do Open Food Facts tem muito texto inserido por OCR; variantes reais
como "lete"/"glutem"/"trico" são frequentes. A hipótese: o casamento exato
(simbólico) quebra com 1 caractere trocado, enquanto o BERTimbau (tokenização
por subpalavra + contexto) degrada de forma mais suave.

Ruído de caractere controlado é injetado no texto do conjunto de teste e a queda
do micro-F1 de cada abordagem é medida em função da taxa de ruído.

Uso:
    python3 robustez_ocr.py
"""

import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from avaliacao_alergenos import (
    normalizar, contar_pares, prf1, split, preditos_por_rotulo,
)
from desambiguacao import detectar_preciso
from deteccao_ontologia import carregar_formas_da_ontologia

MODELO_NEURAL = "modelo_bert_alergenos"
SEED = 42
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFUSOES = {"i": "1", "l": "1", "o": "0", "e": "c", "a": "o",
             "s": "5", "c": "e", "t": "f", "g": "9", "n": "m", "u": "v"}


def ruido_ocr(texto, taxa, rng):
    """Corrompe cada letra com probabilidade `taxa` (troca visual ou remoção)."""
    saida = []
    for ch in texto:
        if ch.isalpha() and rng.random() < taxa:
            baixo = ch.lower()
            if baixo in CONFUSOES and rng.random() < 0.7:
                saida.append(CONFUSOES[baixo])
        else:
            saida.append(ch)
    return "".join(saida)


def micro_f1(pares):
    contagem = contar_pares(pares)
    tp = sum(v[0] for v in contagem.values())
    fp = sum(v[1] for v in contagem.values())
    fn = sum(v[2] for v in contagem.values())
    return prf1(tp, fp, fn)[2]


@torch.no_grad()
def prob_neural(textos, tokenizer, model):
    saidas = []
    for i in range(0, len(textos), 32):
        enc = tokenizer(textos[i:i + 32], truncation=True, padding=True,
                        max_length=192, return_tensors="pt").to(DISPOSITIVO)
        saidas.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.concatenate(saidas)


def main():
    torch.set_num_threads(8)
    _, _, teste = split()
    mapa = carregar_formas_da_ontologia()

    tokenizer = AutoTokenizer.from_pretrained(MODELO_NEURAL)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_NEURAL).to(DISPOSITIVO)
    model.eval()
    limiares = np.load(f"{MODELO_NEURAL}/limiares.npy")

    golds = [p["gold"] for p in teste]
    print(f"{'taxa ruído':>10s}  {'SIMBÓLICO F1':>13s}  {'NEURAL F1':>10s}")
    for taxa in [0.0, 0.05, 0.10, 0.15, 0.20]:
        rng = random.Random(SEED)
        crus = [ruido_ocr(p["cru"].replace("_", " "), taxa, rng) for p in teste]

        f1_sim = micro_f1([(detectar_preciso(normalizar(c), mapa), g)
                           for c, g in zip(crus, golds)])
        preds_neu = preditos_por_rotulo(prob_neural(crus, tokenizer, model), limiares)
        f1_neu = micro_f1(list(zip(preds_neu, golds)))
        print(f"{taxa:10.2f}  {f1_sim:13.3f}  {f1_neu:10.3f}", flush=True)


if __name__ == "__main__":
    main()
