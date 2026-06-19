"""
Avaliação da detecção de alérgenos contra o gabarito do Open Food Facts.

Carrega produtos do export do OFF, divide em treino/validação/teste e calcula
precisão, revocação e F1 (micro e macro) por alérgeno, com IC95% por bootstrap
e teste de significância pareado.

Uso:
    python3 avaliacao_alergenos.py
"""

import csv
import re
import random
import unicodedata
import collections

from conhecimento_alergenos import (
    ALERGENOS, TAG_PARA_ALERGENO, FORMAS, FORMAS_BASE, EXCLUSOES,
)

CSV_PATH = "openfoodfacts_export.csv"
SEED = 42
csv.field_size_limit(10 ** 7)
INDICE = {alergeno: i for i, alergeno in enumerate(ALERGENOS)}


def normalizar(texto):
    """Preserva , . ; como fronteiras de cláusula (escopo da negação)."""
    texto = unicodedata.normalize("NFKD", (texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s,.;]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def gold_do_produto(allergens_tags):
    return {
        TAG_PARA_ALERGENO[t.strip()]
        for t in allergens_tags.split(",")
        if t.strip() in TAG_PARA_ALERGENO
    }


def multirrotulo(gold):
    vetor = [0.0] * len(ALERGENOS)
    for alergeno in gold:
        vetor[INDICE[alergeno]] = 1.0
    return vetor


def carregar_produtos(path=CSV_PATH):
    """`cru` preserva casing para BERTimbau; `norm` é normalizado para detecção simbólica."""
    produtos = []
    with open(path, newline="") as fh:
        leitor = csv.DictReader(fh, delimiter="\t")
        for linha in leitor:
            cru = linha.get("ingredients_text_pt") or ""
            tags = (linha.get("allergens_tags") or "").strip()
            if not cru.strip() or not tags:
                continue
            gold = gold_do_produto(tags)
            if gold:
                produtos.append({"cru": cru, "norm": normalizar(cru),
                                 "gold": gold, "y": multirrotulo(gold)})
    return produtos


def dividir(produtos, seed=SEED):
    produtos = list(produtos)
    random.Random(seed).shuffle(produtos)
    n = len(produtos)
    n_teste = int(n * 0.15)
    n_val = int(n * 0.15)
    teste = produtos[:n_teste]
    validacao = produtos[n_teste:n_teste + n_val]
    treino = produtos[n_teste + n_val:]
    return treino, validacao, teste


def split(path=CSV_PATH):
    return dividir(carregar_produtos(path))


def detectar(texto_norm, mapa_formas):
    detectados = set()
    for alergeno, formas in mapa_formas.items():
        texto = texto_norm
        for trecho in EXCLUSOES.get(alergeno, []):
            texto = texto.replace(normalizar(trecho), " ")
        for forma in formas:
            if re.search(rf"\b{re.escape(normalizar(forma))}\b", texto):
                detectados.add(alergeno)
                break
    return detectados


def prf1(tp, fp, fn):
    """Retorna 0 quando o denominador é zero."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def contar_pares(pares):
    contagem = collections.defaultdict(lambda: [0, 0, 0])
    for pred, gold in pares:
        for alergeno in pred | gold:
            indice = 0 if alergeno in gold else 1
            if alergeno not in pred:
                indice = 2
            contagem[alergeno][indice] += 1
    return contagem


def relatorio(pares, titulo, ic=False):
    """ic=True adiciona IC95% por bootstrap."""
    contagem = contar_pares(pares)

    micro_p, micro_r, micro_f1 = prf1(
        sum(v[0] for v in contagem.values()),
        sum(v[1] for v in contagem.values()),
        sum(v[2] for v in contagem.values()),
    )

    por_alergeno = {
        alergeno: (*prf1(tp_a, fp_a, fn_a), tp_a, fp_a, fn_a)
        for alergeno, (tp_a, fp_a, fn_a) in contagem.items()
    }

    macro_p = sum(v[0] for v in por_alergeno.values()) / len(por_alergeno)
    macro_r = sum(v[1] for v in por_alergeno.values()) / len(por_alergeno)
    macro_f1 = sum(v[2] for v in por_alergeno.values()) / len(por_alergeno)

    print(f"\n### {titulo}")
    print(f"  micro   P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"  macro   P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")
    if ic:
        (mi_lo, mi_hi), (ma_lo, ma_hi) = ic_bootstrap(pares)
        print(f"  IC95%   micro-F1 [{mi_lo:.3f}, {mi_hi:.3f}]  "
              f"macro-F1 [{ma_lo:.3f}, {ma_hi:.3f}]")
    print(f"  {'alérgeno':12s} {'P':>5s} {'R':>5s} {'F1':>5s}   (tp/fp/fn)")
    for alergeno in sorted(por_alergeno, key=lambda a: -(por_alergeno[a][3] + por_alergeno[a][5])):
        p, r, f1, tp_a, fp_a, fn_a = por_alergeno[alergeno]
        print(f"  {alergeno:12s} {p:5.2f} {r:5.2f} {f1:5.2f}   ({tp_a}/{fp_a}/{fn_a})")
    return micro_f1, macro_f1


def micro_f1_pares(pares):
    c = contar_pares(pares)
    return prf1(sum(v[0] for v in c.values()),
                sum(v[1] for v in c.values()),
                sum(v[2] for v in c.values()))[2]


def macro_f1_pares(pares):
    c = contar_pares(pares)
    return sum(prf1(*v)[2] for v in c.values()) / len(c) if c else 0.0


def ic_bootstrap(pares, n=1000, seed=SEED):
    rng = random.Random(seed)
    N = len(pares)
    micros, macros = [], []
    for _ in range(n):
        amostra = [pares[rng.randrange(N)] for _ in range(N)]
        micros.append(micro_f1_pares(amostra))
        macros.append(macro_f1_pares(amostra))
    micros.sort()
    macros.sort()
    lo, hi = int(0.025 * n), int(0.975 * n)
    return (micros[lo], micros[hi]), (macros[lo], macros[hi])


def teste_pareado(pares_a, pares_b, n=1000, seed=SEED):
    rng = random.Random(seed)
    N = len(pares_a)
    difs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        difs.append(micro_f1_pares([pares_a[i] for i in idx])
                    - micro_f1_pares([pares_b[i] for i in idx]))
    media = sum(difs) / n
    p = 2 * min(sum(d <= 0 for d in difs), sum(d >= 0 for d in difs)) / n
    return media, p


def limiares_por_rotulo(probs_val, conjunto_val):
    """Limiar por rótulo (não global) melhora o macro-F1 em classes raras."""
    import numpy as np
    Y = np.array([p["y"] for p in conjunto_val])
    limiares = np.full(len(ALERGENOS), 0.5)
    for j in range(len(ALERGENOS)):
        melhor, melhor_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.05):
            pred = probs_val[:, j] >= t
            tp = int((pred & (Y[:, j] == 1)).sum())
            fp = int((pred & (Y[:, j] == 0)).sum())
            fn = int((~pred & (Y[:, j] == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > melhor_f1:
                melhor, melhor_f1 = t, f1
        limiares[j] = melhor
    return limiares


def preditos_por_rotulo(probs, limiares):
    return [{ALERGENOS[j] for j in range(len(ALERGENOS)) if linha[j] >= limiares[j]}
            for linha in probs]


def avaliar(conjunto, mapa_formas, titulo, fn_detectar=detectar):
    pares = [(fn_detectar(p["norm"], mapa_formas), p["gold"]) for p in conjunto]
    return relatorio(pares, titulo)


def main():
    treino, validacao, teste = split()
    print(f"produtos avaliáveis: {len(treino) + len(validacao) + len(teste)} "
          f"(treino={len(treino)}, validação={len(validacao)}, teste={len(teste)})")
    avaliar(teste, FORMAS_BASE, "BASE (substring da palavra-base ~ abordagem atual)")
    avaliar(teste, FORMAS, "ONTOLOGIA (derivações ancoradas na RDC 26/2015)")


if __name__ == "__main__":
    main()
