"""
Desambiguação contextual de sentido com BERTimbau (WSD).

O sentido de cada termo polissêmico é decidido pelo contexto: o embedding
contextual do token é comparado, por cosseno, a protótipos positivos e negativos.

    "leite de coco"        -> mais próximo de {leite de soja, ...} -> não é leite
    "manteiga de cacau"    -> mais próximo de {manteiga vegetal}   -> não é leite
    "lecitina de girassol" -> não é soja  |  "noz moscada" -> não é castanha

Reusa a detecção simbólica (precaução, negação) e substitui apenas a decisão
de ambiguidade pelo embedding contextual.
"""

import os
import re
import functools

import torch
from transformers import AutoTokenizer, AutoModel

from avaliacao_alergenos import normalizar
from desambiguacao import secao_contem, esta_negado

MODELO = os.environ.get("MODELO_WSD", "neuralmind/bert-base-portuguese-cased")
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMBIGUOS = {
    "leite": "leite",
    "manteiga": "leite",
    "lecitina": "soja",
    "noz": "castanhas",
    "polvo": "moluscos",
}

PROTOTIPOS = {
    "leite": (["leite integral", "leite de vaca", "leite em po", "creme de leite"],
              ["leite de coco", "leite de soja", "leite de amendoa", "leite de aveia"]),
    "manteiga": (["manteiga", "manteiga de leite", "manteiga sem sal"],
                 ["manteiga de cacau", "manteiga de amendoim", "manteiga vegetal",
                  "manteiga de karite"]),
    "lecitina": (["lecitina de soja", "lecitina de soja emulsificante"],
                 ["lecitina de girassol", "lecitina de canola"]),
    "noz": (["nozes", "creme de nozes", "noz pecan", "miolo de noz"],
            ["noz moscada", "noz-moscada ralada"]),
    "polvo": (["polvo grelhado", "salada de polvo", "polvo a lagareiro"],
              ["acucar em polvo", "leite em polvo", "fermento em polvo"]),
}


@functools.lru_cache(maxsize=1)
def _modelo():
    tok = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModel.from_pretrained(MODELO).to(DISPOSITIVO)
    model.eval()
    return tok, model


@torch.no_grad()
def embedding_frase(frase):
    """A discriminação vem do modificador ("de coco", "moscada"), não do token isolado."""
    tok, model = _modelo()
    enc = tok(frase, return_offsets_mapping=True, return_tensors="pt",
              truncation=True, max_length=32)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DISPOSITIVO) for k, v in enc.items()}
    estados = model(**enc).last_hidden_state[0]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a]
    vetor = estados[indices].mean(0) if indices else estados.mean(0)
    return torch.nn.functional.normalize(vetor, dim=0)


def frase_candidata(contexto, termo, antes=1, seguintes=2):
    """Janela ao redor do termo; `antes=1` captura modificadores precedentes ("en polvo")."""
    palavras = contexto.split()
    idx = next((i for i, w in enumerate(palavras) if termo in w), None)
    if idx is None:
        return termo
    return " ".join(palavras[max(0, idx - antes): idx + seguintes + 1])


@functools.lru_cache(maxsize=None)
def _prototipos(termo):
    pos, neg = PROTOTIPOS[termo]
    vp = torch.stack([embedding_frase(c) for c in pos]).mean(0)
    vn = torch.stack([embedding_frase(c) for c in neg]).mean(0)
    return torch.nn.functional.normalize(vp, dim=0), torch.nn.functional.normalize(vn, dim=0)


def e_o_alergeno(termo, contexto):
    """O termo, neste contexto, indica o alérgeno? (mais perto do protótipo +)."""
    emb = embedding_frase(frase_candidata(contexto, termo))
    vp, vn = _prototipos(termo)
    return float(emb @ vp) >= float(emb @ vn)


def _janela(texto, termo, largura=40):
    pos = texto.find(termo)
    if pos < 0:
        return texto
    return texto[max(0, pos - largura): pos + len(termo) + largura]


def _forma_confirma(forma, alergeno, texto, raw_lc):
    """A forma aparece, não negada, e (se ambígua) confirmada pelo BERT no contexto?"""
    fn = normalizar(forma)
    for m in re.finditer(rf"\b{re.escape(fn)}\b", texto):
        if esta_negado(texto, m.start()):
            continue
        if AMBIGUOS.get(fn) == alergeno and not e_o_alergeno(fn, _janela(raw_lc, fn)):
            continue
        return True
    return False


def detectar_hibrido(produto, mapa):
    """Detecção simbólica com a ambiguidade resolvida pelo BERT (não por regras)."""
    texto = secao_contem(produto["norm"])
    raw_lc = produto["cru"].replace("_", " ").lower()
    detectados = set()
    for alergeno, formas in mapa.items():
        if any(_forma_confirma(forma, alergeno, texto, raw_lc) for forma in formas):
            detectados.add(alergeno)
    return detectados


if __name__ == "__main__":
    casos = [
        ("leite", "biscoito com leite integral e acucar"),
        ("leite", "bebida vegetal de leite de coco e agua"),
        ("manteiga", "massa folhada com manteiga e sal"),
        ("manteiga", "recheio com manteiga de cacau e avela"),
        ("lecitina", "chocolate com lecitina de soja"),
        ("lecitina", "chocolate com lecitina de girassol"),
        ("noz", "creme de nozes torradas"),
        ("noz", "molho com noz moscada ralada"),
    ]
    for termo, ctx in casos:
        print(f"  {'É ALÉRGENO' if e_o_alergeno(termo, ctx) else 'não é     '} | {termo:9s} | {ctx}")
