"""
Regras linguísticas de precisão sobre o texto normalizado.

  1. Precaução vs. presença: "pode conter (traços de) X" / "fabricado em
     equipamento que processa X" declaram contaminação cruzada, não ingrediente.
  2. Negação com escopo: "sem glúten", "não contém leite", "zero lactose",
     "isento de soja" invertem o sentido do termo (estilo NegEx).
  3. Homógrafos e contaminação de idioma via EXCLUSOES
     (ex.: "en polvo" = pó em espanhol, não o molusco "polvo").
"""

import re

from avaliacao_alergenos import normalizar
from conhecimento_alergenos import EXCLUSOES

PRECAUCAO = re.compile(
    r"\b(pode(m)? conter|tra[cz]os de|fabricado (em|na|no|num|numa)|"
    r"produzido (em|na|no)|processa(do|dos|mento)?|"
    r"contem (ou |e )?pode|alergicos:? pode)\b"
)

NEGACAO = re.compile(
    r"\b(sem|nao contem|nao possui|nao contendo|isento[a]? de|"
    r"isento[a]? em|livre de|zero|nao adicao de|sem adicao de)\b"
)


def secao_contem(texto):
    m = PRECAUCAO.search(texto)
    return texto[:m.start()] if m else texto


def esta_negado(texto, inicio):
    """Há marcador de negação entre a fronteira de cláusula anterior e o termo?"""
    fronteira = max(texto.rfind(",", 0, inicio),
                    texto.rfind(".", 0, inicio),
                    texto.rfind(";", 0, inicio)) + 1
    return bool(NEGACAO.search(texto[fronteira:inicio]))


def detectar_preciso(texto_norm, mapa):
    texto = secao_contem(texto_norm)
    detectados = set()
    for alergeno, formas in mapa.items():
        alvo = texto
        for trecho in EXCLUSOES.get(alergeno, []):
            alvo = alvo.replace(normalizar(trecho), " ")
        for forma in formas:
            achou = False
            for m in re.finditer(rf"\b{re.escape(normalizar(forma))}\b", alvo):
                if not esta_negado(alvo, m.start()):
                    achou = True
                    break
            if achou:
                detectados.add(alergeno)
                break
    return detectados


if __name__ == "__main__":
    from deteccao_ontologia import carregar_formas_da_ontologia
    mapa = carregar_formas_da_ontologia()
    testes = [
        "biscoito sem gluten, acucar, oleo de soja",
        "chocolate, acucar. pode conter tracos de leite e amendoim",
        "farinha de trigo, leite em po, ovo",
        "harina de trigo, azucar en polvo, sal",
        "creme de avela, acucar, noz moscada",
    ]
    for t in testes:
        print(f"  {t!r}\n   -> {sorted(detectar_preciso(normalizar(t), mapa))}\n")
