"""
Detecção de alérgenos consultando a ontologia OWL via SPARQL.

Uso:
    python3 deteccao_ontologia.py
"""

import collections

from owlready2 import get_ontology, default_world

from avaliacao_alergenos import normalizar, detectar

ONTOLOGIA = "ontologia_alergenos.owl"

CONSULTA_FORMAS = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX food: <http://www.ufsc.br/food#>
SELECT ?forma ?base WHERE {
    {
        ?ind a food:Alergeno .
        ?ind rdfs:label ?forma .
        BIND(?forma AS ?base)
    }
    UNION
    {
        ?ind food:eDerivadoDe ?b .
        ?ind rdfs:label ?forma .
        ?b rdfs:label ?base .
    }
}
"""


def carregar_formas_da_ontologia(caminho=ONTOLOGIA):
    get_ontology(caminho).load()
    mapa = collections.defaultdict(list)
    for forma, base in default_world.sparql(CONSULTA_FORMAS):
        mapa[str(base)].append(str(forma))
    return dict(mapa)


def detectar_via_ontologia(texto, mapa=None):
    if mapa is None:
        mapa = carregar_formas_da_ontologia()
    return detectar(normalizar(texto), mapa)


if __name__ == "__main__":
    mapa = carregar_formas_da_ontologia()
    print(f"alérgenos-base na ontologia: {len(mapa)}")
    print(f"total de formas carregadas: {sum(len(v) for v in mapa.values())}")
    exemplos = [
        "Leite integral, açúcar, LACTOSE, lecitina de soja",
        "Farinha de trigo enriquecida, óleo de soja, sal",
        "Água, leite de coco, açúcar",
        "Castanha de caju torrada e sal",
    ]
    for texto in exemplos:
        print(f"\n  {texto!r}\n   -> {sorted(detectar_via_ontologia(texto, mapa))}")
