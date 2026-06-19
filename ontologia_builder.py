"""
Gera a ontologia OWL de alérgenos a partir da base de conhecimento (idempotente).

Uso:
    python3 ontologia_builder.py
"""

import re
import unicodedata

from owlready2 import get_ontology, Thing, ObjectProperty

from conhecimento_alergenos import ALERGENOS, FORMAS

IRI = "http://www.ufsc.br/food#"
SAIDA = "ontologia_alergenos.owl"


def nome_individuo(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return "".join(p.capitalize() for p in re.split(r"[^a-z0-9]+", texto) if p)


def construir():
    onto = get_ontology(IRI)

    with onto:
        class Alimento(Thing):
            pass

        class Alergeno(Alimento):
            pass

        class eDerivadoDe(ObjectProperty):
            domain = [Alimento]
            range = [Alimento]

        class AlergenoPorDerivacao(Alimento):
            equivalent_to = [Alimento & eDerivadoDe.some(Alergeno)]

    Alergeno.label = ["Alérgeno"]
    Alimento.label = ["Alimento"]
    AlergenoPorDerivacao.label = ["Alérgeno por Derivação"]
    eDerivadoDe.label = ["é derivado de"]

    bases = {}
    for canon in ALERGENOS:
        base = onto.Alergeno(nome_individuo(canon))
        base.label = [canon]
        bases[canon] = base

    usados = {nome_individuo(c) for c in ALERGENOS}
    derivados = 0
    for canon, formas in FORMAS.items():
        for forma in formas:
            if forma == canon:
                continue
            nome = f"{nome_individuo(canon)}_{nome_individuo(forma)}"
            if nome in usados:
                continue
            usados.add(nome)
            derivado = onto.Alimento(nome)
            derivado.label = [forma]
            derivado.eDerivadoDe.append(bases[canon])
            derivados += 1

    onto.save(file=SAIDA, format="rdfxml")
    print(f"ontologia gerada: {SAIDA}")
    print(f"  alérgenos-base: {len(bases)} | formas derivadas: {derivados}")
    return onto


def demonstrar_raciocinio(caminho=SAIDA):
    """Nenhum indivíduo é asserido como AlergenoPorDerivacao; o HermiT os infere."""
    from owlready2 import get_ontology, sync_reasoner

    onto = get_ontology(caminho).load()
    apod = onto.search_one(iri="*AlergenoPorDerivacao")
    antes = len(list(apod.instances()))
    with onto:
        sync_reasoner()
    inferidos = list(apod.instances())

    print("\nRaciocinador HermiT (OWL DL):")
    print(f"  indivíduos asserido(s) como AlergenoPorDerivacao antes: {antes}")
    print(f"  indivíduos inferidos como AlergenoPorDerivacao:        {len(inferidos)}")
    for ind in sorted(inferidos, key=lambda x: x.name)[:8]:
        base = ind.eDerivadoDe[0].label.first() if ind.eDerivadoDe else "?"
        print(f"    {ind.label.first():24s} ⊑ AlergenoPorDerivacao  (é derivado de: {base})")


if __name__ == "__main__":
    import sys
    construir()
    if "raciocinar" in sys.argv:
        demonstrar_raciocinio()
