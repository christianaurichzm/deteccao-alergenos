from owlready2 import default_world, get_ontology, sync_reasoner

ONTOLOGY_PATH = "ontologia.owl"
LOAD_ALLERGENS_QUERY_PATH = "query_alergenos.sparql"

ontology = get_ontology(ONTOLOGY_PATH).load()
sync_reasoner()


def load_allergens_from_ontology():
    with open(LOAD_ALLERGENS_QUERY_PATH, "r") as file:
        query = file.read()
    results = list(default_world.sparql(query))
    allergens = set(label for [label] in results)
    return allergens


def create_allergen_in_ontology(derived_name, allergen_base):
    allergen_base_instance = ontology.search_one(label=allergen_base)
    Alimento = ontology.search_one(label="Alimento")

    new_allergen = Alimento(derived_name.title().replace(" ", ""))
    new_allergen.label = [derived_name.title()]
    new_allergen.eDerivadoDe.append(allergen_base_instance)
