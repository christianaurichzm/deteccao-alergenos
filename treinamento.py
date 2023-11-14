import pandas as pd

from algorithms import Algorithms
from deteccao_alergenos import detect_allergens
from extracao import extract_ingredients
from ontology_operations import load_allergens_from_ontology, create_allergen_in_ontology, ontology, ONTOLOGY_PATH
from pre_processamento import clean_text, preprocess_data


def consultar_alergenos():
    allergens_set = load_allergens_from_ontology()
    cleaned_allergens_set = set()
    allergen_mapping = {}

    for allergen in allergens_set:
        cleaned = clean_text(allergen)
        cleaned_allergens_set.add(cleaned)
        allergen_mapping[cleaned] = allergen
    return cleaned_allergens_set, allergen_mapping


def detectar_alergenos_conjunto_treinamento(df, cleaned_allergens_set, allergen_mapping):
    extracted_ingredients = df['ingredients_text_pt'].progress_apply(extract_ingredients)
    detected_allergens_match = extracted_ingredients.progress_apply(
        lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, Algorithms.MATCH, 0)
    )
    return detected_allergens_match


def popular_ontologia(detected_allergens_match):
    added_derived_allergens = set()

    for allergen_pairs in detected_allergens_match:
        for derivated_allergen, base_allergen in allergen_pairs:
            if derivated_allergen != base_allergen and derivated_allergen not in added_derived_allergens and not ontology.search_one(
                    label=derivated_allergen.title()):
                create_allergen_in_ontology(derivated_allergen, base_allergen)
                added_derived_allergens.add(derivated_allergen)
    ontology.save(ONTOLOGY_PATH)


def load_and_preprocess_data():
    df = pd.read_csv('conjunto_treinamento.csv', sep="\t", low_memory=False,
                     usecols=["product_name_pt", "ingredients_text_pt"])
    df = preprocess_data(df)
    return df


def treinamento(df):
    cleaned_allergens_set, allergen_mapping = consultar_alergenos()

    detected_allergens_match = detectar_alergenos_conjunto_treinamento(df, cleaned_allergens_set, allergen_mapping)

    popular_ontologia(detected_allergens_match)

    return cleaned_allergens_set, allergen_mapping
