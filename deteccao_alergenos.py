import re
from enum import Enum

import pandas as pd
from IPython.core.display_functions import display
from owlready2 import get_ontology, sync_reasoner, default_world
from thefuzz import fuzz
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from unidecode import unidecode
from tqdm import tqdm
import cachetools

cache = cachetools.LFUCache(maxsize=1000)


class Algorithms(Enum):
    MATCH = "match"
    LEVENSHTEIN = "levenshtein"
    BERT = "bert"


tokenizerBert = BertTokenizer.from_pretrained(
    "neuralmind/bert-large-portuguese-cased")
modelBert = BertModel.from_pretrained("neuralmind/bert-large-portuguese-cased")


def load_data(file_path):
    return pd.read_csv(file_path, sep="\t", low_memory=False, usecols=["product_name_pt", "ingredients_text_pt"])


def extract_ingredients(text):
    pattern = r"\s*,\s*|\s+e\s+(?=[a-zA-Z])"
    return [ingredient.strip() for ingredient in re.split(pattern, text)]


def levenshtein_distance(ingredient, allergen):
    return fuzz.ratio(ingredient, allergen)


def sentences_to_embeddings_bert(sentences):
    hash_key = hash(sentences)

    if hash_key in cache:
        return cache[hash_key]

    inputs = tokenizerBert(sentences, return_tensors="pt",
                           padding=True, truncation=True, max_length=128)
    outputs = modelBert(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)

    cache[hash_key] = embedding

    return embedding


def bert_similarity(ingredient, allergen):
    ingredient_embedding = sentences_to_embeddings_bert(ingredient)
    allergen_embedding = sentences_to_embeddings_bert(allergen)

    return cosine_similarity(ingredient_embedding, allergen_embedding)


def is_allergen_present(ingredient, allergen, algorithm):
    match algorithm:
        case Algorithms.MATCH:
            return allergen in ingredient
        case Algorithms.LEVENSHTEIN:
            return levenshtein_distance(ingredient, allergen) > 80
        case Algorithms.BERT:
            return bert_similarity(ingredient, allergen) > 0.92


def detect_allergens(ingredients, cleaned_allergens_set, allergen_mapping, algorithm):
    detected_allergens = []
    for ingredient in ingredients:
        for cleaned_allergen in cleaned_allergens_set:
            if is_allergen_present(ingredient, cleaned_allergen, algorithm):
                original_allergen = allergen_mapping.get(cleaned_allergen, cleaned_allergen)
                detected_allergens.append((ingredient, original_allergen))
                break
    return detected_allergens


def clean_text(text):
    cleaned = unidecode(text.lower().strip())
    cleaned = re.sub(r'[^a-zA-Z\s,]', ' ', cleaned)
    cleaned = re.sub(r',+', ',', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned


def preprocess_data(df):
    df.dropna(subset=['ingredients_text_pt'], inplace=True)
    df['ingredients_text_pt'] = df['ingredients_text_pt'].apply(clean_text)
    return df


def load_allergens_from_ontology(query_path):
    with open(query_path, "r") as file:
        query = file.read()
    results = list(default_world.sparql(query))
    allergens = set(label for [label] in results)
    return allergens


def create_allergen_in_ontology(ontology, ontology_path, derived_name, allergen_base):
    allergen_base_instance = ontology.search_one(label=allergen_base)
    Alimento = ontology.search_one(label="Alimento")

    new_allergen = Alimento(derived_name.title().replace(" ", ""))
    new_allergen.label = [derived_name.title()]
    new_allergen.eDerivadoDe.append(allergen_base_instance)

    ontology.save(ontology_path)


def main():
    tqdm.pandas()
    df = load_data('openfoodfacts_export.csv')
    df = preprocess_data(df)

    ONTOLOGY_PATH = "ontologia.owl"

    ontology = get_ontology(ONTOLOGY_PATH).load()
    sync_reasoner()

    allergens_set = load_allergens_from_ontology("query_alergenos.sparql")
    cleaned_allergens_set = set()
    allergen_mapping = {}

    for allergen in allergens_set:
        cleaned = clean_text(allergen)
        cleaned_allergens_set.add(cleaned)
        allergen_mapping[cleaned] = allergen

    extracted_ingredients = df['ingredients_text_pt'].progress_apply(extract_ingredients)

    detected_allergens_match = extracted_ingredients.progress_apply(
        lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, Algorithms.MATCH)
    )

    added_derived_allergens = set()

    for allergen_pairs in detected_allergens_match:
        for derivated_allergen, base_allergen in allergen_pairs:
            if derivated_allergen != base_allergen and derivated_allergen not in added_derived_allergens and not ontology.search_one(
                    label=derivated_allergen.title()):
                create_allergen_in_ontology(ontology, ONTOLOGY_PATH, derivated_allergen, base_allergen)
                added_derived_allergens.add(derivated_allergen)

    for algorithm in Algorithms:
        detected_allergens = extracted_ingredients.progress_apply(
            lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm)
        )
        df[f'alergenos_{algorithm.value}'] = detected_allergens.apply(lambda x: [detected[0] for detected in x])

    return df


if __name__ == '__main__':
    result_df = main()
    display(result_df)
