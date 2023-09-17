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
    inputs = tokenizerBert(sentences, return_tensors="pt",
                           padding=True, truncation=True, max_length=128)
    outputs = modelBert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


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


def detect_allergens(ingredients, allergens_set, algorithm):
    detected_allergens = [ingredient for ingredient in ingredients
                          if any(is_allergen_present(ingredient, allergen, algorithm) for allergen in allergens_set)]
    return detected_allergens


def preprocess_data(df):
    df.dropna(subset=['ingredients_text_pt'], inplace=True)
    df['ingredients_text_pt'] = (df['ingredients_text_pt'].apply(unidecode)
                                 .str.lower()
                                 .str.strip()
                                 .replace({
                                     '[^a-zA-Z\\s,]': '',
                                     ',+': ',',
                                     ' +': ' '
                                 }, regex=True))
    return df


def load_allergens_from_ontology(ontology_path, query_path="query_alergenos.sparql"):
    get_ontology(ontology_path).load()
    sync_reasoner()
    with open(query_path, "r") as file:
        query = file.read()
    results = list(default_world.sparql(query))
    allergens = set(unidecode(label.lower()) for [label] in results)
    return allergens


def main():
    tqdm.pandas()
    df = load_data('openfoodfacts_export.csv')
    df = preprocess_data(df)

    allergens_set = load_allergens_from_ontology("ontologia.owl")

    for algorithm in Algorithms:
        df[f'alergenos_{algorithm.value}'] = df['ingredients_text_pt'].progress_apply(
            lambda x: detect_allergens(extract_ingredients(
                x), allergens_set, algorithm)
        )

    return df


if __name__ == '__main__':
    result_df = main()
    display(result_df)
