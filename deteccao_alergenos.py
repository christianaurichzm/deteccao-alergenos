import pandas as pd
from IPython.core.display_functions import display
from owlready2 import get_ontology, sync_reasoner, default_world
import re
from unidecode import unidecode


def load_data(file_path):
    return pd.read_csv(file_path, sep="\t", low_memory=False, usecols=["product_name_pt", "ingredients_text_pt"])


def extract_ingredients(text):
    pattern = r"\s*,\s*|\s+e\s+(?=[a-zA-Z])"
    return [ingredient.strip() for ingredient in re.split(pattern, text)]


def detect_allergens(ingredients, allergens_set):
    return [ingredient for ingredient in ingredients if any(allergen in ingredient for allergen in allergens_set)]


def preprocess_data(df):
    df.dropna(subset=['ingredients_text_pt'], inplace=True)
    df['ingredients_text_pt'] = df['ingredients_text_pt'].str.lower()
    df['ingredients_text_pt'] = df['ingredients_text_pt'].str.strip()
    df['ingredients_text_pt'] = df['ingredients_text_pt'].apply(unidecode)
    df['ingredients_text_pt'] = df['ingredients_text_pt'].replace({
        '[^a-zA-Záéíóúçãõôê\\s,]': '',
        ',+': ',',
        ' +': ' '
    }, regex=True)
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
    df = load_data('openfoodfacts_export.csv')
    df = preprocess_data(df)

    allergens_set = load_allergens_from_ontology("ontologia.owl")

    df['alergenos'] = df['ingredients_text_pt'].apply(
        lambda x: detect_allergens(extract_ingredients(x), allergens_set)
    )
    return df


if __name__ == '__main__':
    result_df = main()
    display(result_df)
