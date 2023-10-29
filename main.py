from IPython.core.display_functions import display
from tqdm import tqdm

from gerador_conjuntos import gerador_conjuntos
from ontology_operations import ontology, ONTOLOGY_PATH
from teste import teste, load_and_preprocess_sample
from treinamento import treinamento, load_and_preprocess_data


def preparar_dados():
    gerador_conjuntos()
    tqdm.pandas()
    return load_and_preprocess_data()


def main():
    df = preparar_dados()
    cleaned_allergens_set, allergen_mapping = treinamento(df)
    df_amostra, extracted_ingredients_amostra = load_and_preprocess_sample()
    teste(extracted_ingredients_amostra, cleaned_allergens_set, allergen_mapping, df_amostra)

    return df_amostra


if __name__ == '__main__':
    result_df = main()
    display(result_df)
