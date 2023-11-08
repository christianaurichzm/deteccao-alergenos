from IPython.core.display_functions import display

from data_preparation import preparar_dados
from teste import teste, load_and_preprocess_sample
from treinamento import treinamento, load_and_preprocess_data


def main():
    preparar_dados()
    df_treinamento = load_and_preprocess_data()
    cleaned_allergens_set, allergen_mapping = treinamento(df_treinamento)
    df_teste, extracted_ingredients_amostra = load_and_preprocess_sample()
    teste(extracted_ingredients_amostra, cleaned_allergens_set, allergen_mapping, df_teste)

    return df_teste


if __name__ == '__main__':
    result_df = main()
    display(result_df)
