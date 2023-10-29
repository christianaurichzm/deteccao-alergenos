import ast

import pandas as pd

from algorithms import Algorithms
from cache import cache
from deteccao_alergenos import detect_allergens
from evalutation_metrics import evaluate_algorithm
from extracao import extract_ingredients
from plot import plot_confusion_matrix, plot_metrics
from pre_processamento import preprocess_data, clean_text
from similarity_metrics import model_fasttext


def find_best_threshold(algorithm, extracted_ingredients_amostra, cleaned_allergens_set, allergen_mapping, df_amostra):
    best_threshold = 0
    best_f1 = 0
    for threshold in range(0, 101):
        detected_allergens = extracted_ingredients_amostra.progress_apply(
            lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm, threshold)
        )
        df_amostra[f'alergenos_{algorithm.value}'] = detected_allergens.apply(lambda x: [detected[0] for detected in x])
        metrics = evaluate_algorithm(df_amostra, algorithm)
        avg_f1 = metrics.f1
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold
    return best_threshold


def apply_best_threshold(algorithm, best_threshold, extracted_ingredients_amostra, cleaned_allergens_set,
                         allergen_mapping, df_amostra):
    detected_allergens = extracted_ingredients_amostra.progress_apply(
        lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm, best_threshold)
    )
    df_amostra[f'alergenos_{algorithm.value}'] = detected_allergens.apply(lambda x: [detected[0] for detected in x])


def plotar_graficos(best_confusion_mat, algorithm, best_avg_accuracy, best_avg_precisions, best_avg_recalls,
                    best_avg_f1):
    plot_confusion_matrix(best_confusion_mat, algorithm)
    plot_metrics(best_avg_accuracy, best_avg_precisions, best_avg_recalls, best_avg_f1, algorithm.value)


def evaluate_and_visualize(algorithm, df_amostra):
    best_confusion_mat, best_avg_accuracy, best_avg_precisions, best_avg_recalls, best_avg_f1 = evaluate_algorithm(
        df_amostra, algorithm)
    plotar_graficos(best_confusion_mat, algorithm, best_avg_accuracy, best_avg_precisions, best_avg_recalls,
                    best_avg_f1)


def load_and_preprocess_sample():
    df_amostra = pd.read_csv('conjunto_teste.csv', sep="\t", low_memory=False,
                             usecols=["code", "product_name_pt", "ingredients_text_pt", "gabarito"])

    df_amostra = preprocess_data(df_amostra)

    extracted_ingredients_amostra = df_amostra['ingredients_text_pt'].progress_apply(extract_ingredients)

    df_amostra['gabarito'] = df_amostra['gabarito'].apply(ast.literal_eval).apply(lambda x: [clean_text(i) for i in x])
    return df_amostra, extracted_ingredients_amostra


def teste(extracted_ingredients_amostra, cleaned_allergens_set, allergen_mapping, df_amostra):
    for algorithm in Algorithms:
        if algorithm == Algorithms.FASTTEXT and model_fasttext is None:
            continue

        best_threshold = find_best_threshold(algorithm, extracted_ingredients_amostra, cleaned_allergens_set,
                                             allergen_mapping, df_amostra)
        apply_best_threshold(algorithm, best_threshold, extracted_ingredients_amostra, cleaned_allergens_set,
                             allergen_mapping, df_amostra)
        evaluate_and_visualize(algorithm, df_amostra)

        cache.clear()
