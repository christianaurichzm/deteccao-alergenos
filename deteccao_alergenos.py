import re
from enum import Enum
import ast

import cachetools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from owlready2 import get_ontology, sync_reasoner, default_world
from sklearn.metrics import ConfusionMatrixDisplay
from thefuzz import fuzz
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from unidecode import unidecode

cache = cachetools.LFUCache(maxsize=1000)


class Algorithms(Enum):
    MATCH = "match"
    LEVENSHTEIN = "levenshtein"
    JACCARD = "jaccard"
    BERT = "bert"


bert_arch = "neuralmind/bert-large-portuguese-cased"
tokenizer_bert = BertTokenizer.from_pretrained(bert_arch)
model_bert = BertModel.from_pretrained(bert_arch)


def extract_ingredients(text):
    pattern = r"\s*,\s*|\s+e\s+(?=[a-zA-Z])"
    return [ingredient.strip() for ingredient in re.split(pattern, text)]


def levenshtein_distance(ingredient, allergen):
    return fuzz.ratio(ingredient, allergen)

def jaccard_similarity(ingredient, allergen):
    ingredient_set = set(ingredient)
    allergen_set = set(allergen)
    intersection = ingredient_set.intersection(allergen_set)
    union = ingredient_set.union(allergen_set)
    return (len(intersection) / len(union)) * 100


def sentences_to_embeddings_bert(sentences):
    hash_key = hash(sentences)

    if hash_key in cache:
        return cache[hash_key]

    inputs = tokenizer_bert(sentences, return_tensors="pt",
                            padding=True, truncation=True, max_length=128)
    outputs = model_bert(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)

    cache[hash_key] = embedding

    return embedding


def bert_similarity(ingredient, allergen):
    ingredient_embedding = sentences_to_embeddings_bert(ingredient)
    allergen_embedding = sentences_to_embeddings_bert(allergen)

    return cosine_similarity(ingredient_embedding, allergen_embedding) * 100


def calculate_metrics(predicted_list, true_list, all_ingredients_list):
    TP = sum([1 for item in predicted_list if item in true_list])
    FP = sum([1 for item in predicted_list if item not in true_list])
    FN = sum([1 for item in true_list if item not in predicted_list])
    non_allergens = [item for item in all_ingredients_list if item not in true_list]
    TN = sum([1 for item in non_allergens if item not in predicted_list])

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return TP, FP, TN, FN, accuracy, precision, recall, f1


def plot_confusion_matrix(matrix, algorithm):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap='Purples', values_format='.2f')

    plt.title(f'Matriz de confusão para o algoritmo: {algorithm}')
    plt.tight_layout()
    plt.show()


def plot_metrics(avg_accuracy, avg_precisions, avg_recalls, avg_f1, algorithm):
    metrics = ['Acurácia', 'Precisão', 'Revocação', 'Score F1']
    values = [avg_accuracy, avg_precisions, avg_recalls, avg_f1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#003f5c', '#7a5195', '#ef5675', '#ffa600'])

    plt.ylabel('Pontuação')
    plt.title(f'Métricas para o algoritmo: {algorithm}')
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

    plt.show()


def evaluate_algorithm(df, algorithm):
    predicted_list = df[f'alergenos_{algorithm}'].tolist()
    true_list = df['gabarito'].tolist()
    all_ingredients_list = df['ingredients_text_pt'].apply(extract_ingredients).tolist()

    all_TPs = []
    all_FPs = []
    all_TNs = []
    all_FNs = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for predicted, true, all_ingredients in zip(predicted_list, true_list, all_ingredients_list):
        TP, FP, TN, FN, accuracy, precision, recall, f1 = calculate_metrics(predicted, true, all_ingredients)
        all_TPs.append(TP)
        all_FPs.append(FP)
        all_TNs.append(TN)
        all_FNs.append(FN)
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    avg_TP = np.mean(all_TPs)
    avg_FP = np.mean(all_FPs)
    avg_TN = np.mean(all_TNs)
    avg_FN = np.mean(all_FNs)
    avg_accuracy = np.mean(all_accuracies)
    avg_precisions = np.mean(all_precisions)
    avg_recalls = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    confusion_mat = np.array([[avg_TP, avg_FN], [avg_FP, avg_TN]])

    return confusion_mat, avg_accuracy, avg_precisions, avg_recalls, avg_f1


def is_allergen_present(ingredient, allergen, algorithm, threshold):
    match algorithm:
        case Algorithms.MATCH:
            return allergen in ingredient
        case Algorithms.LEVENSHTEIN:
            return levenshtein_distance(ingredient, allergen) > threshold
        case Algorithms.BERT:
            return bert_similarity(ingredient, allergen) > threshold
        case Algorithms.JACCARD:
            return jaccard_similarity(ingredient, allergen) > threshold


def detect_allergens(ingredients, cleaned_allergens_set, allergen_mapping, algorithm, threshold):
    detected_allergens = []
    for ingredient in ingredients:
        for cleaned_allergen in cleaned_allergens_set:
            if is_allergen_present(ingredient, cleaned_allergen, algorithm, threshold):
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
    df = pd.read_csv('openfoodfacts_export.csv', sep="\t", low_memory=False,
                     usecols=["product_name_pt", "ingredients_text_pt"])
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
        lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, Algorithms.MATCH, 0)
    )

    added_derived_allergens = set()

    for allergen_pairs in detected_allergens_match:
        for derivated_allergen, base_allergen in allergen_pairs:
            if derivated_allergen != base_allergen and derivated_allergen not in added_derived_allergens and not ontology.search_one(
                    label=derivated_allergen.title()):
                create_allergen_in_ontology(ontology, ONTOLOGY_PATH, derivated_allergen, base_allergen)
                added_derived_allergens.add(derivated_allergen)

    df_amostra = pd.read_csv('conjunto_teste.csv', sep="\t", low_memory=False,
                             usecols=["code", "product_name_pt", "ingredients_text_pt", "gabarito"])

    df_amostra = preprocess_data(df_amostra)

    extracted_ingredients_amostra = df_amostra['ingredients_text_pt'].progress_apply(extract_ingredients)

    df_amostra['gabarito'] = df_amostra['gabarito'].apply(ast.literal_eval)
    df_amostra['gabarito'] = df_amostra['gabarito'].apply(lambda x: [clean_text(i) for i in x])

    for algorithm in Algorithms:
        best_threshold = 0
        best_f1 = 0
        for threshold in range(0, 101):
            detected_allergens = extracted_ingredients_amostra.progress_apply(
                lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm, threshold)
            )
            df_amostra[f'alergenos_{algorithm.value}'] = detected_allergens.apply(
                lambda x: [detected[0] for detected in x])
            metrics = evaluate_algorithm(df_amostra, algorithm.value)
            avg_f1 = metrics[-1]
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold

        detected_allergens = extracted_ingredients_amostra.progress_apply(
            lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm, best_threshold)
        )
        df_amostra[f'alergenos_{algorithm.value}'] = detected_allergens.apply(lambda x: [detected[0] for detected in x])
        best_confusion_mat, best_avg_accuracy, best_avg_precisions, best_avg_recalls, best_avg_f1 = evaluate_algorithm(
            df_amostra, algorithm.value)
        plot_confusion_matrix(best_confusion_mat, algorithm.value)
        plot_metrics(best_avg_accuracy, best_avg_precisions, best_avg_recalls, best_avg_f1, algorithm.value)

    return df_amostra


if __name__ == '__main__':
    result_df = main()
    display(result_df)
