import re
from collections import namedtuple
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
from torch.nn.functional import cosine_similarity as cosine_similarity_torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cossine_similarity_sklearn
from transformers import BertTokenizer, BertModel
from unidecode import unidecode
import spacy
import fasttext.util

cache = cachetools.LFUCache(maxsize=1000)

metrics = ['accuracy', 'precision', 'recall', 'f1']
PerformanceIndicators = namedtuple('PerformanceIndicators', ['TP', 'FP', 'TN', 'FN', *metrics])
EvaluationResult = namedtuple('EvaluationResult', ['confusion_mat', *metrics])


class Algorithms(Enum):
    MATCH = "match"
    LEVENSHTEIN = "levenshtein"
    JACCARD = "jaccard"
    SPACY = "spacy"
    FASTTEXT = "fasttext"
    BERT = "bert"


model_spacy = spacy.load('pt_core_news_md')

FASTTEXT_LANG = "pt"
fasttext.util.download_model(FASTTEXT_LANG, if_exists='ignore')
model_fasttext = fasttext.load_model(f'cc.{FASTTEXT_LANG}.300.bin')

BERT_ARCH = "neuralmind/bert-large-portuguese-cased"
tokenizer_bert = BertTokenizer.from_pretrained(BERT_ARCH)
model_bert = BertModel.from_pretrained(BERT_ARCH)


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


def get_cached_transformation(sentence, transformation_func):
    hash_key = hash(sentence)

    if hash_key in cache:
        return cache[hash_key]

    transformed = transformation_func(sentence)

    cache[hash_key] = transformed

    return transformed


def spacy_similarity(ingredient, allergen):
    token1 = get_cached_transformation(ingredient, model_spacy)
    token2 = get_cached_transformation(allergen, model_spacy)

    if token1.has_vector and token2.has_vector:
        similarity = token1.similarity(token2) * 100
    else:
        similarity = 0

    return similarity


def sentence_to_vector_fasttext(sentence):
    words = sentence.split()
    vectors = [model_fasttext.get_word_vector(word) for word in words]

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        vector_dim = model_fasttext.get_dimension()
        return np.zeros(vector_dim)


def fasttext_similarity(ingredient, allergen):
    token1 = get_cached_transformation(ingredient, sentence_to_vector_fasttext)
    token2 = get_cached_transformation(allergen, sentence_to_vector_fasttext)
    token1 = np.array(token1).reshape(1, -1)
    token2 = np.array(token2).reshape(1, -1)
    return cossine_similarity_sklearn(token1, token2) * 100


def sentences_to_embeddings_bert(sentences):
    inputs = tokenizer_bert(sentences, return_tensors="pt",
                            padding=True, truncation=True, max_length=128)
    outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def bert_similarity(ingredient, allergen):
    ingredient_embedding = get_cached_transformation(ingredient, sentences_to_embeddings_bert)
    allergen_embedding = get_cached_transformation(allergen, sentences_to_embeddings_bert)

    return cosine_similarity_torch(ingredient_embedding, allergen_embedding) * 100


def calculate_metrics(predicted, true_list, all_ingredients):
    TP = len(set(predicted) & set(true_list))
    FP = len(set(predicted) - set(true_list))
    FN = len(set(true_list) - set(predicted))
    TN = len(set(all_ingredients) - set(true_list) - set(predicted))

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return PerformanceIndicators(TP, FP, TN, FN, accuracy, precision, recall, f1)

def evaluate_algorithm(df, algorithm):
    indicators_list = []

    for row in df.itertuples():
        predicted = getattr(row, f'alergenos_{algorithm}')
        true_list = row.gabarito
        all_ingredients = extract_ingredients(row.ingredients_text_pt)

        indicators = calculate_metrics(predicted, true_list, all_ingredients)
        indicators_list.append(indicators)

    indicators_array = np.array(indicators_list)

    avg_indicators_values = np.mean(indicators_array, axis=0)
    avg_indicators = PerformanceIndicators(*avg_indicators_values)

    confusion_mat = np.array([[avg_indicators.TP, avg_indicators.FN], [avg_indicators.FP, avg_indicators.TN]])

    return EvaluationResult(confusion_mat, avg_indicators.accuracy, avg_indicators.precision, avg_indicators.recall,
                            avg_indicators.f1)


def plot_confusion_matrix(matrix, algorithm):
    class_labels = ['Alérgeno', 'Não alergeno']
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    disp.plot(cmap='Purples', values_format='.2f')

    ax = plt.gca()
    ax.set_yticklabels(class_labels, rotation=90, va='center')

    plt.xlabel('Rótulo predito')
    plt.ylabel('Rótulo verdadeiro')
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


def is_allergen_present(ingredient, allergen, algorithm, threshold):
    match algorithm:
        case Algorithms.MATCH:
            return allergen in ingredient
        case Algorithms.LEVENSHTEIN:
            return levenshtein_distance(ingredient, allergen) > threshold
        case Algorithms.JACCARD:
            return jaccard_similarity(ingredient, allergen) > threshold
        case Algorithms.SPACY:
            return spacy_similarity(ingredient, allergen) > threshold
        case Algorithms.FASTTEXT:
            return fasttext_similarity(ingredient, allergen) > threshold
        case Algorithms.BERT:
            return bert_similarity(ingredient, allergen) > threshold


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


def create_allergen_in_ontology(ontology, derived_name, allergen_base):
    allergen_base_instance = ontology.search_one(label=allergen_base)
    Alimento = ontology.search_one(label="Alimento")

    new_allergen = Alimento(derived_name.title().replace(" ", ""))
    new_allergen.label = [derived_name.title()]
    new_allergen.eDerivadoDe.append(allergen_base_instance)


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
                create_allergen_in_ontology(ontology, derivated_allergen, base_allergen)
                added_derived_allergens.add(derivated_allergen)
    ontology.save(ONTOLOGY_PATH)

    df_amostra = pd.read_csv('conjunto_teste.csv', sep="\t", low_memory=False,
                             usecols=["code", "product_name_pt", "ingredients_text_pt", "gabarito"])

    df_amostra = preprocess_data(df_amostra)

    extracted_ingredients_amostra = df_amostra['ingredients_text_pt'].progress_apply(extract_ingredients)

    df_amostra['gabarito'] = df_amostra['gabarito'].apply(ast.literal_eval).apply(lambda x: [clean_text(i) for i in x])

    for algorithm in Algorithms:
        if algorithm == Algorithms.FASTTEXT and model_fasttext is None:
            continue

        best_threshold = 0
        best_f1 = 0
        for threshold in range(0, 101):
            detected_allergens = extracted_ingredients_amostra.progress_apply(
                lambda x: detect_allergens(x, cleaned_allergens_set, allergen_mapping, algorithm, threshold)
            )
            df_amostra[f'alergenos_{algorithm.value}'] = detected_allergens.apply(
                lambda x: [detected[0] for detected in x])
            metrics = evaluate_algorithm(df_amostra, algorithm.value)
            avg_f1 = metrics.f1
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
        cache.clear()

    return df_amostra


if __name__ == '__main__':
    result_df = main()
    display(result_df)
