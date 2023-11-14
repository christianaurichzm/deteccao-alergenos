from algorithms import Algorithms
from similarity_metrics import levenshtein_distance, jaccard_similarity, spacy_similarity, \
    fasttext_similarity, bert_similarity


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



