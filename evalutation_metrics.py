from collections import namedtuple

import numpy as np

from extracao import extract_ingredients

evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1']
PerformanceIndicators = namedtuple('PerformanceIndicators', ['TP', 'FP', 'TN', 'FN', *evaluation_metrics])
EvaluationResult = namedtuple('EvaluationResult', ['confusion_mat', *evaluation_metrics])


def calculate_metrics(predicted, true_list, all_ingredients):
    predicted_set = set(predicted)
    true_set = set(true_list)
    all_ingredients_set = set(all_ingredients)

    TP = len(predicted_set & true_set)
    FP = len(predicted_set - true_set)
    FN = len(true_set - predicted_set)
    TN = len(all_ingredients_set - true_set - predicted_set)

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return PerformanceIndicators(TP, FP, TN, FN, accuracy, precision, recall, f1)


def evaluate_algorithm(df, algorithm):
    indicators_list = []

    for row in df.itertuples():
        predicted = getattr(row, f'alergenos_{algorithm.value}')
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
