from enum import Enum


class Algorithms(Enum):
    MATCH = "match"
    LEVENSHTEIN = "levenshtein"
    JACCARD = "jaccard"
    SPACY = "spacy"
    FASTTEXT = "fasttext"
    BERT = "bert"
