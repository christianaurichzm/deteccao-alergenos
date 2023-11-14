import numpy as np
import spacy
import fasttext.util
from thefuzz import fuzz
from transformers import BertTokenizer, BertModel

from torch.nn.functional import cosine_similarity as cosine_similarity_torch
from sklearn.metrics.pairwise import cosine_similarity as cossine_similarity_sklearn

from cache import get_cached_transformation

model_spacy = spacy.load('pt_core_news_md')

MODEL_LANG = "pt"
fasttext.util.download_model(MODEL_LANG, if_exists='ignore')
model_fasttext = fasttext.load_model(f'cc.{MODEL_LANG}.300.bin')

BERT_ARCH = "neuralmind/bert-large-portuguese-cased"
tokenizer_bert = BertTokenizer.from_pretrained(BERT_ARCH)
model_bert = BertModel.from_pretrained(BERT_ARCH)


def normalize_thresold(value):
    return value * 100


def levenshtein_distance(ingredient, allergen):
    return fuzz.ratio(ingredient, allergen)


def jaccard_similarity(ingredient, allergen):
    ingredient_set = set(ingredient)
    allergen_set = set(allergen)
    intersection = ingredient_set.intersection(allergen_set)
    union = ingredient_set.union(allergen_set)
    return normalize_thresold(len(intersection) / len(union))


def spacy_similarity(ingredient, allergen):
    token1 = get_cached_transformation(ingredient, model_spacy)
    token2 = get_cached_transformation(allergen, model_spacy)

    if token1.has_vector and token2.has_vector:
        similarity = normalize_thresold(token1.similarity(token2))
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
    return normalize_thresold(cossine_similarity_sklearn(token1, token2))


def sentences_to_embeddings_bert(sentences):
    inputs = tokenizer_bert(sentences, return_tensors="pt",
                            padding=True, truncation=True, max_length=128)
    outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def bert_similarity(ingredient, allergen):
    ingredient_embedding = get_cached_transformation(ingredient, sentences_to_embeddings_bert)
    allergen_embedding = get_cached_transformation(allergen, sentences_to_embeddings_bert)

    return normalize_thresold(cosine_similarity_torch(ingredient_embedding, allergen_embedding))
