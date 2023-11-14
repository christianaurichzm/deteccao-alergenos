from tqdm import tqdm

from gerador_conjuntos import gerador_conjuntos


def preparar_dados():
    tqdm.pandas()
    gerador_conjuntos()
