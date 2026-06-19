"""
Pipeline de detecção de alérgenos: reconstrói a ontologia OWL e compara os
paradigmas de detecção contra o gabarito do Open Food Facts.

Uso:
    python3 main.py
"""

from ontologia_builder import construir
from comparacao import main as rodar_comparacao


def main():
    construir()
    rodar_comparacao()


if __name__ == "__main__":
    main()
