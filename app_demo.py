"""
Demonstração: dado um código de barras, consulta a API v2 do Open Food Facts
e detecta alérgenos no texto de ingredientes (ontologia + regras linguísticas).

Uso:
    python3 app_demo.py 7894900011517
    python3 app_demo.py            # usa um código de exemplo
"""

import sys
import json
import urllib.request

from avaliacao_alergenos import normalizar
from desambiguacao import detectar_preciso
from deteccao_ontologia import carregar_formas_da_ontologia

API = "https://world.openfoodfacts.org/api/v2/product/{}.json"
CAMPOS = "product_name,brands,ingredients_text_pt,ingredients_text,allergens_tags"
USER_AGENT = "deteccao-alergenos"


def buscar_produto(codigo):
    url = f"{API.format(codigo)}?fields={CAMPOS}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def detectar_no_produto(codigo, mapa=None):
    dados = buscar_produto(codigo)
    if dados.get("status") != 1:
        return None
    produto = dados["product"]
    texto = produto.get("ingredients_text_pt") or produto.get("ingredients_text") or ""
    mapa = mapa or carregar_formas_da_ontologia()
    detectados = detectar_preciso(normalizar(texto), mapa)
    return produto, texto, detectados


def main(codigo):
    resultado = detectar_no_produto(codigo)
    if resultado is None:
        print(f"Produto {codigo} não encontrado no Open Food Facts.")
        return
    produto, texto, detectados = resultado
    nome = produto.get("product_name") or "(sem nome)"
    marca = produto.get("brands") or "-"
    print(f"Produto: {nome}  [{marca}]  (código {codigo})")
    print(f"Ingredientes: {texto[:140]}{'...' if len(texto) > 140 else ''}")
    print(f"\n>>> ALÉRGENOS DETECTADOS: {sorted(detectados) or 'nenhum'}")
    print(f"    (tags declaradas no OFF, p/ conferência: {produto.get('allergens_tags', [])})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "7894900011517")
