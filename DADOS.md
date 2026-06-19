# Proveniência dos dados

Documentação da origem e versão do conjunto de dados, para reprodutibilidade,
o Open Food Facts é uma base **viva**, então os resultados deste trabalho referem-se
ao **snapshot fixo** descrito abaixo (versionado no próprio repositório).

## Fonte

- **Base:** Open Food Facts, <https://br.openfoodfacts.org>
- **Arquivo:** `openfoodfacts_export.csv` (formato TSV, separador `\t`)
- **Obtenção:** exportação **manual** pelo site do Open Food Facts, filtrando
  produtos com texto de ingredientes em **português**.
- **Data do dump:** **agosto/2023** (adicionado ao repositório em 2023-08-27,
  commit `9dc6ead`; última atualização do arquivo em 2023-09-19, commit `e03b153`).

## Por que fixar o snapshot

O Open Food Facts cresce continuamente (mais produtos, mais `allergens_tags`,
correções de OCR pela comunidade). Atualizar o dump mudaria todos os números
medidos. Para um trabalho científico, o snapshot é **congelado e versionado** de
modo que qualquer pessoa reproduza exatamente os mesmos resultados.

## Como atualizar (trabalho futuro)

O pipeline ingere **qualquer** export do OFF sem mudança de código (a taxonomia
`allergens_tags`, ex. `en:milk`, é estável). Para usar um dump mais novo:

1. Exportar um novo CSV do OFF (mesmos filtros) sobre `openfoodfacts_export.csv`.
2. Rodar `python3 main.py` / `bert_classificador.py` / `comparacao.py`.
3. Atualizar a **data do dump** acima.

Um dump mais novo e maior tende a beneficiar sobretudo os modelos **neurais**
(hoje limitados por ~2.393 exemplos de treino) e os **alérgenos raros**
(moluscos, castanhas), podendo, inclusive, alterar o veredito simbólico vs.
neural. Ver §9 do [RESULTADOS.md](RESULTADOS.md).
