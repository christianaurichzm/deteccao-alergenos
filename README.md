# Detecção de alérgenos em listas de ingredientes (PLN + ontologia OWL)

TCC: detecção de alérgenos em listas de ingredientes de alimentos combinando uma
**ontologia OWL** de derivações (ancorada na RDC nº 26/2015 da ANVISA) com
**processamento de linguagem natural**, avaliada contra o gabarito do
**Open Food Facts**.

O trabalho compara três paradigmas no mesmo conjunto de teste e métricas:
**simbólico** (ontologia + regras linguísticas), **híbrido** (ontologia + BERT
para desambiguação contextual) e **neural** (BERTimbau fim-a-fim; baseline
clássico TF-IDF).

**Documentação:** [RESULTADOS.md](RESULTADOS.md) (metodologia, tabelas, conclusões), [FUNDAMENTACAO.md](FUNDAMENTACAO.md) (embasamento teórico de IA: R&N, Luger, neuro-simbólico), [DADOS.md](DADOS.md) (proveniência do dataset).

## Como rodar

```bash
python3 main.py            # reconstrói a ontologia e compara as abordagens simbólicas
python3 bert_classificador.py   # treina o BERTimbau (usa GPU automaticamente, se houver)
python3 comparacao.py      # tabela final dos 7 sistemas (com IC95% e significância)
python3 robustez_ocr.py    # curva de robustez a ruído de OCR
python3 graficos.py        # gera as figuras do estudo em figuras/
```

## Estrutura do projeto

O repositório tem dois blocos, ambos parte do estudo:

**(A) Estudo atual**, o sistema de detecção, a infraestrutura de avaliação e os
experimentos que produzem os resultados (tabela abaixo).

**(B) Abordagem inicial por similaridade**, a primeira forma do trabalho
(comparação de métricas de similaridade), preservada como registro do estudo e
reavaliada com a metodologia atual em `baselines_similaridade.py` (ver nota ao fim).

### (A) Módulos do estudo atual

| Arquivo | Papel |
|---|---|
| `conhecimento_alergenos.py` | base de conhecimento: alérgenos canônicos, derivações (RDC 26/2015), exclusões |
| `ontologia_builder.py` | gera `ontologia_alergenos.owl` a partir do conhecimento (idempotente) |
| `deteccao_ontologia.py` | detecção consultando a ontologia OWL via SPARQL |
| `desambiguacao.py` | regras linguísticas: precaução/presença, negação (NegEx), idioma |
| `desambiguacao_bert.py` | desambiguação contextual de sentido com BERTimbau (WSD) |
| `bert_classificador.py` | BERTimbau multirrótulo (neural fim-a-fim) |
| `avaliacao_alergenos.py` | gabarito do OFF, divisão treino/val/teste, métricas (IC + significância) |
| `comparacao.py` | comparação dos 7 sistemas |
| `robustez_ocr.py` | experimento de robustez a OCR |
| `curva_aprendizado.py` | curva de aprendizado (TF-IDF): mais dados melhorariam os modelos? |
| `graficos.py` | gera as figuras do estudo (robustez, curva de aprendizado, comparação) |
| `baselines_similaridade.py` | baselines por similaridade (Levenshtein, Jaccard, spaCy, fastText, BERT-cosseno), abordagem inicial |
| `app_demo.py` | aplicação: código de barras, consulta a API do OFF e detecta ao vivo |
| `main.py` | ponto de entrada do pipeline |

### (B) Implementação inicial (abordagem por similaridade)

Os módulos `similarity_metrics.py`, `algorithms.py`, `deteccao_alergenos.py`,
`treinamento.py`, `teste.py`, `evalutation_metrics.py`, `gerador_conjuntos.py`,
`data_preparation.py`, `ontology_operations.py`, `extracao.py`,
`pre_processamento.py`, `cache.py`, `plot.py` (+ `ontologia.owl`,
`query_alergenos.sparql`, `amostra.csv`) são a primeira forma do trabalho,
**preservados como registro do estudo**. O `baselines_similaridade.py` reavalia
essas métricas com a metodologia atual (gold do OFF, split, IC, significância).

## Ambiente

Python 3.12. Dependências do estudo atual (Bloco A) em `requirements.txt`
(torch 2.12 +cu126 para GPU, transformers 5.12, scikit-learn 1.9, owlready2 0.50,
matplotlib para as figuras):

```bash
pip install -r requirements.txt
```

A abordagem inicial (Bloco B) tem ambiente próprio de 2023 em
`requirements-inicial.txt`.
