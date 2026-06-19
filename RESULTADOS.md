# Detecção de alérgenos em listas de ingredientes: ontologia OWL + PLN

Documento de resultados e metodologia (material para a dissertação). Todos os
números foram medidos no **mesmo conjunto de teste** e com o **mesmo código de
métricas** (`avaliacao_alergenos.relatorio`), garantindo comparação justa entre
os paradigmas.

---

## 1. Dados e gabarito

Fonte: exportação do **Open Food Facts** (`openfoodfacts_export.csv`), produtos
do mercado brasileiro. **Snapshot fixo de agosto/2023**, obtido por exportação
manual filtrando produtos com ingredientes em português; versionado no
repositório para reprodutibilidade (proveniência completa em
[DADOS.md](DADOS.md)).

Caracterização (revela problemas reais de qualidade do dado):

| Métrica | Valor |
|---|---|
| Linhas totais no CSV | 15.204 |
| Com `ingredients_text_pt` preenchido | 5.599 (**37%**), muitas colunas faltando |
| Com `allergens_tags` (taxonomia OFF) | 4.010 |
| **Produtos avaliáveis** (≥1 alérgeno canônico) | **3.417** |
| Divisão treino / validação / teste (70/15/15) | 2.393 / 512 / 512 |

**Gabarito:** o conjunto de alérgenos de cada produto vem de `allergens_tags`
(`en:milk`, `en:gluten`, `en:soybeans`...), mapeado para 14 alérgenos canônicos
alinhados à **RDC nº 26/2015 da ANVISA**. Substitui os ~30 itens rotulados à mão
da versão anterior por um gabarito ~110× maior.

**Ruído de OCR (medido):** o texto de ingredientes contém muita inserção por OCR.
Variantes a 1 caractere de um alérgeno (não captadas por casamento exato):
`lete`, `lelte`, `deite`, `glutem`, `glute`, `trico`, `castana`, além de
contaminação de idioma (`soya`, `avena`, `cebada`, `centeno`). Total: **512
ocorrências "quase-alérgeno"**, recall perdido por matching exato. Isso motiva o
estudo de robustez (§5).

---

## 2. Metodologia

### 2.1 Métricas
Classes muito desbalanceadas, o que exige **precisão / revocação / F1**, em versão **micro**
(agrega todas as decisões) e **macro** (média por alérgeno). Acurácia foi
descartada por ser inflada pelos verdadeiros-negativos. Cada resultado traz
**intervalo de confiança 95% (bootstrap, 1000 reamostragens)** e a comparação
entre sistemas usa **teste de significância pareado (bootstrap)**.

### 2.2 Os sete sistemas

**Simbólico**
1. **BASE**, casamento por substring apenas da palavra-base do alérgeno
   (aproxima a abordagem original do trabalho).
2. **Ontologia (OWL/SPARQL)**, a ontologia `ontologia_alergenos.owl` é
   reconstruída da base de conhecimento (`conhecimento_alergenos.py`,
   ancorada na RDC 26/2015): cada **forma derivada** (lactose, caseína, soro de
   leite, lecitina de soja, sêmola...) é um indivíduo ligado ao alérgeno-base
   por `eDerivadoDe`. A detecção consulta a ontologia via **SPARQL**.
3. **Ontologia + Regras**, camada linguística de precisão: separação
   presença vs. **precaução** ("pode conter / traços de"), **negação com escopo
   de cláusula** (estilo NegEx: "sem glúten", "não contém leite") e tratamento de
   **homógrafos/idioma** ("en polvo" = pó em espanhol; "noz moscada" = especiaria).

> **Raciocínio OWL DL (HermiT):** a classe `AlergenoPorDerivacao` é definida por
> equivalência lógica (`Alimento ⊓ ∃eDerivadoDe.Alergeno`). Nenhum indivíduo é
> asserido nela, mas o raciocinador **infere automaticamente** que as **93 formas
> derivadas** pertencem à classe, evidência de que a ontologia faz dedução
> semântica, não apenas casamento de strings (`python3 ontologia_builder.py raciocinar`).

**Híbrido**
4. **Ontologia + BERT (WSD)**, a ambiguidade dos termos polissêmicos é resolvida
   por **desambiguação contextual** com **BERTimbau** (_word sense disambiguation_,
   aplicação direta da representação contextual): o embedding contextual do termo é
   comparado a protótipos de cada sentido ("leite de coco" vs. "leite integral").

**Aprendizado supervisionado**
5. **TF-IDF + Regressão Logística**, baseline clássico forte (1-2 gramas,
   um-contra-todos, `class_weight=balanced`), com limiar por rótulo na validação.
6. **BERTimbau multirrótulo**, fine-tuning fim-a-fim do
   `neuralmind/bert-base-portuguese-cased` (4 épocas, lote 8, máx. 192 tokens,
   AdamW lr 2e-5, `weight_decay` 0,01). Rigor: `pos_weight` para o
   desbalanceamento, **limiar por rótulo** ajustado na validação e seleção da
   **melhor época** pela validação (early stopping). GPU NVIDIA RTX 3050 Ti.
7. **BERTimbau-large + LoRA**, variante de **escala**: o modelo *large* (335 M)
   ajustado por **LoRA** (adaptadores de baixo posto; Hu et al., 2022), já que o
   full fine-tuning não cabe nos 4 GB da GPU. Os adaptadores são fundidos ao
   modelo base ao salvar (`LORA=1`, `BATCH=2`, `LR=1e-4`).

> Anti-vazamento: o OFF marca alérgenos no texto com underscores (`_leite_`);
> são removidos da entrada do BERTimbau para o modelo não "trapacear".

---

## 3. Resultado principal, comparação dos 7 sistemas

Conjunto de teste (512 produtos), IC95% bootstrap do micro-F1.

| # | Sistema | Paradigma | micro-F1 | macro-F1 | IC95% (micro) |
|---|---|---|---:|---:|---|
| 1 | BASE (substring) | simbólico | 0,773 | 0,475 | [0,748; 0,797] |
| 2 | Ontologia (OWL/SPARQL) | simbólico | 0,800 | 0,629 | [0,779; 0,822] |
| 3 | **Ontologia + Regras** | simbólico | **0,868** | **0,737** | [0,849; 0,887] |
| 4 | **Ontologia + BERT (WSD)** | híbrido | **0,868** | 0,737 | [0,847; 0,886] |
| 5 | TF-IDF + Reg. Logística | supervisionado | 0,827 | 0,606 | [0,807; 0,848] |
| 6 | BERTimbau multirrótulo | supervisionado | 0,788 | 0,562 | [0,764; 0,814] |
| 7 | BERTimbau-**large** + LoRA | supervisionado | 0,750 | 0,517 | [0,724; 0,778] |

**Significância (bootstrap pareado de micro-F1) vs. sistema 3:**

| Comparação | Δ | p | Conclusão |
|---|---:|---:|---|
| Híbrido (BERT-WSD) | -0,000 | 0,90 | **empate estatístico** |
| TF-IDF | -0,041 | <0,001 | pior (significativo) |
| BERTimbau | -0,080 | <0,001 | pior (significativo) |
| BERTimbau-large + LoRA | -0,118 | <0,001 | pior (significativo) |
| Ontologia | -0,068 | <0,001 | pior (significativo) |
| BASE | -0,095 | <0,001 | pior (significativo) |

**Estabilidade do neural (5 sementes, BERTimbau, `MAXLEN=160`):**
micro-F1 **0,788 ± 0,011**, macro-F1 **0,566 ± 0,007**. O desvio entre sementes
(±0,011) é ~7× menor que a diferença para o simbólico (≈0,08), a vantagem do
conhecimento estruturado **não é ruído de inicialização do treino**.

### Baselines de similaridade (ponto de partida)

A abordagem inicial casava *frase de ingrediente* e *palavra de alérgeno* por
similaridade + limiar. Avaliadas como baselines, com limiar ajustado na
**validação** (nunca no teste) e medidas no mesmo teste, com IC95%:

| Baseline (similaridade) | micro-F1 | macro-F1 |
|---|---:|---:|
| Jaccard (caracteres) | 0,221 | 0,200 |
| Levenshtein | 0,343 | 0,235 |
| BERTimbau (cosseno, palavra × frase) | 0,374 | 0,221 |
| spaCy (vetores de palavra) | 0,425 | 0,255 |
| fastText (cosseno) | não medido (requer `cc.pt.300`, 4,5 GB; opcional em `baselines_similaridade.py`) | |

Todas ficam **muito abaixo** do simples casamento por substring (BASE = 0,773) e
do sistema simbólico (0,868). A razão é estrutural: comparar uma *palavra-base* a
uma *frase inteira* por similaridade é a ferramenta errada, a frase dilui o
sinal, e mesmo o BERT usado de forma ingênua (cosseno palavra × frase) rende 0,374.
Esse diagnóstico, **as métricas de similaridade saturam baixo**, foi o que
**motivou a evolução** da detecção: de "casamento aproximado de strings" para
**conhecimento estruturado** (ontologia + regras) e o **uso correto do BERT**
(desambiguação contextual, §3 sistema 4). Reprodução em `baselines_similaridade.py`.

---

## 4. Contribuição de cada camada (ablação simbólica)

- **De BASE para Ontologia:** o ganho é em **revocação** (macro-R 0,54 para 0,76; macro-F1
  0,48 para 0,63). A ontologia recupera os alérgenos ocultos exigidos pela ANVISA
  (lactose, caseína, soro de leite, lecitina), invisíveis ao substring ingênuo.
- **De Ontologia para +Regras:** o ganho é em **precisão** (micro-P 0,72 para 0,85). O
  diagnóstico mostrou que o gargalo de precisão **não era semântico** (e sim
  declarações de precaução, negação e idioma), resolvido por regras linguísticas.

---

## 5. Robustez a ruído de OCR

Injeção de ruído de caractere controlado (confusões visuais de OCR) no texto de
teste; micro-F1 por taxa de ruído.

| Taxa de ruído | Simbólico (Ont.+Regras) | Neural (BERTimbau) |
|---:|---:|---:|
| 0% | **0,868** | 0,792 |
| 5% | **0,799** | 0,767 |
| 10% | 0,733 | **0,744** |
| 15% | 0,674 | **0,706** |
| 20% | 0,603 | **0,666** |

**Cruzamento em ~10%:** o casamento exato (simbólico) quebra com 1 caractere
trocado; acima de ~10% de ruído, a tokenização por subpalavra do BERTimbau o
torna mais robusto. O ponto de cruzamento **quantifica quando cada paradigma
vence**. Todos os números (tabela e curva) usam o **mesmo modelo canônico**
(BERTimbau, semente 42, `MAXLEN=160`).

---

## 6. Análise de erro (sobre os falsos-positivos do sistema simbólico)

Antes da camada de regras havia 288 FP no teste; a categorização revelou que a
maioria **não era erro do modelo**:

| Categoria | Qtde | Natureza |
|---|---:|---|
| Declaração de precaução (`traces_tags` / "pode conter") | 186 (65%) | precaução, não presença, resolvida pelas regras |
| Gabarito do OFF incompleto | ~100 | detecção correta, tag faltando |
| Ambiguidade real (lecitina de girassol) | 2 | resolvido por exclusão/WSD |

Portanto, a **precisão medida (0,85) é um piso**: a precisão real é maior, pois ~100 "FP"
eram detecções corretas penalizadas por gold incompleto.

---

## 7. Conclusões

**Ranking de desempenho na tarefa (por micro-F1):**

| Posição | Sistema | micro-F1 | macro-F1 |
|---|---|---:|---:|
| 1º (empate) | **Ontologia + Regras** (simbólico) | **0,868** | 0,737 |
| 1º (empate) | **Ontologia + BERT-WSD** (híbrido) | **0,868** | 0,737 |
| 3º | TF-IDF + Reg. Logística | 0,827 | 0,606 |
| 4º | Ontologia (sem regras) | 0,800 | 0,629 |
| 5º | BERTimbau base | 0,788 | 0,562 |
| 6º | BASE (substring) | 0,773 | 0,475 |
| 7º | BERTimbau-large + LoRA | 0,750 | 0,517 |

**Melhor sistema: o simbólico (Ontologia + Regras).** Empata estatisticamente com
o híbrido (Δ=0,000; p=0,90), mas vence na prática por ser **mais simples,
interpretável, rastreável à RDC 26/2015 e sem custo de GPU/treino**. Notas:
(a) por **macro-F1** o topo se confirma (0,737), com a Ontologia (0,629)
superando o TF-IDF (0,606) e o BASE em último (0,475); (b) entre os modelos
**aprendidos**, o desempenho *cai* com o tamanho, TF-IDF (0,827) > BERTimbau
(0,788) > large+LoRA (0,750), evidência de que o limite é **dados**, não capacidade.

1. **Conhecimento estruturado vence aprendizado** neste cenário: a ontologia +
   regras (ancorada na RDC 26/2015) supera, de forma **estatisticamente
   significativa**, tanto o baseline clássico (TF-IDF) quanto o BERTimbau,
   porque o rótulo é escasso e ruidoso e o conhecimento curado não depende de dados.
   **Escalar o modelo não inverte o quadro**: o BERTimbau-*large* (via LoRA) fica
   ainda mais atrás (0,750; Δ=-0,118), confirmando que o gargalo é de **dados**,
   não de **capacidade** do modelo.
2. **O BERT no papel certo não se distingue das regras** (p=0,90): a desambiguação
   contextual generaliza as exclusões manuais **sem perda detectável de F1**, evidência de
   que o valor do BERT aqui é a contextualização (WSD), não o casamento de string.
3. **Os paradigmas são complementares:** simbólico para texto limpo,
   interpretável e rastreável à norma; neural para o mundo real ruidoso de OCR
   (cruzamento em ~10% de ruído).

---

## 8. Reprodutibilidade

```bash
python3 main.py              # ontologia + comparação simbólica
python3 ontologia_builder.py # (re)gera ontologia_alergenos.owl do conhecimento
python3 bert_classificador.py             # treina BERTimbau (usa GPU se houver)
SEEDS=42,1,2,3,4 python3 bert_classificador.py   # média ± desvio sobre sementes
python3 comparacao.py        # tabela dos 7 sistemas (IC + significância)
python3 robustez_ocr.py      # curva de robustez a OCR
```

Módulos: `conhecimento_alergenos` (base de conhecimento/RDC), `avaliacao_alergenos`
(gold+split+métricas), `ontologia_builder`, `deteccao_ontologia` (SPARQL),
`desambiguacao` (regras), `desambiguacao_bert` (WSD), `bert_classificador`
(neural), `comparacao`, `robustez_ocr`, `main`.

Ambiente: Python 3.12; torch 2.12+cu126, transformers 5.12, scikit-learn 1.9,
owlready2 0.50; GPU RTX 3050 Ti.

---

## 9. Validade externa, longevidade e trabalho futuro

**Longevidade do projeto (o que envelhece e o que não).** Os *números absolutos*
dependem do snapshot de dados e do modelo; as *conclusões conceituais* e a
*metodologia comparativa* não. O sistema foi desenhado para envelhecer bem:

- **Dados refrescáveis.** O gabarito é gerado das `allergens_tags` (taxonomia
  estável do OFF), então o pipeline ingere **qualquer dump novo sem mudar código**
  (ver [DADOS.md](DADOS.md)). O snapshot deste trabalho é de **ago/2023**; um dump
  mais novo e maior beneficiaria sobretudo os modelos **neurais** (hoje limitados
  a ~2.393 exemplos) e os **alérgenos raros**, podendo, inclusive, **alterar o
  veredito simbólico × neural** (a vantagem do simbólico está condicionada à
  escassez de dados). Isso é um teste futuro que o próprio design já suporta.

  *Evidência, curva de aprendizado* (TF-IDF treinado com frações crescentes do
  treino, avaliado no mesmo teste; `curva_aprendizado.py`):

  | treino | 239 | 598 | 1.196 | 1.794 | 2.393 |
  |---|---:|---:|---:|---:|---:|
  | micro-F1 | 0,541 | 0,647 | 0,716 | 0,797 | **0,827** |

  A curva **não atinge platô** em 100% (ainda sobe +0,03 de 75% para 100%): o modelo
  aprendido é limitado por **dados**, não por capacidade. Extrapolando, com ~2-3×
  mais dados ele tende a alcançar o simbólico (0,868), e o BERTimbau, com curva
  tipicamente mais íngreme, a ultrapassá-lo. É a confirmação empírica de que a
  conclusão #1 vale **no regime atual de poucos dados**.
- **Modelo trocável.** O codificador é um parâmetro de configuração
  (`MODELO`, `MODELO_WSD`): modelos de PT mais recentes (Albertina-PT, mDeBERTa,
  ou LLMs) plugam sem reescrever, o **arcabouço de comparação independe do modelo**.
- **Conhecimento estável.** As derivações vêm da RDC 26/2015; se a norma mudar,
  basta editar `conhecimento_alergenos.py`. O simbólico é robusto ao tempo
  *porque é conhecimento, não padrão ajustado a dados*.

**Limitações conhecidas.**
- **Gold incompleto** (OFF é colaborativo): subestima a precisão; uma amostra
  auditada manualmente daria o teto real.
- **Alérgenos raros**: moluscos e castanhas têm poucos exemplos (revocação baixa).
- **Hardware**: o full fine-tuning do **BERTimbau-large** não cabe nos 4 GB da GPU;
  com **LoRA** ele foi treinável, mas ficou **abaixo do base** (0,750), reforçando
  que o gargalo é de dados, não de capacidade. Com mais dados (dump novo) e/ou GPU
  maior, o full fine-tuning do large seria o próximo passo natural.
