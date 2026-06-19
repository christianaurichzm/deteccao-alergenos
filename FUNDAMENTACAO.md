# Fundamentação Teórica

Este capítulo posiciona o trabalho no mapa da Inteligência Artificial. A tese
central é que o sistema desenvolvido é, simultaneamente, **(i)** um **sistema
neuro-simbólico** para detecção de alérgenos e **(ii)** um **estudo empírico
comparativo** entre os paradigmas simbólico, estatístico e híbrido. As seções a
seguir dão o embasamento de cada uma dessas dimensões, ancorando-se nas obras de
referência de Russell & Norvig (2021) e Luger (2009).

> Convenção: ao longo do texto, cada conceito teórico é amarrado ao componente
> correspondente da implementação (arquivos do repositório), para tornar
> explícito *onde* cada ideia da IA se materializa no sistema.

---

## 1. Inteligência Artificial e o agente racional

Russell & Norvig (2021) organizam o campo a partir de duas perguntas: a IA deve
*pensar* ou *agir*, e fazê-lo *como humanos* ou *racionalmente*? Disso resultam
quatro abordagens, agir humanamente (Teste de Turing), pensar humanamente
(modelagem cognitiva), pensar racionalmente (tradição **logicista**, as "leis do
pensamento") e **agir racionalmente** (o **agente racional**). Os autores adotam
o agente racional como fio condutor: um **agente** é tudo aquilo que percebe seu
ambiente por meio de **sensores** e age sobre ele por meio de **atuadores**,
buscando maximizar uma medida de desempenho.

Russell & Norvig (2021, cap. 2) hierarquizam os agentes em **reflexo simples**,
**reflexo baseado em modelo**, **baseado em objetivos**, **baseado em utilidade**
e **com aprendizagem**. Tratam ainda, na Parte de agentes lógicos, dos **agentes
baseados em conhecimento**, que mantêm uma **base de conhecimento (KB)** e
decidem por **inferência** sobre ela.

**Posicionamento.** O detector simbólico deste trabalho (o que a aplicação
`app_demo.py` invoca) realiza **inferência de tiro único**, percebe um texto de
ingredientes e produz um conjunto de alérgenos, sem estado interno persistente,
metas ou utilidade. Em termos estritos, seu núcleo é melhor descrito como um
**sistema baseado em conhecimento** (uma função de classificação por inferência)
do que como um agente autônomo. Quando encapsulado na aplicação (percebe o
ambiente via API do Open Food Facts, decide via base de conhecimento e reporta),
o conjunto caracteriza um **agente reflexo baseado em conhecimento**: reativo,
condição-ação, apoiado em uma KB. É o limite inferior, e honesto, da noção de
agente, não há planejamento nem ação sequencial.

---

## 2. IA simbólica e a hipótese do sistema de símbolos físicos

Luger (2009) estrutura a IA em torno de **representação (structures)** e
**inferência/busca (strategies)**, sob a **hipótese do sistema de símbolos
físicos** de Newell & Simon (1976): um sistema de símbolos físicos tem os meios
necessários e suficientes para a ação inteligente geral. Nessa tradição, hoje
chamada de **IA simbólica** ou *GOFAI* (*Good Old-Fashioned AI*), conhecimento
é representado **explicitamente** (em símbolos manipuláveis) e a inteligência
emerge da **manipulação formal** desses símbolos.

**Posicionamento.** A camada simbólica do sistema é um exemplar direto desse
paradigma: o **conhecimento** sobre derivações de alérgenos (ex.: *caseína* indica
leite, *lecitina de soja* indica soja), ancorado na RDC nº 26/2015 da ANVISA, é
representado explicitamente (`conhecimento_alergenos.py`) e materializado em uma
ontologia formal; a **inferência** ocorre por raciocínio lógico e por regras.
Representação + estratégia, exatamente o eixo de Luger.

---

## 3. Representação de conhecimento e raciocínio: ontologias e lógica de descrição

Uma **ontologia** é uma "especificação explícita de uma conceitualização"
(Gruber, 1993): formaliza classes, indivíduos, propriedades e axiomas de um
domínio. Russell & Norvig (2021) dedicam o capítulo de **Representação de
Conhecimento** às ontologias e às **lógicas de descrição** (*Description Logics*,
DL), formalismos que equilibram expressividade e decidibilidade da inferência
(Baader et al., 2007). A **OWL** (*Web Ontology Language*; W3C, 2012) é a
linguagem-padrão da Web Semântica para ontologias, com semântica fundada em DL;
**raciocinadores** como o **HermiT** (Glimm et al., 2014) computam, por dedução,
classificações e consistência; **SPARQL** (W3C) é a linguagem de consulta a esses
grafos de conhecimento.

**Posicionamento.** Este é o coração simbólico do trabalho:
- A ontologia (`ontologia_builder.py`, que gera `ontologia_alergenos.owl`) define as
  classes `Alimento`, `Alergeno` e `AlergenoPorDerivacao` e a propriedade
  `eDerivadoDe`, com a classe de derivação especificada por **equivalência
  lógica** (`Alimento ⊓ ∃eDerivadoDe.Alergeno`).
- O **raciocinador** (HermiT, via `owlready2`; Lamy, 2017) **infere** que as 93
  formas derivadas pertencem a `AlergenoPorDerivacao`, nenhuma é asserida
  manualmente, demonstrando dedução em DL (`python3 ontologia_builder.py raciocinar`).
- A detecção consulta a ontologia por **SPARQL** (`deteccao_ontologia.py`),
  fazendo da OWL a fonte de conhecimento em tempo de execução.

---

## 4. Sistemas baseados em conhecimento, sistemas especialistas e regras

A IA simbólica consolidou-se aplicada na forma de **sistemas baseados em
conhecimento** e **sistemas especialistas**, cuja arquitetura canônica é uma
**base de conhecimento** somada a um **motor de inferência** (Luger, 2009),
frequentemente com **sistemas de produção** (regras condição-ação). No
processamento de linguagem natural, regras simbólicas seguem relevantes para
fenômenos bem delimitados; o algoritmo **NegEx** (Chapman et al., 2001), por
exemplo, identifica termos **negados** pelo escopo da cláusula, técnica nascida
no domínio clínico e diretamente aplicável a rótulos de alimentos.

**Posicionamento.** A camada de precisão (`desambiguacao.py`) é um sistema de
regras: separa **presença** de **precaução** ("pode conter / traços de"), trata
**negação com escopo** ("sem glúten", "não contém leite") em moldes NegEx, e
resolve homógrafos/idioma ("en polvo" = pó em espanhol). Em conjunto com a
ontologia, constitui um **sistema baseado em conhecimento** para a tarefa
estreita de identificação de alérgenos, a linhagem clássica do sistema
especialista.

---

## 5. IA estatística e conexionista: aprendizado, redes neurais e Transformers

O segundo grande paradigma que Russell & Norvig (2021) e Luger (2009) apresentam
é o **estatístico/conexionista**: em vez de conhecimento codificado à mão, o
sistema **aprende** padrões a partir de dados. Inclui o **aprendizado de máquina**
(supervisionado, em particular) e as **redes neurais artificiais**. A arquitetura
**Transformer** (Vaswani et al., 2017) e o modelo **BERT** (Devlin et al., 2019)
inauguraram as **representações contextuais** pré-treinadas: o mesmo token recebe
vetores distintos conforme o contexto, habilitando, entre outras tarefas, a
**desambiguação de sentido**. Para o português do Brasil, o **BERTimbau** (Souza
et al., 2020) fornece tais modelos. Quando o custo de ajuste fino completo é
proibitivo, técnicas de **ajuste parametricamente eficiente** como **LoRA** (Hu
et al., 2022) congelam o modelo base e treinam pequenos adaptadores. Anteriores ao
Transformer, *embeddings* estáticos como *word2vec* (Mikolov et al., 2013) e
*fastText* (Bojanowski et al., 2017) já capturavam similaridade distribucional.

**Posicionamento.** Dois componentes pertencem a este paradigma:
- O **classificador neural** (`bert_classificador.py`): BERTimbau ajustado como
  classificador **multirrótulo**, com `pos_weight` para o desbalanceamento,
  limiar por rótulo e seleção da melhor época; suporta **LoRA** para o modelo
  *large* sob restrição de memória de GPU.
- O **desambiguador contextual** (`desambiguacao_bert.py`): usa o **embedding
  contextual** do BERTimbau para decidir, por proximidade a protótipos de
  sentido, se um termo polissêmico indica o alérgeno, empregando a representação
  contextual para desambiguação de sentido, não casamento de strings.

---

## 6. Processamento de linguagem natural: tarefas e técnicas

A detecção mobiliza tarefas clássicas de PLN. A **desambiguação de sentido de
palavra** (*Word Sense Disambiguation*, WSD), determinar a acepção correta de um
termo ambíguo no contexto, é tarefa de longa data na área (Navigli, 2009) e é
exatamente o que a abordagem híbrida realiza ("leite de coco" vs. "leite
integral"). A **tokenização por subpalavra** dos modelos Transformer confere
robustez a variações ortográficas e ruído de OCR. Por fim, o gabarito deste
trabalho é construído por **supervisão distante** (Mintz et al., 2009): rótulos
derivados automaticamente de uma fonte estruturada (as `allergens_tags` do Open
Food Facts) em vez de anotação manual.

---

## 7. Sistemas neuro-simbólicos

A integração entre o simbólico (conhecimento, lógica, interpretabilidade) e o
conexionista (aprendizado, robustez a ruído) é o objeto da **IA neuro-simbólica**,
apontada como uma terceira onda da área (Garcez & Lamb, 2023). A motivação é
complementaridade: sistemas simbólicos são transparentes e rastreáveis a normas,
porém rígidos; sistemas neurais generalizam e toleram ruído, porém são opacos e
dependentes de dados.

**Posicionamento.** O sistema deste trabalho é deliberadamente **neuro-simbólico**:
a ontologia e as regras provêm conhecimento e interpretabilidade (alinhamento à
RDC 26/2015), enquanto o BERTimbau provê desambiguação contextual e robustez a
OCR. O estudo quantifica essa complementaridade, inclusive o ponto de
**cruzamento de robustez** a partir do qual o neural supera o simbólico sob ruído.

---

## 8. Avaliação empírica como método científico

Para além dos artefatos, a contribuição é um **estudo empírico**. Russell &
Norvig (2021) enquadram a IA moderna como disciplina **empírica**: hipóteses
testadas contra dados. O trabalho adota o instrumental correspondente, gabarito
em escala, separação treino/validação/teste, métricas adequadas a classes
desbalanceadas (precisão/revocação/F1, micro e macro), **intervalos de confiança**
por *bootstrap*, **testes de significância pareados**, **ablação** por camada e um
**baseline clássico forte** (TF-IDF + Regressão Logística). É essa montagem
metodológica que converte "parece funcionar" em afirmação mensurável.

---

## 9. Síntese: onde este trabalho se situa

| Componente (arquivo) | Paradigma de IA | Conceito (referência) |
|---|---|---|
| `conhecimento_alergenos.py` | simbólico | Representação de conhecimento (Gruber, 1993) |
| `ontologia_builder.py`, `ontologia_alergenos.owl` | simbólico | Ontologia / OWL / DL (W3C, 2012; Baader et al., 2007) |
| raciocinador (HermiT) | simbólico | Inferência em DL (Glimm et al., 2014) |
| `deteccao_ontologia.py` | simbólico | Consulta a grafo de conhecimento (SPARQL) |
| `desambiguacao.py` | simbólico | Sistema de regras / negação NegEx (Chapman et al., 2001) |
| `desambiguacao_bert.py` | híbrido | WSD por representação contextual (Devlin et al., 2019; Navigli, 2009) |
| `bert_classificador.py` | conexionista | BERT / BERTimbau / LoRA (Souza et al., 2020; Hu et al., 2022) |
| `comparacao.py`, `avaliacao_alergenos.py` | método | Avaliação empírica e estatística |
| `app_demo.py` | aplicação | Agente reflexo baseado em conhecimento (Russell & Norvig, 2021) |

**Classificação do artefato.** O núcleo invocado pela aplicação é um **sistema
baseado em conhecimento** (ontologia em lógica de descrição + raciocínio +
regras), na linhagem da IA simbólica / sistemas especialistas; o projeto como um
todo é um **sistema neuro-simbólico**. Cientificamente, a contribuição é um
**estudo empírico comparativo** de paradigmas, não "um modelo" isolado.

---

## Referências

- BAADER, F. et al. *The Description Logic Handbook*. 2. ed. Cambridge University Press, 2007.
- BOJANOWSKI, P. et al. Enriching Word Vectors with Subword Information. *TACL*, 2017.
- CHAPMAN, W. W. et al. A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries. *Journal of Biomedical Informatics*, 2001.
- DEVLIN, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*, 2019.
- GARCEZ, A. d'A.; LAMB, L. Neurosymbolic AI: The 3rd Wave. *Artificial Intelligence Review*, 2023.
- GLIMM, B. et al. HermiT: An OWL 2 Reasoner. *Journal of Automated Reasoning*, 2014.
- GRUBER, T. R. A Translation Approach to Portable Ontology Specifications. *Knowledge Acquisition*, 1993.
- HU, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022.
- LAMY, J.-B. Owlready: Ontology-oriented programming in Python. *Artificial Intelligence in Medicine*, 2017.
- LUGER, G. F. *Artificial Intelligence: Structures and Strategies for Complex Problem Solving*. 6. ed. Pearson, 2009.
- MIKOLOV, T. et al. Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*, 2013.
- MINTZ, M. et al. Distant Supervision for Relation Extraction without Labeled Data. *ACL-IJCNLP*, 2009.
- NAVIGLI, R. Word Sense Disambiguation: A Survey. *ACM Computing Surveys*, 2009.
- NEWELL, A.; SIMON, H. A. Computer Science as Empirical Inquiry: Symbols and Search. *Communications of the ACM*, 1976.
- RUSSELL, S.; NORVIG, P. *Artificial Intelligence: A Modern Approach*. 4. ed. Pearson, 2021.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. *BRACIS*, 2020.
- VASWANI, A. et al. Attention Is All You Need. *NeurIPS*, 2017.
- W3C. *OWL 2 Web Ontology Language Document Overview*. 2. ed. 2012.
- BRASIL. ANVISA. *Resolução RDC nº 26, de 2 de julho de 2015*. (Rotulagem obrigatória de alergênicos.)

> Nota: confira edição/ano de cada referência na sua norma de citação (ABNT), as
> entradas acima usam as edições mais usuais. O DOI/URL de cada uma pode ser
> adicionado conforme o template da sua instituição.
