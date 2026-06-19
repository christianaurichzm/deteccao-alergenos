"""
Base de conhecimento de alérgenos alimentares, fonte única de verdade.

Ancorada na RDC nº 26/2015 da ANVISA (rotulagem obrigatória de alergênicos no
Brasil) e alinhada à taxonomia de alérgenos do Open Food Facts, usada como
gabarito para avaliação.

A ideia central do trabalho está aqui: além da palavra-base do alérgeno
("leite", "trigo"...), as *formas derivadas/ocultas* ("lactose",
"caseína", "lecitina de soja", "albumina"...) são mapeadas explicitamente,
pois o casamento por substring ingênuo não as captura, mas a regulação exige declarar. Este dicionário é o que
popula a ontologia OWL (cada forma derivada vira um indivíduo ligado ao alérgeno
base pela propriedade `eDerivadoDe`).
"""

ALERGENOS = [
    "leite", "gluten", "soja", "ovo", "amendoim", "castanhas",
    "peixe", "crustaceos", "moluscos", "gergelim", "sulfitos",
    "mostarda", "aipo", "tremoco",
]

TAG_PARA_ALERGENO = {
    "en:milk": "leite", "es:leite": "leite", "es:nata": "leite",
    "pt:fermentos-lacteos": "leite", "pt:nata": "leite", "pt:natas": "leite",
    "pt:leitelho": "leite", "pt:soro-de-leite": "leite", "pt:solidos-lacteos": "leite",
    "en:gluten": "gluten", "es:gluten-de-trigo": "gluten", "pt:trigo-duro": "gluten",
    "es:avena": "gluten", "fr:avoine": "gluten",
    "en:soybeans": "soja",
    "en:eggs": "ovo",
    "en:nuts": "castanhas", "pt:amendoa": "castanhas", "pt:avela": "castanhas",
    "en:fish": "peixe",
    "en:peanuts": "amendoim",
    "en:sulphur-dioxide-and-sulphites": "sulfitos", "pt:metabissulfito": "sulfitos",
    "en:sesame-seeds": "gergelim",
    "en:mustard": "mostarda",
    "en:celery": "aipo",
    "en:molluscs": "moluscos",
    "en:crustaceans": "crustaceos",
    "en:lupin": "tremoco",
}

FORMAS = {
    "leite": ["leite", "lactose", "caseina", "caseinato", "soro de leite",
              "lactosoro", "nata", "creme de leite", "manteiga", "queijo",
              "requeijao", "iogurte", "coalhada", "composto lacteo",
              "solidos lacteos", "fermento lacteo", "fermentos lacteos",
              "leitelho", "whey", "lacteo", "lactea", "doce de leite",
              "leite condensado", "chantilly"],
    "gluten": ["trigo", "gluten", "semola", "semolina", "farinha de trigo",
               "farelo de trigo", "cevada", "centeio", "aveia", "malte",
               "triticale", "farinha de rosca", "cuscuz"],
    "soja": ["soja", "lecitina de soja", "lecitina", "proteina de soja",
             "extrato de soja", "oleo de soja", "farinha de soja", "tofu", "shoyu"],
    "ovo": ["ovo", "ovos", "clara de ovo", "gema", "albumina", "ovalbumina",
            "lisozima", "clara"],
    "amendoim": ["amendoim", "manteiga de amendoim", "oleo de amendoim",
                 "pasta de amendoim"],
    "castanhas": ["castanha", "castanha de caju", "castanha do para",
                  "castanha do brasil", "nozes", "noz", "amendoa", "avela",
                  "pistache", "macadamia", "peca", "pinoli"],
    "peixe": ["peixe", "bacalhau", "atum", "sardinha", "anchova", "salmao",
              "pescado"],
    "crustaceos": ["camarao", "caranguejo", "lagosta", "siri", "lagostim",
                   "crustaceo"],
    "moluscos": ["mexilhao", "ostra", "lula", "polvo", "molusco", "vieira",
                 "caracol"],
    "gergelim": ["gergelim", "sesamo", "tahine"],
    "sulfitos": ["sulfito", "metabissulfito", "bissulfito", "dioxido de enxofre",
                 "anidrido sulfuroso"],
    "mostarda": ["mostarda"],
    "aipo": ["aipo", "salsao"],
    "tremoco": ["tremoco", "lupino"],
}

FORMAS_BASE = {
    "leite": ["leite"], "gluten": ["trigo"], "soja": ["soja"], "ovo": ["ovo"],
    "amendoim": ["amendoim"],
    "castanhas": ["castanha", "noz", "amendoa", "avela"],
    "peixe": ["peixe"], "crustaceos": ["camarao"], "moluscos": ["molusco"],
    "gergelim": ["gergelim"], "sulfitos": ["sulfito"], "mostarda": ["mostarda"],
    "aipo": ["aipo"], "tremoco": ["tremoco"],
}

EXCLUSOES = {
    "leite": ["leite de coco", "leite de soja", "leite de amendoa",
              "leite de aveia", "leite de arroz", "leite de castanha",
              "manteiga de cacau", "manteiga de amendoim", "manteiga vegetal",
              "manteiga de karite"],
    "castanhas": ["noz moscada", "noz-moscada", "nozes moscada"],
    "moluscos": ["en polvo", "azucar en polvo"],
    "soja": ["lecitina de girassol", "lecitina de canola", "lecitina girassol"],
    "amendoim": [],
}
