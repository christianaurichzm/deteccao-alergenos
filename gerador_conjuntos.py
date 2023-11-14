import pandas as pd

GRUPOS = {
    "facil": [
        {"code": "7894904219650", "gabarito": ['proteína de soja', 'trigo']},
        {"code": "7896005804629", "gabarito": ['Leite em pó integral', 'Soro de leite em pó']},
        {"code": "7896050411230",
         "gabarito": ['leite em pó integral', 'farinha de trigo enriquecida com ferro e ácido fólico',
                      'soro de leite em pó']},
        {"code": "7896068000075", "gabarito": ['leite', 'creme de leite', 'fermento lácteo']},
        {"code": "7897846901232", "gabarito": ['fibra de trigo', 'farinha de soja', 'flocos de soja']},
        {"code": "7898067340404",
         "gabarito": ['Farinha de trigo enriquecida com ferro e ácido fólico', 'margarina', 'queijo mussarela',
                      'requeijão cremoso']},
        {"code": "7898409951824", "gabarito": ['soro de leite em pó', 'leite em pó integral', 'ovo em pó']},
        {"code": "7898666240068", "gabarito": ['Leite de cabra']},
        {"code": "7898686950305", "gabarito": ['Proteína de soja']},
        {"code": "7897517206116", "gabarito": ['proteína texturizada de soja']}
    ],

    "medio": [
        {"code": "7891095006250", "gabarito": ['leite', 'soro de leite']},
        {"code": "7896022086114",
         "gabarito": ['Sêmola de trigo enriquecida com ferro e ácido fólico', 'sêmola de trigo durum',
                      'farelo de trigo', 'fibra de trigo']},
        {"code": "7896283005664", "gabarito": ['extrato de soja']},
        {"code": "7896423480894", "gabarito": []},
        {"code": "7898614670343", "gabarito": ['Amendoim triturado', 'castanha triturada']},
        {"code": "7891103212420", "gabarito": ['FARINHA DE TRIGO ENRIQUECIDA COM FERRO E ÁCIDO FÓLICO']},
        {"code": "7894904261154", "gabarito": ['soja']},
        {"code": "7896024800602",
         "gabarito": ['farinha de trigo enriquecida com ferro e ácido fólico', 'farinha integral de soja',
                      'óleo essencial noz moscada']},
        {"code": "7897517206727", "gabarito": ['Extrato de soja', 'noz moscada']},
        {"code": "7898958161583", "gabarito": ['castanha de caju']}
    ],

    "dificil": [
        {"code": "5601038002209", "gabarito": ['Proteína de soja']},
        {"code": "7622300800529",
         "gabarito": ['farinha de trigo enriquecida com ferro e ácido fólico', 'farinha de trigo integral',
                      'aveia em flocos', 'farinha de cevada', 'farinha de centeio', 'leite em pó desnatado']},
        {"code": "7891025115199", "gabarito": ['pasta de amêndoas']},
        {"code": "7891097000133",
         "gabarito": ['Leite desnatado e/ou leite desnatado reconstituído', 'creme de leite', 'leite em pó desnatado']},
        {"code": "7892840814175", "gabarito": ['soro de leite']},
        {"code": "7895000483730",
         "gabarito": ['Farinha de trigo enriquecida com ferro e ácido fólico', 'fibra de trigo', 'ovo em pó',
                      'clara de ovo em pó', 'Queijo em pó', 'composto lácteo', 'leite em pó', 'soro de leite']},
        {"code": "7896003706598", "gabarito": ['FARINHA DE TRIGO FORTIFICADA COM FERRO E ÁCIDO FÓLICO', 'SOJA']},
        {"code": "7896080870274",
         "gabarito": ['FARINHA DE TRIGO ENRIQUECIDA COM FERRO E ÁCIDO FÓLICO', 'GORDURA VEGETAL HIDROGENADA DE SOJA']},
        {"code": "7896327514114", "gabarito": []},
        {"code": "7898215157311", "gabarito": ['Leite integral']}
    ]
}


def gerador_conjuntos():
    df = pd.read_csv(
        "openfoodfacts_export.csv",
        sep="\t",
        low_memory=False,
        usecols=["code", "product_name_pt", "ingredients_text_pt"]
    )

    all_codes = [item['code'] for group in GRUPOS.values() for item in group]
    present_codes = df['code'].unique().tolist()
    testing_codes = [code for code in all_codes if code in present_codes]

    if testing_codes:
        df_treinamento = df[~df['code'].isin(testing_codes)]

        gabaritos = [
            {'code': item['code'], 'gabarito': str(item['gabarito']), 'Grupo': grupo}
            for grupo, items in GRUPOS.items()
            for item in items
        ]

        df_teste = df.merge(pd.DataFrame(gabaritos), on='code', how='inner')

        df_teste['Grupo'] = pd.Categorical(df_teste['Grupo'], ["facil", "medio", "dificil"])
        df_teste = df_teste.sort_values('Grupo')

        df_treinamento.to_csv("conjunto_treinamento.csv", sep='\t', index=False)
        df_teste.to_csv("conjunto_teste.csv", sep='\t', index=False)
