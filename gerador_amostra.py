import pandas as pd

df = pd.read_csv(
    "openfoodfacts_export.csv",
    sep="\t",
    low_memory=False,
    usecols=["code", "product_name_pt", "ingredients_text_pt"]
)

df.dropna(subset=['ingredients_text_pt'], inplace=True)

GRUPOS = {
    "facil": [
        "7894904219650",
        "7896005804629",
        "7896050411230",
        "7896068000075",
        "7897846901232",
        "7898067340404",
        "7898409951824",
        "7898666240068",
        "7898686950305",
        "7897517206116"
    ],

    "medio": [
        "7891095006250",
        "7896022086114",
        "7896283005664",
        "7896423480894",
        "7898614670343",
        "7891103212420",
        "7894904261154",
        "7896024800602",
        "7897517206727",
        "7898958161583"
    ],

    "dificil": [
        "5601038002209",
        "7622300800529",
        "7891025115199",
        "7891097000133",
        "7892840814175",
        "7895000483730",
        "7896003706598",
        "7896080870274",
        "7896327514114",
        "7898215157311"
    ]
}

grupos_dfs = [
    df[df["code"].isin(codigos)].assign(Grupo=nome_grupo)
    for nome_grupo, codigos in GRUPOS.items()
]

result_df = pd.concat(grupos_dfs, ignore_index=True)

result_df.to_csv("amostra.csv", sep='\t')
