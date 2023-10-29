import re

from unidecode import unidecode


def clean_text(text):
    cleaned = unidecode(text.lower().strip())
    cleaned = re.sub(r'[^a-zA-Z\s,]', ' ', cleaned)
    cleaned = re.sub(r',+', ',', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned


def preprocess_data(df):
    df.dropna(subset=['ingredients_text_pt'], inplace=True)
    df['ingredients_text_pt'] = df['ingredients_text_pt'].apply(clean_text)
    return df


