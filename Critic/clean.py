import pandas as pd
from langdetect import detect


def clean_review_data(df, dropna=True, drop_duplicated=True, remove_spoiler=True,
                      all_languages=False, selected_languages=['en']):
    """
    Cleans the DataFrame of reviews:
      - drops NA values and duplicates,
      - removes standardized spoiler messages,
      - and optionally filters by language.
    """
    if dropna:
        df = df.dropna()
    if drop_duplicated:
        df = df.drop_duplicates()
    if remove_spoiler:
        df = df[df['review_text'] != "[SPOILER ALERT: This review contains spoilers.]"]

    if not all_languages:
        languages = []
        for text in df['review_text']:
            try:
                languages.append(detect(text))
            except Exception:
                languages.append('unknown')
        df['language'] = languages
        df = df[df['language'].isin(selected_languages)]

    if len(selected_languages) <= 1 and 'language' in df.columns:
        df.drop(columns=['language'], inplace=True)

    return df
