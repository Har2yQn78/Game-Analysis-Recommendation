import pandas as pd
import re
from langdetect import detect

def clean_text(text):
    """Clean and preprocess the review text."""
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters (except punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common review boilerplate
    text = re.sub(r'reviewed by.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'review copy provided.*?(\.|$)', '', text, flags=re.IGNORECASE)
    return text

def clean_review_data(df, dropna=True, drop_duplicated=True, remove_spoiler=True,
                      all_languages=False, selected_languages=['en']):
    """
    Clean the DataFrame containing reviews.
    """
    if dropna:
        df = df.dropna(subset=['review_text'])
    if drop_duplicated:
        df = df.drop_duplicates(subset=['review_text'])
    df['review_text'] = df['review_text'].apply(clean_text)
    # Remove very short reviews
    df = df[df['review_text'].str.len() > 100]
    if remove_spoiler:
        df = df[~df['review_text'].str.contains('spoiler alert|spoiler warning', case=False)]
    if not all_languages:
        try:
            languages = []
            for text in df['review_text']:
                try:
                    languages.append(detect(text))
                except:
                    languages.append('unknown')
            df['language'] = languages
            df = df[df['language'].isin(selected_languages)]
            if len(selected_languages) <= 1:
                df.drop(columns=['language'], inplace=True)
        except Exception as e:
            print(f"Language detection error: {e}")
    df = df.reset_index(drop=True)
    return df
