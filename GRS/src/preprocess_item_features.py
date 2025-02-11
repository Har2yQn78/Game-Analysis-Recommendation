import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def load_games_data():
    return pd.read_parquet('data/games_cleaned.parquet')

def prepare_features():
    gamesdata = load_games_data()

    # --- TF-IDF features ---
    gamesdata['genres'] = gamesdata['genres'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['categories'] = gamesdata['categories'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['combined_text'] = gamesdata['genres'] + ' ' + gamesdata['categories']

    tfidf = TfidfVectorizer(max_features=200)
    tfidf_features = tfidf.fit_transform(gamesdata['combined_text'])
    # Convert to dense float32 array
    tfidf_dense = tfidf_features.toarray().astype(np.float32)

    # --- Developer feature ---
    gamesdata['developers'] = gamesdata['developers'].fillna('Unknown')
    unique_devs = gamesdata['developers'].unique().tolist()
    dev_to_idx = {dev: idx for idx, dev in enumerate(unique_devs)}
    developer_indices = gamesdata['developers'].map(lambda d: dev_to_idx[d]).values.astype(np.int64)

    # --- Metadata feature: sentiment and metacritic ---
    gamesdata['positive'] = gamesdata['positive'].fillna(0)
    gamesdata['negative'] = gamesdata['negative'].fillna(0)
    total_votes = gamesdata['positive'] + gamesdata['negative']
    sentiment = np.where(
        total_votes > 0,
        (gamesdata['positive'] - gamesdata['negative']) / total_votes,
        0
    ).astype(np.float32)
    metacritic = gamesdata['metacritic_score'].fillna(gamesdata['metacritic_score'].median())
    scaler = MinMaxScaler()
    metacritic_scaled = scaler.fit_transform(metacritic.values.reshape(-1, 1)).astype(np.float32)
    metadata = np.hstack([sentiment.reshape(-1, 1), metacritic_scaled])

    # --- Tags feature ---
    all_tags = set()
    for tag_dict in gamesdata['tags']:
        if isinstance(tag_dict, dict):
            all_tags.update(tag_dict.keys())
    all_tags = list(all_tags)
    tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}

    tag_indices = []  # flattened list of tag indices for each game
    offsets = [0]     # start index for each game in the flattened array
    for tag_dict in gamesdata['tags']:
        if isinstance(tag_dict, dict):
            for tag, votes in tag_dict.items():
                if tag in tag_to_idx:
                    tag_indices.append(tag_to_idx[tag])
        offsets.append(len(tag_indices))
    tag_indices = np.array(tag_indices, dtype=np.int64)
    offsets = np.array(offsets[:-1], dtype=np.int64)  # one offset per game

    features = {
        "tfidf": tfidf_dense,              # [n_games, tfidf_dim]
        "developer": developer_indices,    # [n_games]
        "metadata": metadata,              # [n_games, 2]
        "tags_indices": tag_indices,       # 1D array (variable length overall)
        "tags_offsets": offsets            # 1D array with one offset per game
    }
    objects = {
        "tfidf": tfidf,
        "scaler": scaler,
        "dev_to_idx": dev_to_idx,
        "all_tags": all_tags
    }
    return features, objects
