import streamlit as st
import torch
import pandas as pd
import numpy as np
import pickle
import ast
from model import GameRecommendationModel


@st.cache_resource
def load_model():
    with open("vocabularies.pkl", "rb") as f:
        vocabs = pickle.load(f)

    dev_vocab_size = len(vocabs['dev_vocab'])
    pub_vocab_size = len(vocabs['pub_vocab'])
    gen_vocab_size = len(vocabs['gen_vocab'])
    tag_vocab_size = len(vocabs['tag_vocab'])

    model = GameRecommendationModel(dev_vocab_size, pub_vocab_size, gen_vocab_size, tag_vocab_size)
    model.load_state_dict(torch.load("game_recommendation_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, vocabs


@st.cache_data
def load_data():
    df = pd.read_parquet("data/games_cleaned.parquet")
    return df


def process_input(row, vocabs):
    """
    Process a DataFrame row into the feature dictionary expected by the model.
    """

    def parse_list(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except:
            return []

    dev = vocabs['dev_vocab'].get(row['developers'], 0)
    pub = vocabs['pub_vocab'].get(row['publishers'], 0)
    gens_list = parse_list(row['genres'])
    gens = [vocabs['gen_vocab'].get(g, 0) for g in gens_list if g in vocabs['gen_vocab']]
    if not gens:
        gens = [0]
    tag_dict = row['tags']
    if isinstance(tag_dict, str):
        try:
            tag_dict = ast.literal_eval(tag_dict)
        except:
            tag_dict = {}
    elif not isinstance(tag_dict, dict):
        tag_dict = {}
    sorted_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True) if tag_dict else []
    top_tags = [tag for tag, _ in sorted_tags[:5]]
    if not top_tags:
        top_tags = [0]
    tags = [vocabs['tag_vocab'].get(t, 0) for t in top_tags]
    numeric = [row['pct_pos_total'], row['num_reviews_total']]

    return {
        'developer': dev,
        'publisher': pub,
        'genres': gens,
        'tags': tags,
        'numeric': numeric
    }


def get_embedding(features, model):
    """
    Given a features dictionary, convert to tensors and obtain the game embedding.
    """
    dev = torch.tensor([features['developer']], dtype=torch.long)
    pub = torch.tensor([features['publisher']], dtype=torch.long)
    gens = torch.tensor([features['genres']], dtype=torch.long)
    tags = torch.tensor([features['tags']], dtype=torch.long)
    numeric = torch.tensor([features['numeric']], dtype=torch.float)

    with torch.no_grad():
        emb = model(dev, pub, gens, tags, numeric)
    return emb.numpy().squeeze()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    st.title("Game Recommendation System")
    model, vocabs = load_model()
    df = load_data()

    game_names = df['name'].tolist()
    selected_game = st.selectbox("Select a game", game_names)

    row = df[df['name'] == selected_game].iloc[0]
    features = process_input(row, vocabs)
    anchor_emb = get_embedding(features, model)

    @st.cache_data
    def compute_all_embeddings():
        embeddings = []
        for _, row in df.iterrows():
            feats = process_input(row, vocabs)
            emb = get_embedding(feats, model)
            embeddings.append(emb)
        return np.array(embeddings)

    all_embeddings = compute_all_embeddings()

    sims = np.array([cosine_sim(anchor_emb, emb) for emb in all_embeddings])

    indices = sims.argsort()[::-1]
    recommendations = []
    count = 0
    for idx in indices:
        if df.iloc[idx]['name'] == selected_game:
            continue
        recommendations.append(df.iloc[idx]['name'])
        count += 1
        if count >= 5:
            break

    st.subheader("Recommended Games:")
    for rec in recommendations:
        st.write(rec)


if __name__ == "__main__":
    main()
