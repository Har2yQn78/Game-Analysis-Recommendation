# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
from torchrec_item_model import GameEmbeddingModel
from preprocess_item_features import prepare_features

# --- Load game metadata (e.g., a CSV mapping game index to title) ---
game_metadata = pd.read_csv('data/games_cleaned.parquet')  # columns: game_id, title

# --- Load preprocessed features and related objects ---
features_np, proc_objs = prepare_features()  # numpy arrays and dictionaries

# Convert numpy arrays into torch tensors.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_torch = {
    "tfidf": torch.tensor(features_np["tfidf"], dtype=torch.float32, device=device),
    "developer": torch.tensor(features_np["developer"], dtype=torch.long, device=device),
    "metadata": torch.tensor(features_np["metadata"], dtype=torch.float32, device=device),
    # For tags, we assume the arrays were saved as numpy arrays:
    "tags_indices": torch.tensor(features_np["tags_indices"], dtype=torch.long, device=device),
    "tags_offsets": torch.tensor(features_np["tags_offsets"], dtype=torch.long, device=device)
}

# --- Initialize and load the trained model ---
tfidf_dim = features_np["tfidf"].shape[1]
num_developers = len(proc_objs["dev_to_idx"])
num_tags = len(proc_objs["all_tags"])
model = GameEmbeddingModel(tfidf_dim, num_developers, num_tags, metadata_dim=2, embedding_dim=128)
model.load_state_dict(torch.load('models/torchrec_game_recommender_item.pth', map_location=device))
model.to(device)
model.eval()

def recommend_similar_games(selected_idx, num_recs=5):
    similar_indices = get_similar_games(model, features_torch, selected_idx, num_recommendations=num_recs, device=device)
    # Map indices to game titles using your metadata DataFrame.
    return game_metadata.iloc[similar_indices]['title'].tolist()

# --- Streamlit UI ---
st.title("Similar Game Recommender")
st.write("Enter a game index to find similar games:")

selected_game_idx = st.number_input("Game Index", min_value=0, max_value=features_torch["tfidf"].shape[0]-1, value=0, step=1)
if st.button("Get Similar Games"):
    recs = recommend_similar_games(selected_game_idx, num_recs=5)
    st.write("Games similar to game index", selected_game_idx, ":")
    for rec in recs:
        st.write(rec)
