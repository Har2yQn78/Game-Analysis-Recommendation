import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from src.utils import (create_interaction_matrix, create_user_dict,
                       create_item_dict, get_recs)

# Load Data
recdata = pd.read_csv('data/recdata.csv', index_col=0)
recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
gamesdata = pd.read_csv('data/gamesdata.csv', index_col=0)

interactions, user_ids, item_ids = create_interaction_matrix(
    df=recdata, user_col='uid', item_col='id', rating_col='owned'
)

user_dict = create_user_dict(user_ids)
games_dict = create_item_dict(gamesdata, 'id', 'title')

models_dir = 'models'
user_embeddings = np.load(os.path.join(models_dir, 'user_embeddings.npy'))
item_embeddings = np.load(os.path.join(models_dir, 'item_embeddings.npy'))
embedding_dim = np.load(os.path.join(models_dir, 'embedding_dim.npy'))[0]

user_embedding_layer = tf.keras.layers.Embedding(
    input_dim=user_embeddings.shape[0],
    output_dim=embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(user_embeddings),
    trainable=False
)

item_embedding_layer = tf.keras.layers.Embedding(
    input_dim=item_embeddings.shape[0],
    output_dim=embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(item_embeddings),
    trainable=False
)

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=[str(x) for x in user_ids], mask_token=None),
    user_embedding_layer
])

item_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=[str(x) for x in item_ids], mask_token=None),
    item_embedding_layer
])


class SimpleRecommender:
    def __init__(self, user_model, item_model):
        self.user_model = user_model
        self.item_model = item_model


recommender_model = SimpleRecommender(user_model, item_model)

# Streamlit UI
st.title('Game Recommendation System')

# Create reverse mapping from game title to ID
title_to_id = {v: k for k, v in games_dict.items()}

# Sidebar with two tabs
tab1, tab2 = st.tabs(["User-based Recommendations", "Game-based Recommendations"])

with tab1:
    user_options = [f"User {uid}" for uid in sorted(user_ids)]
    selected_user = st.selectbox('Select User', user_options)
    user_id = int(selected_user.split()[1])

    if st.button('Get User Recommendations'):
        recommendations = get_recs(
            model=recommender_model,
            user_id=user_id,
            item_ids=item_ids,
            item_dict=games_dict,
            num_items=5
        )

        st.write(f"Recommendations for {selected_user}:")
        for idx, game in enumerate(recommendations, 1):
            st.write(f"{idx}. {game}")

with tab2:
    try:
        game_titles = sorted(games_dict.values(), key=str)  # Convert all values to string before sorting
    except TypeError as e:
        st.error(f"Error occurred while sorting games: {e}")
        game_titles = []
    selected_games = st.multiselect('Select Games', game_titles)

    if st.button('Get Similar Games') and selected_games:
        similar_games = set()
        for game in selected_games:
            game_id = title_to_id[game]
            recs = get_recs(
                model=recommender_model,
                user_id=game_id,  # Using game_id as user_id for similarity
                item_ids=item_ids,
                item_dict=games_dict,
                num_items=3
            )
            similar_games.update(recs)

        # Remove selected games from recommendations
        similar_games = similar_games - set(selected_games)

        st.write("Similar games:")
        for idx, game in enumerate(similar_games, 1):
            st.write(f"{idx}. {game}")