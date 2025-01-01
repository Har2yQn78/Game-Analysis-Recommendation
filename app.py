import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.utils import (create_interaction_matrix, create_user_dict,
                       create_item_dict, load_model_and_embeddings, get_recs)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import os


class ContentBasedModel(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(ContentBasedModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load data
@st.cache_data
def load_data():
    recdata = pd.read_parquet('data/recdata.parquet')  # Read .parquet file
    recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
    gamesdata = pd.read_parquet('data/gamesdata.parquet')  # Read .parquet file
    numgames = pd.read_parquet('data/numgames.parquet')  # Read .parquet file
    return recdata, gamesdata, numgames


@st.cache_data
def load_cf_model():
    models_dir = 'models'
    device = torch.device("cpu")  # Force CPU usage
    model = load_model_and_embeddings(models_dir, device)
    return model


@st.cache_resource
def load_cb_model_and_preprocessing():
    save_dir = 'models'

    tfidf = np.load(os.path.join(save_dir, 'tfidf.npy'), allow_pickle=True).item()
    scaler = np.load(os.path.join(save_dir, 'scaler.npy'), allow_pickle=True).item()
    developer_columns = np.load(os.path.join(save_dir, 'developer_columns.npy'), allow_pickle=True)

    input_dim = tfidf.get_feature_names_out().shape[0] + len(developer_columns) + 2  # +2 for sentiment and metascore

    model = ContentBasedModel(input_dim)

    model.load_state_dict(torch.load(os.path.join(save_dir, 'content_based_model.pth')))
    model.eval()

    return model, tfidf, scaler, developer_columns


def preprocess_input(game_data, tfidf, scaler, developer_columns):
    game_data['genres'] = game_data['genres'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['tags'] = game_data['tags'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['specs'] = game_data['specs'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['developer'] = game_data['developer'].fillna('Unknown')
    game_data['metascore'] = game_data['metascore'].fillna(game_data['metascore'].median())

    sentiment_mapping = {
        "Very Negative": 0,
        "Overwhelmingly Negative": 0,
        "Negative": 1,
        "Mostly Negative": 1,
        "Mixed": 2,
        "1 user reviews": 3,
        "2 user reviews": 3,
        "3 user reviews": 3,
        "4 user reviews": 3,
        "5 user reviews": 3,
        "6 user reviews": 3,
        "7 user reviews": 3,
        "8 user reviews": 3,
        "9 user reviews": 3,
        "Positive": 4,
        "Mostly Positive": 4,
        "Very Positive": 5,
        "Overwhelmingly Positive": 5,
        None: 3
    }
    game_data['sentiment'] = game_data['sentiment'].map(sentiment_mapping).fillna(3)

    game_data['metascore'] = scaler.transform(game_data[['metascore']])

    game_data['combined_text'] = game_data['genres'] + ' ' + game_data['tags'] + ' ' + game_data['specs']

    tfidf_features = tfidf.transform(game_data['combined_text']).toarray()

    developer_encoded = pd.get_dummies(game_data['developer'], prefix='dev')
    developer_encoded = developer_encoded.reindex(columns=developer_columns, fill_value=0)

    tfidf_features *= 2.0
    developer_encoded *= 0.5
    sentiment = game_data[['sentiment']].values * 1.0
    metascore = game_data[['metascore']].values * 1.0

    features = np.hstack([tfidf_features, developer_encoded, sentiment, metascore])

    features = features.astype(np.float32)
    return features


def main():
    st.title("Game Recommendation System")
    st.write("This app provides game recommendations based on user preferences or game content.")

    recdata, gamesdata, numgames = load_data()

    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio("Choose Recommendation Model", ("Collaborative Filtering", "Content-Based"))

    if model_type == "Collaborative Filtering":
        st.subheader("Collaborative Filtering Recommendations")
        interactions, user_ids, item_ids = create_interaction_matrix(df=recdata, user_col='uid', item_col='id',
                                                                     rating_col='owned')
        user_dict = create_user_dict(user_ids)
        games_dict = create_item_dict(df=gamesdata, id_col='id', name_col='title')

        if len(numgames.columns) >= 2:
            numgames.columns = ['uid', 'user_id'] + list(numgames.columns[2:])
        else:
            st.error("The 'numgames.parquet' file must have at least two columns.")
            return

        uid_to_user_id = dict(zip(numgames['uid'], numgames['user_id']))

        model = load_cf_model()

        st.sidebar.header("User Input")

        user_ids_list = list(user_dict.keys())
        user_id_display = [uid_to_user_id.get(uid, f"Unknown User ({uid})") for uid in user_ids_list]
        selected_user_display = st.sidebar.selectbox("Select User", user_id_display)

        selected_uid = user_ids_list[user_id_display.index(selected_user_display)]

        num_recs = st.sidebar.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

        if st.sidebar.button("Get User Recommendations"):
            st.subheader(f"Recommendations for User {selected_user_display}")
            rec_list = get_recs(model=model, user_id=selected_uid, item_ids=item_ids, item_dict=games_dict,
                                num_items=num_recs, device="cpu")
            for i, rec in enumerate(rec_list, start=1):
                st.write(f"{i}. {rec}")

    elif model_type == "Content-Based":
        st.subheader("Content-Based Recommendations")
        model, tfidf, scaler, developer_columns = load_cb_model_and_preprocessing()

        st.subheader("Select a Game")
        game_names = gamesdata['title'].tolist()  # Get the list of game names
        selected_game_name = st.selectbox("Select a game by name", game_names)

        selected_game = gamesdata[gamesdata['title'] == selected_game_name].iloc[0]

        features = preprocess_input(pd.DataFrame([selected_game]), tfidf, scaler, developer_columns)

        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            embedding = model(features_tensor)

        st.subheader("Similar Games")
        with torch.no_grad():
            all_features = preprocess_input(gamesdata, tfidf, scaler, developer_columns)
            all_features_tensor = torch.tensor(all_features, dtype=torch.float32)
            all_embeddings = model(all_features_tensor)
            similarities = torch.nn.functional.cosine_similarity(embedding, all_embeddings)
            top_indices = similarities.argsort(descending=True)[1:]  # Exclude itself

            num_similar_games = st.slider("Number of similar games to display", min_value=1, max_value=8, value=5)
            similar_games = gamesdata.iloc[top_indices[:num_similar_games]]

            st.write(similar_games[['title', 'developer', 'release_date', 'genres']])


if __name__ == "__main__":
    main()
