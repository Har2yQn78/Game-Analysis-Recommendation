import streamlit as st
import pandas as pd
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
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


@st.cache_data
def load_cf_data():
    """Load collaborative filtering data from parquet files."""
    recdata = pd.read_parquet('data/recdata.parquet')
    recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
    gamesdata = pd.read_parquet('data/gamesdata.parquet')
    numgames = pd.read_parquet('data/numgames.parquet')
    return recdata, gamesdata, numgames


@st.cache_data
def load_cb_data():
    """Load content-based filtering data from parquet file."""
    return pd.read_parquet('data/cleanedgamesdata.parquet')


@st.cache_data
def load_cf_model():
    """Load collaborative filtering model."""
    models_dir = 'models'
    device = torch.device("cpu")  # Force CPU usage
    return load_model_and_embeddings(models_dir, device)


@st.cache_resource
def load_cb_model_and_preprocessing():
    """Load content-based model and preprocessing objects."""
    save_dir = 'models'

    # Load preprocessing objects
    tfidf = np.load(os.path.join(save_dir, 'tfidf.npy'), allow_pickle=True).item()
    scaler = np.load(os.path.join(save_dir, 'scaler.npy'), allow_pickle=True).item()
    developer_columns = np.load(os.path.join(save_dir, 'developer_columns.npy'), allow_pickle=True)
    all_tags = np.load(os.path.join(save_dir, 'all_tags.npy'), allow_pickle=True)

    # Calculate input dimension
    input_dim = (tfidf.get_feature_names_out().shape[0] +
                 len(developer_columns) + 2 + len(all_tags))

    # Initialize and load model
    model = ContentBasedModel(input_dim)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'content_based_model.pth')))
    model.eval()

    return model, tfidf, scaler, developer_columns, all_tags


def preprocess_input(game_data, tfidf, scaler, developer_columns, all_tags):
    """Preprocess input data efficiently using sparse matrices."""
    # Basic preprocessing
    game_data['genres'] = game_data['genres'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['tags'] = game_data['tags'].fillna('').apply(
        lambda x: x if isinstance(x, dict) else {})
    game_data['categories'] = game_data['categories'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['developers'] = game_data['developers'].fillna('Unknown')
    game_data['metacritic_score'] = game_data['metacritic_score'].fillna(
        game_data['metacritic_score'].median())

    # Calculate sentiment
    game_data['total_votes'] = game_data['positive'] + game_data['negative']
    game_data['sentiment'] = (game_data['positive'] - game_data['negative']) / game_data['total_votes']
    game_data['sentiment'] = game_data['sentiment'].fillna(0)

    # Scale metacritic score
    game_data['metacritic_score'] = scaler.transform(game_data[['metacritic_score']])

    # Get TF-IDF features (sparse)
    game_data['combined_text'] = game_data['genres'] + ' ' + game_data['categories']
    tfidf_features = tfidf.transform(game_data['combined_text'])

    # Create sparse developers matrix
    n_samples = len(game_data)
    n_developers = len(developer_columns)
    developer_to_idx = {dev.replace('dev_', ''): idx for idx, dev in enumerate(developer_columns)}

    dev_rows, dev_cols, dev_data = [], [], []
    for idx, developer in enumerate(game_data['developers']):
        if developer in developer_to_idx:
            dev_rows.append(idx)
            dev_cols.append(developer_to_idx[developer])
            dev_data.append(1)

    developers_encoded = csr_matrix(
        (dev_data, (dev_rows, dev_cols)),
        shape=(n_samples, n_developers)
    )

    # Create sparse tags matrix
    tag_rows, tag_cols, tag_data = [], [], []
    for i, tag_dict in enumerate(game_data['tags']):
        if isinstance(tag_dict, dict):
            for tag, votes in tag_dict.items():
                if tag in all_tags:
                    tag_rows.append(i)
                    tag_cols.append(all_tags.index(tag))
                    tag_data.append(float(votes))

    tags_features = csr_matrix(
        (tag_data, (tag_rows, tag_cols)),
        shape=(n_samples, len(all_tags))
    )

    # Normalize tags if present
    if len(all_tags) > 0 and tags_features.nnz > 0:
        max_val = tags_features.max()
        if max_val != 0:
            tags_features = tags_features / max_val

    # Convert sentiment and metacritic to sparse
    sentiment = csr_matrix(game_data[['sentiment']].values)
    metacritic_score = csr_matrix(game_data[['metacritic_score']].values)

    # Apply feature weights
    tfidf_features = tfidf_features.multiply(1.5)
    developers_encoded = developers_encoded.multiply(0.5)
    tags_features = tags_features.multiply(1.2)

    # Combine features
    features = hstack([
        tfidf_features,
        developers_encoded,
        sentiment,
        metacritic_score,
        tags_features
    ]).tocsr()

    # Convert to float32
    features = features.astype(np.float32)
    return features.toarray()


def get_content_based_recommendations(selected_game, cleanedgamesdata, model, tfidf,
                                      scaler, developer_columns, all_tags, num_recommendations):
    """Get content-based recommendations using batched processing."""
    # Process selected game
    selected_features = preprocess_input(
        pd.DataFrame([selected_game]),
        tfidf, scaler, developer_columns, all_tags
    )

    with torch.no_grad():
        selected_embedding = model(torch.tensor(selected_features, dtype=torch.float32))

    # Process games in batches
    batch_size = 1000
    num_games = len(cleanedgamesdata)
    similarities = torch.zeros(num_games)

    with torch.no_grad():
        for i in range(0, num_games, batch_size):
            batch_end = min(i + batch_size, num_games)
            batch_data = cleanedgamesdata.iloc[i:batch_end]

            batch_features = preprocess_input(
                batch_data, tfidf, scaler, developer_columns, all_tags
            )
            batch_embeddings = model(torch.tensor(batch_features, dtype=torch.float32))

            batch_similarities = torch.nn.functional.cosine_similarity(
                selected_embedding,
                batch_embeddings
            )

            similarities[i:batch_end] = batch_similarities

    # Get top similar games
    top_indices = similarities.argsort(descending=True)[1:num_recommendations + 1]
    return cleanedgamesdata.iloc[top_indices]


def main():
    st.title("Game Recommendation System")
    st.write("This app provides game recommendations based on user preferences or game content.")

    # Load data
    recdata, gamesdata, numgames = load_cf_data()
    cleanedgamesdata = load_cb_data()

    # Model selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio(
        "Choose Recommendation Model",
        ("Collaborative Filtering", "Content-Based")
    )

    if model_type == "Collaborative Filtering":
        st.subheader("Collaborative Filtering Recommendations")

        # Create interaction matrix and dictionaries
        interactions, user_ids, item_ids = create_interaction_matrix(
            df=recdata,
            user_col='uid',
            item_col='id',
            rating_col='owned'
        )
        user_dict = create_user_dict(user_ids)
        games_dict = create_item_dict(df=gamesdata, id_col='id', name_col='title')

        # Handle numgames columns
        if len(numgames.columns) >= 2:
            numgames.columns = ['uid', 'user_id'] + list(numgames.columns[2:])
        else:
            st.error("The 'numgames.parquet' file must have at least two columns.")
            return

        uid_to_user_id = dict(zip(numgames['uid'], numgames['user_id']))

        # Load model
        model = load_cf_model()

        # User selection
        st.sidebar.header("User Input")
        user_ids_list = list(user_dict.keys())
        user_id_display = [uid_to_user_id.get(uid, f"Unknown User ({uid})")
                           for uid in user_ids_list]
        selected_user_display = st.sidebar.selectbox("Select User", user_id_display)
        selected_uid = user_ids_list[user_id_display.index(selected_user_display)]

        num_recs = st.sidebar.number_input(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5
        )

        if st.sidebar.button("Get User Recommendations"):
            st.subheader(f"Recommendations for User {selected_user_display}")
            rec_list = get_recs(
                model=model,
                user_id=selected_uid,
                item_ids=item_ids,
                item_dict=games_dict,
                num_items=num_recs,
                device="cpu"
            )
            for i, rec in enumerate(rec_list, start=1):
                st.write(f"{i}. {rec}")

    elif model_type == "Content-Based":
        st.subheader("Content-Based Recommendations")

        # Load model and preprocessing objects
        model, tfidf, scaler, developer_columns, all_tags = load_cb_model_and_preprocessing()

        # Game selection
        st.subheader("Select a Game")
        game_names = cleanedgamesdata['name'].tolist()
        selected_game_name = st.selectbox("Select a game by name", game_names)
        selected_game = cleanedgamesdata[cleanedgamesdata['name'] == selected_game_name].iloc[0]

        # Get recommendations
        num_similar_games = st.slider(
            "Number of similar games to display",
            min_value=1,
            max_value=8,
            value=5
        )

        similar_games = get_content_based_recommendations(
            selected_game,
            cleanedgamesdata,
            model,
            tfidf,
            scaler,
            developer_columns,
            all_tags,
            num_similar_games
        )

        st.write(similar_games[['name', 'developers', 'release_date', 'genres']])


if __name__ == "__main__":
    main()