import streamlit as st
import pandas as pd
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from src.utils import (create_interaction_matrix, create_user_dict,
                       create_item_dict, load_model_and_embeddings, get_recs)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import os


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


@st.cache_resource
def load_cb_model_and_preprocessing():
    """Load content-based model and preprocessing objects."""
    save_dir = 'models'

    tfidf = np.load(os.path.join(save_dir, 'tfidf.npy'), allow_pickle=True).item()
    scaler = np.load(os.path.join(save_dir, 'scaler.npy'), allow_pickle=True).item()
    developer_columns = np.load(os.path.join(save_dir, 'developer_columns.npy'), allow_pickle=True)
    all_tags = np.load(os.path.join(save_dir, 'all_tags.npy'), allow_pickle=True)

    input_dim = (tfidf.get_feature_names_out().shape[0] +
                 len(developer_columns) + 2 + len(all_tags))

    model = ContentBasedModel(input_dim)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'content_based_model.pth')))
    model.eval()

    return model, tfidf, scaler, developer_columns, all_tags


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


@st.cache_resource
def load_or_create_features():
    """Load preprocessed features from disk or create if not exists."""
    feature_path = 'models/preprocessed_features.npz'
    if os.path.exists(feature_path):
        return load_npz(feature_path)

    cleanedgamesdata = load_cb_data()
    model, tfidf, scaler, developer_columns, all_tags = load_cb_model_and_preprocessing()
    features = create_sparse_features(cleanedgamesdata, tfidf, scaler, developer_columns, all_tags)

    os.makedirs('models', exist_ok=True)
    save_npz(feature_path, features)
    return features


@st.cache_resource
def load_or_compute_embeddings(batch_size=1000):
    """Load or compute embeddings in batches."""
    embedding_path = 'models/game_embeddings.pt'
    if os.path.exists(embedding_path):
        return torch.load(embedding_path)

    features = load_or_create_features()
    model, *_ = load_cb_model_and_preprocessing()
    n_samples = features.shape[0]
    embeddings = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_features = torch.tensor(
                features[i:end_idx].toarray(),
                dtype=torch.float32
            )
            batch_embeddings = model(batch_features)
            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    os.makedirs('models', exist_ok=True)
    torch.save(embeddings, embedding_path)
    return embeddings


def create_sparse_features(game_data, tfidf, scaler, developer_columns, all_tags):
    """Create sparse feature matrix for all games."""
    game_data['genres'] = game_data['genres'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['categories'] = game_data['categories'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['combined_text'] = game_data['genres'] + ' ' + game_data['categories']

    # TF-IDF features
    tfidf_features = tfidf.transform(game_data['combined_text'])

    # Developers encoding
    dev_matrix = encode_developers(game_data['developers'].fillna('Unknown'), developer_columns)

    tags_matrix = encode_tags(game_data['tags'].fillna({}), all_tags)

    metadata_features = create_metadata_features(game_data, scaler)

    features = hstack([
        tfidf_features.multiply(1.5),
        dev_matrix.multiply(0.5),
        metadata_features,
        tags_matrix.multiply(1.2)
    ]).tocsr()

    return features


def encode_developers(developers, developer_columns):
    """Encode developers as sparse matrix."""
    n_samples = len(developers)
    n_developers = len(developer_columns)
    developer_to_idx = {dev.replace('dev_', ''): idx for idx, dev in enumerate(developer_columns)}

    rows, cols = [], []
    for idx, developer in enumerate(developers):
        if developer in developer_to_idx:
            rows.append(idx)
            cols.append(developer_to_idx[developer])

    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(n_samples, n_developers))


def encode_tags(tags_series, all_tags):
    """Encode tags as sparse matrix."""
    n_samples = len(tags_series)
    n_tags = len(all_tags)
    tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}

    rows, cols, data = [], [], []
    for i, tag_dict in enumerate(tags_series):
        if isinstance(tag_dict, dict):
            for tag, votes in tag_dict.items():
                if tag in tag_to_idx:
                    rows.append(i)
                    cols.append(tag_to_idx[tag])
                    data.append(float(votes))

    tags_matrix = csr_matrix((data, (rows, cols)), shape=(n_samples, n_tags))

    if tags_matrix.nnz > 0:
        max_val = tags_matrix.max()
        if max_val != 0:
            tags_matrix = tags_matrix / max_val

    return tags_matrix


def create_metadata_features(game_data, scaler):
    """Create metadata features (sentiment and metacritic score)."""
    game_data['positive'] = game_data['positive'].fillna(0)
    game_data['negative'] = game_data['negative'].fillna(0)
    total_votes = game_data['positive'] + game_data['negative']
    sentiment = np.where(
        total_votes > 0,
        (game_data['positive'] - game_data['negative']) / total_votes,
        0
    )

    # Scale metacritic score
    metacritic = game_data['metacritic_score'].fillna(
        game_data['metacritic_score'].median()
    )
    metacritic_scaled = scaler.transform(metacritic.values.reshape(-1, 1))

    metadata = np.column_stack([sentiment, metacritic_scaled])
    return csr_matrix(metadata)


@st.cache_data
def get_recommendations(_embeddings, selected_game_idx, num_recommendations):
    """Get recommendations using pre-computed embeddings."""
    selected_embedding = _embeddings[selected_game_idx].unsqueeze(0)

    batch_size = 1000
    n_games = len(_embeddings)
    similarities = torch.zeros(n_games)

    for i in range(0, n_games, batch_size):
        end_idx = min(i + batch_size, n_games)
        batch_similarities = torch.nn.functional.cosine_similarity(
            selected_embedding,
            _embeddings[i:end_idx]
        )
        similarities[i:end_idx] = batch_similarities

    # Get top similar games
    top_indices = similarities.argsort(descending=True)[1:num_recommendations + 1]
    return top_indices.tolist()


def main():
    st.title("Game Recommendation System")
    st.write("This app provides game recommendations based on user preferences or game content.")

    # Load all necessary data
    recdata, gamesdata, numgames = load_cf_data()
    cleanedgamesdata = load_cb_data()

    with st.spinner("Loading game embeddings..."):
        embeddings = load_or_compute_embeddings()

    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio(
        "Choose Recommendation Model",
        ("Collaborative Filtering", "Content-Based")
    )

    if model_type == "Content-Based":
        st.subheader("Content-Based Recommendations")

        # Game selection
        st.subheader("Select a Game")
        game_names = cleanedgamesdata['name'].tolist()
        selected_game_name = st.selectbox("Select a game by name", game_names)
        selected_game_idx = cleanedgamesdata[cleanedgamesdata['name'] == selected_game_name].index[0]

        initial_recommendations = 5
        similar_games_indices = get_recommendations(
            embeddings,
            selected_game_idx,
            initial_recommendations
        )

        similar_games = cleanedgamesdata.iloc[similar_games_indices]
        st.write("Top 5 Similar Games:")
        st.write(similar_games[['name', 'developers', 'release_date', 'genres']])

        if st.button("Show More"):
            additional_recommendations = get_recommendations(
                embeddings,
                selected_game_idx,
                initial_recommendations + 3
            )
            additional_games = cleanedgamesdata.iloc[additional_recommendations[initial_recommendations:]]
            st.write("Additional Similar Games:")
            st.write(additional_games[['name', 'developers', 'release_date', 'genres']])

    else:
        st.subheader("Collaborative Filtering Recommendations")

        interactions, user_ids, item_ids = create_interaction_matrix(
            df=recdata,
            user_col='uid',
            item_col='id',
            rating_col='owned'
        )
        user_dict = create_user_dict(user_ids)
        games_dict = create_item_dict(df=gamesdata, id_col='id', name_col='title')

        if len(numgames.columns) >= 2:
            numgames.columns = ['uid', 'user_id'] + list(numgames.columns[2:])
            uid_to_user_id = dict(zip(numgames['uid'], numgames['user_id']))
        else:
            st.error("The 'numgames.parquet' file must have at least two columns.")
            return

        with st.spinner("Loading collaborative filtering model..."):
            model = load_model_and_embeddings('models', torch.device("cpu"))

        st.sidebar.header("User Input")
        user_ids_list = list(user_dict.keys())
        user_id_display = [uid_to_user_id.get(uid, f"Unknown User ({uid})")
                           for uid in user_ids_list]
        selected_user_display = st.selectbox(
            "Select User",
            user_id_display,
            key="cf_user_select"
        )
        selected_uid = user_ids_list[user_id_display.index(selected_user_display)]

        num_recs = st.number_input(
            "Number of Recommendations",
            min_value=1,
            max_value=8,
            value=5,
            step=1,
            key="cf_num_input"
        )

        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                rec_list = get_recs(
                    model=model,
                    user_id=selected_uid,
                    item_ids=item_ids,
                    item_dict=games_dict,
                    num_items=num_recs,
                    device="cpu"
                )

                st.subheader(f"Top {num_recs} Recommendations for User {selected_user_display}")

                rec_df = pd.DataFrame({
                    'Game': rec_list
                }).reset_index(names=['Rank'])
                rec_df['Rank'] = rec_df['Rank'] + 1

                st.dataframe(rec_df, hide_index=True)

                if st.checkbox("Show detailed game information"):
                    for game_name in rec_list:
                        game_info = gamesdata[gamesdata['title'] == game_name].iloc[0]
                        with st.expander(f"Details for {game_name}"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Developers:**", game_info.get('developers', 'N/A'))
                                st.write("**Release Date:**", game_info.get('release_date', 'N/A'))
                            with cols[1]:
                                st.write("**Genres:**",
                                         ', '.join(game_info['genres']) if isinstance(game_info.get('genres'),
                                                                                      list) else 'N/A')
                                st.write("**Tags:**",
                                         ', '.join(list(game_info['tags'].keys())[:5]) if isinstance(
                                             game_info.get('tags'), dict) else 'N/A')


if __name__ == "__main__":
    main()
