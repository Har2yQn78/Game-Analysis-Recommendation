import pandas as pd
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def load_cb_data():
    return pd.read_parquet('data/cleanedgamesdata.parquet')


class SimpleRecommender:
    def __init__(self, user_embeddings, item_embeddings, embedding_dim, device):
        self.user_embeddings = torch.tensor(user_embeddings, device=device)
        self.item_embeddings = torch.tensor(item_embeddings, device=device)
        self.embedding_dim = embedding_dim
        self.device = device

    def get_user_embedding(self, user_id):
        user_id = int(user_id)
        if user_id < 0 or user_id >= self.user_embeddings.shape[0]:
            raise ValueError(f"Invalid user_id: {user_id}")
        return self.user_embeddings[user_id]

    def get_item_embedding(self, item_id):
        if isinstance(item_id, torch.Tensor):
            item_id = item_id.cpu().numpy().astype(int)
            if np.any(item_id < 0) or np.any(item_id >= self.item_embeddings.shape[0]):
                raise ValueError(f"Invalid item_id in tensor: {item_id}")
            return self.item_embeddings[item_id]
        else:
            item_id = int(item_id)
            if item_id < 0 or item_id >= self.item_embeddings.shape[0]:
                raise ValueError(f"Invalid item_id: {item_id}")
            return self.item_embeddings[item_id]

    def eval(self):
        pass


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


def create_sparse_features(game_data, tfidf, scaler, developer_columns, all_tags):
    game_data['genres'] = game_data['genres'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['categories'] = game_data['categories'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['combined_text'] = game_data['genres'] + ' ' + game_data['categories']

    tfidf_features = tfidf.transform(game_data['combined_text'])
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
    game_data['positive'] = game_data['positive'].fillna(0)
    game_data['negative'] = game_data['negative'].fillna(0)
    total_votes = game_data['positive'] + game_data['negative']
    sentiment = np.where(
        total_votes > 0,
        (game_data['positive'] - game_data['negative']) / total_votes,
        0
    )

    metacritic = game_data['metacritic_score'].fillna(
        game_data['metacritic_score'].median()
    )
    metacritic_scaled = scaler.transform(metacritic.values.reshape(-1, 1))

    metadata = np.column_stack([sentiment, metacritic_scaled])
    return csr_matrix(metadata)


def get_recommendations(_embeddings, selected_game_idx, num_recommendations):
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

    top_indices = similarities.argsort(descending=True)[1:num_recommendations + 1]
    return top_indices.tolist()


def load_cb_model_and_preprocessing():
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


def load_or_create_features():
    feature_path = 'models/preprocessed_features.npz'
    if os.path.exists(feature_path):
        return load_npz(feature_path)

    cleanedgamesdata = load_cb_data()
    model, tfidf, scaler, developer_columns, all_tags = load_cb_model_and_preprocessing()
    features = create_sparse_features(cleanedgamesdata, tfidf, scaler, developer_columns, all_tags)

    os.makedirs('models', exist_ok=True)
    save_npz(feature_path, features)
    return features


def load_or_compute_embeddings(batch_size=1000):
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


# Data Preprocessing Functions
def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    user_ids = df[user_col].unique()
    item_ids = df[item_col].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    users = df[user_col].map(user_to_idx)
    items = df[item_col].map(item_to_idx)
    ratings = df[rating_col].values

    if norm:
        ratings = np.where(ratings > threshold, 1, 0)

    matrix = sparse.csr_matrix((ratings, (users, items)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_ids, item_ids


def create_user_dict(user_ids):
    return {uid: idx for idx, uid in enumerate(user_ids)}


def create_item_dict(df, id_col, name_col):
    return dict(zip(df[id_col], df[name_col]))


# Model Loading and Recommendation Functions
def load_model_and_embeddings(models_dir, device):
    user_embeddings = np.load(os.path.join(models_dir, 'user_embeddings.npy'))
    item_embeddings = np.load(os.path.join(models_dir, 'item_embeddings.npy'))
    embedding_dim = np.load(os.path.join(models_dir, 'embedding_dim.npy'))[0]

    return SimpleRecommender(user_embeddings, item_embeddings, embedding_dim, device)


def get_recs(model, user_id, item_ids, item_dict, num_items=5, device="cpu", user_vector=None):
    model.device = torch.device(device)

    if user_vector is not None:
        user_embedding = torch.tensor(user_vector, device=model.device).float()
    else:
        user_embedding = model.get_user_embedding(user_id).to(model.device)

    valid_item_ids = [item_id for item_id in item_ids if 0 <= item_id < model.item_embeddings.shape[0]]
    if not valid_item_ids:
        raise ValueError("No valid item_ids found.")

    batch_size = 500
    all_scores = []

    for i in range(0, len(valid_item_ids), batch_size):
        batch_items = valid_item_ids[i:i + batch_size]
        item_ids_tensor = torch.tensor(batch_items, device=model.device)
        item_embeddings = model.get_item_embedding(item_ids_tensor).to(model.device)

        chunk_size = 100
        for j in range(0, len(item_embeddings), chunk_size):
            chunk_embeddings = item_embeddings[j:j + chunk_size]
            scores = torch.matmul(user_embedding, chunk_embeddings.T)
            all_scores.extend(scores.cpu().detach().numpy().flatten())

    item_scores = pd.Series(all_scores, index=valid_item_ids)
    top_items = item_scores.nlargest(num_items).index

    return [item_dict[i] for i in top_items]


def create_item_embedding_matrix(model, interactions):
    item_ids = interactions.columns.tolist()
    item_ids_tensor = torch.tensor(item_ids, device=model.device)
    item_embeddings = model.get_item_embedding(item_ids_tensor).cpu().detach().numpy()
    similarities = cosine_similarity(item_embeddings)
    item_embedding_matrix = pd.DataFrame(similarities, index=interactions.columns, columns=interactions.columns)
    return item_embedding_matrix


def get_item_recs(item_embedding_matrix, item_id, item_dict, num_items=10, show=True):
    recommended_items = item_embedding_matrix.loc[item_id].sort_values(ascending=False).head(num_items + 1).index[
                        1:num_items + 1]
    if show:
        print(f"Item of interest: {item_dict[item_id]}")
        print("Similar items:")
        for idx, item in enumerate(recommended_items, start=1):
            print(f"{idx}- {item_dict[item]}")
    return recommended_items
