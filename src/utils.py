import pandas as pd
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import os


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
            # Check if any item_id is out of bounds
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


# Data Preprocessing Functions
def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    """
    Creates a sparse interaction matrix from user-item interaction data.
    """
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
    """
    Creates a user dictionary mapping user IDs to indices.
    """
    return {uid: idx for idx, uid in enumerate(user_ids)}


def create_item_dict(df, id_col, name_col):
    """
    Creates an item dictionary mapping item IDs to item names.
    """
    return dict(zip(df[id_col], df[name_col]))


# Model Loading and Recommendation Functions
def load_model_and_embeddings(models_dir, device):
    """
    Load the saved model and embeddings.
    """
    user_embeddings = np.load(os.path.join(models_dir, 'user_embeddings.npy'))
    item_embeddings = np.load(os.path.join(models_dir, 'item_embeddings.npy'))
    embedding_dim = np.load(os.path.join(models_dir, 'embedding_dim.npy'))[0]

    return SimpleRecommender(user_embeddings, item_embeddings, embedding_dim, device)


def get_recs(model, user_id, item_ids, item_dict, num_items=5, device="cpu", user_vector=None):
    """
    Generate recommendations for a user.
    """
    # Ensure the model is on the correct device
    model.device = torch.device(device)

    if user_vector is not None:
        user_embedding = torch.tensor(user_vector, device=model.device).float()
    else:
        user_embedding = model.get_user_embedding(user_id).to(model.device)

    # Filter out invalid item_ids
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
    """
    Create an item-item similarity matrix using cosine similarity.
    """
    item_ids = interactions.columns.tolist()
    item_ids_tensor = torch.tensor(item_ids, device=model.device)
    item_embeddings = model.get_item_embedding(item_ids_tensor).cpu().detach().numpy()
    similarities = cosine_similarity(item_embeddings)
    item_embedding_matrix = pd.DataFrame(similarities, index=interactions.columns, columns=interactions.columns)
    return item_embedding_matrix


def get_item_recs(item_embedding_matrix, item_id, item_dict, num_items=10, show=True):
    """
    Get similar items for a given item.
    """
    recommended_items = item_embedding_matrix.loc[item_id].sort_values(ascending=False).head(num_items + 1).index[
                        1:num_items + 1]
    if show:
        print(f"Item of interest: {item_dict[item_id]}")
        print("Similar items:")
        for idx, item in enumerate(recommended_items, start=1):
            print(f"{idx}- {item_dict[item]}")
    return recommended_items
