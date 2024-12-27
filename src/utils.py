import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    user_ids = df[user_col].unique()
    item_ids = df[item_col].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    users = df[user_col].map(user_to_idx)
    items = df[item_col].map(item_to_idx)
    ratings = df[rating_col].values

    matrix = csr_matrix((ratings, (users, items)),
                        shape=(len(user_ids), len(item_ids)))

    return matrix, user_ids, item_ids


def create_user_dict(user_ids):
    return {uid: idx for idx, uid in enumerate(user_ids)}


def create_item_dict(df, id_col, name_col):
    return dict(zip(df[id_col], df[name_col]))


class RecommenderModel(tfrs.Model):
    def __init__(self, user_model, item_model):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["item_id"])

        return self.task(
            user_embeddings,
            item_embeddings,
            compute_metrics=not training
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "user_model": tf.keras.layers.serialize(self.user_model),
            "item_model": tf.keras.layers.serialize(self.item_model),
        })
        return config

    @classmethod
    def from_config(cls, config):
        user_model = tf.keras.layers.deserialize(config.pop("user_model"))
        item_model = tf.keras.layers.serialize(config.pop("item_model"))
        return cls(user_model=user_model, item_model=item_model)


def run_model(interactions, embedding_dim=32, epoch=30, batch_size=128):
    user_ids = interactions.index.tolist()
    item_ids = interactions.columns.tolist()

    user_ids = [str(uid) for uid in user_ids]
    item_ids = [str(iid) for iid in item_ids]

    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dim)
    ])

    item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(item_ids) + 1, embedding_dim)
    ])

    model = RecommenderModel(user_model, item_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    def data_generator():
        while True:
            for user in interactions.index:
                user_items = interactions.loc[user]
                positive_items = user_items[user_items > 0].index
                if len(positive_items) > 0:
                    for item in positive_items:
                        yield {
                            "user_id": str(user),
                            "item_id": str(item)
                        }

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            "user_id": tf.TensorSpec(shape=(), dtype=tf.string),
            "item_id": tf.TensorSpec(shape=(), dtype=tf.string)
        }
    )

    n_positive = sum(interactions.values > 0).sum()
    steps_per_epoch = n_positive // batch_size

    dataset = dataset.shuffle(10000).batch(batch_size).cache()
    model.fit(dataset, epochs=epoch, steps_per_epoch=steps_per_epoch)

    return model


def get_recs(model, user_id, item_ids, item_dict, num_items=5):
    user_embedding = model.user_model(tf.constant([str(user_id)]))

    batch_size = 1000
    all_scores = []

    for i in range(0, len(item_ids), batch_size):
        batch_items = [str(x) for x in item_ids[i:i + batch_size]]
        item_embeddings = model.item_model(tf.constant(batch_items))
        scores = tf.linalg.matmul(user_embedding, item_embeddings, transpose_b=True)
        all_scores.extend(scores.numpy().flatten())

    # Get top recommendations
    item_scores = pd.Series(all_scores, index=item_ids)
    top_items = item_scores.nlargest(num_items).index

    return [item_dict[i] for i in top_items]


def create_item_embedding_matrix(model, interactions):
    item_embeddings = model.item_model(tf.convert_to_tensor(interactions.columns.tolist())).numpy()
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
