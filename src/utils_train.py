import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from sklearn.metrics.pairwise import cosine_similarity


def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    interactions = df.groupby([user_col, item_col])[rating_col] \
        .sum().unstack().reset_index() \
        .fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions


def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {i: idx for idx, i in enumerate(user_id)}
    return user_dict


def create_item_dict(df, id_col, name_col):
    item_dict = {df.loc[i, id_col]: df.loc[i, name_col] for i in range(df.shape[0])}
    return item_dict


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


def run_model(interactions, embedding_dim=32, epoch=30, batch_size=128):
    user_ids = interactions.index.tolist()
    item_ids = interactions.columns.tolist()

    # Ensure the IDs are strings
    user_ids = [str(uid) for uid in user_ids]
    item_ids = [str(iid) for iid in item_ids]

    # Create user and item models
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dim)
    ])

    item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(item_ids) + 1, embedding_dim)
    ])

    # Create and compile model
    model = RecommenderModel(user_model, item_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # Create training data generator
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

    # Create dataset
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
