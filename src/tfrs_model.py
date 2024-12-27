import tensorflow as tf
import tensorflow_recommenders as tfrs


class RecommenderModel(tfrs.Model):
    def __init__(self, user_model, item_model, task):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["item_id"])
        return self.task(user_embeddings, item_embeddings)

def run_model(interactions, embedding_dim=32, epoch=30, batch_size=128):
    user_ids = interactions.index.tolist()
    item_ids = interactions.columns.tolist()
    ratings = interactions.values

    user_model = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=user_ids), tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dim)])
    item_model = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=item_ids), tf.keras.layers.Embedding(len(item_ids) + 1, embedding_dim)])

    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=item_ids))

    model = RecommenderModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    users = np.array([user for user in interactions.index for _ in interactions.columns])
    items = np.array([item for _ in interactions.index for item in interactions.columns])
    ratings = np.array([rating for row in interactions.values for rating in row])

    dataset = tf.data.Dataset.from_tensor_slices({"user_id": users, "item_id": items, "rating": ratings})
    dataset = dataset.shuffle(len(users)).batch(batch_size).cache()

    model.fit(dataset, epochs=epoch)

    return model
