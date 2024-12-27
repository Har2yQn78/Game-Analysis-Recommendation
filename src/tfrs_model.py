import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from utils import create_interaction_matrix, run_model, create_user_dict, create_item_dict

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Data
recdata_path = os.path.join('..', 'data', 'recdata.csv')
gamesdata_path = os.path.join('..', 'data', 'gamesdata.csv')

recdata = pd.read_csv(recdata_path, index_col=0)
recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
gamesdata = pd.read_csv(gamesdata_path, index_col=0)

interactions = create_interaction_matrix(df=recdata, user_col='uid', item_col='id', rating_col='owned')

interactions.index = interactions.index.astype(str)
interactions.columns = interactions.columns.astype(str)

# Train the model
recommender_model = run_model(interactions=interactions, embedding_dim=32, epoch=30, batch_size=128)

save_dir = os.path.join('..', 'models')
os.makedirs(save_dir, exist_ok=True)

dummy_input = tf.constant([interactions.index[0]])
_ = recommender_model.user_model(dummy_input)
user_embeddings = recommender_model.user_model.layers[1].embeddings.numpy()

dummy_input = tf.constant([interactions.columns[0]])
_ = recommender_model.item_model(dummy_input)
item_embeddings = recommender_model.item_model.layers[1].embeddings.numpy()

np.save(os.path.join(save_dir, 'user_embeddings.npy'), user_embeddings)
np.save(os.path.join(save_dir, 'item_embeddings.npy'), item_embeddings)
np.save(os.path.join(save_dir, 'user_ids.npy'), np.array(interactions.index.tolist(), dtype=object))
np.save(os.path.join(save_dir, 'item_ids.npy'), np.array(interactions.columns.tolist(), dtype=object))
np.save(os.path.join(save_dir, 'embedding_dim.npy'), np.array([32], dtype=np.int32))
