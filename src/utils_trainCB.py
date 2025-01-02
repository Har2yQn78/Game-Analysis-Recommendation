import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, hstack, csc_matrix
from tqdm import tqdm


class ContentBasedDataset(Dataset):
    def __init__(self, features):
        self.features = features  # Keep as sparse matrix

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Convert only the required batch to dense
        start_idx = idx
        end_idx = start_idx + 1
        batch_dense = self.features[start_idx:end_idx].toarray()
        return torch.tensor(batch_dense, dtype=torch.float32).squeeze(0)


class ContentBasedModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(ContentBasedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess_data(gamesdata):
    print("Preprocessing data...")
    # Fill missing values
    gamesdata['genres'] = gamesdata['genres'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['tags'] = gamesdata['tags'].fillna('').apply(lambda x: x if isinstance(x, dict) else {})
    gamesdata['categories'] = gamesdata['categories'].fillna('').apply(
        lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['developers'] = gamesdata['developers'].fillna('Unknown')
    gamesdata['metacritic_score'] = gamesdata['metacritic_score'].fillna(gamesdata['metacritic_score'].median())

    # Handle positive and negative votes to derive sentiment
    gamesdata['total_votes'] = gamesdata['positive'] + gamesdata['negative']
    gamesdata['sentiment'] = (gamesdata['positive'] - gamesdata['negative']) / gamesdata['total_votes']
    gamesdata['sentiment'] = gamesdata['sentiment'].fillna(0)

    # Normalize metacritic_score
    scaler = MinMaxScaler()
    normalized_score = scaler.fit_transform(gamesdata[['metacritic_score']])

    # Combine textual features
    gamesdata['combined_text'] = gamesdata['genres'] + ' ' + gamesdata['categories']

    # TF-IDF with reduced features
    print("Applying TF-IDF to combined text...")
    tfidf = TfidfVectorizer(max_features=200)  # Reduced from 500
    tfidf_features = tfidf.fit_transform(gamesdata['combined_text'])

    # Memory-efficient one-hot encoding for developers
    print("One-hot encoding developers column...")
    unique_developers = gamesdata['developers'].unique()
    developer_to_idx = {dev: idx for idx, dev in enumerate(unique_developers)}
    developer_columns = [f'dev_{dev}' for dev in unique_developers]

    rows = range(len(gamesdata))
    cols = [developer_to_idx[dev] for dev in gamesdata['developers']]
    data = np.ones(len(gamesdata))
    developers_encoded = csr_matrix((data, (rows, cols)),
                                    shape=(len(gamesdata), len(unique_developers)))

    # Process tags more efficiently
    print("Processing tags with player votes...")
    all_tags = set()
    for tag_dict in gamesdata['tags']:
        if isinstance(tag_dict, dict):
            all_tags.update(tag_dict.keys())
    all_tags = list(all_tags)

    # Create sparse matrix for tags
    tag_rows = []
    tag_cols = []
    tag_data = []

    for i, tag_dict in enumerate(gamesdata['tags']):
        if isinstance(tag_dict, dict):
            for tag, votes in tag_dict.items():
                if tag in all_tags:
                    tag_rows.append(i)
                    tag_cols.append(all_tags.index(tag))
                    tag_data.append(float(votes))

    tags_features = csr_matrix((tag_data, (tag_rows, tag_cols)),
                               shape=(len(gamesdata), len(all_tags)))

    # Normalize tag features using sparse operations
    if len(all_tags) > 0:
        tag_max = tags_features.max()
        if tag_max != 0:
            tags_features = tags_features / tag_max

    # Convert sentiment and metacritic_score to sparse matrices
    sentiment = csr_matrix(gamesdata[['sentiment']].values)
    metacritic_score = csr_matrix(normalized_score)

    # Combine all features (keeping everything sparse)
    print("Combining all features...")
    if len(all_tags) > 0:
        features = hstack([
            tfidf_features * 1.5,
            developers_encoded * 0.5,
            sentiment * 1.0,
            metacritic_score * 1.0,
            tags_features * 1.2
        ], format='csr')
    else:
        features = hstack([
            tfidf_features * 1.5,
            developers_encoded * 0.5,
            sentiment * 1.0,
            metacritic_score * 1.0
        ], format='csr')

    return features, tfidf, scaler, developer_columns, all_tags


def train_model(features, embedding_dim=128, batch_size=32, max_epochs=10, device="cpu"):
    print("Creating dataset and dataloader...")
    # Keep features as sparse matrix
    dataset = ContentBasedDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing model...")
    model = ContentBasedModel(features.shape[1], embedding_dim).to(device)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch + 1}/{max_epochs}") as tepoch:
            for batch in tepoch:
                batch = batch.to(device)
                optimizer.zero_grad()
                embeddings = model(batch)
                target = torch.ones(batch.shape[0]).to(device)
                loss = criterion(embeddings, embeddings, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")

    return model