import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ContentBasedDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class ContentBasedModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(ContentBasedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)  # Additional layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess_data(gamesdata):
    print("Preprocessing data...")
    # Fill missing values
    gamesdata['genres'] = gamesdata['genres'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['tags'] = gamesdata['tags'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['specs'] = gamesdata['specs'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    gamesdata['developer'] = gamesdata['developer'].fillna('Unknown')
    gamesdata['metascore'] = gamesdata['metascore'].fillna(gamesdata['metascore'].median())

    # Map sentiment to numerical values
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
        None: 3  # Map missing values to Neutral
    }
    gamesdata['sentiment'] = gamesdata['sentiment'].map(sentiment_mapping).fillna(3)

    # Normalize metascore
    scaler = MinMaxScaler()
    gamesdata['metascore'] = scaler.fit_transform(gamesdata[['metascore']])  # Fixed typo: game_data -> gamesdata

    # Combine textual features
    gamesdata['combined_text'] = gamesdata['genres'] + ' ' + gamesdata['tags'] + ' ' + gamesdata['specs']

    # TF-IDF for combined text
    print("Applying TF-IDF to combined text...")
    tfidf = TfidfVectorizer(max_features=500)  # Adjust max_features as needed
    tfidf_features = tfidf.fit_transform(gamesdata['combined_text']).toarray()

    # One-hot encoding for developer
    print("One-hot encoding developer column...")
    developer_encoded = pd.get_dummies(gamesdata['developer'], prefix='dev')

    # Assign weights to features
    tfidf_features *= 1.5  # Higher weight for genres, specs, and tags
    developer_encoded *= 0.5  # Lower weight for developer
    sentiment = gamesdata[['sentiment']].values * 1.0  # Neutral weight for sentiment
    metascore = gamesdata[['metascore']].values * 1.0  # Neutral weight for metascore

    # Combine all features
    print("Combining all features...")
    features = np.hstack([tfidf_features, developer_encoded, sentiment, metascore])
    return features, tfidf, scaler, developer_encoded.columns


def train_model(features, embedding_dim=128, batch_size=128, max_epochs=10, device="cpu"):
    print("Creating dataset and dataloader...")
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

                # Debugging: Print embeddings and loss for the first batch of the first epoch
                if epoch == 0 and tepoch.n == 1:
                    print("Sample Embeddings:", embeddings[:5])
                    print("Batch Loss:", loss.item())

        print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")

    return model
