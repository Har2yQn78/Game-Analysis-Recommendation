import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    def __init__(self, data, user_col, item_col, rating_col):
        self.data = data
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.user_dict = {user: idx for idx, user in enumerate(data[user_col].unique())}
        self.item_dict = {item: idx for idx, item in enumerate(data[item_col].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = self.user_dict[row[self.user_col]]
        item = self.item_dict[row[self.item_col]]
        rating = row[self.rating_col]
        return user, item, rating


def create_batch_interaction_matrix(data, user_col, item_col, rating_col):
    user_dict = {user: idx for idx, user in enumerate(data[user_col].unique())}
    item_dict = {item: idx for idx, item in enumerate(data[item_col].unique())}
    return user_dict, item_dict


class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        output = self.fc(concat_emb)
        return output


def run_model(train_data, val_data, user_col, item_col, rating_col, embedding_dim=32, max_epochs=2, batch_size=128,
              patience=1, device="cpu"):
    user_dict, item_dict = create_batch_interaction_matrix(train_data, user_col, item_col, rating_col)
    num_users = len(user_dict)
    num_items = len(item_dict)

    model = RecommenderModel(num_users, num_items, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = InteractionDataset(train_data, user_col, item_col, rating_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(user_ids, item_ids)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}")

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_data, user_col, item_col, rating_col, device=device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model


def evaluate_model(model, data, user_col, item_col, rating_col, device="cpu"):
    model.eval()
    dataset = InteractionDataset(data, user_col, item_col, rating_col)
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids, item_ids, labels = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device)

            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()

    return total_loss / len(test_loader)
