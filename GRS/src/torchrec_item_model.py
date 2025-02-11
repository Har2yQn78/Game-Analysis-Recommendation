import torch
import torch.nn as nn


class GameEmbeddingModel(nn.Module):
    def __init__(self, tfidf_dim, num_developers, num_tags, metadata_dim=2, embedding_dim=128):
        super().__init__()
        # Process TF-IDF (dense) features
        self.tfidf_net = nn.Sequential(
            nn.Linear(tfidf_dim, embedding_dim),
            nn.ReLU()
        )
        # Developer: an embedding lookup
        self.developer_emb = nn.Embedding(num_developers, embedding_dim)
        # Process metadata (sentiment and metacritic)
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, embedding_dim),
            nn.ReLU()
        )
        # Tags: using EmbeddingBag to handle variable number of tags per game
        # (If there are no tags, you can skip this branch)
        self.tags_emb = nn.EmbeddingBag(num_tags, embedding_dim, mode='mean')

        # Combine all branches.
        # (We have 4 branches: tfidf, developer, metadata, and tags)
        self.final_layer = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU()
        )

    def forward(self, features):
        """
        Expects a dict with keys:
          - "tfidf": [B, tfidf_dim] float tensor
          - "developer": [B] long tensor
          - "metadata": [B, 2] float tensor
          - "tags_indices": 1D long tensor (flattened tags indices for the batch)
          - "tags_offsets": 1D long tensor (start index per sample in the batch)
        """
        tfidf_emb = self.tfidf_net(features["tfidf"])  # [B, emb]
        dev_emb = self.developer_emb(features["developer"])  # [B, emb]
        meta_emb = self.metadata_net(features["metadata"])  # [B, emb]
        tags_emb = self.tags_emb(features["tags_indices"], features["tags_offsets"])  # [B, emb]

        # Concatenate the branch outputs and project
        combined = torch.cat([tfidf_emb, dev_emb, meta_emb, tags_emb], dim=1)
        game_embedding = self.final_layer(combined)
        return game_embedding
