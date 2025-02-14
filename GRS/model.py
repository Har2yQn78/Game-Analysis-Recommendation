import torch
import torch.nn as nn


class GameRecommendationModel(nn.Module):
    def __init__(self, dev_vocab_size, pub_vocab_size, gen_vocab_size, tag_vocab_size, emb_dim=32):
        """
        Creates embeddings for developer, publisher, genres, and tags.
        Developer and publisher embeddings are given extra weight.
        Combines these with numerical features (pct_pos_total and num_reviews_total).
        """
        super().__init__()
        self.developer_embedding = nn.Embedding(dev_vocab_size, emb_dim)
        self.publisher_embedding = nn.Embedding(pub_vocab_size, emb_dim)

        self.genre_embedding = nn.Embedding(gen_vocab_size, emb_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, emb_dim)

        self.num_linear = nn.Linear(2, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(5 * emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, developer, publisher, genres, tags, numeric_feats):
        dev_emb = self.developer_embedding(developer) * 1.5
        pub_emb = self.publisher_embedding(publisher) * 1.5

        gen_emb = self.genre_embedding(genres).mean(dim=1)
        tag_emb = self.tag_embedding(tags).mean(dim=1)

        num_emb = self.num_linear(numeric_feats)

        combined = torch.cat([dev_emb, pub_emb, gen_emb, tag_emb, num_emb], dim=1)
        out = self.fc(combined)
        return out
