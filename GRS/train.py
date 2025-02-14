import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GameRecommendationModel
from data import GameDataset, collate_fn


def train_model(parquet_file, num_epochs=5, batch_size=32, lr=1e-3):
    dataset = GameDataset(parquet_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dev_vocab_size = len(dataset.dev_vocab)
    pub_vocab_size = len(dataset.pub_vocab)
    gen_vocab_size = len(dataset.gen_vocab)
    tag_vocab_size = len(dataset.tag_vocab)

    model = GameRecommendationModel(dev_vocab_size, pub_vocab_size, gen_vocab_size, tag_vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for batch in pbar:
            anchor = batch['anchor']
            positive = batch['positive']
            negative = batch['negative']

            anchor_emb = model(anchor['developer'], anchor['publisher'],
                               anchor['genres'], anchor['tags'], anchor['numeric'])
            pos_emb = model(positive['developer'], positive['publisher'],
                            positive['genres'], positive['tags'], positive['numeric'])
            neg_emb = model(negative['developer'], negative['publisher'],
                            negative['genres'], negative['tags'], negative['numeric'])

            pos_score = (anchor_emb * pos_emb).sum(dim=1)
            neg_score = (anchor_emb * neg_emb).sum(dim=1)

            target = torch.ones_like(pos_score)
            loss = F.margin_ranking_loss(pos_score, neg_score, target, margin=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "game_recommendation_model.pth")

    import pickle
    with open("vocabularies.pkl", "wb") as f:
        pickle.dump({
            'dev_vocab': dataset.dev_vocab,
            'pub_vocab': dataset.pub_vocab,
            'gen_vocab': dataset.gen_vocab,
            'tag_vocab': dataset.tag_vocab
        }, f)

    print("Training complete and model saved.")


if __name__ == "__main__":
    train_model("data/games_cleaned.parquet", num_epochs=5, batch_size=32)
