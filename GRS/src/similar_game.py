import torch


def get_similar_games(model, features, selected_game_idx, num_recommendations=5, device="cpu"):
    """
    Given a trained model and features for all games (each feature is a torch.Tensor
    with batch dimension = number of games), compute the embedding for each game,
    then return the indices of the top similar games (excluding the selected one).
    """
    model.eval()
    with torch.no_grad():
        # Compute all game embeddings.
        # Note: For EmbeddingBag, ensure that "tags_indices" and "tags_offsets" refer to the full dataset.
        all_embeddings = model(features)  # shape: [n_games, embedding_dim]
        selected_emb = all_embeddings[selected_game_idx].unsqueeze(0)  # [1, embedding_dim]

        # Compute cosine similarity.
        emb_norms = all_embeddings.norm(dim=1, keepdim=True)
        selected_norm = selected_emb.norm(dim=1, keepdim=True)
        cosine_sim = (all_embeddings @ selected_emb.T) / (emb_norms * selected_norm + 1e-8)
        cosine_sim = cosine_sim.squeeze(1)

        # Exclude the selected game itself by setting its similarity to a very low value.
        cosine_sim[selected_game_idx] = -1.0

        # Get the indices of the top similar games.
        top_indices = torch.topk(cosine_sim, num_recommendations).indices.cpu().tolist()
    return top_indices
