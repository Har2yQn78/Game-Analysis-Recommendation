import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from utils_train import create_batch_interaction_matrix, run_model, evaluate_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    recdata_path = os.path.join('..', 'data', 'recdata.parquet')
    gamesdata_path = os.path.join('..', 'data', 'gamesdata.parquet')

    recdata = pd.read_parquet(recdata_path)
    gamesdata = pd.read_parquet(gamesdata_path)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_results = []

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define embedding_dim
    embedding_dim = 32  # You can change this value if needed

    for train_index, val_index in kf.split(recdata):
        train_data = recdata.iloc[train_index]
        val_data = recdata.iloc[val_index]

        print("Train data shape:", train_data.shape)
        print("Validation data shape:", val_data.shape)

        print("Starting training...")
        recommender_model = run_model(
            train_data=train_data,
            val_data=val_data,
            user_col='uid',
            item_col='id',
            rating_col='owned',
            embedding_dim=embedding_dim,  # Pass embedding_dim here
            max_epochs=1,
            batch_size=128,
            patience=1,
            device=device
        )

        val_loss = evaluate_model(recommender_model, val_data, user_col='uid', item_col='id', rating_col='owned',
                                  device=device)
        fold_results.append(val_loss)

    # Save the model and embeddings
    save_dir = os.path.join('..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    # Save the model state
    torch.save(recommender_model.state_dict(), os.path.join(save_dir, 'recommender_model.pth'))

    user_embeddings = recommender_model.user_embedding.weight.data.cpu().numpy()
    item_embeddings = recommender_model.item_embedding.weight.data.cpu().numpy()
    np.save(os.path.join(save_dir, 'user_embeddings.npy'), user_embeddings)
    np.save(os.path.join(save_dir, 'item_embeddings.npy'), item_embeddings)

    np.save(os.path.join(save_dir, 'embedding_dim.npy'), np.array([embedding_dim]))


if __name__ == '__main__':
    main()
