import os
import pandas as pd
import numpy as np
import torch
from utils_trainCB import preprocess_data, train_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load the dataset
    print("Loading dataset...")
    gamesdata_path = os.path.join('..', 'data', 'gamesdata.parquet')
    gamesdata = pd.read_parquet(gamesdata_path)
    print("Dataset loaded successfully.")

    # Preprocess the data
    features, tfidf, scaler, developer_columns = preprocess_data(gamesdata)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    model = train_model(features, device=device)

    # Save the model and preprocessing objects
    save_dir = os.path.join('..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    print("Saving model and preprocessing objects...")
    torch.save(model.state_dict(), os.path.join(save_dir, 'content_based_model.pth'))
    np.save(os.path.join(save_dir, 'tfidf.npy'), tfidf)
    np.save(os.path.join(save_dir, 'scaler.npy'), scaler)
    np.save(os.path.join(save_dir, 'developer_columns.npy'), developer_columns)

    print("Model and preprocessing objects saved.")


if __name__ == '__main__':
    main()
