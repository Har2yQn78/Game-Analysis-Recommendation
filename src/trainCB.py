import os
import pandas as pd
import numpy as np
import torch
from utils_trainCB import preprocess_data, train_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading dataset...")
    gamesdata_path = os.path.join('..', 'data', 'cleanedgamesdata.parquet')
    gamesdata = pd.read_parquet(gamesdata_path)
    print("Dataset loaded successfully.")

    features, tfidf, scaler, developer_columns, all_tags = preprocess_data(gamesdata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_model(features, device=device)

    save_dir = os.path.join('..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    print("Saving model and preprocessing objects...")
    torch.save(model.state_dict(), os.path.join(save_dir, 'content_based_model.pth'))
    np.save(os.path.join(save_dir, 'tfidf.npy'), tfidf)
    np.save(os.path.join(save_dir, 'scaler.npy'), scaler)
    np.save(os.path.join(save_dir, 'developer_columns.npy'), developer_columns)
    np.save(os.path.join(save_dir, 'all_tags.npy'), all_tags)

    print("Model and preprocessing objects saved.")


if __name__ == '__main__':
    main()