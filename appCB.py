import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import os

# Debugging: Print the current working directory
st.write("Current Working Directory:", os.getcwd())

# Define the model architecture
class ContentBasedModel(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(ContentBasedModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessing():
    save_dir = 'models'  # Path to the models directory

    # Load preprocessing objects
    tfidf = np.load(os.path.join(save_dir, 'tfidf.npy'), allow_pickle=True).item()
    scaler = np.load(os.path.join(save_dir, 'scaler.npy'), allow_pickle=True).item()
    developer_columns = np.load(os.path.join(save_dir, 'developer_columns.npy'), allow_pickle=True)

    # Calculate the input dimension
    input_dim = tfidf.get_feature_names_out().shape[0] + len(developer_columns) + 2  # +2 for sentiment and metascore

    # Load the model architecture
    model = ContentBasedModel(input_dim)

    # Load the state dictionary
    model.load_state_dict(torch.load(os.path.join(save_dir, 'content_based_model.pth')))
    model.eval()  # Set the model to evaluation mode

    return model, tfidf, scaler, developer_columns

# Preprocess input data
def preprocess_input(game_data, tfidf, scaler, developer_columns):
    # Fill missing values
    game_data['genres'] = game_data['genres'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['tags'] = game_data['tags'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['specs'] = game_data['specs'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    game_data['developer'] = game_data['developer'].fillna('Unknown')
    game_data['metascore'] = game_data['metascore'].fillna(game_data['metascore'].median())

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
    game_data['sentiment'] = game_data['sentiment'].map(sentiment_mapping).fillna(3)

    # Normalize metascore
    game_data['metascore'] = scaler.transform(game_data[['metascore']])

    # Combine textual features
    game_data['combined_text'] = game_data['genres'] + ' ' + game_data['tags'] + ' ' + game_data['specs']

    # TF-IDF for combined text
    tfidf_features = tfidf.transform(game_data['combined_text']).toarray()

    # One-hot encoding for developer
    developer_encoded = pd.get_dummies(game_data['developer'], prefix='dev')
    developer_encoded = developer_encoded.reindex(columns=developer_columns, fill_value=0)

    # Assign weights to features
    tfidf_features *= 2.0  # Higher weight for genres, specs, and tags
    developer_encoded *= 0.5  # Lower weight for developer
    sentiment = game_data[['sentiment']].values * 1.0  # Neutral weight for sentiment
    metascore = game_data[['metascore']].values * 1.0  # Neutral weight for metascore

    # Combine all features
    features = np.hstack([tfidf_features, developer_encoded, sentiment, metascore])

    # Ensure all features are of type float32
    features = features.astype(np.float32)
    return features

# Streamlit app
def main():
    st.title("Content-Based Game Recommendation System")

    # Load the dataset
    gamesdata_path = 'data/gamesdata.parquet'  # Path to the gamesdata.parquet file
    if not os.path.exists(gamesdata_path):
        st.error(f"File not found: {gamesdata_path}")
        st.write("Please ensure the file exists in the correct location.")
        return

    gamesdata = pd.read_parquet(gamesdata_path)

    # Load the model and preprocessing objects
    model, tfidf, scaler, developer_columns = load_model_and_preprocessing()

    # Select a game by name
    st.subheader("Select a Game")
    game_names = gamesdata['title'].tolist()  # Get the list of game names
    selected_game_name = st.selectbox("Select a game by name", game_names)

    # Get the selected game's data
    selected_game = gamesdata[gamesdata['title'] == selected_game_name].iloc[0]

    # Preprocess the selected game
    features = preprocess_input(pd.DataFrame([selected_game]), tfidf, scaler, developer_columns)

    # Get the model's embedding for the selected game
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32)
        embedding = model(features_tensor)

    # Compute similarity with other games
    st.subheader("Similar Games")
    with torch.no_grad():
        all_features = preprocess_input(gamesdata, tfidf, scaler, developer_columns)
        all_features_tensor = torch.tensor(all_features, dtype=torch.float32)
        all_embeddings = model(all_features_tensor)
        similarities = torch.nn.functional.cosine_similarity(embedding, all_embeddings)
        top_indices = similarities.argsort(descending=True)[1:]  # Exclude itself

        # Let the user choose how many similar games to display (default is 5, max is 8)
        num_similar_games = st.slider("Number of similar games to display", min_value=1, max_value=8, value=5)
        similar_games = gamesdata.iloc[top_indices[:num_similar_games]]

        # Display only the required columns
        st.write(similar_games[['title', 'developer', 'release_date', 'genres']])

if __name__ == '__main__':
    main()