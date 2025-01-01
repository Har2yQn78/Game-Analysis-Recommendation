import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.utils import (create_interaction_matrix, create_user_dict,
                       create_item_dict, load_model_and_embeddings, get_recs)


# Load data
@st.cache_data
def load_data():
    recdata = pd.read_parquet('data/recdata.parquet')  # Read .parquet file
    recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
    gamesdata = pd.read_parquet('data/gamesdata.parquet')  # Read .parquet file
    numgames = pd.read_parquet('data/numgames.parquet')  # Read .parquet file
    return recdata, gamesdata, numgames


@st.cache_data
def load_model():
    models_dir = 'models'
    device = torch.device("cpu")  # Force CPU usage
    model = load_model_and_embeddings(models_dir, device)
    return model


def main():
    st.title("Game Recommendation System")
    st.write("This app provides game recommendations based on user preferences.")

    recdata, gamesdata, numgames = load_data()

    interactions, user_ids, item_ids = create_interaction_matrix(df=recdata, user_col='uid', item_col='id',
                                                                 rating_col='owned')
    user_dict = create_user_dict(user_ids)
    games_dict = create_item_dict(df=gamesdata, id_col='id', name_col='title')

    # Handle numgames: the first column is unnamed (uid), and the second column is the id to display
    # Check the number of columns in numgames
    if len(numgames.columns) >= 2:
        # Rename the first two columns
        numgames.columns = ['uid', 'user_id'] + list(numgames.columns[2:])
    else:
        st.error("The 'numgames.parquet' file must have at least two columns.")
        return

    # Map uid to user_id using numgames
    uid_to_user_id = dict(zip(numgames['uid'], numgames['user_id']))

    # Load PyTorch model
    model = load_model()

    # Sidebar for user input
    st.sidebar.header("User Input")

    # Create a dropdown for user selection
    user_ids_list = list(user_dict.keys())
    user_id_display = [uid_to_user_id.get(uid, f"Unknown User ({uid})") for uid in user_ids_list]
    selected_user_display = st.sidebar.selectbox("Select User", user_id_display)

    selected_uid = user_ids_list[user_id_display.index(selected_user_display)]

    num_recs = st.sidebar.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.sidebar.button("Get User Recommendations"):
        st.subheader(f"Recommendations for User {selected_user_display}")
        rec_list = get_recs(model=model, user_id=selected_uid, item_ids=item_ids, item_dict=games_dict,
                            num_items=num_recs, device="cpu")
        for i, rec in enumerate(rec_list, start=1):
            st.write(f"{i}. {rec}")


if __name__ == "__main__":
    main()
