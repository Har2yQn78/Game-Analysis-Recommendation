import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.utils import (
    create_interaction_matrix, create_user_dict, create_item_dict, load_model_and_embeddings, get_recs,
    load_cb_model_and_preprocessing, load_or_create_features, load_or_compute_embeddings, get_recommendations,
    load_cb_data
)


@st.cache_data
def load_cf_data():
    """Load collaborative filtering data from parquet files."""
    recdata = pd.read_parquet('data/recdata.parquet')
    recdata = recdata.rename(columns={'variable': 'id', 'value': 'owned'})
    gamesdata = pd.read_parquet('data/gamesdata.parquet')
    numgames = pd.read_parquet('data/numgames.parquet')
    return recdata, gamesdata, numgames


@st.cache_data
def load_cb_data():
    """Load content-based filtering data from parquet file."""
    from src.utils import load_cb_data as _load_cb_data
    return _load_cb_data()


def main():
    st.title("Game Recommendation System")
    st.write("This app provides game recommendations based on user preferences or game content.")

    # Load all necessary data
    recdata, gamesdata, numgames = load_cf_data()
    cleanedgamesdata = load_cb_data()

    with st.spinner("Loading game embeddings..."):
        embeddings = load_or_compute_embeddings()

    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio(
        "Choose Recommendation Model",
        ("Collaborative Filtering", "Content-Based")
    )

    if model_type == "Content-Based":
        st.subheader("Content-Based Recommendations")

        # Game selection
        st.subheader("Select a Game")
        game_names = cleanedgamesdata['name'].tolist()
        selected_game_name = st.selectbox("Select a game by name", game_names)
        selected_game_idx = cleanedgamesdata[cleanedgamesdata['name'] == selected_game_name].index[0]

        initial_recommendations = 5
        similar_games_indices = get_recommendations(
            embeddings,
            selected_game_idx,
            initial_recommendations
        )

        similar_games = cleanedgamesdata.iloc[similar_games_indices]
        st.write("Top 5 Similar Games:")
        st.write(similar_games[['name', 'developers', 'release_date', 'genres']])

        if st.button("Show More"):
            additional_recommendations = get_recommendations(
                embeddings,
                selected_game_idx,
                initial_recommendations + 3
            )
            additional_games = cleanedgamesdata.iloc[additional_recommendations[initial_recommendations:]]
            st.write("Additional Similar Games:")
            st.write(additional_games[['name', 'developers', 'release_date', 'genres']])

    else:
        st.subheader("Collaborative Filtering Recommendations")

        interactions, user_ids, item_ids = create_interaction_matrix(
            df=recdata,
            user_col='uid',
            item_col='id',
            rating_col='owned'
        )
        user_dict = create_user_dict(user_ids)
        games_dict = create_item_dict(df=gamesdata, id_col='id', name_col='title')

        if len(numgames.columns) >= 2:
            numgames.columns = ['uid', 'user_id'] + list(numgames.columns[2:])
            uid_to_user_id = dict(zip(numgames['uid'], numgames['user_id']))
        else:
            st.error("The 'numgames.parquet' file must have at least two columns.")
            return

        with st.spinner("Loading collaborative filtering model..."):
            model = load_model_and_embeddings('models', torch.device("cpu"))

        st.sidebar.header("User Input")
        user_ids_list = list(user_dict.keys())
        user_id_display = [uid_to_user_id.get(uid, f"Unknown User ({uid})")
                           for uid in user_ids_list]
        selected_user_display = st.selectbox(
            "Select User",
            user_id_display,
            key="cf_user_select"
        )
        selected_uid = user_ids_list[user_id_display.index(selected_user_display)]

        num_recs = st.number_input(
            "Number of Recommendations",
            min_value=1,
            max_value=8,
            value=5,
            step=1,
            key="cf_num_input"
        )

        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                rec_list = get_recs(
                    model=model,
                    user_id=selected_uid,
                    item_ids=item_ids,
                    item_dict=games_dict,
                    num_items=num_recs,
                    device="cpu"
                )

                st.subheader(f"Top {num_recs} Recommendations for User {selected_user_display}")

                rec_df = pd.DataFrame({
                    'Game': rec_list
                }).reset_index(names=['Rank'])
                rec_df['Rank'] = rec_df['Rank'] + 1

                st.dataframe(rec_df, hide_index=True)

                if st.checkbox("Show detailed game information"):
                    for game_name in rec_list:
                        game_info = gamesdata[gamesdata['title'] == game_name].iloc[0]
                        with st.expander(f"Details for {game_name}"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Developers:**", game_info.get('developers', 'N/A'))
                                st.write("**Release Date:**", game_info.get('release_date', 'N/A'))
                            with cols[1]:
                                st.write("**Genres:**",
                                         ', '.join(game_info['genres']) if isinstance(game_info.get('genres'),
                                                                                      list) else 'N/A')
                                st.write("**Tags:**",
                                         ', '.join(list(game_info['tags'].keys())[:5]) if isinstance(
                                             game_info.get('tags'), dict) else 'N/A')


if __name__ == "__main__":
    main()
