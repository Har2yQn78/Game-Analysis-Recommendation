import streamlit as st
import pandas as pd
import logging
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

from analysis import comment_analysis_with

st.set_page_config(page_title="Game Review Analysis", layout="wide")

st.title("🎮 Game Review Analysis Dashboard")
st.write("Enter a game name and select the aspects you want to analyze using OpenRouter's R1 model.")

with st.sidebar:
    st.header("Analysis Settings")
    default_game = "Red Dead Redemption 2"
    default_aspects = ["graphics", "gameplay", "story", "performance"]

    game_name = st.text_input("Enter game name:", default_game)
    aspects = st.multiselect(
        "Select aspects to analyze:",
        ["graphics", "gameplay", "story", "performance"],
        default=default_aspects,
        max_selections=4
    )

if st.button("Run Analysis"):
    if not game_name or not aspects:
        st.error("Please enter both a game name and at least one aspect.")
    else:
        try:
            with st.spinner("Analysis in progress..."):
                combined_results, df_reviews = comment_analysis_with(game_name, aspects)

            if df_reviews.empty:
                st.error("No reviews found or analysis failed.")
            else:
                tab1, tab2, tab3 = st.tabs(["Summary", "Aspect Analysis", "Raw Data"])

                with tab1:
                    st.subheader("Overall Summary")
                    st.write(combined_results.get("summary", "No summary available."))

                with tab2:
                    st.subheader("Overall Aspect Analysis (Gauge Charts)")
                    overall = combined_results.get("overall", {"score": 0, "explanation": ""})
                    fig_overall = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall.get("score", 0),
                        title={"text": "Overall"},
                        gauge={
                            'axis': {'range': [0, 10]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 3], 'color': 'red'},
                                {'range': [3, 7], 'color': 'yellow'},
                                {'range': [7, 10], 'color': 'green'}
                            ],
                        }
                    ))
                    st.plotly_chart(fig_overall, use_container_width=True)
                    st.write(f"**Explanation:** {overall.get('explanation', '')}")

                    for aspect in aspects:
                        aspect_data = combined_results.get(aspect, {"score": 0, "explanation": ""})
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=aspect_data.get("score", 0),
                            title={"text": aspect.title()},
                            gauge={
                                'axis': {'range': [0, 10]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 3], 'color': 'red'},
                                    {'range': [3, 7], 'color': 'yellow'},
                                    {'range': [7, 10], 'color': 'green'}
                                ],
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(f"**Explanation for {aspect.title()}:** {aspect_data.get('explanation', '')}")

                with tab3:
                    st.subheader("Raw Review Data")
                    st.dataframe(df_reviews)
                    csv = df_reviews.to_csv(index=False)
                    st.download_button(
                        "Download Reviews CSV",
                        csv,
                        "game_reviews.csv",
                        "text/csv",
                        key='download-csv'
                    )
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
