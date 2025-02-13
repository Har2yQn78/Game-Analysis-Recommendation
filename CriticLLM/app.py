import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from analysis import comment_analysis_with

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Game Review Analysis", layout="wide")

st.title("🎮 Game Review Analysis Dashboard")
st.write("Enter a game name and select the aspects you want to analyze using OpenRouter's R1 model.")

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Settings")
    default_game = "Red Dead Redemption 2"
    default_aspects = ["graphics", "gameplay", "story"]

    game_name = st.text_input("Enter game name:", default_game)
    aspects = st.multiselect(
        "Select aspects to analyze:",
        ["graphics", "gameplay", "story", "performance"],
        default=default_aspects,
        max_selections=4
    )

    st.divider()
    st.markdown("### Advanced Settings")
    min_confidence = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5)
    show_raw_data = st.checkbox("Show Raw Review Data", False)

if st.button("Run Analysis"):
    if not game_name or not aspects:
        st.error("Please enter both a game name and at least one aspect.")
    else:
        try:
            with st.spinner("Analysis in progress..."):
                overall_aspect_scores, overall_summary, df_reviews = comment_analysis_with(game_name, aspects)

            if df_reviews.empty:
                st.error("No reviews found or analysis failed.")
            else:
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Analysis", "Raw Data"])

                with tab1:
                    st.subheader("Overall Summary")
                    st.write(overall_summary)

                    st.subheader("Aspect Scores")
                    cols = st.columns(len(overall_aspect_scores))
                    for idx, (aspect, score) in enumerate(overall_aspect_scores.items()):
                        with cols[idx]:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=score,
                                title={'text': f"{aspect.title()}"},
                                gauge={
                                    'axis': {'range': [-1, 1]},
                                    'bar': {'color': "green" if score >= 0 else "red"},
                                    'steps': [
                                        {'range': [-1, -0.33], 'color': "red"},
                                        {'range': [-0.33, 0.33], 'color': "yellow"},
                                        {'range': [0.33, 1], 'color': "green"}
                                    ],
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.subheader("Detailed Aspect Analysis")
                    for aspect in aspects:
                        st.write(f"### {aspect.title()} Analysis")
                        # Filter reviews with a high confidence value for the given aspect
                        high_conf_reviews = df_reviews[df_reviews[f'{aspect}_analysis'].apply(
                            lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0) >= min_confidence]

                        if not high_conf_reviews.empty:
                            # Prepare data for a scatter plot
                            scores = []
                            confidences = []
                            websites = []
                            for _, row in high_conf_reviews.iterrows():
                                analysis = row.get(f'{aspect}_analysis', {})
                                score = analysis.get("score", 0)
                                confidence = analysis.get("confidence", 0)
                                scores.append(score)
                                confidences.append(confidence)
                                websites.append(row.get("website", "N/A"))

                            fig = px.scatter(
                                x=scores,
                                y=confidences,
                                size=[len(row.get("review_text", "")) for _, row in high_conf_reviews.iterrows()],
                                hover_data={"website": websites},
                                labels={"x": "Score", "y": "Confidence"},
                                title=f"{aspect.title()} Score vs Confidence"
                            )
                            st.plotly_chart(fig)

                            st.write("Most Relevant Review Excerpts:")
                            sorted_reviews = high_conf_reviews.sort_values(
                                by=f'{aspect}_analysis',
                                key=lambda col: col.apply(
                                    lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0),
                                ascending=False
                            )
                            for _, row in sorted_reviews.head(3).iterrows():
                                analysis = row.get(f'{aspect}_analysis', {})
                                st.info(
                                    f"Source: {row.get('website', 'N/A')}\n"
                                    f"Score: {analysis.get('score', 0):.2f}, Confidence: {analysis.get('confidence', 0):.2f}\n\n"
                                    f"{row.get('review_text', '')[:300]}..."
                                )
                        else:
                            st.write("No high-confidence reviews available for this aspect.")

                with tab3:
                    if show_raw_data:
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
            logger.error(f"Error during analysis: {str(e)}")
            st.error(f"An error occurred during analysis: {str(e)}")
