import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
import numpy as np

from analysis import comment_analysis_with_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Game Review Analysis", layout="wide")

st.title("🎮 Game Review Analysis Dashboard")
st.write("Enter a game name and the aspects you want to analyze.")

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
                overall_aspect_scores, overall_summary, df_reviews = comment_analysis_with_summary(game_name, aspects)

            if not overall_aspect_scores:
                st.error("No reviews found or analysis failed.")
            else:
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Analysis", "Raw Data"])

                with tab1:
                    st.subheader("Overall Summary")
                    st.write(overall_summary)

                    # Display aspect scores with confidence
                    st.subheader("Aspect Scores & Confidence")
                    cols = st.columns(len(overall_aspect_scores))

                    for idx, (aspect, data) in enumerate(overall_aspect_scores.items()):
                        with cols[idx]:
                            score = data['score']
                            confidence = data['confidence']
                            mentions = data['total_mentions']

                            # Create gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=score,
                                title={'text': f"{aspect.title()}\n(Confidence: {confidence:.2f})"},
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
                            st.metric("Total Mentions", mentions)

                with tab2:
                    st.subheader("Detailed Aspect Analysis")

                    # Create detailed visualizations for each aspect
                    for aspect in aspects:
                        st.write(f"### {aspect.title()} Analysis")

                        # Filter for high-confidence mentions
                        high_conf_reviews = df_reviews[
                            df_reviews[f'{aspect}_confidence'] >= min_confidence
                            ]

                        if not high_conf_reviews.empty:
                            # Create scatter plot of scores vs confidence
                            fig = px.scatter(
                                high_conf_reviews,
                                x=f'{aspect}_score',
                                y=f'{aspect}_confidence',
                                size=f'{aspect}_mentions',
                                hover_data=['website'],
                                title=f"{aspect.title()} Score vs Confidence"
                            )
                            st.plotly_chart(fig)

                            # Show most relevant review excerpts
                            st.write("Most Relevant Review Excerpts:")
                            for _, row in high_conf_reviews.nlargest(3, f'{aspect}_confidence').iterrows():
                                st.info(
                                    f"Source: {row['website']}\nConfidence: {row[f'{aspect}_confidence']:.2f}\n\n{row['review_text'][:300]}...")

                with tab3:
                    if show_raw_data:
                        st.subheader("Raw Review Data")
                        st.dataframe(df_reviews)

                        # Add download button for CSV
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