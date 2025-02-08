import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logging

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
st.write("Enter a game name and the aspects you want to analyze (e.g., graphics, gameplay, story).")
default_game = "Red Dead Redemption 2"
default_aspects = ["graphics", "gameplay", "story"]

# Sidebar inputs for game name and aspects
game_name = st.text_input("Enter game name:", default_game, placeholder="Type a game name here...")
aspects = st.multiselect(
    "Select up to 3 aspects to analyze:",
    ["graphics", "gameplay", "story", "performance"],
    default=default_aspects,
    max_selections=3
)

if st.button("Run Analysis"):
    if not game_name or not aspects:
        st.error("Please enter both a game name and at least one aspect.")
    else:
        progress_text = st.empty()

        try:
            # Create a progress container
            progress_container = st.container()

            with st.spinner("Analysis in progress..."):
                # Step 1: Scraping reviews
                progress_text.text("Step 1/4: Scraping game reviews from multiple sources...")
                logger.info(f"Starting review scraping for game: {game_name}")

                # Step 2: Processing text
                progress_text.text("Step 2/4: Processing and cleaning review text...")
                logger.info("Processing review text data")

                # Step 3: Analyzing aspects
                progress_text.text(f"Step 3/4: Analyzing aspects: {', '.join(aspects)}...")
                logger.info(f"Analyzing selected aspects: {aspects}")

                # Perform the actual analysis
                overall_aspect_scores, overall_summary, df_reviews = comment_analysis_with_summary(game_name, aspects)

                # Step 4: Generating visualizations
                progress_text.text("Step 4/4: Generating visualizations and summary...")
                logger.info("Generating final results and visualizations")

            if not overall_aspect_scores:
                st.error("No reviews found or analysis failed.")
                logger.error("Analysis returned no results")
            else:
                progress_text.text("Analysis complete!")
                st.success("Analysis complete!")
                logger.info("Analysis completed successfully")

                # Display results
                st.subheader("Overall Summary")
                st.write(overall_summary)

                st.subheader("Aspect Sentiment Scores")
                cols = st.columns(len(overall_aspect_scores))
                for idx, (aspect, score) in enumerate(overall_aspect_scores.items()):
                    with cols[idx]:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': aspect.title()},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "green" if score >= 0 else "red"},
                                'steps': [
                                    {'range': [-1, 0], 'color': "red"},
                                    {'range': [0, 1], 'color': "green"}
                                ],
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Scraped Reviews Data")
                st.dataframe(df_reviews)

        except Exception as e:
            progress_text.text("Analysis failed!")
            logger.error(f"Error during analysis: {str(e)}")
            st.error(f"An error occurred during analysis: {str(e)}")