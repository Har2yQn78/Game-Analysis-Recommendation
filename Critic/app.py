import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from dotenv import load_dotenv

load_dotenv()

from analysis import comment_analysis_with

# Page configuration
st.set_page_config(
    page_title="Game Review Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    /* Main container background */
    .stApp {
        background-color: #0e1117;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #262730;
    }

    /* Cards and containers */
    .stMarkdown, .stButton, .stSelectbox {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Tables */
    .dataframe {
        background-color: #262730;
        border: 1px solid #374151;
    }
    .dataframe th {
        background-color: #374151;
        color: white;
    }
    .dataframe td {
        background-color: #1f2937;
        color: #d1d5db;
    }

    /* Plotly charts */
    .js-plotly-plot {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #374151;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem 0.3rem 0 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def create_gauge_chart(value, title, color_ranges=None):
    if color_ranges is None:
        color_ranges = [
            {'range': [0, 3], 'color': '#ef4444'},
            {'range': [3, 7], 'color': '#eab308'},
            {'range': [7, 10], 'color': '#22c55e'}
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': 'white', 'size': 24}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4CAF50"},
            'steps': color_ranges,
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig


with st.sidebar:
    st.header("Analysis Settings")
    default_game = "Elden Ring"
    default_aspects = ["graphics", "gameplay", "story", "performance"]

    game_name = st.text_input("Enter game name:", default_game, help="Enter the name of the game you want to analyze")
    aspects = st.multiselect(
        "Select aspects to analyze:",
        ["graphics", "gameplay", "story", "performance", "music", "value", "innovation"],
        default=default_aspects,
        help="Choose which aspects of the game to analyze"
    )

st.title("🎮 Game Review Analysis Dashboard")
st.write("Analyze game reviews and view comprehensive summaries and aspect breakdowns.")

if st.button("Run Analysis"):
    if not game_name or not aspects:
        st.error("Please enter a game name and select at least one aspect.")
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
                    summary = combined_results.get("summary", "No summary available.")
                    if isinstance(summary, dict):
                        st.markdown("### Overall")
                        st.write(summary.get("overall", "No overall summary available."))
                        st.markdown("### Technical Details")
                        st.write(summary.get("technical", "No technical summary available."))
                        for aspect in aspects:
                            with st.expander(f"📌 {aspect.title()}"):
                                st.write(summary.get(aspect, "No summary available."))
                    else:
                        st.write(summary)

                with tab2:
                    st.subheader("Overall Aspect Analysis")
                    overall = combined_results.get("overall", {"score": 0, "explanation": "No data available."})
                    st.plotly_chart(
                        create_gauge_chart(overall.get("score", 0), "Overall Score"),
                        use_container_width=True
                    )
                    st.markdown(f"**Explanation:** {overall.get('explanation', '')}")

                    st.markdown("### Detailed Aspect Scores")
                    cols = st.columns(2)
                    for idx, aspect in enumerate(aspects):
                        aspect_data = combined_results.get(aspect, {"score": 0, "explanation": "No data available."})
                        with cols[idx % 2]:
                            st.plotly_chart(
                                create_gauge_chart(aspect_data.get("score", 0), aspect.title()),
                                use_container_width=True
                            )
                            st.markdown(f"**Analysis:** {aspect_data.get('explanation', '')}")

                with tab3:
                    st.subheader("Raw Review Data")
                    st.dataframe(df_reviews, use_container_width=True)
                    csv = df_reviews.to_csv(index=False)
                    st.download_button(
                        "📥 Download Reviews CSV",
                        csv,
                        "game_reviews.csv",
                        "text/csv",
                        key='download-csv'
                    )
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")