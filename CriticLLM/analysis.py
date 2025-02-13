import logging
import pandas as pd
import numpy as np
from scrape import scrape_critic_reviews
from clean import clean_review_data
from summarization import summarize_text
from aspect_analysis import aspect_analysis


def comment_analysis_with(game_name, aspects, headers=None):
    """
    Analysis pipeline that scrapes critic reviews, cleans the data,
    and uses OpenRouter's ChatGPT API for both summarization and aspect-based sentiment analysis.
    """
    logging.info("Scraping critic reviews...")
    df_reviews = scrape_critic_reviews(game_name, headers=headers)
    if df_reviews.empty:
        logging.error("No reviews scraped. Exiting analysis.")
        return {}, "", df_reviews

    logging.info("Cleaning review data...")
    df_reviews = clean_review_data(df_reviews)

    aspect_results = []
    for text in df_reviews['review_text']:
        analysis = aspect_analysis(text, aspects)
        aspect_results.append(analysis)

    # Store the analysis results in the DataFrame
    for aspect in aspects:
        df_reviews[f'{aspect}_analysis'] = [result.get(aspect, {}) for result in aspect_results]

    logging.info("Generating overall summary using LLM...")
    combined_text = "\n\n".join(df_reviews['review_text'])
    overall_summary = summarize_text(combined_text)

    # Aggregate aspect scores (simple average)
    overall_aspect_scores = {}
    for aspect in aspects:
        scores = []
        for result in aspect_results:
            try:
                score = float(result.get(aspect, {}).get("score", 0))
            except Exception:
                score = 0.0
            scores.append(score)
        if scores:
            overall_aspect_scores[aspect] = sum(scores) / len(scores)
        else:
            overall_aspect_scores[aspect] = 0.0

    return overall_aspect_scores, overall_summary, df_reviews
