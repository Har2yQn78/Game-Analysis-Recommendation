import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

from scrape import scrape_critic_reviews
from clean import clean_review_data
from sentiment import find_aspects
from summary import summarize_text

# Set the device id (adjust if needed)
device_id = 0


def comment_analysis_with_summary(game_name, aspects, headers=None):
    """
    Performs the following steps:
      1. Loads the ABSA model and a T5-small summarizer.
      2. Scrapes critic reviews for the game.
      3. Cleans the review data.
      4. For each review, computes aspect sentiment scores.
      5. Combines all review texts into one aggregated text and summarizes it.
      6. Returns overall aspect scores (averaged over reviews), the combined summary, and the DataFrame.
    """
    logging.info("Loading ABSA model for aspect analysis...")
    model_name_absa = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer_absa = AutoTokenizer.from_pretrained(model_name_absa, use_fast=False)
    model_absa = AutoModelForSequenceClassification.from_pretrained(model_name_absa, torch_dtype=torch.float16).to(
        "cuda")
    classifier = pipeline("text-classification", model=model_absa, tokenizer=tokenizer_absa, device=device_id)
    logging.info("ABSA model loaded on GPU.")

    logging.info("Initializing summarization pipeline (T5-small)...")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device_id)
    logging.info("Summarization model loaded on GPU.")

    logging.info("Scraping critic reviews...")
    df_reviews = scrape_critic_reviews(game_name, headers=headers)
    if df_reviews.empty:
        logging.error("No reviews scraped. Exiting analysis.")
        return {}, "", df_reviews

    logging.info("Cleaning review data...")
    df_reviews = clean_review_data(df_reviews)

    logging.info("Performing aspect-based sentiment analysis...")
    # Compute aspect scores for each review and average them for overall scores
    scores_df = df_reviews['review_text'].apply(lambda x: pd.Series(find_aspects(x, aspects, classifier=classifier)))
    df_reviews = pd.concat([df_reviews, scores_df], axis=1)
    logging.info("Aspect analysis complete!")

    # Combine all review texts into one aggregated text
    combined_text = "\n\n".join(df_reviews['review_text'].tolist())
    logging.info("Generating overall summary from combined reviews...")
    overall_summary = summarize_text(combined_text, summarizer=summarizer)
    logging.info("Overall summarization complete!")

    overall_aspect_scores = scores_df.mean().to_dict()  # Averaging scores for better interpretability

    return overall_aspect_scores, overall_summary, df_reviews
