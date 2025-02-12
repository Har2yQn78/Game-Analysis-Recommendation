import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

from scrape import scrape_critic_reviews
from clean import clean_review_data
from sentiment import find_aspects
from summary import summarize_text

device_id = 0


def weighted_average(scores, weights):
    """Calculate weighted average of scores."""
    return np.average(scores, weights=weights) if len(scores) > 0 else 0.0


def comment_analysis_with_summary(game_name, aspects, headers=None):
    """
    Enhanced analysis pipeline with weighted scoring and confidence measures.
    """
    logging.info("Loading ABSA model for aspect analysis...")
    model_name_absa = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer_absa = AutoTokenizer.from_pretrained(model_name_absa, use_fast=False)
    model_absa = AutoModelForSequenceClassification.from_pretrained(
        model_name_absa,
        torch_dtype=torch.float16
    ).to("cuda")
    classifier = pipeline(
        "text-classification",
        model=model_absa,
        tokenizer=tokenizer_absa,
        device=device_id
    )
    logging.info("ABSA model loaded on GPU.")

    logging.info("Initializing summarization pipeline...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=device_id,
        torch_dtype=torch.float16
    )
    logging.info("Summarization model loaded on GPU.")

    logging.info("Scraping critic reviews...")
    df_reviews = scrape_critic_reviews(game_name, headers=headers)
    if df_reviews.empty:
        logging.error("No reviews scraped. Exiting analysis.")
        return {}, "", df_reviews

    logging.info("Cleaning review data...")
    df_reviews = clean_review_data(df_reviews)

    logging.info("Performing enhanced aspect-based sentiment analysis...")
    # Compute aspect scores with confidence for each review
    scores_list = []
    for text in df_reviews['review_text']:
        scores = find_aspects(text, aspects, classifier=classifier)
        scores_list.append(scores)

    # Convert scores to DataFrame columns
    for aspect in aspects:
        df_reviews[f'{aspect}_score'] = [s[aspect]['score'] for s in scores_list]
        df_reviews[f'{aspect}_confidence'] = [s[aspect]['confidence'] for s in scores_list]
        df_reviews[f'{aspect}_mentions'] = [s[aspect]['mention_count'] for s in scores_list]

    # Calculate weighted average scores
    overall_aspect_scores = {}
    for aspect in aspects:
        scores = df_reviews[f'{aspect}_score']
        confidences = df_reviews[f'{aspect}_confidence']
        mentions = df_reviews[f'{aspect}_mentions']

        # Combine confidence and mention count for weighting
        weights = confidences * (mentions + 1)  # Add 1 to avoid zero weights

        overall_aspect_scores[aspect] = {
            'score': weighted_average(scores, weights),
            'confidence': np.mean(confidences),
            'total_mentions': sum(mentions)
        }

    logging.info("Generating overall summary from combined reviews...")
    # Prioritize reviews with higher confidence scores for summary
    weighted_texts = []
    for _, row in df_reviews.iterrows():
        avg_confidence = np.mean([row[f'{aspect}_confidence'] for aspect in aspects])
        if avg_confidence > 0.5:  # Only include reviews with decent confidence
            weighted_texts.append(row['review_text'])

    combined_text = "\n\n".join(weighted_texts)
    overall_summary = summarize_text(combined_text, summarizer=summarizer)
    logging.info("Analysis complete!")

    return overall_aspect_scores, overall_summary, df_reviews