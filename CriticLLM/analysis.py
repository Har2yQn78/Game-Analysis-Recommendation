import os
import json
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scrape import scrape_critic_reviews
from clean import clean_review_data

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)


def combined_analysis(text, aspects, max_summary_length=450):
    """
    Constructs a prompt that instructs the LLM to:
      1. Produce a concise technical summary of the review text,
         emphasizing key performance details, technical insights, and industry-specific terminology.
      2. Provide a detailed aspect-based evaluation including an overall score and individual scores for the specified aspects.
         For each score, include a rating between 0 (very negative) and 10 (very positive) and a brief explanation.
    The response must be returned as valid JSON in the format shown in the example.
    """
    aspects_list = ", ".join(aspects)
    prompt = (
        "You are an expert video game journalist.\n\n"
        "Read the following review text carefully and perform two tasks:\n\n"
        "1. **Technical Summary:** Provide a concise technical summary of the review. Focus on key performance metrics, "
        "technical details, and industry-specific insights. Keep the summary under "
        f"{max_summary_length} words.\n\n"
        "2. **Aspect-Based Analysis:** Evaluate the review by assigning scores on a scale from 0 (very negative) to 10 (very positive). "
        "Provide an overall score and individual scores for the following aspects: "
        f"{aspects_list}. For each, include a brief explanation justifying your rating.\n\n"
        "Return your answer **ONLY** as valid JSON (with no additional commentary) using the exact format shown in the example below:\n\n"
        "{\n"
        '  "summary": "A concise technical summary of the review text.",\n'
        '  "overall": {"score": 8.0, "explanation": "Overall evaluation of the review."},\n'
    )
    for aspect in aspects:
        prompt += f'  "{aspect}": {{"score": 7.0, "explanation": "Explanation for {aspect}."}},\n'
    prompt = prompt.rstrip(",\n") + "\n}\n\n"
    prompt += f"Review text:\n{text}\n"

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", ""),
                "X-Title": os.getenv("SITE_NAME", ""),
            },
            extra_body={},
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "You are an expert video game journalist."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"LLM combined analysis raw response: {response_text}")

        match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()

        analysis = json.loads(response_text)
    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        analysis = {"summary": f"Error: {str(e)}", "overall": {"score": 0.0, "explanation": f"Error: {str(e)}"}}
        for aspect in aspects:
            analysis[aspect] = {"score": 0.0, "explanation": f"Error: {str(e)}"}
    return analysis


def comment_analysis_with(game_name, aspects, headers=None):
    """
    Analysis pipeline that scrapes critic reviews, cleans the data, and uses a single API call
    (via combined_analysis) to obtain both a technical summary and an aspect-based evaluation.
    """
    logger.info(f"Starting review scraping for game: {game_name}")
    df_reviews = scrape_critic_reviews(game_name, headers=headers)
    if df_reviews.empty:
        logger.error("No reviews scraped. Exiting analysis.")
        return {}, df_reviews

    logger.info("Cleaning review data...")
    df_reviews = clean_review_data(df_reviews)

    combined_text = "\n\n".join(df_reviews['review_text'])

    logger.info("Generating combined analysis using LLM...")
    combined_results = combined_analysis(combined_text, aspects)

    return combined_results, df_reviews
