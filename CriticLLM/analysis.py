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


def combined_analysis(text, aspects):
    """
    Constructs a prompt that instructs the LLM to:
      1. Provide a comprehensive summary of the review that includes:
         - A detailed overall summary (~200-350 words) covering general impressions.
         - A technical summary (~20-50 words) focusing on performance metrics, system insights, and other technical details.
         - Brief aspect-specific summaries (~20-50 words each) for the specified aspects.
      2. Provide an aspect-based evaluation by assigning scores (0 to 10) with brief explanations.
    The response must be returned as valid JSON using the exact format shown in the example.
    """
    aspects_list = ", ".join(aspects)

    prompt = (
        "You are an expert video game journalist.\n\n"
        "Read the following review text carefully and perform the following tasks:\n\n"
        "1. **Comprehensive Summary:** Provide a comprehensive summary of the review that includes:\n"
        "   - **Overall Summary:** A detailed summary of the review covering general impressions, context, and main points. "
        "Ensure this summary is approximately 200-350 words.\n"
        "   - **Technical Summary:** A concise summary of technical details (such as performance metrics, system insights, etc.) "
        "in about 20-50 words.\n"
        f"   - **Aspect-Specific Summaries:** For the following aspects ({aspects_list}), provide a brief summary of the review points "
        "related to each aspect in about 20-50 words:\n"
    )
    for aspect in aspects:
        prompt += f'      - "{aspect}"\n'

    prompt += (
        "\n2. **Aspect-Based Evaluation:** Evaluate the review by assigning scores on a scale from 0 (very negative) to 10 (very positive). "
        "Provide an overall score and individual scores for each specified aspect. For each, include a brief explanation.\n\n"
        "Return your answer ONLY as valid JSON (with no additional commentary) using the exact format shown in the example below:\n\n"
        "{\n"
        '  "summary": {\n'
        '    "overall": "Detailed overall summary of the review (approximately 200-350 words).",\n'
        '    "technical": "Concise summary of technical aspects (approximately 20-50 words).",\n'
    )
    for aspect in aspects:
        prompt += f'    "{aspect}": "Brief summary for {aspect} (approximately 20-50 words).",\n'
    prompt += (
        "  },\n"
        '  "overall": {"score": 8.0, "explanation": "Overall evaluation of the review."},\n'
    )
    for aspect in aspects:
        prompt += f'  "{aspect}": {{"score": 7.0, "explanation": "Evaluation of {aspect}."}},\n'
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
    (via combined_analysis) to obtain a comprehensive summary (covering overall, technical, and aspect-specific points)
    and an aspect-based evaluation.
    """
    logger.info(f"Starting review scraping for game: {game_name}")
    df_reviews = scrape_critic_reviews(game_name, headers=headers)
    if df_reviews.empty:
        logger.error("No reviews scraped. Exiting analysis.")
        return {}, df_reviews

    logger.info("Cleaning review data...")
    df_reviews = clean_review_data(df_reviews)

    # Combine all review texts for overall analysis.
    combined_text = "\n\n".join(df_reviews['review_text'])

    logger.info("Generating combined analysis using LLM...")
    combined_results = combined_analysis(combined_text, aspects)

    return combined_results, df_reviews
