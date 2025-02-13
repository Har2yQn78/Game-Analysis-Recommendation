import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)


def aspect_analysis(text, aspects):
    """
    As an expert video game journalist, analyze the review text and evaluate its overall tone for the following aspects:
    (e.g., graphics, gameplay, story, etc.)

    For each aspect, provide:
      - a sentiment score between -1 (very negative) and 1 (very positive),
      - a brief explanation for your evaluation.

    Base your analysis on the entire review text.

    Please provide your analysis in JSON format exactly as shown in the example below:

    {
      "graphics": {"score": 0.8, "explanation": "The visuals are outstanding and immersive."},
      "gameplay": {"score": 0.6, "explanation": "The gameplay is solid but has minor flaws."},
      "story": {"score": 0.9, "explanation": "The narrative is compelling and well-developed."}
    }
    """
    aspects_list = ", ".join(aspects)
    prompt = (
        f"As an expert video game journalist, analyze the following review text and evaluate its overall tone for the following aspects: {aspects_list}.\n"
        "For each aspect, provide a sentiment score between -1 (very negative) and 1 (very positive) based on the review content, and include a brief explanation for your evaluation. "
        "Base your analysis on the entire review text. \n"
        "Please provide your analysis in JSON format exactly as shown in the example below:\n"
        '{\n'
        '  "graphics": {"score": 0.8, "explanation": "The visuals are outstanding and immersive."},\n'
        '  "gameplay": {"score": 0.6, "explanation": "The gameplay is solid but has minor flaws."},\n'
        '  "story": {"score": 0.9, "explanation": "The narrative is compelling and well-developed."}\n'
        '}\n\n'
        f"Review text:\n{text}\n"
    )

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
        analysis_text = completion.choices[0].message.content.strip()
        analysis = json.loads(analysis_text)
    except Exception as e:
        analysis = {aspect: {"score": 0.0, "explanation": f"Error: {str(e)}"} for aspect in aspects}
    return analysis
