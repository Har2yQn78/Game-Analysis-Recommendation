import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)


def summarize_text(text, max_summary_length=450):
    """
    Summarize the given text using ChatGPT with the specified model.
    """
    prompt = (
        f"Please provide a concise summary of the following review text. "
        f"Keep the summary under {max_summary_length} words.\n\n"
        f"Review text:\n{text}\n\nSummary:"
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
                {"role": "system", "content": "You are a Video Game Journalist"},
                {"role": "user", "content": prompt}
            ]
        )
        summary = completion.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"
    return summary
