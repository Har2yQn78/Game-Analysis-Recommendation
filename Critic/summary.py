import logging
import re
from transformers import pipeline

device_id = 0


def summarize_text(text, summarizer=None):
    """
    Summarizes the given text using a pretrained summarization model.
    If the text is too long for the model (i.e. more tokens than the maximum input length),
    the text is split into chunks, each chunk is summarized, and the summaries are combined.
    """
    if summarizer is None:
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device_id)
    try:
        tokenizer = summarizer.tokenizer
        max_input_length = tokenizer.model_max_length  # e.g., 512 tokens for T5-small
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_length = inputs.input_ids.shape[1]

        if input_length <= max_input_length:
            summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        else:
            # Split text into chunks by sentences
            sentences = re.split(r'(?<=[.?!])\s+', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                test_tokens = tokenizer.encode(test_chunk, add_special_tokens=True)
                if len(test_tokens) <= max_input_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)

            # Summarize each chunk and combine summaries
            summaries = []
            for chunk in chunks:
                summ = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                summaries.append(summ[0]['summary_text'])
            combined_summary = " ".join(summaries)

            # Re-summarize if necessary
            inputs_combined = tokenizer(combined_summary, return_tensors="pt", truncation=False)
            if inputs_combined.input_ids.shape[1] > max_input_length:
                final_summary = summarizer(combined_summary, max_length=150, min_length=40, do_sample=False)
                return final_summary[0]['summary_text']
            else:
                return combined_summary
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return ""
