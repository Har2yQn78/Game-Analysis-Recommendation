import logging
import re
import torch
from transformers import pipeline, AutoTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

device_id = 0


def preprocess_text(text):
    """Clean and prepare text for summarization."""
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove common review boilerplate
    text = re.sub(r'reviewed on \w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'review copy provided by.*', '', text, flags=re.IGNORECASE)

    return text.strip()


def get_key_sentences(text, n=3):
    """Extract key sentences using TF-IDF."""
    sentences = sent_tokenize(text)
    if len(sentences) <= n:
        return sentences

    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get average TF-IDF score for each sentence
    sentence_scores = np.array([tfidf_matrix[i].mean() for i in range(len(sentences))])

    # Get indices of top n sentences
    top_indices = sentence_scores.argsort()[-n:][::-1]

    return [sentences[i] for i in sorted(top_indices)]


def chunk_text_semantic(text, max_length):
    """Split text into semantic chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def remove_redundancy(summary):
    """Remove redundant information from summary."""
    sentences = sent_tokenize(summary)
    if len(sentences) <= 1:
        return summary

    # Calculate sentence similarities using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate pairwise similarities
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    # Keep sentences with low similarity to previous ones
    unique_sentences = [sentences[0]]
    for i in range(1, len(sentences)):
        max_similarity = max(similarity_matrix[i][:i])
        if max_similarity < 0.7:  # Threshold for similarity
            unique_sentences.append(sentences[i])

    return ' '.join(unique_sentences)


def summarize_text(text, summarizer=None, max_summary_length=150):
    """
    Enhanced text summarization with semantic chunking and redundancy removal.
    """
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # Using a better model
            tokenizer="facebook/bart-large-cnn",
            device=device_id,
            torch_dtype=torch.float16  # Use fp16 for memory efficiency
        )

    try:
        # Preprocess text
        text = preprocess_text(text)

        # Get tokenizer max length
        max_input_length = summarizer.tokenizer.model_max_length

        # Generate summary based on text length
        if len(text.split()) <= max_input_length:
            summary = summarizer(
                text,
                max_length=max_summary_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
        else:
            # Split into semantic chunks
            chunks = chunk_text_semantic(text, max_input_length)

            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = summarizer(
                    chunk,
                    max_length=max_summary_length // len(chunks),
                    min_length=20,
                    do_sample=False
                )[0]['summary_text']
                chunk_summaries.append(summary)

            # Combine summaries
            combined_summary = ' '.join(chunk_summaries)

            # If combined summary is too long, summarize again
            if len(combined_summary.split()) > max_summary_length:
                summary = summarizer(
                    combined_summary,
                    max_length=max_summary_length,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
            else:
                summary = combined_summary

        # Remove redundancy
        summary = remove_redundancy(summary)

        # Add key sentences if summary is too short
        if len(summary.split()) < 50:
            key_sentences = get_key_sentences(text)
            summary = summary + ' ' + ' '.join(key_sentences)

        return summary.strip()

    except Exception as e:
        logging.error(f"Summarization error: {str(e)}")
        return ""