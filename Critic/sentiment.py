import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set the device id (adjust if needed)
device_id = 0


def find_aspects(text, aspects, classifier=None):
    """
    Evaluates sentiment for each aspect keyword in the review text.
    A positive sentiment adds the classifier's confidence score; negative subtracts it.
    """
    if classifier is None:
        model_name = "yangheng/deberta-v3-base-absa-v1.1"
        tokenizer_absa = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model_absa = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).to(
            "cuda")
        classifier = pipeline("text-classification", model=model_absa, tokenizer=tokenizer_absa, device=device_id)

    aspect_scores = {}
    for aspect in aspects:
        try:
            result = classifier(text, text_pair=aspect)[0]
            score = result['score'] if result['label'] == 'Positive' else -result['score']
            aspect_scores[aspect] = score
        except Exception as e:
            logging.warning(f"ABSA error for aspect '{aspect}': {e}")
            aspect_scores[aspect] = 0.0
    return aspect_scores
