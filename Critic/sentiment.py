import logging
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import defaultdict

# Load spaCy for better text processing
try:
    nlp = spacy.load('en_core_web_sm')
except:
    logging.warning("Please install spaCy model: python -m spacy download en_core_web_sm")

device_id = 0

# Enhanced aspect keywords with related terms
ASPECT_KEYWORDS = {
    'graphics': [
        'graphics', 'visual', 'visuals', 'graphic', 'texture', 'textures',
        'animation', 'animations', 'rendering', 'resolution', 'fps', 'frame rate',
        'ray tracing', 'lighting', 'shadows', 'effects'
    ],
    'gameplay': [
        'gameplay', 'mechanic', 'mechanics', 'control', 'controls', 'difficulty',
        'combat', 'system', 'systems', 'movement', 'interaction', 'interactions',
        'design', 'balance', 'progression', 'features'
    ],
    'story': [
        'story', 'narrative', 'plot', 'character', 'characters', 'writing',
        'dialogue', 'cutscene', 'cutscenes', 'storytelling', 'pacing',
        'ending', 'story arc', 'lore', 'world-building'
    ],
    'performance': [
        'performance', 'optimization', 'fps', 'frame rate', 'lag', 'stutter',
        'crash', 'crashes', 'bug', 'bugs', 'loading', 'stable', 'stability',
        'technical', 'running', 'runs'
    ]
}


def get_aspect_context(text, aspect_term, window_size=50):
    """Extract context around aspect mentions."""
    doc = nlp(text.lower())
    contexts = []

    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in ASPECT_KEYWORDS[aspect_term]):
            contexts.append(sent.text)

    return ' '.join(contexts)


def calculate_confidence(scores):
    """Calculate confidence score based on variance and sample size."""
    if not scores:
        return 0.0

    mean_score = sum(scores) / len(scores)
    variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
    confidence = 1 / (1 + variance)  # Higher variance = lower confidence
    sample_size_factor = min(len(scores) / 5, 1)  # More samples = higher confidence

    return confidence * sample_size_factor


def find_aspects(text, aspects, classifier=None):
    """
    Enhanced aspect-based sentiment analysis with confidence scores and context.
    """
    if classifier is None:
        model_name = "yangheng/deberta-v3-base-absa-v1.1"
        tokenizer_absa = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model_absa = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda")
        classifier = pipeline(
            "text-classification",
            model=model_absa,
            tokenizer=tokenizer_absa,
            device=device_id
        )

    results = {}
    for aspect in aspects:
        try:
            # Get all relevant context for this aspect
            context = get_aspect_context(text, aspect)
            if not context:
                results[aspect] = {
                    'score': 0.0,
                    'confidence': 0.0,
                    'mention_count': 0
                }
                continue

            # Split context into manageable chunks
            sentences = context.split('.')
            scores = []
            confidences = []

            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue

                classification = classifier(sentence, text_pair=aspect)[0]
                score = classification['score']

                if classification['label'] == 'Negative':
                    score = -score

                scores.append(score)
                confidences.append(classification['score'])  # Original confidence

            if scores:
                avg_score = sum(scores) / len(scores)
                confidence = calculate_confidence(scores)

                results[aspect] = {
                    'score': avg_score,
                    'confidence': confidence,
                    'mention_count': len(scores)
                }
            else:
                results[aspect] = {
                    'score': 0.0,
                    'confidence': 0.0,
                    'mention_count': 0
                }

        except Exception as e:
            logging.error(f"Error analyzing aspect '{aspect}': {str(e)}")
            results[aspect] = {
                'score': 0.0,
                'confidence': 0.0,
                'mention_count': 0
            }

    return results