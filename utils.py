"""
ClaimLens — Utility Functions Module
Provides text preprocessing, input validation, negation detection,
and linguistic feature extraction.
"""

import re
from typing import Dict, List, Tuple


def clean_text(text: str) -> str:
    """
    Standardize text by stripping extra whitespace, normalizing unicode quotes,
    and removing raw markup or trailing control characters.
    Preserves alphanumeric text and basic punctuation for semantic parsing.
    """
    if not text:
        return ""
    # Replace non-standard apostrophes and quotation marks
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple whitespaces/newlines into single space
    text = re.sub(r"\s+", " ", text).strip()
    # Clean spaces directly before punctuation
    text = re.sub(r"\s+([?.!,:;])", r"\1", text)
    return text


def validate_input(text: str, min_chars: int = 15, min_words: int = 3) -> Tuple[bool, str]:
    """
    Validate user-submitted claim or article text.
    Returns (is_valid, error_message).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return False, "Please enter a news headline, claim, or article text to analyze."

    words = cleaned.split()
    if len(cleaned) < min_chars or len(words) < min_words:
        return False, (
            f"Input is too brief for analysis ({len(words)} words, {len(cleaned)} chars). "
            f"Please provide at least a complete headline or sentence ({min_words}+ words)."
        )

    if len(cleaned) > 20000:
        return False, "Input exceeds maximum limit (20,000 characters). Please provide an article excerpt."

    return True, ""


def get_text_statistics(text: str) -> Dict[str, int]:
    """Compute basic text statistics: word count, character count, sentence count."""
    cleaned = clean_text(text)
    words = cleaned.split() if cleaned else []
    sentences = re.split(r"[.!?]+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    return {
        "char_count": len(cleaned),
        "word_count": len(words),
        "sentence_count": max(1, len(sentences)) if words else 0,
    }


def detect_negation_conflict(text_a: str, text_b: str) -> bool:
    """
    Check if one text contains strong negation terms that the other lacks.
    Useful for flagging semantic inversion between a claim and an evidence item
    (e.g., 'NASA confirmed darkness' vs 'NASA did NOT confirm darkness').
    """
    negation_tokens = {
        "not",
        "never",
        "no",
        "neither",
        "nor",
        "false",
        "hoax",
        "debunked",
        "fake",
        "untrue",
        "refuted",
        "denied",
        "disproven",
    }
    words_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    words_b = set(re.findall(r"\b\w+\b", text_b.lower()))

    neg_in_a = bool(words_a & negation_tokens)
    neg_in_b = bool(words_b & negation_tokens)

    # Returns True if one text has negation markers while the other does not
    return neg_in_a != neg_in_b


def extract_key_terms(text: str, max_terms: int = 5) -> List[str]:
    """
    Extract meaningful topical keywords, filtering standard English stop words.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
        "that", "these", "those", "it", "its", "they", "them", "their", "we", "us",
        "our", "you", "your", "he", "him", "his", "she", "her", "will", "would",
        "shall", "should", "may", "might", "must", "can", "could", "of", "as"
    }
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in stop_words]
    # Keep unique in order of appearance
    seen = set()
    result = []
    for token in filtered:
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= max_terms:
            break
    return result
