"""
ClaimLens — Evidence Relevance Detection Engine
Utilizes TF-IDF vectorization and Cosine Similarity to evaluate topical
relatedness between a user claim and retrieved evidence passages.

CRITICAL DISTINCTION:
Relevance measures semantic/topical similarity, NOT factual correctness.
A relevant passage can support, contradict, or neutrally discuss a claim.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default heuristic thresholds for relevance categorization
# Calibrated for short single-sentence claims evaluated against passage snippets
THRESHOLD_RELEVANT = 0.10
THRESHOLD_POSSIBLY_RELEVANT = 0.03


class EvidenceRelevanceEngine:
    """
    Engine to assess relevance of textual evidence passages against a target claim.
    """

    def __init__(
        self,
        threshold_relevant: float = THRESHOLD_RELEVANT,
        threshold_possible: float = THRESHOLD_POSSIBLY_RELEVANT,
    ):
        self.threshold_relevant = threshold_relevant
        self.threshold_possible = threshold_possible

    def evaluate_passages(
        self,
        claim_text: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of evidence passage dictionaries against the claim_text.

        Each item in passages should contain at least:
        - 'text': the passage body string
        - 'source' or 'publisher': optional provenance label

        Returns the passage list augmented with:
        - 'similarity_score': float rounded to 4 decimal places
        - 'similarity_percentage': float percentage (0-100%)
        - 'classification': 'RELEVANT', 'POSSIBLY RELEVANT', or 'NOT RELEVANT'
        - 'explanation': contextual explanation of the relevance classification
        - 'shared_terms': list of top overlapping topical terms
        """
        if not claim_text or not passages:
            return []

        clean_claim = claim_text.strip()
        corpus = [clean_claim] + [p.get("text", "").strip() for p in passages]

        # Use unigram and bigram TF-IDF with sublinear scaling
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            lowercase=True,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = np.array(vectorizer.get_feature_names_out())
            claim_vec = tfidf_matrix[0:1]
            passage_vecs = tfidf_matrix[1:]

            similarities = cosine_similarity(claim_vec, passage_vecs)[0]
        except ValueError:
            # Occurs if vocabulary is empty (e.g. only stopwords or symbols)
            similarities = [0.0] * len(passages)
            feature_names = np.array([])

        results = []
        for idx, passage in enumerate(passages):
            score = float(similarities[idx]) if idx < len(similarities) else 0.0
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1

            if score >= self.threshold_relevant:
                classification = "RELEVANT"
                rationale = (
                    "High topical overlap. The passage directly discusses the key "
                    "entities and subject matter referenced in the claim."
                )
            elif score >= self.threshold_possible:
                classification = "POSSIBLY RELEVANT"
                rationale = (
                    "Moderate topical overlap. The passage shares related contextual concepts "
                    "or background terminology, but may not directly evaluate the claim."
                )
            else:
                classification = "NOT RELEVANT"
                rationale = (
                    "Low or negligible similarity. The passage pertains to a different "
                    "topic or context and provides no direct evidence regarding the claim."
                )

            # Find shared top n-grams if feature names are available
            shared_terms = []
            if feature_names.size > 0 and idx + 1 < tfidf_matrix.shape[0]:
                prod = tfidf_matrix[0].multiply(tfidf_matrix[idx + 1]).toarray()[0]
                top_indices = prod.argsort()[::-1]
                for ti in top_indices:
                    if prod[ti] > 0:
                        shared_terms.append(str(feature_names[ti]))
                    if len(shared_terms) >= 4:
                        break

            enriched = dict(passage)
            enriched.update(
                {
                    "similarity_score": round(score, 4),
                    "similarity_percentage": round(score * 100, 1),
                    "classification": classification,
                    "explanation": rationale,
                    "shared_terms": shared_terms,
                }
            )
            results.append(enriched)

        # Sort by similarity score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results


def evaluate_evidence_relevance(
    claim_text: str,
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convenience function using default thresholds."""
    engine = EvidenceRelevanceEngine()
    return engine.evaluate_passages(claim_text, passages)
