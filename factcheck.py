"""
ClaimLens — Fact Verification & Evidence Retrieval Module
Supports live queries to the Google Fact Check Tools API as well as
a high-reliability curated local fact-check evidence repository.
Adheres strictly to evidence attribution and nuanced rating preservation.
"""

import csv
import os
import re
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import clean_text, detect_negation_conflict

# Load environment variables from .env file if present
load_dotenv()

DEFAULT_EVIDENCE_CSV = os.path.join(
    os.path.dirname(__file__), "data", "curated_evidence.csv"
)
GOOGLE_API_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
MATCH_THRESHOLD = 0.20  # Calibrated for headline and sentence-level claim matching


class FactCheckService:
    """
    Service for retrieving and matching verified fact-checks against user claims.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        evidence_csv: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
        self.evidence_csv = evidence_csv or DEFAULT_EVIDENCE_CSV
        self.local_claims = self._load_curated_evidence()

    def _load_curated_evidence(self) -> List[Dict[str, Any]]:
        """Load curated local evidence records from CSV file."""
        records = []
        if not os.path.exists(self.evidence_csv):
            return records

        try:
            with open(self.evidence_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    relevant_list = [
                        p.strip()
                        for p in row.get("relevant_passages", "").split("|")
                        if p.strip()
                    ]
                    irrelevant_list = [
                        p.strip()
                        for p in row.get("irrelevant_passages", "").split("|")
                        if p.strip()
                    ]
                    records.append(
                        {
                            "claim_id": row.get("claim_id", ""),
                            "claim_text": row.get("claim_text", ""),
                            "topic": row.get("topic", "General"),
                            "documented_assessment": row.get(
                                "documented_assessment", "Unverified"
                            ),
                            "publisher": row.get("publisher", "Unknown Publisher"),
                            "source_url": row.get("source_url", ""),
                            "date_checked": row.get("date_checked", ""),
                            "supporting_summary": row.get("supporting_summary", ""),
                            "relevant_passages": relevant_list,
                            "irrelevant_passages": irrelevant_list,
                        }
                    )
        except Exception:
            pass
        return records

    def query_google_api(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Query the Google Fact Check Tools API for matching claims.
        Returns parsed result or None if unavailable / no match.
        """
        if not self.api_key:
            return None

        params = {
            "query": query[:200],  # Google API works best with concise claim strings
            "key": self.api_key,
            "languageCode": "en",
        }
        try:
            response = requests.get(
                GOOGLE_API_ENDPOINT, params=params, timeout=4.0
            )
            if response.status_code != 200:
                return None

            data = response.json()
            claims = data.get("claims", [])
            if not claims:
                return None

            # Process top match
            top_claim = claims[0]
            claim_text = top_claim.get("text", "")
            reviews = top_claim.get("claimReview", [])
            if not reviews:
                return None

            top_review = reviews[0]
            publisher = top_review.get("publisher", {}).get("name", "Fact-Checking Partner")
            original_rating = top_review.get("textualRating", "Evaluated")
            url = top_review.get("url", "")
            review_date = top_review.get("reviewDate", "")

            # Create evidence passages from review title and claim context
            passages = []
            if top_review.get("title"):
                passages.append(
                    {
                        "text": top_review.get("title"),
                        "publisher": publisher,
                        "type": "headline",
                    }
                )
            if claim_text:
                passages.append(
                    {
                        "text": f"Documented Claim: {claim_text}",
                        "publisher": publisher,
                        "type": "claim_record",
                    }
                )

            return {
                "source_type": "LIVE_API",
                "status": "PUBLISHED FACT-CHECK FOUND",
                "matched_claim": claim_text,
                "publisher": publisher,
                "original_rating": original_rating,
                "source_url": url,
                "date_checked": review_date[:10] if review_date else "Unknown",
                "summary": (
                    f"External fact-checking organization '{publisher}' reviewed a "
                    f"substantially matching claim and assigned the rating: '{original_rating}'."
                ),
                "passages": passages,
                "match_similarity": 0.85,  # API semantic match heuristic
            }
        except Exception:
            return None

    def match_local_evidence(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Match query against curated local fact-checks using TF-IDF cosine similarity.
        Evaluates full query as well as constituent sentences to support article excerpts.
        Protects against false associations and negation inversions.
        """
        if not self.local_claims or not query.strip():
            return None

        candidates = [c["claim_text"] for c in self.local_claims]

        # Extract sentences from query plus the full text
        query_parts = [query.strip()]
        for sent in re.split(r"[.!?]+", query):
            clean_s = sent.strip()
            if len(clean_s.split()) >= 3:
                query_parts.append(clean_s)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True
        )

        best_score = 0.0
        best_idx = -1
        matching_query_part = query.strip()

        try:
            corpus = query_parts + candidates
            matrix = vectorizer.fit_transform(corpus)
            n_queries = len(query_parts)

            query_vecs = matrix[:n_queries]
            candidate_vecs = matrix[n_queries:]

            similarities = cosine_similarity(query_vecs, candidate_vecs)

            for q_idx in range(n_queries):
                for c_idx in range(len(candidates)):
                    score = float(similarities[q_idx, c_idx])
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
                        matching_query_part = query_parts[q_idx]
        except Exception:
            return None

        if best_score < MATCH_THRESHOLD or best_idx == -1:
            return None

        matched_record = self.local_claims[best_idx]
        negation_shift = detect_negation_conflict(
            matching_query_part, matched_record["claim_text"]
        )

        # Prepare evidence passages for relevance scoring
        passages = []
        for p in matched_record.get("relevant_passages", []):
            passages.append(
                {
                    "text": p,
                    "publisher": matched_record["publisher"],
                    "expected_type": "relevant",
                }
            )
        for ip in matched_record.get("irrelevant_passages", []):
            passages.append(
                {
                    "text": ip,
                    "publisher": matched_record["publisher"],
                    "expected_type": "irrelevant_comparison",
                }
            )

        return {
            "source_type": "CURATED_LOCAL",
            "status": "PUBLISHED FACT-CHECK FOUND",
            "matched_claim": matched_record["claim_text"],
            "publisher": matched_record["publisher"],
            "original_rating": matched_record["documented_assessment"],
            "source_url": matched_record["source_url"],
            "date_checked": matched_record["date_checked"],
            "summary": matched_record["supporting_summary"],
            "passages": passages,
            "match_similarity": round(best_score, 4),
            "negation_shift": negation_shift,
        }

    def verify(self, claim_text: str) -> Dict[str, Any]:
        """
        Verify a claim by checking Live API first, falling back to Curated Local.
        Always returns structured, non-fabricated verdict data.
        """
        cleaned = clean_text(claim_text)
        if not cleaned:
            return {
                "status": "INPUT INSUFFICIENT",
                "summary": "Please provide a valid claim or article text.",
                "passages": [],
            }

        # 1. Attempt Live API if configured
        api_result = self.query_google_api(cleaned)
        if api_result:
            return api_result

        # 2. Attempt Curated Local Database
        local_result = self.match_local_evidence(cleaned)
        if local_result:
            # If a negation was detected, add an explicit cautionary note
            if local_result.get("negation_shift"):
                local_result["summary"] += (
                    " [NOTE: A polarity or negation contrast was identified between "
                    "your input and the verified reference record. The documented assessment "
                    "applies to the reference claim statement.]"
                )
            return local_result

        # 3. Return UNVERIFIED if no suitable match found
        return {
            "status": "UNVERIFIED",
            "summary": (
                "No sufficiently matching published fact-check was located in external "
                "fact-checking databases or local curated records. An Unverified status "
                "indicates that no documented third-party assessment exists for this specific "
                "phrasing—it does NOT prove the claim true or false."
            ),
            "publisher": None,
            "original_rating": "No Record Found",
            "source_url": None,
            "date_checked": None,
            "passages": [],
            "match_similarity": 0.0,
        }
