"""
Unit tests for Evidence Relevance Engine.
Validates TF-IDF cosine similarity computation, classification thresholds,
and detection of relevant vs irrelevant passages.
"""

import unittest
from relevance import EvidenceRelevanceEngine


class TestRelevanceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EvidenceRelevanceEngine(
            threshold_relevant=0.10, threshold_possible=0.03
        )

    def test_relevant_vs_irrelevant_passages(self):
        claim = "NASA confirmed that Earth will experience 15 days of total darkness in November."

        passages = [
            {
                "text": "NASA confirmed no such darkness event will occur and the claim is a complete hoax.",
                "source": "Snopes",
            },
            {
                "text": "Planetary alignments between Venus and Jupiter occur periodically without blocking sunlight.",
                "source": "Astronomy Journal",
            },
            {
                "text": "The Eiffel Tower in Paris was constructed between 1887 and 1889 as the entrance to the World Fair.",
                "source": "History Archive",
            },
        ]

        results = self.engine.evaluate_passages(claim, passages)
        self.assertEqual(len(results), 3)

        # First result should be the highest similarity (direct match on NASA & darkness)
        top = results[0]
        self.assertIn("NASA confirmed", top["text"])
        self.assertEqual(top["classification"], "RELEVANT")
        self.assertGreaterEqual(top["similarity_score"], 0.10)
        self.assertTrue(len(top["shared_terms"]) > 0)

        # Last result should be the Eiffel Tower passage (completely unrelated)
        last = results[-1]
        self.assertIn("Eiffel Tower", last["text"])
        self.assertEqual(last["classification"], "NOT RELEVANT")
        self.assertLess(last["similarity_score"], 0.03)

    def test_empty_inputs(self):
        # Empty claim
        res1 = self.engine.evaluate_passages("", [{"text": "Sample text"}])
        self.assertEqual(res1, [])

        # Empty passages list
        res2 = self.engine.evaluate_passages("Some claim", [])
        self.assertEqual(res2, [])


if __name__ == "__main__":
    unittest.main()
