"""
Unit tests for ClaimLens inference, input validation, and fact-checking.
Tests validation bounds, negation conflict detection, model prediction loading,
and curated evidence matching.
"""

import os
import unittest
import joblib

from factcheck import FactCheckService
from utils import clean_text, detect_negation_conflict, get_text_statistics, validate_input


class TestInferenceAndValidation(unittest.TestCase):
    def test_clean_text(self):
        dirty = "  Breaking   news:   NASA’s “secret”   dossier <p>revealed</p>!  "
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "Breaking news: NASA's \"secret\" dossier revealed!")

    def test_validate_input(self):
        # Empty input
        val, msg = validate_input("")
        self.assertFalse(val)

        # Too short input (1 word)
        val, msg = validate_input("Hello")
        self.assertFalse(val)
        self.assertIn("too brief", msg.lower())

        # Valid input
        val, msg = validate_input(
            "The Federal Reserve announced it will hold interest rates steady."
        )
        self.assertTrue(val)
        self.assertEqual(msg, "")

    def test_text_statistics(self):
        text = "NASA confirmed no darkness. Scientists investigated the hoax."
        stats = get_text_statistics(text)
        self.assertEqual(stats["word_count"], 8)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertGreater(stats["char_count"], 20)

    def test_negation_conflict(self):
        affirmed = "NASA confirmed total darkness in November."
        negated = "NASA confirmed no darkness in November and called it a hoax."
        has_conflict = detect_negation_conflict(affirmed, negated)
        self.assertTrue(has_conflict)

        both_affirm = "ISRO launched Chandrayaan-3 successfully to the Moon."
        other_affirm = "The Indian space agency reached the lunar south pole."
        self.assertFalse(detect_negation_conflict(both_affirm, other_affirm))

    def test_curated_factcheck_matching(self):
        service = FactCheckService()

        # Query matching EV-001 (NASA 15 days darkness)
        res = service.verify("NASA announced 15 days of total darkness across Earth.")
        self.assertEqual(res["status"], "PUBLISHED FACT-CHECK FOUND")
        self.assertEqual(res["publisher"], "Snopes")
        self.assertIn("snopes.com", res["source_url"])
        self.assertGreater(len(res["passages"]), 0)

        # Query with unknown claim -> UNVERIFIED
        unknown = service.verify(
            "Springfield local municipal council voted on street lighting repairs."
        )
        self.assertEqual(unknown["status"], "UNVERIFIED")
        self.assertIsNone(unknown["publisher"])

    def test_trained_model_pipeline(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "claimlens_model.joblib"
        )
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            sample_real = (
                "The European Central Bank lowered interest rates by 25 basis points "
                "following a sustained decline in eurozone inflation metrics."
            )
            sample_fake = (
                "SHOCKING: Leaked secret government dossier reveals microchips dissolved "
                "in drinking water caps to control human thoughts and brain waves!"
            )

            pred_real = pipeline.predict([sample_real])[0]
            pred_fake = pipeline.predict([sample_fake])[0]
            proba_real = pipeline.predict_proba([sample_real])[0]
            proba_fake = pipeline.predict_proba([sample_fake])[0]

            self.assertIn(pred_real, [0, 1])
            self.assertIn(pred_fake, [0, 1])
            self.assertEqual(len(proba_real), 2)
            self.assertEqual(len(proba_fake), 2)


if __name__ == "__main__":
    unittest.main()
