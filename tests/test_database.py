"""
Unit tests for ClaimLens database operations (SQLite).
Tests initialization, insertion, querying, search, deletion, clearing, and statistics.
"""

import os
import tempfile
import unittest

from database import (
    clear_history,
    delete_analysis,
    get_analysis,
    get_dashboard_stats,
    get_history,
    init_db,
    save_analysis,
)


class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        # Create a unique temporary database file for isolation
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_claimlens.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_analysis(self):
        rec_id = save_analysis(
            claim_text="NASA confirmed 15 days of darkness in November.",
            model_prediction="LIKELY FAKE",
            model_score=0.89,
            factcheck_status="PUBLISHED FACT-CHECK FOUND",
            factcheck_publisher="Snopes",
            factcheck_url="https://www.snopes.com/fact-check/15-days-darkness-november/",
            relevance_summary="Evaluated 2 passages",
            db_path=self.db_path,
        )
        self.assertGreater(rec_id, 0)

        record = get_analysis(rec_id, db_path=self.db_path)
        self.assertIsNotNone(record)
        self.assertEqual(record["model_prediction"], "LIKELY FAKE")
        self.assertEqual(record["factcheck_publisher"], "Snopes")
        self.assertAlmostEqual(record["model_score"], 0.89, places=2)

    def test_search_history(self):
        save_analysis(
            claim_text="ISRO Chandrayaan-3 soft touchdown on lunar surface",
            model_prediction="LIKELY REAL",
            model_score=0.92,
            factcheck_status="PUBLISHED FACT-CHECK FOUND",
            db_path=self.db_path,
        )
        save_analysis(
            claim_text="Drinking lemon juice cures all cancer",
            model_prediction="LIKELY FAKE",
            model_score=0.95,
            factcheck_status="PUBLISHED FACT-CHECK FOUND",
            db_path=self.db_path,
        )

        # Search for Chandrayaan
        results = get_history(search_query="Chandrayaan", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertIn("Chandrayaan", results[0]["claim_text"])

        # Search for FAKE
        fake_results = get_history(search_query="LIKELY FAKE", db_path=self.db_path)
        self.assertEqual(len(fake_results), 1)
        self.assertEqual(fake_results[0]["model_prediction"], "LIKELY FAKE")

    def test_delete_analysis(self):
        rec_id = save_analysis(
            claim_text="Test claim to be deleted",
            model_prediction="LIKELY REAL",
            db_path=self.db_path,
        )
        success = delete_analysis(rec_id, db_path=self.db_path)
        self.assertTrue(success)

        record = get_analysis(rec_id, db_path=self.db_path)
        self.assertIsNone(record)

    def test_clear_history(self):
        for i in range(5):
            save_analysis(
                claim_text=f"Batch test claim #{i}",
                model_prediction="LIKELY REAL",
                db_path=self.db_path,
            )
        cleared_count = clear_history(db_path=self.db_path)
        self.assertEqual(cleared_count, 5)

        remaining = get_history(db_path=self.db_path)
        self.assertEqual(len(remaining), 0)

    def test_dashboard_statistics(self):
        save_analysis(
            claim_text="Real claim 1",
            model_prediction="LIKELY REAL",
            factcheck_status="PUBLISHED FACT-CHECK FOUND",
            db_path=self.db_path,
        )
        save_analysis(
            claim_text="Fake claim 1",
            model_prediction="LIKELY FAKE",
            factcheck_status="PUBLISHED FACT-CHECK FOUND",
            db_path=self.db_path,
        )
        save_analysis(
            claim_text="Unverified claim 1",
            model_prediction="LIKELY REAL",
            factcheck_status="UNVERIFIED",
            db_path=self.db_path,
        )

        stats = get_dashboard_stats(db_path=self.db_path)
        self.assertEqual(stats["total_analyses"], 3)
        self.assertEqual(stats["likely_real_count"], 2)
        self.assertEqual(stats["likely_fake_count"], 1)
        self.assertEqual(stats["unverified_count"], 1)
        self.assertEqual(len(stats["recent_analyses"]), 3)


if __name__ == "__main__":
    unittest.main()
