import unittest

from xquik_export import normalize_xquik_records


DEFAULTS = {
    "tweet_id": "0",
    "airline_sentiment_confidence": 1.0,
    "negativereason": "nan",
    "negativereason_confidence": "nan",
    "airline": "Delta",
    "name": "anonymous",
    "retweet_count": 0,
    "text": "",
    "tweet_created": "2015-02-23 15:29:23-0800",
    "latitude": 0.0,
    "longitude": 0.0,
}


class XquikExportTests(unittest.TestCase):
    def test_maps_export_text_and_metadata_to_model_defaults(self):
        rows = normalize_xquik_records(
            [
                {
                    "id": "abc",
                    "full_text": "Flight delay update",
                    "author_username": "ops",
                    "created_at": "2026-07-04",
                }
            ],
            DEFAULTS,
        )

        self.assertEqual(rows[0]["tweet_id"], "abc")
        self.assertEqual(rows[0]["text"], "Flight delay update")
        self.assertEqual(rows[0]["name"], "ops")
        self.assertEqual(rows[0]["source_id"], "abc")

    def test_skips_blank_text_rows(self):
        rows = normalize_xquik_records(
            [
                {"text": "   ", "id": "skip"},
                {"message": "Useful row"},
            ],
            DEFAULTS,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tweet_id"], "2")


if __name__ == "__main__":
    unittest.main()
