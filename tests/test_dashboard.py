import io
import unittest

import pandas as pd

from app.app import process_csv
from src.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class StubPredictor:
    def __init__(self):
        self.batch_sizes = []

    def predict_batches(self, input_batches):
        for batch in input_batches:
            self.batch_sizes.append(len(batch))
            result = batch.copy()
            result["predicted_label"] = ["Normal"] * len(batch)
            result["prediction_confidence"] = [0.9] * len(batch)
            yield result


class DashboardCsvTests(unittest.TestCase):
    def make_csv(self, rows):
        columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        data = {
            column: (
                ["tcp"] * rows if column in CATEGORICAL_FEATURES else [0] * rows
            )
            for column in columns
        }
        return io.StringIO(pd.DataFrame(data).to_csv(index=False))

    def test_processes_csv_incrementally_and_preserves_features(self):
        predictor = StubPredictor()

        csv_content, preview, row_count, counts = process_csv(
            self.make_csv(5), predictor, batch_size=2
        )

        self.assertEqual(row_count, 5)
        self.assertEqual(predictor.batch_sizes, [2, 2, 1])
        self.assertEqual(counts, {"Normal": 5})
        self.assertIn("duration", preview.columns)
        self.assertIn("predicted_label", preview.columns)
        self.assertEqual(len(pd.read_csv(io.StringIO(csv_content))), 5)

    def test_rejects_csv_missing_feature_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required NSL-KDD"):
            process_csv(
                io.StringIO("duration\n1\n"),
                StubPredictor(),
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
