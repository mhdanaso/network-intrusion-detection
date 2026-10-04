import unittest

import numpy as np
import pandas as pd

from src.predict import NetworkIntrusionPredictor


class StubPredictor(NetworkIntrusionPredictor):
    def __init__(self):
        pass

    def predict(self, input_data):
        return {
            "predictions": ["Normal"] * len(input_data),
            "probabilities": [[0.8, 0.2]] * len(input_data),
        }


class PredictBatchesTests(unittest.TestCase):
    def test_yields_predictions_and_confidence_for_each_batch(self):
        predictor = StubPredictor()
        batches = [
            pd.DataFrame({"duration": [1, 2]}),
            pd.DataFrame({"duration": [3]}),
        ]

        results = list(predictor.predict_batches(iter(batches)))

        self.assertEqual([len(result) for result in results], [2, 1])
        self.assertEqual(
            results[0]["predicted_label"].tolist(),
            ["Normal", "Normal"],
        )
        np.testing.assert_allclose(
            results[1]["prediction_confidence"].to_numpy(),
            [0.8],
        )
        self.assertEqual(batches[0].columns.tolist(), ["duration"])

    def test_rejects_reserved_prediction_columns(self):
        predictor = StubPredictor()
        batch = pd.DataFrame({"predicted_label": ["existing"]})

        with self.assertRaisesRegex(ValueError, "reserved output column"):
            next(predictor.predict_batches([batch]))


if __name__ == "__main__":
    unittest.main()
