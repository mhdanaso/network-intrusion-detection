"""
predict.py — Inference Module
==============================
Load a trained model + preprocessor and make predictions
on new network traffic data.

Usage:
    from src.predict import NetworkIntrusionPredictor
    
    predictor = NetworkIntrusionPredictor("models/best_model.joblib")
    result = predictor.predict(input_data)
"""

import joblib
import numpy as np
import pandas as pd
from src.data_loader import COLUMN_NAMES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


class NetworkIntrusionPredictor:
    """
    End-to-end predictor for network intrusion detection.
    Loads a saved model and preprocessor, then provides
    prediction and probability estimation on new data.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        label_encoder_path: str = None,
    ):
        """
        Initialize the predictor.

        Parameters
        ----------
        model_path : str
            Path to the saved trained model (.joblib).
        preprocessor_path : str
            Path to the saved preprocessor (.joblib).
        label_encoder_path : str, optional
            Path to the saved label encoder (.joblib) for multi-class.
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.label_encoder = joblib.load(label_encoder_path) if label_encoder_path else None

    def predict(self, input_data: pd.DataFrame) -> dict:
        """
        Make predictions on input data.

        Parameters
        ----------
        input_data : pd.DataFrame
            DataFrame with the same feature columns as the training data.

        Returns
        -------
        dict
            {
                "predictions": list of predicted labels,
                "probabilities": list of probability arrays (if available),
            }
        """
        # Ensure correct columns
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = input_data[feature_cols]

        # Preprocess
        X_processed = self.preprocessor.transform(X)

        # Predict
        predictions = self.model.predict(X_processed)

        # Decode labels if label encoder exists
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        # Probabilities
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_processed)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
        }

    def predict_single(self, features: dict) -> dict:
        """
        Predict for a single network connection (from a dict of features).

        Parameters
        ----------
        features : dict
            Dictionary of feature name → value.

        Returns
        -------
        dict
            {"prediction": str, "confidence": float or None}
        """
        df = pd.DataFrame([features])
        result = self.predict(df)

        prediction = result["predictions"][0]
        confidence = None
        if result["probabilities"] is not None:
            confidence = max(result["probabilities"][0])

        return {
            "prediction": prediction,
            "confidence": confidence,
        }
