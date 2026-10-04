"""Streamlit dashboard for model evaluation and batched CSV predictions."""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.predict import NetworkIntrusionPredictor


MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = MODEL_DIR / "training_results.json"
OUTPUT_COLUMNS = {"predicted_label", "prediction_confidence"}
PREVIEW_ROWS = 500
SPOOL_LIMIT_BYTES = 8 * 1024 * 1024


def process_csv(uploaded_file, predictor, batch_size):
    """Run inference in bounded row batches and spool the downloadable CSV."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    prediction_counts = Counter()
    preview_frames = []
    preview_rows = 0
    row_count = 0
    output = tempfile.SpooledTemporaryFile(
        max_size=SPOOL_LIMIT_BYTES,
        mode="w+",
        encoding="utf-8",
        newline="",
    )

    with output:
        chunks = pd.read_csv(uploaded_file, chunksize=batch_size)
        for input_chunk in chunks:
            missing_columns = sorted(set(feature_columns) - set(input_chunk.columns))
            if missing_columns:
                raise ValueError(
                    "CSV is missing required NSL-KDD feature columns: "
                    + ", ".join(missing_columns)
                )

            collisions = OUTPUT_COLUMNS.intersection(input_chunk.columns)
            if collisions:
                raise ValueError(
                    "CSV uses reserved prediction output column(s): "
                    + ", ".join(sorted(collisions))
                )

            result_chunk = next(predictor.predict_batches([input_chunk]))
            result_chunk.to_csv(output, index=False, header=(row_count == 0))
            row_count += len(result_chunk)
            prediction_counts.update(result_chunk["predicted_label"].astype(str))

            preview_count = min(PREVIEW_ROWS - preview_rows, len(result_chunk))
            if preview_count > 0:
                preview_frames.append(result_chunk.iloc[:preview_count])
                preview_rows += preview_count

        if row_count == 0:
            raise ValueError("The uploaded CSV has no data rows.")

        output.seek(0)
        csv_content = output.read()

    preview = pd.concat(preview_frames, ignore_index=True)
    return csv_content, preview, row_count, prediction_counts


def show_model_results():
    st.header("Model performance")
    if not RESULTS_PATH.is_file():
        st.info("Train the models to create a results report: `python -m src.train --task both`.")
        return

    try:
        with RESULTS_PATH.open(encoding="utf-8") as results_file:
            results = json.load(results_file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read model results from {RESULTS_PATH}: {error}")
        return

    available_tasks = [name for name in ("binary", "multiclass") if name in results]
    if not available_tasks:
        st.info("No binary or multiclass evaluation results are available yet.")
        return

    task = st.selectbox(
        "Classification task",
        options=available_tasks,
    )
    comparison = results.get(task, {}).get("comparison", [])
    if not comparison:
        st.info(f"No {task} evaluation results are available yet.")
        return

    comparison_frame = pd.DataFrame(comparison)
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
    if "model_name" in comparison_frame and "f1_score" in comparison_frame:
        chart_data = comparison_frame.set_index("model_name")[["f1_score"]]
        st.bar_chart(chart_data)


def show_csv_predictions():
    st.header("Predict from a CSV")
    st.write(
        "Upload a headered CSV containing the 41 NSL-KDD feature columns. "
        "Rows are processed in batches; additional input columns are preserved."
    )
    st.caption(
        "This benchmark model expects NSL-KDD features and is not a live network sensor."
    )

    selected_task = st.selectbox(
        "Prediction task",
        options=("binary", "multiclass"),
        format_func=lambda task: "Normal vs. attack" if task == "binary" else "Attack category",
    )
    batch_size = st.number_input(
        "Rows per inference batch",
        min_value=100,
        max_value=20_000,
        value=2_000,
        step=100,
        help="Lower this value to reduce peak inference memory for large files.",
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if st.button("Run predictions", type="primary", disabled=uploaded_file is None):
        task_artifacts = {
            "model": MODEL_DIR / f"{selected_task}_best_model.joblib",
            "preprocessor": MODEL_DIR / f"{selected_task}_preprocessor.joblib",
        }
        if selected_task == "multiclass":
            task_artifacts["label encoder"] = MODEL_DIR / "multiclass_label_encoder.joblib"
        missing_artifacts = [
            str(path) for path in task_artifacts.values() if not path.is_file()
        ]
        if missing_artifacts:
            st.error(
                "Trained model files are missing. Run "
                "`python -m src.train --task both` first. Missing: "
                + ", ".join(missing_artifacts)
            )
            return

        try:
            predictor = NetworkIntrusionPredictor(
                str(task_artifacts["model"]),
                str(task_artifacts["preprocessor"]),
                str(task_artifacts["label encoder"])
                if "label encoder" in task_artifacts
                else None,
            )
            with st.spinner("Processing CSV in batches..."):
                csv_content, preview, row_count, counts = process_csv(
                    uploaded_file, predictor, int(batch_size)
                )
        except (
            OSError,
            EOFError,
            ImportError,
            UnicodeDecodeError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            st.error(f"Prediction failed: {error}")
            return

        st.success(f"Predictions completed for {row_count:,} rows.")
        st.write("Predicted label counts")
        st.dataframe(
            pd.DataFrame(
                sorted(counts.items()),
                columns=["predicted_label", "rows"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.write(f"Preview (first {len(preview):,} rows)")
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions CSV",
            data=csv_content.encode("utf-8"),
            file_name=f"nids_{selected_task}_predictions.csv",
            mime="text/csv",
        )


def main():
    st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")
    st.title("Network Intrusion Detection")
    st.caption("Explore trained model results and classify NSL-KDD-compatible traffic records.")

    results_tab, predictions_tab = st.tabs(["Model results", "CSV predictions"])
    with results_tab:
        show_model_results()
    with predictions_tab:
        show_csv_predictions()


if __name__ == "__main__":
    main()
