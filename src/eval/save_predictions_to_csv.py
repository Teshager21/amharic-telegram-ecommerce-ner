# 📁 src/eval/save_predictions_to_csv.py
import sys
import pathlib
import csv
from src.eval.evaluate import evaluate_and_return_preds

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

LABEL_LIST = ["B-PRODUCT", "I-LOC", "I-PRICE", "I-PRODUCT", "O"]

MODEL_PATH = "models/ner"
TEST_DATA_PATH = "data/labeled/eval_split.conll"
OUTPUT_CSV = "reports/ner_predictions.csv"

# Run evaluation and get token-level predictions
try:
    print("📊 Evaluating and collecting predictions...")
    predictions = evaluate_and_return_preds(
        model_path=MODEL_PATH,
        data_path=TEST_DATA_PATH,
        label_list=LABEL_LIST,
        return_tokens=True,  # ensure it includes tokens
    )

    print(f"💾 Saving predictions to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Token", "True_Label", "Predicted_Label"])
        for sample in predictions:
            for token, true_label, pred_label in zip(
                sample["tokens"], sample["true_labels"], sample["pred_labels"]
            ):
                writer.writerow([token, true_label, pred_label])

    print("✅ Done. Predictions saved.")

except Exception as e:
    print(f"❌ Evaluation or save failed: {e}")
