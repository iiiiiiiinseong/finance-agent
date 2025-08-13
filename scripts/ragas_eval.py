# scripts/ragas_eval.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ragas.metrics import answer_relevancy, faithfulness, context_precision
from ragas import evaluate
import pandas as pd, json

from config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_PATH = PROJECT_ROOT / "predictions.jsonl"
GT_PATH = DATA_DIR / "processed" / "faq_woori_structured.jsonl"

refs = [json.loads(l) for l in open(GT_PATH, encoding="utf-8")]
preds = [json.loads(l) for l in open(PREDICT_PATH, encoding="utf-8")]

df = pd.DataFrame({
    "question": [p["question"] for p in preds],
    "ground_truth": [r["answer"] for r in refs],
    "context": [p["context"] for p in preds],
    "answer": [p["answer"] for p in preds],
})

report = evaluate(
    df,
    metrics=[answer_relevancy, faithfulness, context_precision]
)
print(report)
