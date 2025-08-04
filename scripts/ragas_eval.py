from ragas.metrics import answer_relevancy, faithfulness, context_precision
from ragas import evaluate
import pandas as pd, json

PREDICT = "predictions.jsonl"    # run demo questions & store outputs
GT      = "data/processed/faq_woori_structured.jsonl"
refs = [json.loads(l) for l in open(GT)]
preds = [json.loads(l) for l in open(PREDICT)]

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
