import json, re, sys, pathlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ROOT_DIR, DATA_DIR

# FAQ txt → JSONL 변환
# 스키마 설명 & 샘플 2 레코드
# 스키마 { "topic", "subcategory", "question", "answer" }
# retriever에서 metadata={topic, subcategory} 로 필터링 가능
# 후속 파이프라인 loader → embedder → FAISS/BM25 hybrid 인덱스

# Load the txt file uploaded by the user
txt_path = ROOT_DIR / "data/raw_docs/자주하는질문(FAQ).txt"
text = txt_path.read_text(encoding="utf-8")

records = []
current_category = None
lines = text.splitlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    cat_match = re.match(r"카테고리:\s*(.*)", line)
    if cat_match:
        current_category = cat_match.group(1).strip()
        i += 1
        continue

    if line == "질문":
        # Collect question lines
        i += 1
        question_lines = []
        while i < len(lines) and lines[i].strip() not in ("답변",):
            question_lines.append(lines[i].rstrip())
            i += 1

        # Skip "답변" line
        if i < len(lines) and lines[i].strip() == "답변":
            i += 1

        # Collect answer lines
        answer_lines = []
        while i < len(lines):
            next_line = lines[i]
            if (
                next_line.startswith("=")
                or next_line.strip() == "질문"
                or re.match(r"카테고리:\s*", next_line)
            ):
                break
            answer_lines.append(next_line.rstrip())
            i += 1

        question_text = "\n".join([l.strip() for l in question_lines]).strip()
        answer_text = "\n".join([l.strip() for l in answer_lines]).strip()

        # Split category into topic & subcategory if possible
        if current_category and " - " in current_category:
            topic, subcat = current_category.split(" - ", 1)
        else:
            topic, subcat = current_category, ""

        records.append(
            {
                "topic": topic.strip(),
                "subcategory": subcat.strip(),
                "question": question_text,
                "answer": answer_text,
            }
        )

        # Skip delimiter lines
        while i < len(lines) and lines[i].startswith("="):
            i += 1
        continue

    i += 1

# Save to JSONL
output_path = DATA_DIR / "processed" / "faq_woori_structured.jsonl"

with output_path.open("w", encoding="utf-8") as f:
    for rec in records:
        json.dump(rec, f, ensure_ascii=False)
        f.write("\n")
