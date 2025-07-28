# config.py
"""
프로젝트 전체에서 공통으로 쓰는
 - 환경 변수 (.env) 로딩
 - 모델·경로 상수 정의
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# .env 읽어오기
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# 필수값
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set in .env"

# 모델관련
LLM_MODEL: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 공통 경로
DATA_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "index"
