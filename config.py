# config.py
"""
프로젝트 전체에서 공통으로 쓰는
 - 환경 변수 (.env) 로딩
 - 모델·경로 상수 정의
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# 로컬 개발 시 .env 파일 로드
load_dotenv()

# --- AI / LLM 관련 ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
LLM_MODEL: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- MCP 서비스 관련 ---
MCP_URL: str = os.getenv("MCP_URL")
MCP_API_KEY: str = os.getenv("MCP_API_KEY")
MCP_SECRET_KEY: str = os.getenv("MCP_SECRET_KEY")
MCP_MAX_BYTES: int = int(os.getenv("MCP_MAX_BYTES", 10485760)) # 기본값 10MB

# --- SMTP (이메일 발송) 관련 ---
SMTP_HOST: str = os.getenv("SMTP_HOST")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587)) # 기본값 587
SMTP_USER: str = os.getenv("SMTP_USER")
SMTP_PASS: str = os.getenv("SMTP_PASS")
SMTP_FROM: str = os.getenv("SMTP_FROM")
SMTP_SECURITY: str = os.getenv("SMTP_SECURITY", "TLS") # 기본값 TLS
SMTP_TIMEOUT: int = int(os.getenv("SMTP_TIMEOUT", 10)) # 기본값 10초

# 공통 경로
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"