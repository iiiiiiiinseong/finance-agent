# mcp_server/main.py
"""
MCP server 실행. uvicorn mcp_server.main:app --reload
"""
from fastapi import FastAPI, UploadFile, status, Depends, Form, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from smtplib import SMTP, SMTP_SSL, SMTPAuthenticationError, SMTPSenderRefused
from email.message import EmailMessage
from starlette.concurrency import run_in_threadpool

import os, ssl, asyncio, logging
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).resolve().parents[1] / ".env.mcp"
load_dotenv(dotenv_path=dotenv_path)

# ---------- ENV ----------
MCP_SECRET_KEY = os.getenv("MCP_SECRET_KEY") 
API_KEY_NAME = "Authorization"
api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

MAX_BYTES = int(os.getenv("MCP_MAX_BYTES", str(10 * 1024 * 1024)))  # 기본 10MB
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)        # 표시 From (없으면 USER)
SMTP_SECURITY = os.getenv("SMTP_SECURITY", "TLS")    # "TLS" | "SSL"
SMTP_TIMEOUT = float(os.getenv("SMTP_TIMEOUT", "10"))
SMTP_SEND_TIMEOUT = float(os.getenv("SMTP_SEND_TIMEOUT", "20"))


class SendResult(BaseModel):
    ok: bool
    message_id: str | None = None
    detail: str | None = None

async def get_api_key(api_key_header: str = Depends(api_key_header_scheme)):
    if not api_key_header or not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    token = api_key_header.split("Bearer ")[1]

    if token != MCP_SECRET_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return token

app = FastAPI(title="MCP Email Server")
logger = logging.getLogger("mcp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def _send_via_smtp(to_addr: str, subject: str, text: str | None, html: str | None,
                   attachment: tuple[str, bytes, str]) -> str:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise RuntimeError("SMTP env not configured (SMTP_HOST/PORT/USER/PASS)")

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    if html:
        msg.set_content(text or "첨부를 확인해 주세요.")
        msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(text or "첨부를 확인해 주세요.")

    filename, file_bytes, mime = attachment
    maintype, subtype = (mime or "application/pdf").split("/", 1)
    msg.add_attachment(file_bytes, maintype=maintype, subtype=subtype, filename=filename)

    context = ssl.create_default_context()

    if SMTP_SECURITY.upper() == "SSL":
        # e.g. Gmail 465, Naver 465
        with SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        # TLS (STARTTLS) e.g. Gmail 587, Naver 587
        with SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

    # 간단한 의사 message-id
    return f"smtp:{abs(hash((to_addr, subject, len(file_bytes)))) & 0xffffffff:08x}"

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/v1/messages", dependencies=[Depends(get_api_key)])
async def send_message(
    to: EmailStr = Form(...),
    subject: str = Form(...),
    text: str | None = Form(None),
    html: str | None = Form(None),
    attachment: UploadFile | None = None,
):
    try:
        # 파일 스트리밍 수신 + 사이즈 제한
        total = 0
        file_bytes = b""
        filename = "attachment.pdf"
        mime = "application/pdf"
        if attachment is not None:
            filename = attachment.filename or filename
            mime = attachment.content_type or mime
            chunks = []
            while True:
                chunk = await attachment.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Attachment exceeds size limit")
                chunks.append(chunk)
            file_bytes = b"".join(chunks)

        logger.info("recv to=%s subject=%s size=%d mime=%s", to, subject, total, mime)

        message_id = await asyncio.wait_for(
            run_in_threadpool(_send_via_smtp, str(to), subject, text, html, (filename, file_bytes, mime)),
            timeout=SMTP_SEND_TIMEOUT,
        )
        return JSONResponse(status_code=200, content={"ok": True, "message_id": message_id})
    except SMTPAuthenticationError:
        # Gmail: App Password 미설정/오입력, Naver: SMTP 허용 안 됨 등
        raise HTTPException(status_code=401, detail="SMTP authentication failed. Check password/app password & provider settings.")
    except SMTPSenderRefused as e:
        raise HTTPException(status_code=403, detail=f"Sender refused: {e}")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="SMTP send timed out")
    except Exception as e:
        logger.exception("send error")
        return JSONResponse(status_code=500, content={"ok": False, "detail": str(e)})
