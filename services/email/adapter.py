# services/email/adapter.py
from __future__ import annotations
import os, logging
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from .schemas import EmailSendRequest, EmailSendResponse

logger = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/v1/messages")  # 로컬 기본
MCP_API_KEY = os.getenv("MCP_API_KEY", "")

# 10MB 제한 기준
MAX_BYTES = 10 * 1024 * 1024

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _progress_cb(monitor: MultipartEncoderMonitor):
    # 필요하면 진행률 로깅
    pass

def _multipart_payload(req: EmailSendRequest, pdf_path: Path) -> MultipartEncoderMonitor:
    fields = {
        "to": req.to,
        "subject": req.subject,
    }
    if req.text:
        fields["text"] = req.text
    if req.html:
        fields["html"] = req.html

    fp = open(pdf_path, "rb")
    encoder = MultipartEncoder(
        fields={
            **fields,
            "attachment": ("product.pdf", fp, "application/pdf"),
        }
    )
    monitor = MultipartEncoderMonitor(encoder, _progress_cb)
    return monitor

def send_product_email(to_addr: str, product_name: str, pdf_path: str) -> bool:
    """
    기존 app.py와의 호환을 위해 bool만 반환합니다.
    세부 응답이 필요하면 아래 send_email_with_resp를 사용하세요.
    """
    resp = send_email_with_resp(
        to_addr=to_addr,
        subject=f"[우리은행] {product_name} 상품설명서",
        pdf_path=pdf_path,
        text="첨부된 상품설명서를 확인해 주세요.",
        html="첨부된 상품설명서를 확인해 주세요.",
    )
    if not resp.ok:
        logger.warning("Email send failed: %s %s", resp.status_code, resp.detail)
    return resp.ok

def send_email_with_resp(
    to_addr: str, subject: str, pdf_path: str, text: str | None = None, html: str | None = None
) -> EmailSendResponse:
    path = Path(pdf_path)
    if not path.exists():
        return EmailSendResponse(ok=False, status_code=400, detail="PDF file not found")

    if path.stat().st_size > MAX_BYTES:
        return EmailSendResponse(ok=False, status_code=413, detail="Attachment exceeds 10MB")

    req = EmailSendRequest(to=to_addr, subject=subject, text=text, html=html)

    monitor = _multipart_payload(req, path)
    headers = {"Content-Type": monitor.content_type}
    auth = None
    # 필요 시 Header 인증으로 변경하세요.
    if MCP_API_KEY:
        headers["Authorization"] = f"Bearer {MCP_API_KEY}"

    try:
        s = _session()
        r = s.post(MCP_URL, data=monitor, headers=headers, timeout=30)
        status = r.status_code
        if status >= 400:
            return EmailSendResponse(ok=False, status_code=status, detail=r.text)
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        return EmailSendResponse(
            ok=True, status_code=status, message_id=data.get("message_id"), detail=data.get("detail")
        )
    except requests.RequestException as e:
        return EmailSendResponse(ok=False, status_code=0, detail=str(e))
