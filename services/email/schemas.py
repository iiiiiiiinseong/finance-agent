# services/email/schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class EmailSendRequest(BaseModel):
    to: EmailStr
    subject: str = Field(..., min_length=1)
    text: Optional[str] = None
    html: Optional[str] = None

class EmailSendResponse(BaseModel):
    ok: bool
    status_code: int
    message_id: Optional[str] = None
    detail: Optional[str] = None
