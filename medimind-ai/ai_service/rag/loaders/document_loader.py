from io import BytesIO

import pdfplumber


def load_pdf(content: bytes, metadata: dict | None = None) -> dict:
    with pdfplumber.open(BytesIO(content)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return {"text": text.strip(), "metadata": metadata or {}}
