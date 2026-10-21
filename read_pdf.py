import io
import os
import time

import PyPDF2
import requests
from langchain_core.tools import tool

# ~4 chars per token. Lower this if your LLM provider has a small tokens-per-minute limit.
MAX_CHARS = int(os.getenv("READ_PDF_MAX_CHARS", "20000"))
MAX_PAGES = 15


@tool
def read_pdf(url: str) -> str:
    """Read and extract text from a PDF file given its URL.

    Args:
        url: The URL of the PDF file to read

    Returns:
        The extracted text content from the PDF (truncated for long papers)
    """
    t0 = time.monotonic()
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "research-compass/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        total = len(reader.pages)

        text = ""
        for i, page in enumerate(reader.pages[:MAX_PAGES], 1):
            text += (page.extract_text() or "") + "\n"
            if len(text) >= MAX_CHARS:
                break

        text = text[:MAX_CHARS].strip()
        note = f"\n\n[Truncated: read first {min(MAX_PAGES, total)} of {total} pages]" if total > MAX_PAGES or len(text) >= MAX_CHARS else ""
        print(f"read_pdf: {len(text)} chars in {time.monotonic() - t0:.1f}s")
        return text + note
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return f"Could not read PDF: {e}"