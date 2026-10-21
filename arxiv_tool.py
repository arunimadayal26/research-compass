# Step1: Access arXiv using URL (rate-limit safe)
import re
import time
import threading
import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool

ARXIV_URL = "https://export.arxiv.org/api/query"  # https, not http (avoids a redirect)
HEADERS = {"User-Agent": "research-compass/0.1 (mailto:arunimadayal2608@gmail.com)"}

MIN_INTERVAL = 3.1   # arXiv asks for max 1 request every 3 seconds
MAX_RETRIES = 2      # fail fast; the LLM gets the error and can tell the user
MAX_DELAY = 8        # never sleep longer than this between retries

# ToolNode runs tool calls in parallel threads, so the limiter must be global + locked.
_lock = threading.Lock()
_last_request_time = 0.0
_cache: dict[tuple, dict] = {}  # (query, max_results) -> parsed result


def _clean_query(topic: str) -> str:
    # Keep letters, digits and spaces only, then join words with "+"
    topic = re.sub(r"[^\w\s-]", " ", topic.lower())
    return "+".join(topic.split())


def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    query = _clean_query(topic)
    if not query:
        raise ValueError("Empty search topic")

    key = (query, max_results)
    if key in _cache:
        print(f"arXiv cache hit: {query}")
        return _cache[key]

    url = (
        f"{ARXIV_URL}?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )

    global _last_request_time
    resp = None
    for attempt in range(MAX_RETRIES):
        # Only one request at a time, always >= MIN_INTERVAL apart
        with _lock:
            wait = MIN_INTERVAL - (time.monotonic() - _last_request_time)
            if wait > 0:
                time.sleep(wait)
            print(f"Making request to arXiv API (attempt {attempt + 1}): {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            _last_request_time = time.monotonic()

        if resp.status_code == 429 or resp.status_code >= 500:
            # Respect Retry-After if present, otherwise exponential backoff
            retry_after = resp.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 4 * (2 ** attempt)
            delay = min(delay, MAX_DELAY)
            print(f"arXiv returned {resp.status_code}; retrying in {delay}s")
            time.sleep(delay)
            continue
        break

    if resp is None or not resp.ok:
        raise ValueError(f"arXiv API error: {getattr(resp, 'status_code', 'no response')}")

    data = parse_arxiv_xml(resp.text)
    _cache[key] = data
    return data


# Step2: Parse XML
def parse_arxiv_xml(xml_content: str) -> dict:
    """Parse the XML content from arXiv API response."""
    entries = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_content)
    for entry in root.findall("atom:entry", ns):
        authors = [
            a.findtext("atom:name", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        categories = [c.attrib.get("term") for c in entry.findall("atom:category", ns)]

        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        entries.append({
            "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
            "summary": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
            "authors": authors,
            "categories": categories,
            "pdf": pdf_link,
        })

    return {"entries": entries}


# Step3: Convert the functionality into a tool
@tool
def arxiv_search(topic: str) -> dict:
    """Search for recently uploaded arXiv papers

    Args:
        topic: The topic to search for papers about

    Returns:
        Dict with an "entries" list of papers (title, authors, summary, categories, pdf link)
    """
    print("ARXIV Agent called")
    print(f"Searching arXiv for papers about: {topic}")
    try:
        papers = search_arxiv_papers(topic)
    except Exception as e:
        # Return the error as text so the LLM can react instead of crashing the whole graph
        return {"entries": [], "error": f"arXiv search failed: {e}. Tell the user and try again later."}

    if not papers["entries"]:
        return {"entries": [], "error": f"No papers found for topic: {topic}. Try different keywords."}

    print(f"Found {len(papers['entries'])} papers about {topic}")
    return papers