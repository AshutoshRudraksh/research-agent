from langchain_core.tools import tool
from ddgs import DDGS
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("reports")
READ_FAIL_PREFIX = "Failed to read URL:"
WRITE_OK_PREFIX = "Report saved to "
SEARCH_EMPTY = "SEARCH_EMPTY"
SEARCH_ERROR = "SEARCH_ERROR"


def report_path(topic: str, when: datetime | None = None, reports_dir: Path | None = None) -> Path:
    reports = (reports_dir or REPORTS_DIR).resolve()
    reports.mkdir(exist_ok=True)
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M")
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:40] or "report"
    path = (reports / f"{slug}_{stamp}.md").resolve()
    if path.parent != reports:
        raise ValueError("refusing to write outside reports/")
    return path


@tool
def search_web(query: str) -> str:
    """Search the web for a query. Returns top 5 results with titles, URLs, and snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"{SEARCH_EMPTY}: No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"{SEARCH_ERROR}: {str(e)}"


@tool
def read_url(url: str) -> str:
    """Fetch and extract the main text content from a URL. Returns up to 3000 characters."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        clean = "\n".join(lines)

        return clean[:3000] + ("..." if len(clean) > 3000 else "")
    except Exception as e:
        return f"{READ_FAIL_PREFIX} {str(e)}"


@tool
def write_report(topic: str, content: str) -> str:
    """Save a research report as a markdown file. Returns the file path."""
    path = report_path(topic)
    report = f"""# Research Report: {topic}
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

{content}
"""
    path.write_text(report)
    return f"{WRITE_OK_PREFIX}{path}"


tools = [search_web, read_url, write_report]
