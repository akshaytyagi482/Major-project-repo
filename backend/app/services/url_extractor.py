from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup


@dataclass
class UrlExtractionResult:
    url: str
    text: str
    title: Optional[str]


def extract_main_text(url: str, timeout_seconds: int = 15) -> UrlExtractionResult:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout_seconds, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError("Unable to fetch URL content.") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "header", "footer", "form", "aside"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None

    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p, main p, p") if p.get_text(strip=True)]
    text = "\n".join(paragraphs)

    if len(text.strip()) < 80:
        text = " ".join(s.strip() for s in soup.stripped_strings)

    if len(text.strip()) < 80:
        raise ValueError("Could not extract enough textual content from URL.")

    return UrlExtractionResult(url=url, text=text.strip(), title=title)
