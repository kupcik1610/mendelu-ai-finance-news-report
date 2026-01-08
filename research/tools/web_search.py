"""
Web search functionality using DuckDuckGo.
"""

from dataclasses import dataclass
from ddgs import DDGS


@dataclass
class SearchResult:
    """Represents a single search result."""
    url: str
    title: str
    snippet: str
    date: str
    source: str


def search_news(query: str, max_results: int = 20) -> list[SearchResult]:
    """
    Search for news articles about a topic using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of SearchResult objects
    """
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    date=r.get("date", ""),
                    source=r.get("source", "")
                ))
    except Exception as e:
        print(f"Search failed: {e}")

    return results


def search_web(query: str, max_results: int = 10) -> list[SearchResult]:
    """
    General web search using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of SearchResult objects
    """
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    date="",
                    source=r.get("href", "").split("/")[2] if r.get("href") else ""
                ))
    except Exception as e:
        print(f"Search failed: {e}")

    return results
