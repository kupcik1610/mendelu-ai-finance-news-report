"""
Web scraping functionality for extracting article content.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from urllib.parse import urlparse
from django.conf import settings


@dataclass
class Article:
    """Represents scraped article content."""
    url: str
    title: str
    content: str
    author: str
    date: str
    source: str
    word_count: int


class WebScraper:
    """Scrapes and extracts content from web pages."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Elements to remove (ads, navigation, etc.)
    REMOVE_TAGS = [
        "script", "style", "nav", "header", "footer",
        "aside", "advertisement", "iframe", "noscript",
        "form", "button", "input", "select", "textarea"
    ]

    def __init__(self):
        self.timeout = getattr(settings, 'SCRAPE_TIMEOUT', 15)
        self.min_length = getattr(settings, 'MIN_ARTICLE_LENGTH', 200)
        self.max_length = getattr(settings, 'MAX_ARTICLE_LENGTH', 10000)

    def fetch(self, url: str) -> Article | None:
        """
        Fetch and parse article from URL.

        Args:
            url: URL to fetch

        Returns:
            Article object or None if failed
        """
        try:
            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None

            soup = BeautifulSoup(response.text, "lxml")

            # Remove unwanted elements
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extract title
            title = self._extract_title(soup)

            # Extract main content
            content = self._extract_content(soup)

            if len(content) < self.min_length:
                return None

            # Truncate if too long
            if len(content) > self.max_length:
                content = content[:self.max_length]

            # Extract metadata
            author = self._extract_author(soup)
            date = self._extract_date(soup)
            source = urlparse(url).netloc.replace("www.", "")

            return Article(
                url=url,
                title=title,
                content=content,
                author=author,
                date=date,
                source=source,
                word_count=len(content.split())
            )

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        # Try OpenGraph title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()[:200]

        # Try common title elements
        for selector in ["h1", "title", ".headline", ".title", ".article-title"]:
            element = soup.select_one(selector)
            if element and element.text.strip():
                return element.text.strip()[:200]

        return "Untitled"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content."""
        # Try common content containers
        content_selectors = [
            "article",
            "[role='main']",
            "main",
            ".post-content",
            ".article-content",
            ".article-body",
            ".story-body",
            ".entry-content",
            ".content-body",
            "#article-body",
        ]

        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                paragraphs = element.find_all("p")
                if paragraphs:
                    text = " ".join(p.text.strip() for p in paragraphs if p.text.strip())
                    if len(text) >= self.min_length:
                        return text

        # Fallback: all paragraphs in body
        body = soup.find("body")
        if body:
            paragraphs = body.find_all("p")
            return " ".join(p.text.strip() for p in paragraphs if p.text.strip())

        return ""

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author name."""
        # Try meta tags
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            return author_meta["content"].strip()

        # Try common author elements
        for selector in [".author", "[rel='author']", ".byline", ".author-name", ".post-author"]:
            element = soup.select_one(selector)
            if element:
                text = element.get("content") or element.text
                if text:
                    return text.strip()[:100]

        return ""

    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date."""
        # Try meta tags
        for meta_name in ["article:published_time", "publication_date", "date"]:
            meta = soup.find("meta", property=meta_name) or soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                return meta["content"][:10]  # Just the date part

        # Try time element
        time_el = soup.find("time")
        if time_el:
            return time_el.get("datetime", time_el.text)[:10]

        # Try common date elements
        for selector in [".date", ".published", ".post-date", ".article-date"]:
            element = soup.select_one(selector)
            if element:
                return element.text.strip()[:30]

        return ""
