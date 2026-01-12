"""
Article Curator Agent - LLM-powered article selection and evaluation.

Key difference from traditional approach:
- NO hardcoded "reputable sources" list
- LLM evaluates EACH article based on headline and source
- Includes diverse sources: major outlets, niche blogs, industry publications
- Transparent about source type in the final report
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from research.logging_config import get_logger
from research.tools.llm import LLM
from research.tools.web_search import search_news
from research.tools.web_scraper import WebScraper

logger = get_logger('article_curator')


@dataclass
class CuratedArticle:
    """Represents a curated article."""
    url: str
    title: str
    source: str
    source_name: str
    content: str
    author: str
    date: str
    word_count: int


class ArticleCuratorAgent:
    """
    LLM-powered article curation agent.

    Finds articles about a company and evaluates each one for:
    - Relevance to the company
    - Source credibility
    - Source type (for transparency in report)
    """

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()
        self.scraper = WebScraper()
        self.name = "Article Curator"

    def run(
        self,
        company_name: str,
        max_articles: int = 10
    ) -> list[CuratedArticle]:
        """
        Find, evaluate, and curate articles about the company.

        Args:
            company_name: Company to search for
            max_articles: Maximum articles to return

        Returns:
            List of CuratedArticle objects
        """
        logger.info(f"Finding articles about {company_name}...")

        # 1. Broad news search (get ~30-40 results to filter from)
        search_results = self._broad_search(company_name)

        if not search_results:
            logger.warning("No articles found")
            return []

        logger.info(f"Found {len(search_results)} potential articles")

        # 2. Scrape ALL articles in parallel first
        logger.info("Scraping article content...")
        scraped = self._scrape_all(search_results)
        logger.info(f"Successfully scraped {len(scraped)} articles")

        if not scraped:
            logger.warning("No articles could be scraped")
            return []

        # 3. LLM selects best articles in single pass
        logger.info("Selecting best articles for sentiment analysis...")
        selected = self._llm_select_articles(scraped, company_name, max_articles)
        logger.info(f"Selected {len(selected)} articles for analysis")

        return selected

    def _broad_search(self, company_name: str) -> list:
        """
        Search for financial/business news about the company.
        Uses finance-focused queries to find market-moving, substantive articles.
        """
        # Finance-focused search queries
        queries = [
            f"{company_name} stock",
            f"{company_name} earnings revenue",
            f"{company_name} investor",
            f"{company_name} SEC filing",
            f"{company_name} acquisition merger",
            f"{company_name} financial results",
            f"{company_name} analyst rating",
            f"{company_name} quarterly report",
        ]

        all_results = []
        seen_urls = set()

        for query in queries:
            results = search_news(query, max_results=10)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        return all_results[:50]  # Cap at 50 for scraping

    def _scrape_all(self, search_results: list) -> list[dict]:
        """
        Scrape ALL search results in parallel.
        Returns list of dicts with scraped content + original search metadata.
        """
        scraped = []

        def scrape_one(result):
            content = self.scraper.fetch(result.url)
            if content and len(content.content) >= 200:
                return {
                    'url': result.url,
                    'title': result.title or content.title,
                    'source': content.source,
                    'source_name': result.source,
                    'content': content.content,
                    'author': content.author,
                    'date': content.date or result.date,
                    'word_count': content.word_count,
                }
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scrape_one, r): r for r in search_results}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scraped.append(result)

        return scraped

    def _llm_select_articles(
        self,
        scraped: list[dict],
        company_name: str,
        max_articles: int
    ) -> list[CuratedArticle]:
        """
        LLM selects the best articles in a single pass.
        Returns list of CuratedArticle objects.
        """
        # Format articles for LLM
        articles_text = ""
        for i, article in enumerate(scraped):
            content_preview = article['content'][:500]
            articles_text += f"""{i+1}. "{article['title']}"
   Source: {article['source_name']}
   Preview: {content_preview}...

"""

        prompt = f"""You are filtering articles for stock sentiment analysis of {company_name}.

Select the {max_articles} BEST articles based on these criteria:

1. PRIMARY FOCUS
   - {company_name} is the MAIN subject of the article
   - NOT a roundup, comparison piece, or industry overview
   - NOT just a brief mention within a larger story

2. STOCK-MOVING POTENTIAL
   The news could plausibly affect the stock price:
   - Earnings, revenue, guidance
   - Acquisitions, mergers, partnerships, major contracts
   - Product launches, recalls, failures
   - Leadership changes, layoffs, restructuring
   - Lawsuits, regulatory actions, investigations
   - Analyst upgrades/downgrades

3. CREDIBLE SOURCE
   - Established news organization or financial publication
   - NOT press releases, sponsored content, or anonymous blogs

---

ARTICLES:
{articles_text}

---

Return EXACTLY {max_articles} indices (or fewer if not enough qualify).
Best first, comma-separated. Example: 5, 2, 14, 8, 11, 3, 9, 7, 12, 1"""

        response = self.llm.generate(prompt)

        # Parse comma-separated indices
        selected_indices = []
        try:
            # Handle "NONE" response
            if response.strip().upper() == "NONE":
                return []

            # Parse indices
            for part in response.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1  # Convert to 0-based
                    if 0 <= idx < len(scraped):
                        selected_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: return first max_articles
            selected_indices = list(range(min(max_articles, len(scraped))))

        # Convert to CuratedArticle objects
        return [
            CuratedArticle(
                url=scraped[i]['url'],
                title=scraped[i]['title'],
                source=scraped[i]['source'],
                source_name=scraped[i]['source_name'],
                content=scraped[i]['content'],
                author=scraped[i]['author'],
                date=scraped[i]['date'],
                word_count=scraped[i]['word_count'],
            )
            for i in selected_indices
        ]
