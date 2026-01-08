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
    """Represents a curated article with LLM-assessed metadata."""
    url: str
    title: str
    source: str
    source_name: str
    content: str
    author: str
    date: str
    word_count: int
    # LLM-assessed metadata
    source_type: str           # major_outlet, industry_publication, niche_blog, etc.
    credibility_score: float   # 0-1
    credibility_note: str      # Brief explanation
    relevance_score: float     # 0-1


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
        industry: str,
        max_articles: int = 10
    ) -> list[CuratedArticle]:
        """
        Find, evaluate, and curate articles about the company.

        Args:
            company_name: Company to search for
            industry: Company's industry (for context)
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

        # 3. LLM evaluates scraped articles (now with full content)
        logger.info("Evaluating article relevance and source credibility...")
        evaluated = self._llm_evaluate_articles(scraped, company_name, industry)

        # 4. Select best mix and return top max_articles
        selected = self._select_best_mix(evaluated, max_articles)
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

    def _llm_evaluate_articles(
        self,
        scraped: list[dict],
        company_name: str,
        industry: str
    ) -> list[dict]:
        """
        LLM evaluates scraped articles for relevance and source credibility.
        Returns list of scraped articles with added evaluation metadata.
        """
        # Format articles for LLM (now with actual content)
        articles_text = ""
        for i, article in enumerate(scraped):
            content_preview = article['content'][:500]
            articles_text += f"""{i+1}. "{article['title']}"
   Source: {article['source_name']}
   Content preview: {content_preview}...

"""

        prompt = f"""You are evaluating news articles about {company_name} (industry: {industry}).

For EACH article below, assess:
1. RELEVANCE (0-1): Is this actually about {company_name}? Is it substantive news?
2. SOURCE_TYPE: One of: major_outlet, industry_publication, niche_blog, press_release, unknown
3. CREDIBILITY (0-1): How trustworthy is this source?
4. CREDIBILITY_NOTE: Brief 1-sentence explanation

ARTICLES TO EVALUATE:
{articles_text}

Respond with a JSON array:
[
  {{
    "index": 1,
    "relevance": 0.9,
    "source_type": "major_outlet",
    "credibility": 0.95,
    "credibility_note": "Reuters is a major international news agency"
  }},
  ...
]

Scoring guidelines:
- Relevance: 0.9+ = directly about the company, 0.5-0.8 = mentions company, <0.5 = tangential
- Credibility: Major outlets (Reuters, Bloomberg, BBC) = 0.85-0.95
- Industry publications = 0.7-0.85
- Niche blogs can be 0.5-0.7 if they appear legitimate
- Press releases = 0.5-0.6 (biased but factual)
- Unknown/suspicious = 0.3-0.5

Include ALL {len(scraped)} articles in your response."""

        evaluations = self.llm.generate_json(prompt)

        # Handle case where LLM returns dict instead of list
        if isinstance(evaluations, dict):
            evaluations = evaluations.get('articles', [])

        # Build evaluation map
        eval_map = {}
        if isinstance(evaluations, list):
            for e in evaluations:
                if isinstance(e, dict) and 'index' in e:
                    eval_map[e['index']] = e

        # Merge evaluations into scraped articles
        enriched = []
        for i, article in enumerate(scraped):
            eval_data = eval_map.get(i + 1, {})
            article['relevance'] = float(eval_data.get('relevance', 0.5))
            article['source_type'] = eval_data.get('source_type', 'unknown')
            article['credibility'] = float(eval_data.get('credibility', 0.5))
            article['credibility_note'] = eval_data.get('credibility_note', '')
            enriched.append(article)

        return enriched

    def _select_best_mix(self, evaluated: list[dict], max_articles: int) -> list[CuratedArticle]:
        """
        Select best mix of articles balancing:
        - Relevance (must be about the company)
        - Credibility (prefer higher, but don't exclude all niche sources)
        - Source diversity (mix of major outlets + niche publications)

        Returns CuratedArticle objects.
        """
        # Filter out very low relevance (< 0.3)
        candidates = [e for e in evaluated if e['relevance'] >= 0.3]

        # Sort by combined score (relevance weighted higher)
        for c in candidates:
            c['combined_score'] = (c['relevance'] * 0.6) + (c['credibility'] * 0.4)

        candidates.sort(key=lambda x: x['combined_score'], reverse=True)

        # Select with diversity in mind
        selected = []
        source_types_used = {'major_outlet': 0, 'industry_publication': 0, 'niche_blog': 0}
        max_major = int(max_articles * 0.6)  # Max 60% from major outlets

        for c in candidates:
            if len(selected) >= max_articles:
                break

            source_type = c['source_type']

            # Ensure we get some diversity
            if source_type == 'major_outlet' and source_types_used['major_outlet'] >= max_major:
                continue

            selected.append(c)
            if source_type in source_types_used:
                source_types_used[source_type] += 1

        # If we don't have enough, fill with remaining best candidates
        if len(selected) < max_articles:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                if len(selected) >= max_articles:
                    break

        # Convert to CuratedArticle objects
        return [
            CuratedArticle(
                url=article['url'],
                title=article['title'],
                source=article['source'],
                source_name=article['source_name'],
                content=article['content'],
                author=article['author'],
                date=article['date'],
                word_count=article['word_count'],
                source_type=article['source_type'],
                credibility_score=article['credibility'],
                credibility_note=article['credibility_note'],
                relevance_score=article['relevance']
            )
            for article in selected
        ]
