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

from research.tools.llm import LLM
from research.tools.web_search import search_news
from research.tools.web_scraper import WebScraper


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
        print(f"[{self.name}] Finding articles about {company_name}...")

        # 1. Broad news search (get ~30-40 results to filter from)
        search_results = self._broad_search(company_name)

        if not search_results:
            print(f"[{self.name}] No articles found")
            return []

        print(f"[{self.name}] Found {len(search_results)} potential articles")

        # 2. LLM evaluates ALL candidates
        print(f"[{self.name}] Evaluating article relevance and source credibility...")
        evaluated = self._llm_evaluate_articles(search_results, company_name, industry)

        # 3. Select best mix (balance relevance, credibility, source diversity)
        selected = self._select_best_mix(evaluated, max_articles)
        print(f"[{self.name}] Selected {len(selected)} articles for analysis")

        # 4. Scrape content from selected articles
        print(f"[{self.name}] Scraping article content...")
        articles = self._scrape_and_enrich(selected)

        print(f"[{self.name}] Successfully retrieved {len(articles)} articles")
        return articles

    def _broad_search(self, company_name: str) -> list:
        """
        Search broadly for company news - NO source filtering.
        Returns raw search results from DuckDuckGo news.
        """
        # Multiple search queries to get diverse results
        queries = [
            f"{company_name} news",
            f"{company_name} latest",
            f"{company_name}",
        ]

        all_results = []
        seen_urls = set()

        for query in queries:
            results = search_news(query, max_results=15)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        return all_results[:40]  # Cap at 40 for LLM evaluation

    def _llm_evaluate_articles(
        self,
        results: list,
        company_name: str,
        industry: str
    ) -> list:
        """
        LLM evaluates each article for relevance and source credibility.
        Returns list of results with added evaluation metadata.
        """
        # Format results for LLM
        articles_text = ""
        for i, r in enumerate(results):
            articles_text += f"""{i+1}. "{r.title}"
   Source: {r.source}
   Snippet: {r.snippet[:200] if r.snippet else 'N/A'}

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

Include ALL {len(results)} articles in your response."""

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

        # Merge evaluations back into results
        enriched = []
        for i, r in enumerate(results):
            eval_data = eval_map.get(i + 1, {})
            enriched.append({
                'result': r,
                'relevance': float(eval_data.get('relevance', 0.5)),
                'source_type': eval_data.get('source_type', 'unknown'),
                'credibility': float(eval_data.get('credibility', 0.5)),
                'credibility_note': eval_data.get('credibility_note', '')
            })

        return enriched

    def _select_best_mix(self, evaluated: list, max_articles: int) -> list:
        """
        Select best mix of articles balancing:
        - Relevance (must be about the company)
        - Credibility (prefer higher, but don't exclude all niche sources)
        - Source diversity (mix of major outlets + niche publications)
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

        return selected

    def _scrape_and_enrich(self, selected: list) -> list[CuratedArticle]:
        """
        Scrape content from selected URLs and create CuratedArticle objects.
        Uses parallel scraping for speed.
        """
        articles = []

        def scrape_one(item):
            r = item['result']
            content = self.scraper.fetch(r.url)
            if content and len(content.content) >= 200:
                return CuratedArticle(
                    url=r.url,
                    title=r.title or content.title,
                    source=content.source,
                    source_name=r.source,
                    content=content.content,
                    author=content.author,
                    date=content.date or r.date,
                    word_count=content.word_count,
                    source_type=item['source_type'],
                    credibility_score=item['credibility'],
                    credibility_note=item['credibility_note'],
                    relevance_score=item['relevance']
                )
            return None

        # Parallel scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(scrape_one, item): item for item in selected}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    articles.append(result)

        return articles
