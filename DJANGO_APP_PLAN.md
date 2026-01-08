# Company Sentiment Research Agent - Django Application Plan

## Overview

A Django web application that provides a **24-hour briefing** on any company. Think of it as your morning catch-up: what's happening with this company, what's the news saying, and what's the overall sentiment?

### Workflow:

1. **User inputs company name** via web form
2. **Company Research Agent** gathers overview (stock price with historical chart, recent developments, industry context)
3. **Article Curator Agent** (LLM-powered) finds ~10 articles, evaluating each for relevance and source credibility
4. **Sentiment Analyzer** runs multi-model analysis on each article
5. **Briefing Report** displayed as a clean, readable daily catch-up

### Key Philosophy:

- **No hardcoded "reputable sources"** - the LLM evaluates each article's relevance and source quality dynamically
- **Include diverse sources** - mainstream media, niche industry blogs, lesser-known publications all welcome
- **Transparency** - clearly indicate source type (major outlet vs. niche blog) in the report
- **Briefing format** - the report should read like a quick catch-up, not a data dump

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DJANGO APPLICATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   FRONTEND   │────>│    VIEWS     │────>│    AGENTS    │        │
│  │  (Templates) │<────│  (API/Pages) │<────│   (Backend)  │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                   │                  │
│                                                   v                  │
│                              ┌────────────────────────────────┐     │
│                              │         AGENT PIPELINE         │     │
│                              │                                │     │
│                              │  1. CompanyResearchAgent       │     │
│                              │     └─> LLM + Web Search       │     │
│                              │     └─> Stock data (yfinance)  │     │
│                              │                                │     │
│                              │  2. ArticleCuratorAgent         │     │
│                              │     └─> Broad news search      │     │
│                              │     └─> LLM evaluates headlines│     │
│                              │     └─> LLM rates source quality│    │
│                              │     └─> Article scraping       │     │
│                              │                                │     │
│                              │  3. SentimentAnalyzerAgent     │     │
│                              │     └─> FinBERT               │     │
│                              │     └─> VADER                 │     │
│                              │     └─> TextBlob              │     │
│                              │     └─> RoBERTa               │     │
│                              │     └─> LLM Commentary        │     │
│                              └────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
company_sentiment/
├── manage.py
├── requirements.txt
├── README.md
│
├── company_sentiment/              # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── research/                       # Main Django app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py                   # Database models
│   ├── views.py                    # View logic
│   ├── urls.py                     # App routes
│   ├── forms.py                    # Form definitions
│   │
│   ├── agents/                     # AI Agent logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main coordinator
│   │   ├── company_researcher.py   # Company info + stock data
│   │   ├── article_curator.py      # LLM-curated article selection
│   │   └── sentiment_analyzer.py   # Multi-model sentiment
│   │
│   ├── tools/                      # Utility modules
│   │   ├── __init__.py
│   │   ├── llm.py                  # Ollama wrapper
│   │   ├── web_search.py           # DuckDuckGo search
│   │   ├── web_scraper.py          # Article content extraction
│   │   └── stock_data.py           # yfinance wrapper
│   │
│   └── templates/                  # HTML templates
│       └── research/
│           ├── base.html           # Base template
│           ├── index.html          # Home page / search form
│           ├── loading.html        # Processing status page
│           ├── results.html        # Full results dashboard
│           └── partials/
│               ├── company_card.html
│               ├── article_card.html
│               └── sentiment_chart.html
│
└── static/                         # Static assets
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## Database Models

```python
# research/models.py

from django.db import models
from django.utils import timezone

class CompanyResearch(models.Model):
    """Stores a complete research session for a company"""

    # Basic info
    company_name = models.CharField(max_length=200)
    industry = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    # Company overview (from LLM research)
    overview = models.TextField(blank=True)  # LLM-generated writeup
    stock_ticker = models.CharField(max_length=20, blank=True)  # e.g., "TSLA"
    stock_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    stock_change_percent = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    stock_change_1w = models.DecimalField(max_digits=5, decimal_places=2, null=True)  # 1 week change
    stock_change_1m = models.DecimalField(max_digits=5, decimal_places=2, null=True)  # 1 month change
    market_cap = models.CharField(max_length=50, blank=True)
    stock_history = models.TextField(blank=True)  # JSON: last 30 days of prices for chart
    recent_developments = models.TextField(blank=True)  # JSON list of key points

    # Aggregate sentiment
    overall_sentiment = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    sentiment_label = models.CharField(max_length=20, blank=True)  # positive/negative/neutral
    positive_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)

    # LLM summary
    llm_summary = models.TextField(blank=True)

    # Status
    status = models.CharField(max_length=20, default='pending')  # pending/processing/completed/failed
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.company_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class Article(models.Model):
    """Individual article analyzed for a company"""

    research = models.ForeignKey(CompanyResearch, on_delete=models.CASCADE, related_name='articles')

    # Article metadata
    url = models.URLField(max_length=500)
    title = models.CharField(max_length=500)
    source = models.CharField(max_length=200)  # Domain name
    source_name = models.CharField(max_length=200, blank=True)  # Human-readable name
    author = models.CharField(max_length=200, blank=True)
    published_date = models.DateField(null=True, blank=True)
    content_preview = models.TextField(blank=True)  # First 300 chars
    word_count = models.IntegerField(default=0)

    # Source credibility (LLM-assessed)
    source_type = models.CharField(max_length=50, blank=True)  # major_outlet, industry_publication, niche_blog, unknown
    credibility_score = models.DecimalField(max_digits=3, decimal_places=2, null=True)  # 0-1, LLM assessment
    credibility_note = models.CharField(max_length=300, blank=True)  # Brief note about the source
    relevance_score = models.DecimalField(max_digits=3, decimal_places=2, null=True)  # 0-1, how relevant to company

    # Sentiment scores (all -1 to 1 scale)
    finbert_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    vader_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    textblob_polarity = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    textblob_subjectivity = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    roberta_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)

    # LLM analysis
    llm_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    llm_commentary = models.TextField(blank=True)

    # Aggregated
    ensemble_score = models.DecimalField(max_digits=4, decimal_places=3, null=True)
    model_agreement = models.DecimalField(max_digits=3, decimal_places=2, null=True)  # 0-1
    sentiment_label = models.CharField(max_length=20, blank=True)

    class Meta:
        ordering = ['-ensemble_score']

    def __str__(self):
        return f"{self.title[:50]}... ({self.sentiment_label})"
```

---

## URL Routes

```python
# research/urls.py

from django.urls import path
from . import views

app_name = 'research'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),                           # Home page with search form
    path('search/', views.search, name='search'),                  # POST: Start new research
    path('results/<int:pk>/', views.results, name='results'),      # View results
    path('status/<int:pk>/', views.status, name='status'),         # AJAX: Check processing status

    # History
    path('history/', views.history, name='history'),               # List past researches

    # API endpoints (for AJAX updates)
    path('api/research/<int:pk>/', views.api_research, name='api_research'),
]
```

---

## Views

```python
# research/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import CompanyResearch, Article
from .forms import CompanySearchForm
from .agents.orchestrator import Orchestrator
import threading

def index(request):
    """Home page with company search form"""
    form = CompanySearchForm()
    recent_searches = CompanyResearch.objects.filter(status='completed')[:5]
    return render(request, 'research/index.html', {
        'form': form,
        'recent_searches': recent_searches
    })

@require_POST
def search(request):
    """Start a new company research"""
    form = CompanySearchForm(request.POST)
    if form.is_valid():
        company_name = form.cleaned_data['company_name']

        # Create research record
        research = CompanyResearch.objects.create(
            company_name=company_name,
            status='processing'
        )

        # Start research in background thread
        thread = threading.Thread(
            target=run_research_pipeline,
            args=(research.id,)
        )
        thread.start()

        # Redirect to loading/status page
        return redirect('research:results', pk=research.id)

    return redirect('research:index')

def results(request, pk):
    """Display research results (or loading state if still processing)"""
    research = get_object_or_404(CompanyResearch, pk=pk)
    articles = research.articles.all()

    return render(request, 'research/results.html', {
        'research': research,
        'articles': articles
    })

def status(request, pk):
    """AJAX endpoint for checking research status"""
    research = get_object_or_404(CompanyResearch, pk=pk)
    return JsonResponse({
        'status': research.status,
        'error': research.error_message
    })

def history(request):
    """List all past researches"""
    researches = CompanyResearch.objects.filter(status='completed')
    return render(request, 'research/history.html', {
        'researches': researches
    })

def api_research(request, pk):
    """Full research data as JSON"""
    research = get_object_or_404(CompanyResearch, pk=pk)
    articles = research.articles.all()

    return JsonResponse({
        'company': research.company_name,
        'status': research.status,
        'overview': research.overview,
        'stock_price': str(research.stock_price) if research.stock_price else None,
        'stock_change': str(research.stock_change_percent) if research.stock_change_percent else None,
        'overall_sentiment': str(research.overall_sentiment) if research.overall_sentiment else None,
        'sentiment_label': research.sentiment_label,
        'llm_summary': research.llm_summary,
        'articles': [
            {
                'title': a.title,
                'source': a.source,
                'url': a.url,
                'sentiment_label': a.sentiment_label,
                'ensemble_score': str(a.ensemble_score),
                'llm_commentary': a.llm_commentary
            }
            for a in articles
        ]
    })

def run_research_pipeline(research_id):
    """Background task: run the full research pipeline"""
    from .agents.orchestrator import Orchestrator

    research = CompanyResearch.objects.get(pk=research_id)

    try:
        orchestrator = Orchestrator()
        orchestrator.run(research)
        research.status = 'completed'
    except Exception as e:
        research.status = 'failed'
        research.error_message = str(e)

    research.save()
```

---

## Agent Pipeline

### 1. Company Research Agent

```python
# research/agents/company_researcher.py

"""
CompanyResearchAgent: Gathers company overview information

Responsibilities:
1. Search for basic company info (what they do, industry)
2. Get current stock price and recent performance (yfinance)
3. Find recent key developments/news highlights
4. Generate a concise company overview/writeup

Output: Company overview dict with:
- industry
- overview (LLM-generated 2-3 paragraph summary)
- stock_price, stock_change_percent, market_cap
- recent_developments (list of 3-5 key points)
"""

class CompanyResearchAgent:

    def __init__(self, llm):
        self.llm = llm
        self.name = "Company Researcher"

    def run(self, company_name: str) -> dict:
        """
        Research a company and return overview data

        Steps:
        1. Get stock data (price, change, market cap)
        2. Search for recent company news/info
        3. Have LLM synthesize into overview
        """

        # 1. Stock data
        stock_data = self._get_stock_data(company_name)

        # 2. Web search for company info
        search_results = self._search_company_info(company_name)

        # 3. LLM generates overview
        overview = self._generate_overview(company_name, stock_data, search_results)

        return {
            'industry': overview.get('industry', ''),
            'overview': overview.get('overview', ''),
            'stock_price': stock_data.get('price'),
            'stock_change_percent': stock_data.get('change_percent'),
            'market_cap': stock_data.get('market_cap', ''),
            'recent_developments': overview.get('developments', [])
        }

    def _get_stock_data(self, company_name: str) -> dict:
        """Get stock price using yfinance"""
        # Implementation uses tools/stock_data.py
        pass

    def _search_company_info(self, company_name: str) -> list:
        """Search for company info via web search"""
        # Implementation uses tools/web_search.py
        pass

    def _generate_overview(self, company_name: str, stock_data: dict, search_results: list) -> dict:
        """LLM generates company overview"""

        prompt = f"""Research summary for: {company_name}

Stock Data:
- Current Price: ${stock_data.get('price', 'N/A')}
- Change: {stock_data.get('change_percent', 'N/A')}%
- Market Cap: {stock_data.get('market_cap', 'N/A')}

Recent Search Results:
{self._format_search_results(search_results)}

Based on this information, provide a JSON response:
{{
    "industry": "<primary industry/sector>",
    "overview": "<2-3 paragraph overview of the company, what they do, their market position, recent performance>",
    "developments": ["<key development 1>", "<key development 2>", "<key development 3>"]
}}

Focus on factual, recent information. Be concise but informative."""

        return self.llm.generate_json(prompt)
```

### 2. Article Curator Agent (LLM-Powered)

```python
# research/agents/article_curator.py

"""
ArticleCuratorAgent: LLM-powered article selection and evaluation

KEY DIFFERENCE FROM TRADITIONAL APPROACH:
- NO hardcoded "reputable sources" list
- LLM evaluates EACH article based on headline and source
- Includes diverse sources: major outlets, niche blogs, industry publications
- Transparent about source type in the final report

Responsibilities:
1. Cast a wide net - search broadly for company news
2. LLM evaluates each result:
   - Is the headline relevant to the company?
   - What type of source is this? (major outlet, industry pub, niche blog, etc.)
   - How credible does this source appear?
3. Select diverse mix of ~10 best articles
4. Scrape content and return with source metadata

Output: List of CuratedArticle objects with:
- url, title, source, author, date
- full_content (for sentiment analysis)
- source_type (major_outlet, industry_publication, niche_blog, unknown)
- credibility_score (0-1)
- credibility_note (why this score)
- relevance_score (0-1)
"""

from dataclasses import dataclass

@dataclass
class CuratedArticle:
    url: str
    title: str
    source: str
    source_name: str
    content: str
    author: str
    date: str
    word_count: int
    # LLM-assessed metadata
    source_type: str           # major_outlet, industry_publication, niche_blog, unknown
    credibility_score: float   # 0-1
    credibility_note: str      # Brief explanation
    relevance_score: float     # 0-1


class ArticleCuratorAgent:

    def __init__(self, llm):
        self.llm = llm
        self.scraper = WebScraper()
        self.name = "Article Curator"

    def run(self, company_name: str, industry: str, max_articles: int = 10) -> list[CuratedArticle]:
        """
        Find, evaluate, and curate articles about the company

        Steps:
        1. Broad search - cast a wide net (no source filtering)
        2. LLM evaluates batch of headlines + sources
        3. Select top candidates based on relevance + credibility
        4. Scrape content
        5. Return curated list with source metadata
        """

        # 1. Broad news search (get ~30-40 results to filter from)
        search_results = self._broad_search(company_name)

        if not search_results:
            return []

        # 2. LLM evaluates ALL candidates in batches
        evaluated = self._llm_evaluate_articles(search_results, company_name, industry)

        # 3. Select best mix (balance relevance, credibility, source diversity)
        selected = self._select_best_mix(evaluated, max_articles)

        # 4. Scrape content from selected articles
        articles = self._scrape_and_enrich(selected)

        return articles

    def _broad_search(self, company_name: str) -> list:
        """
        Search broadly for company news - NO source filtering
        Returns raw search results from DuckDuckGo news
        """
        from tools.web_search import search_news

        # Multiple search queries to get diverse results
        queries = [
            f"{company_name} news",
            f"{company_name} latest",
            f"{company_name} announcement",
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

    def _llm_evaluate_articles(self, results: list, company_name: str, industry: str) -> list:
        """
        LLM evaluates each article for relevance and source credibility
        Returns list of results with added evaluation metadata
        """

        # Format results for LLM
        articles_text = ""
        for i, r in enumerate(results):
            articles_text += f"""
{i+1}. "{r.title}"
   Source: {r.source}
   URL: {r.url}
   Snippet: {r.snippet[:200] if r.snippet else 'N/A'}
"""

        prompt = f"""You are evaluating news articles about {company_name} (industry: {industry}).

For EACH article below, assess:
1. RELEVANCE (0-1): Is this actually about {company_name}? Is it substantive news or fluff?
2. SOURCE_TYPE: Classify as one of: major_outlet, industry_publication, niche_blog, press_release, unknown
3. CREDIBILITY (0-1): How trustworthy is this source? Consider:
   - Is it a known news organization?
   - Does it appear to be an industry-specific publication?
   - Is it a personal blog or unknown site?
   - Could it be biased (e.g., company press release)?
4. CREDIBILITY_NOTE: Brief 1-sentence explanation of the credibility score

ARTICLES TO EVALUATE:
{articles_text}

Respond with JSON array:
[
  {{
    "index": 1,
    "relevance": 0.9,
    "source_type": "major_outlet",
    "credibility": 0.95,
    "credibility_note": "Reuters is a major international news agency known for factual reporting"
  }},
  ...
]

Be fair to lesser-known sources - a niche industry blog can still be credible (0.6-0.8) if it appears legitimate.
Only give low credibility (< 0.4) to clearly dubious sources.
Include ALL {len(results)} articles in your response."""

        evaluations = self.llm.generate_json(prompt)

        # Merge evaluations back into results
        eval_map = {e['index']: e for e in evaluations}
        enriched = []

        for i, r in enumerate(results):
            eval_data = eval_map.get(i + 1, {})
            enriched.append({
                'result': r,
                'relevance': eval_data.get('relevance', 0.5),
                'source_type': eval_data.get('source_type', 'unknown'),
                'credibility': eval_data.get('credibility', 0.5),
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

        for c in candidates:
            if len(selected) >= max_articles:
                break

            source_type = c['source_type']

            # Ensure we get some diversity - don't take more than 60% from major outlets
            if source_type == 'major_outlet' and source_types_used['major_outlet'] >= max_articles * 0.6:
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
        Scrape content from selected URLs and create CuratedArticle objects
        """
        articles = []

        for item in selected:
            r = item['result']
            content = self.scraper.fetch(r.url)

            if content and len(content.content) >= 200:
                articles.append(CuratedArticle(
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
                ))

        return articles
```

### 3. Sentiment Analyzer Agent

```python
# research/agents/sentiment_analyzer.py

"""
SentimentAnalyzerAgent: Multi-model sentiment analysis

Uses 4 ML models + LLM:
1. FinBERT - Financial sentiment (trained on financial text)
2. VADER - Lexicon-based (good for intensity/hype detection)
3. TextBlob - General polarity + subjectivity measure
4. RoBERTa - General news sentiment

Then LLM provides:
- Its own sentiment score
- Commentary explaining why models agree/disagree
- Context-aware interpretation

Output per article:
- Individual model scores (-1 to 1)
- Ensemble score (weighted average)
- Model agreement metric (0-1)
- LLM commentary
- Sentiment label (positive/negative/neutral)
"""

# (Implementation as per original spec section 5.6)
```

### 4. Orchestrator

```python
# research/agents/orchestrator.py

"""
Orchestrator: Coordinates the full research pipeline

Workflow:
1. Validate LLM availability (Ollama)
2. Run CompanyResearchAgent -> get overview, stock data
3. Run ArticleGathererAgent -> get 10 relevant articles
4. Run SentimentAnalyzerAgent -> analyze each article
5. Calculate aggregate sentiment
6. Generate LLM summary
7. Save results to database
"""

class Orchestrator:

    def __init__(self):
        self.llm = LLM()
        self.company_researcher = CompanyResearchAgent(self.llm)
        self.article_gatherer = ArticleGathererAgent(self.llm)
        self.sentiment_analyzer = SentimentAnalyzerAgent(self.llm)

    def run(self, research: CompanyResearch):
        """Execute full pipeline and update research record"""

        # 1. Validate Ollama
        if not self.llm.is_available():
            raise Exception("Ollama is not running. Start with: ollama serve")

        company_name = research.company_name

        # 2. Research company
        company_data = self.company_researcher.run(company_name)
        research.industry = company_data['industry']
        research.overview = company_data['overview']
        research.stock_price = company_data['stock_price']
        research.stock_change_percent = company_data['stock_change_percent']
        research.market_cap = company_data['market_cap']
        research.recent_developments = json.dumps(company_data['recent_developments'])
        research.save()

        # 3. Gather articles
        articles = self.article_gatherer.run(
            company_name,
            company_data['industry'],
            max_articles=10
        )

        if not articles:
            raise Exception("No articles found")

        # 4. Analyze sentiment for each article
        analyzed_articles = []
        for article in articles:
            sentiment = self.sentiment_analyzer.analyze(article)

            # Save to database
            Article.objects.create(
                research=research,
                url=article.url,
                title=article.title,
                source=article.source,
                author=article.author,
                published_date=article.date,
                content_preview=article.content[:300],
                word_count=article.word_count,
                finbert_score=sentiment.finbert,
                vader_score=sentiment.vader,
                textblob_polarity=sentiment.textblob_polarity,
                textblob_subjectivity=sentiment.textblob_subjectivity,
                roberta_score=sentiment.roberta,
                llm_score=sentiment.llm_score,
                llm_commentary=sentiment.llm_commentary,
                ensemble_score=sentiment.ensemble,
                model_agreement=sentiment.agreement,
                sentiment_label=sentiment.label
            )
            analyzed_articles.append(sentiment)

        # 5. Calculate aggregates
        self._calculate_aggregates(research, analyzed_articles)

        # 6. Generate overall summary
        research.llm_summary = self._generate_summary(research)
        research.save()

    def _calculate_aggregates(self, research, sentiments):
        """Calculate aggregate sentiment metrics"""
        n = len(sentiments)
        if n == 0:
            return

        research.overall_sentiment = sum(s.ensemble for s in sentiments) / n

        # Distribution
        research.positive_count = sum(1 for s in sentiments if s.label == 'positive')
        research.negative_count = sum(1 for s in sentiments if s.label == 'negative')
        research.neutral_count = sum(1 for s in sentiments if s.label == 'neutral')

        # Overall label
        if research.overall_sentiment > 0.1:
            research.sentiment_label = 'positive'
        elif research.overall_sentiment < -0.1:
            research.sentiment_label = 'negative'
        else:
            research.sentiment_label = 'neutral'
```

---

## Frontend Templates

### Base Template

```html
<!-- templates/research/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Company Sentiment Research{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block extra_head %}{% endblock %}
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-white shadow-sm">
        <div class="max-w-7xl mx-auto px-4 py-4">
            <div class="flex justify-between items-center">
                <a href="{% url 'research:index' %}" class="text-xl font-bold text-gray-800">
                    Company Sentiment Research
                </a>
                <a href="{% url 'research:history' %}" class="text-gray-600 hover:text-gray-800">
                    History
                </a>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>

    {% block extra_js %}{% endblock %}
</body>
</html>
```

### Home Page

```html
<!-- templates/research/index.html -->
{% extends 'research/base.html' %}

{% block content %}
<div class="max-w-2xl mx-auto">
    <!-- Hero Section -->
    <div class="text-center mb-12">
        <h1 class="text-4xl font-bold text-gray-900 mb-4">
            Company Sentiment Analysis
        </h1>
        <p class="text-gray-600 text-lg">
            Enter a company name to get an AI-powered analysis of recent news sentiment
        </p>
    </div>

    <!-- Search Form -->
    <div class="bg-white rounded-lg shadow-md p-8">
        <form method="post" action="{% url 'research:search' %}">
            {% csrf_token %}
            <div class="mb-6">
                <label for="company_name" class="block text-sm font-medium text-gray-700 mb-2">
                    Company Name
                </label>
                <input
                    type="text"
                    name="company_name"
                    id="company_name"
                    placeholder="e.g., Tesla, Apple, Microsoft"
                    class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-lg"
                    required
                >
            </div>
            <button
                type="submit"
                class="w-full bg-blue-600 text-white py-3 px-6 rounded-lg text-lg font-medium hover:bg-blue-700 transition"
            >
                Analyze Sentiment
            </button>
        </form>
    </div>

    <!-- Recent Searches -->
    {% if recent_searches %}
    <div class="mt-12">
        <h2 class="text-xl font-semibold text-gray-800 mb-4">Recent Analyses</h2>
        <div class="grid gap-4">
            {% for research in recent_searches %}
            <a href="{% url 'research:results' research.pk %}"
               class="bg-white rounded-lg shadow-sm p-4 hover:shadow-md transition flex justify-between items-center">
                <div>
                    <span class="font-medium text-gray-900">{{ research.company_name }}</span>
                    <span class="text-gray-500 text-sm ml-2">{{ research.created_at|timesince }} ago</span>
                </div>
                <span class="px-3 py-1 rounded-full text-sm font-medium
                    {% if research.sentiment_label == 'positive' %}bg-green-100 text-green-800
                    {% elif research.sentiment_label == 'negative' %}bg-red-100 text-red-800
                    {% else %}bg-gray-100 text-gray-800{% endif %}">
                    {{ research.sentiment_label|title }}
                </span>
            </a>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>
{% endblock %}
```

### Results Dashboard

```html
<!-- templates/research/results.html -->
{% extends 'research/base.html' %}

{% block content %}
{% if research.status == 'processing' %}
    <!-- Loading State -->
    <div class="max-w-2xl mx-auto text-center py-16">
        <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-6"></div>
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Analyzing {{ research.company_name }}...</h2>
        <p class="text-gray-600" id="status-text">Researching company information</p>
    </div>

    <script>
        // Poll for status updates
        setInterval(function() {
            fetch("{% url 'research:status' research.pk %}")
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'completed' || data.status === 'failed') {
                        location.reload();
                    }
                });
        }, 3000);
    </script>

{% elif research.status == 'failed' %}
    <!-- Error State -->
    <div class="max-w-2xl mx-auto text-center py-16">
        <div class="text-red-600 text-6xl mb-6">!</div>
        <h2 class="text-2xl font-bold text-gray-900 mb-2">Analysis Failed</h2>
        <p class="text-gray-600">{{ research.error_message }}</p>
        <a href="{% url 'research:index' %}" class="mt-6 inline-block text-blue-600 hover:underline">
            Try again
        </a>
    </div>

{% else %}
    <!-- DAILY BRIEFING STYLE RESULTS -->

    <!-- Briefing Header -->
    <div class="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg shadow-lg p-8 mb-8 text-white">
        <div class="flex justify-between items-start">
            <div>
                <p class="text-gray-400 text-sm uppercase tracking-wide mb-1">Daily Briefing</p>
                <h1 class="text-4xl font-bold">{{ research.company_name }}</h1>
                <p class="text-gray-300 mt-2">{{ research.industry }} | {{ research.created_at|date:"F j, Y" }}</p>
            </div>
            <!-- Sentiment Badge -->
            <div class="text-right">
                <div class="text-6xl font-bold
                    {% if research.sentiment_label == 'positive' %}text-green-400
                    {% elif research.sentiment_label == 'negative' %}text-red-400
                    {% else %}text-gray-400{% endif %}">
                    {% if research.overall_sentiment >= 0 %}+{% endif %}{{ research.overall_sentiment|floatformat:2 }}
                </div>
                <div class="text-lg uppercase tracking-wide
                    {% if research.sentiment_label == 'positive' %}text-green-300
                    {% elif research.sentiment_label == 'negative' %}text-red-300
                    {% else %}text-gray-300{% endif %}">
                    {{ research.sentiment_label }} sentiment
                </div>
            </div>
        </div>
    </div>

    <!-- Two Column Layout: Stock + Key Points -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

        <!-- Stock Price Card with Chart -->
        {% if research.stock_price %}
        <div class="bg-white rounded-lg shadow-md p-6">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <p class="text-gray-500 text-sm uppercase tracking-wide">Stock Price</p>
                    <p class="text-sm text-gray-400">{{ research.stock_ticker }}</p>
                </div>
                <div class="text-right">
                    <div class="text-3xl font-bold text-gray-900">${{ research.stock_price }}</div>
                    <div class="flex gap-3 text-sm mt-1">
                        <span class="{% if research.stock_change_percent >= 0 %}text-green-600{% else %}text-red-600{% endif %}">
                            Today: {% if research.stock_change_percent >= 0 %}+{% endif %}{{ research.stock_change_percent }}%
                        </span>
                        {% if research.stock_change_1w %}
                        <span class="{% if research.stock_change_1w >= 0 %}text-green-600{% else %}text-red-600{% endif %}">
                            1W: {% if research.stock_change_1w >= 0 %}+{% endif %}{{ research.stock_change_1w }}%
                        </span>
                        {% endif %}
                        {% if research.stock_change_1m %}
                        <span class="{% if research.stock_change_1m >= 0 %}text-green-600{% else %}text-red-600{% endif %}">
                            1M: {% if research.stock_change_1m >= 0 %}+{% endif %}{{ research.stock_change_1m }}%
                        </span>
                        {% endif %}
                    </div>
                </div>
            </div>
            <!-- Stock Chart -->
            <div class="h-48">
                <canvas id="stockChart"></canvas>
            </div>
        </div>
        {% endif %}

        <!-- Key Developments -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <p class="text-gray-500 text-sm uppercase tracking-wide mb-4">Key Developments</p>
            <ul class="space-y-3">
                {% for dev in research.get_developments %}
                <li class="flex items-start gap-3">
                    <span class="text-blue-500 mt-1">&#8226;</span>
                    <span class="text-gray-700">{{ dev }}</span>
                </li>
                {% empty %}
                <li class="text-gray-500 italic">No recent developments identified</li>
                {% endfor %}
            </ul>
        </div>
    </div>

    <!-- Executive Summary -->
    <div class="bg-blue-50 border-l-4 border-blue-500 rounded-r-lg p-6 mb-8">
        <h3 class="text-lg font-semibold text-blue-900 mb-3">Executive Summary</h3>
        <p class="text-blue-800 leading-relaxed text-lg">{{ research.llm_summary }}</p>
    </div>

    <!-- Company Overview -->
    <div class="bg-white rounded-lg shadow-md p-6 mb-8">
        <h3 class="text-lg font-semibold text-gray-800 mb-3">Company Overview</h3>
        <p class="text-gray-700 leading-relaxed">{{ research.overview }}</p>
    </div>

    <!-- Sentiment Breakdown -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-green-50 rounded-lg p-6 text-center">
            <div class="text-4xl font-bold text-green-600">{{ research.positive_count }}</div>
            <div class="text-green-700 font-medium">Positive Articles</div>
        </div>
        <div class="bg-gray-50 rounded-lg p-6 text-center">
            <div class="text-4xl font-bold text-gray-600">{{ research.neutral_count }}</div>
            <div class="text-gray-700 font-medium">Neutral Articles</div>
        </div>
        <div class="bg-red-50 rounded-lg p-6 text-center">
            <div class="text-4xl font-bold text-red-600">{{ research.negative_count }}</div>
            <div class="text-red-700 font-medium">Negative Articles</div>
        </div>
    </div>

    <!-- News Coverage Section -->
    <div class="mb-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-2">News Coverage</h2>
        <p class="text-gray-600 mb-6">{{ articles.count }} articles analyzed from diverse sources</p>

        <div class="space-y-4">
            {% for article in articles %}
            <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition">
                <!-- Article Header -->
                <div class="flex justify-between items-start mb-3">
                    <div class="flex-1">
                        <a href="{{ article.url }}" target="_blank"
                           class="text-xl font-semibold text-gray-900 hover:text-blue-600 block mb-2">
                            {{ article.title }}
                        </a>
                        <div class="flex items-center gap-3 text-sm">
                            <!-- Source with Type Badge -->
                            <span class="font-medium text-gray-700">{{ article.source_name|default:article.source }}</span>

                            <!-- Source Type Badge -->
                            <span class="px-2 py-0.5 rounded text-xs font-medium
                                {% if article.source_type == 'major_outlet' %}bg-blue-100 text-blue-700
                                {% elif article.source_type == 'industry_publication' %}bg-purple-100 text-purple-700
                                {% elif article.source_type == 'niche_blog' %}bg-yellow-100 text-yellow-700
                                {% elif article.source_type == 'press_release' %}bg-orange-100 text-orange-700
                                {% else %}bg-gray-100 text-gray-700{% endif %}">
                                {% if article.source_type == 'major_outlet' %}Major Outlet
                                {% elif article.source_type == 'industry_publication' %}Industry Pub
                                {% elif article.source_type == 'niche_blog' %}Niche Blog
                                {% elif article.source_type == 'press_release' %}Press Release
                                {% else %}Unknown{% endif %}
                            </span>

                            <!-- Credibility Indicator -->
                            <span class="text-gray-400" title="{{ article.credibility_note }}">
                                {% if article.credibility_score >= 0.8 %}
                                    <span class="text-green-500">&#9679;&#9679;&#9679;</span>
                                {% elif article.credibility_score >= 0.5 %}
                                    <span class="text-yellow-500">&#9679;&#9679;</span><span class="text-gray-300">&#9679;</span>
                                {% else %}
                                    <span class="text-red-500">&#9679;</span><span class="text-gray-300">&#9679;&#9679;</span>
                                {% endif %}
                                <span class="text-xs text-gray-500 ml-1">credibility</span>
                            </span>

                            {% if article.published_date %}
                            <span class="text-gray-400">|</span>
                            <span class="text-gray-500">{{ article.published_date }}</span>
                            {% endif %}
                        </div>

                        <!-- Credibility Note (collapsible) -->
                        {% if article.credibility_note %}
                        <p class="text-xs text-gray-500 mt-1 italic">{{ article.credibility_note }}</p>
                        {% endif %}
                    </div>

                    <!-- Sentiment Score -->
                    <div class="ml-4 text-center">
                        <div class="text-2xl font-bold
                            {% if article.sentiment_label == 'positive' %}text-green-600
                            {% elif article.sentiment_label == 'negative' %}text-red-600
                            {% else %}text-gray-600{% endif %}">
                            {% if article.ensemble_score >= 0 %}+{% endif %}{{ article.ensemble_score|floatformat:2 }}
                        </div>
                        <div class="text-xs uppercase text-gray-500">sentiment</div>
                    </div>
                </div>

                <!-- Content Preview -->
                {% if article.content_preview %}
                <p class="text-gray-600 text-sm mb-4">{{ article.content_preview }}...</p>
                {% endif %}

                <!-- Model Scores (collapsible/expandable) -->
                <details class="mt-3">
                    <summary class="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
                        View model breakdown
                    </summary>
                    <div class="grid grid-cols-5 gap-4 mt-3 pt-3 border-t text-sm">
                        <div>
                            <div class="text-gray-500">FinBERT</div>
                            <div class="font-medium">{{ article.finbert_score|floatformat:2 }}</div>
                        </div>
                        <div>
                            <div class="text-gray-500">VADER</div>
                            <div class="font-medium">{{ article.vader_score|floatformat:2 }}</div>
                        </div>
                        <div>
                            <div class="text-gray-500">TextBlob</div>
                            <div class="font-medium">{{ article.textblob_polarity|floatformat:2 }}</div>
                        </div>
                        <div>
                            <div class="text-gray-500">RoBERTa</div>
                            <div class="font-medium">{{ article.roberta_score|floatformat:2 }}</div>
                        </div>
                        <div>
                            <div class="text-gray-500">LLM</div>
                            <div class="font-medium">{{ article.llm_score|floatformat:2 }}</div>
                        </div>
                    </div>
                </details>

                <!-- LLM Commentary -->
                {% if article.llm_commentary %}
                <div class="bg-gray-50 rounded p-3 mt-3 text-sm text-gray-700">
                    <span class="font-medium">AI Analysis:</span> {{ article.llm_commentary }}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>

    <!-- Source Diversity Summary -->
    <div class="bg-gray-50 rounded-lg p-6 mb-8">
        <h3 class="text-lg font-semibold text-gray-800 mb-3">Source Diversity</h3>
        <p class="text-gray-600 text-sm mb-4">
            This briefing includes articles from a mix of sources to provide balanced coverage.
            Source credibility is assessed by AI based on publication reputation and content quality.
        </p>
        <div class="flex flex-wrap gap-2">
            <span class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">Major Outlets</span>
            <span class="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">Industry Publications</span>
            <span class="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm">Niche Blogs</span>
        </div>
    </div>

{% endif %}
{% endblock %}

{% block extra_js %}
{% if research.status == 'completed' and research.stock_history %}
<script>
    // Stock price chart
    const stockData = {{ research.stock_history|safe }};
    const ctx = document.getElementById('stockChart').getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: stockData.dates,
            datasets: [{
                label: 'Stock Price',
                data: stockData.prices,
                borderColor: stockData.prices[stockData.prices.length-1] >= stockData.prices[0] ? '#22c55e' : '#ef4444',
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: true,
                    grid: { display: false },
                    ticks: { maxTicksLimit: 5 }
                },
                y: {
                    display: true,
                    grid: { color: '#f3f4f6' },
                    ticks: {
                        callback: function(value) { return '$' + value; }
                    }
                }
            }
        }
    });
</script>
{% endif %}
{% endblock %}
```

---

## Implementation Steps

### Phase 1: Project Setup
1. Create Django project: `django-admin startproject company_sentiment`
2. Create research app: `python manage.py startapp research`
3. Configure settings.py (database, static files, templates)
4. Set up requirements.txt with all dependencies
5. Create directory structure for agents/ and tools/

### Phase 2: Core Tools
1. Implement `tools/llm.py` - Ollama wrapper
2. Implement `tools/web_search.py` - DuckDuckGo news search
3. Implement `tools/web_scraper.py` - Article content extraction
4. Implement `tools/stock_data.py` - yfinance wrapper

### Phase 3: Database
1. Define models in `research/models.py`
2. Create and run migrations
3. Set up admin interface for debugging

### Phase 4: Agents
1. Implement CompanyResearchAgent (with stock history for charts)
2. Implement ArticleCuratorAgent (LLM-powered source evaluation)
3. Implement SentimentAnalyzerAgent (port from spec)
4. Implement Orchestrator to coordinate all agents

### Phase 5: Views & URLs
1. Create URL routes
2. Implement index view (search form)
3. Implement search view (start research)
4. Implement results view (display results)
5. Add status polling for async updates

### Phase 6: Templates
1. Create base template with navigation
2. Create index template (search form)
3. Create results template (loading state + results)
4. Add CSS styling (Tailwind or custom)

### Phase 7: Testing & Polish
1. Test full pipeline with sample companies
2. Add error handling throughout
3. Optimize performance (model loading, caching)
4. Add history page
5. Final UI polish

---

## Dependencies

```txt
# requirements.txt

# Django
Django>=4.2

# LLM
ollama

# Sentiment Models
transformers
torch
vaderSentiment
textblob

# Web Scraping
requests
beautifulsoup4
lxml
duckduckgo-search

# Stock Data
yfinance

# Utilities
rich
```

---

## Configuration

```python
# company_sentiment/settings.py additions

# Add research app
INSTALLED_APPS = [
    ...
    'research',
]

# Use SQLite for simplicity (or PostgreSQL for production)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Agent Configuration
OLLAMA_MODEL = "mistral"  # or "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
MAX_ARTICLES = 10
ARTICLE_SEARCH_DAYS = 7
```

---

## Notes

- **Async Processing**: The research pipeline runs in a background thread. For production, use Celery + Redis for proper task queue management.
- **Model Loading**: Sentiment models (FinBERT, RoBERTa) are loaded lazily to avoid slow startup times.
- **Caching**: Consider caching stock data and search results to avoid repeated API calls.
- **Rate Limiting**: Add delays between web requests to be respectful of sources.
- **Error Handling**: Each agent should handle failures gracefully and provide meaningful error messages.
