"""
Orchestrator - Coordinates the full research pipeline.

Workflow:
1. Validate LLM availability (Ollama)
2. Run CompanyResearchAgent -> get overview, stock data
3. Run ArticleCuratorAgent -> get curated articles
4. Run SentimentAnalyzerAgent -> analyze each article
5. Calculate aggregate sentiment
6. Generate LLM summary
7. Save results to database
"""

import json

from research.logging_config import get_logger
from research.tools.llm import LLM
from research.agents.company_researcher import CompanyResearchAgent
from research.agents.article_curator import ArticleCuratorAgent
from research.agents.sentiment_analyzer import SentimentAnalyzerAgent

logger = get_logger('orchestrator')


class Orchestrator:
    """
    Main coordinator for the research pipeline.

    Takes a CompanyResearch model instance and runs the full analysis,
    updating the database record as it progresses.
    """

    def __init__(self):
        self.llm = LLM()
        self.company_researcher = CompanyResearchAgent(self.llm)
        self.article_curator = ArticleCuratorAgent(self.llm)
        self.sentiment_analyzer = SentimentAnalyzerAgent(self.llm)

    def _update_progress(self, research, message: str):
        """Update progress message for frontend display."""
        research.progress_message = message
        research.save(update_fields=['progress_message'])

    def run(self, research):
        """
        Execute full pipeline and update research record.

        Args:
            research: CompanyResearch model instance to populate
        """
        from research.models import Article

        company_name = research.company_name

        # 1. Validate Ollama is available
        self._update_progress(research, "Connecting to LLM...")
        if not self.llm.is_available():
            raise Exception(
                "Ollama is not running or model not found. "
                "Start it with: ollama serve && ollama pull mistral"
            )

        logger.info("=" * 60)
        logger.info(f"Starting research on: {company_name}")
        logger.info("=" * 60)

        # 2. Research company
        self._update_progress(research, "Fetching stock data and company info...")
        logger.info("[Phase 1/4] Researching company...")
        company_data = self.company_researcher.run(company_name)

        # Update research record with company data
        research.industry = company_data['industry']
        research.overview = company_data['overview']
        research.stock_ticker = company_data['stock_ticker']
        research.stock_price = company_data['stock_price']
        research.stock_change_percent = company_data['stock_change_percent']
        research.stock_change_1w = company_data['stock_change_1w']
        research.stock_change_1m = company_data['stock_change_1m']
        research.market_cap = company_data['market_cap']
        research.stock_history = json.dumps(company_data['stock_history'])
        research.recent_developments = json.dumps(company_data['recent_developments'])
        research.save()

        # 3. Gather articles
        self._update_progress(research, "Searching for news articles...")
        logger.info("[Phase 2/4] Curating articles...")
        articles = self.article_curator.run(
            company_name,
            max_articles=10
        )

        if not articles:
            raise Exception("No articles found for analysis")

        # 4. Analyze sentiment for each article
        logger.info(f"[Phase 3/4] Analyzing sentiment for {len(articles)} articles...")
        analyzed_articles = []

        for i, article in enumerate(articles, 1):
            self._update_progress(research, f"Analyzing article {i}/{len(articles)}...")
            logger.info(f"Analyzing article {i}/{len(articles)}: {article.title[:50]}...")

            sentiment = self.sentiment_analyzer.analyze(article.content, article.title)

            # Save to database
            Article.objects.create(
                research=research,
                url=article.url,
                title=article.title,
                source=article.source,
                source_name=article.source_name,
                author=article.author,
                published_date=article.date,
                content_preview=article.content[:300],
                word_count=article.word_count,
                finbert_score=sentiment.finbert,
                llm_score=sentiment.llm_score,
                llm_reasoning=sentiment.llm_reasoning,
                ensemble_score=sentiment.ensemble,
                model_agreement=sentiment.agreement,
                sentiment_label=sentiment.label
            )
            analyzed_articles.append(sentiment)

        # 5. Calculate aggregates
        self._update_progress(research, "Generating executive summary...")
        logger.info("[Phase 4/4] Generating summary...")
        self._calculate_aggregates(research, analyzed_articles)

        # 6. Generate overall summary
        research.llm_summary = self._generate_summary(research, articles, analyzed_articles)
        research.progress_message = "Complete!"
        research.save()

        logger.info("=" * 60)
        logger.info("Research complete!")
        logger.info(f"Overall sentiment: {research.overall_sentiment:.2f} ({research.sentiment_label})")
        logger.info(f"Articles analyzed: {len(articles)}")
        logger.info("=" * 60)

    def _calculate_aggregates(self, research, sentiments):
        """Calculate aggregate sentiment metrics."""
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

        research.save()

    def _generate_summary(self, research, articles, sentiments) -> str:
        """LLM generates overall summary of company sentiment."""

        # Prepare article summaries for LLM
        article_summaries = []
        for article, sentiment in zip(articles[:10], sentiments[:10]):
            article_summaries.append(
                f"- {article.title} ({article.source_name}): "
                f"{sentiment.label} ({sentiment.ensemble:.2f})"
            )

        prompt = f"""Based on the sentiment analysis of recent news about {research.company_name}, write a brief executive summary.

COMPANY: {research.company_name}
INDUSTRY: {research.industry}

OVERALL METRICS:
- Average sentiment: {research.overall_sentiment:.2f} ({research.sentiment_label})
- Positive articles: {research.positive_count}
- Neutral articles: {research.neutral_count}
- Negative articles: {research.negative_count}

ARTICLES ANALYZED:
{chr(10).join(article_summaries)}

Write 3-4 sentences summarizing the overall sentiment around {research.company_name}.
Focus on:
- What's the general tone of recent coverage?
- Any notable patterns or themes?
- What might this mean for the company?

Be objective and data-driven. Write in a professional tone suitable for a business briefing."""

        return self.llm.generate(prompt)[:800]
