"""
Database models for the company sentiment research application.
"""

import json
from django.db import models
from django.utils import timezone


class CompanyResearch(models.Model):
    """Stores a complete research session for a company."""

    # Basic info
    company_name = models.CharField(max_length=200)
    industry = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    # Company overview (from LLM research)
    overview = models.TextField(blank=True)
    stock_ticker = models.CharField(max_length=20, blank=True)
    stock_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    stock_change_percent = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    stock_change_1w = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    stock_change_1m = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    market_cap = models.CharField(max_length=50, blank=True)
    stock_history = models.TextField(blank=True)  # JSON: {"dates": [...], "prices": [...]}
    recent_developments = models.TextField(blank=True)  # JSON list of key points

    # Aggregate sentiment
    overall_sentiment = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    sentiment_label = models.CharField(max_length=20, blank=True)  # positive/negative/neutral
    positive_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)

    # LLM summary
    llm_summary = models.TextField(blank=True)

    # Status
    status = models.CharField(max_length=20, default='pending')  # pending/processing/completed/failed
    error_message = models.TextField(blank=True)
    progress_message = models.CharField(max_length=200, blank=True)  # Live progress updates

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Company researches"

    def __str__(self):
        return f"{self.company_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    def get_developments(self):
        """Return recent developments as a list."""
        if not self.recent_developments:
            return []
        try:
            return json.loads(self.recent_developments)
        except json.JSONDecodeError:
            return []

    def get_stock_history(self):
        """Return stock history as a dict for chart."""
        if not self.stock_history:
            return {"dates": [], "prices": []}
        try:
            return json.loads(self.stock_history)
        except json.JSONDecodeError:
            return {"dates": [], "prices": []}


class Article(models.Model):
    """Individual article analyzed for a company."""

    research = models.ForeignKey(
        CompanyResearch,
        on_delete=models.CASCADE,
        related_name='articles'
    )

    # Article metadata
    url = models.URLField(max_length=500)
    title = models.CharField(max_length=500)
    source = models.CharField(max_length=200)  # Domain name
    source_name = models.CharField(max_length=200, blank=True)  # Human-readable name
    author = models.CharField(max_length=200, blank=True)
    published_date = models.CharField(max_length=50, blank=True)  # Keep as string for flexibility
    content_preview = models.TextField(blank=True)  # First 300 chars
    word_count = models.IntegerField(default=0)

    # Sentiment scores (all -1 to 1 scale)
    finbert_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    llm_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    llm_reasoning = models.TextField(blank=True)

    # Aggregated
    ensemble_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    model_agreement = models.DecimalField(max_digits=3, decimal_places=2, null=True, blank=True)  # 0-1
    sentiment_label = models.CharField(max_length=20, blank=True)

    class Meta:
        ordering = ['-ensemble_score']

    def __str__(self):
        return f"{self.title[:50]}... ({self.sentiment_label})"
