"""
Views for the research application.
"""

import threading
import json

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .logging_config import get_logger
from .models import CompanyResearch
from .forms import CompanySearchForm

logger = get_logger('views')


def index(request):
    """Home page with company search form."""
    form = CompanySearchForm()
    recent_searches = CompanyResearch.objects.filter(status='completed')[:5]
    return render(request, 'research/index.html', {
        'form': form,
        'recent_searches': recent_searches
    })


@require_POST
def search(request):
    """Start a new company research."""
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
        thread.daemon = True
        thread.start()

        # Redirect to results page (will show loading state)
        return redirect('research:results', pk=research.id)

    return redirect('research:index')


def results(request, pk):
    """Display research results (or loading state if still processing)."""
    research = get_object_or_404(CompanyResearch, pk=pk)
    articles = research.articles.all()

    # Prepare stock history for JavaScript
    stock_history_json = research.stock_history if research.stock_history else '{}'

    return render(request, 'research/results.html', {
        'research': research,
        'articles': articles,
        'stock_history_json': stock_history_json,
    })


def status(request, pk):
    """AJAX endpoint for checking research status."""
    research = get_object_or_404(CompanyResearch, pk=pk)
    return JsonResponse({
        'status': research.status,
        'error': research.error_message
    })


def history(request):
    """List all past researches."""
    researches = CompanyResearch.objects.filter(status='completed')
    return render(request, 'research/history.html', {
        'researches': researches
    })


def api_research(request, pk):
    """Full research data as JSON."""
    research = get_object_or_404(CompanyResearch, pk=pk)
    articles = research.articles.all()

    return JsonResponse({
        'company': research.company_name,
        'status': research.status,
        'industry': research.industry,
        'overview': research.overview,
        'stock_ticker': research.stock_ticker,
        'stock_price': str(research.stock_price) if research.stock_price else None,
        'stock_change': str(research.stock_change_percent) if research.stock_change_percent else None,
        'overall_sentiment': str(research.overall_sentiment) if research.overall_sentiment else None,
        'sentiment_label': research.sentiment_label,
        'llm_summary': research.llm_summary,
        'positive_count': research.positive_count,
        'negative_count': research.negative_count,
        'neutral_count': research.neutral_count,
        'articles': [
            {
                'title': a.title,
                'source': a.source,
                'source_name': a.source_name,
                'source_type': a.source_type,
                'url': a.url,
                'sentiment_label': a.sentiment_label,
                'ensemble_score': str(a.ensemble_score) if a.ensemble_score else None,
                'credibility_score': str(a.credibility_score) if a.credibility_score else None,
                'credibility_note': a.credibility_note,
                'llm_commentary': a.llm_commentary
            }
            for a in articles
        ]
    })


def run_research_pipeline(research_id):
    """Background task: run the full research pipeline."""
    # Import here to avoid circular imports
    from .agents.orchestrator import Orchestrator

    research = CompanyResearch.objects.get(pk=research_id)

    try:
        orchestrator = Orchestrator()
        orchestrator.run(research)
        research.status = 'completed'
    except Exception as e:
        research.status = 'failed'
        research.error_message = str(e)
        logger.error(f"Research failed: {e}")

    research.save()
