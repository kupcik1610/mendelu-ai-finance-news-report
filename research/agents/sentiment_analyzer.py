"""
Sentiment Analyzer Agent - FinBERT + LLM sentiment analysis.

Uses 2 complementary approaches:
1. FinBERT - Financial sentiment (trained on financial text)
2. LLM - Context-aware sentiment with reasoning
"""

import os
import warnings

# Suppress transformers and torch warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from dataclasses import dataclass

from research.logging_config import get_logger
from research.tools.llm import LLM

logger = get_logger('sentiment_analyzer')


@dataclass
class SentimentScores:
    """Sentiment scores from FinBERT + LLM for one article."""
    finbert: float  # -1 to 1
    llm_score: float  # -1 to 1
    llm_reasoning: str

    ensemble: float  # weighted average
    agreement: float  # 0-1, how much models agree
    label: str  # positive/negative/neutral


class SentimentAnalyzerAgent:
    """
    FinBERT + LLM sentiment analysis agent.

    Combines FinBERT (financial domain) with LLM (contextual reasoning)
    for accurate financial news sentiment analysis.
    """

    FINBERT_CHUNK_SIZE = 500
    _finbert = None  # Class-level cache

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()
        self.name = "Sentiment Analyzer"

    # ----- Lazy Loading -----

    @property
    def finbert(self):
        """Lazy load FinBERT model."""
        if SentimentAnalyzerAgent._finbert is None:
            logger.info("Loading FinBERT model...")
            from transformers import pipeline
            SentimentAnalyzerAgent._finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
        return SentimentAnalyzerAgent._finbert

    # ----- Model Scores -----

    def _finbert_score(self, text: str) -> float:
        """
        FinBERT: Financial sentiment (-1 to 1).
        Chunks long text and averages scores.
        """
        chunks = self._chunk_text(text, self.FINBERT_CHUNK_SIZE)
        scores = []

        for chunk in chunks:
            try:
                result = self.finbert(chunk)[0]
                label = result["label"]
                score = result["score"]

                if label == "positive":
                    scores.append(score)
                elif label == "negative":
                    scores.append(-score)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _llm_score(self, title: str, content: str) -> dict:
        """
        LLM: Independent sentiment score + reasoning.
        """
        prompt = f"""Analyze this news article's sentiment for investors.

TITLE: {title}

CONTENT:
{content}

Respond as JSON:
{{
    "score": <-1.0 to 1.0>,
    "reasoning": "<3-4 sentences: Is this genuinely positive/negative news, or neutral factual reporting?>"
}}"""

        result = self.llm.generate_json(prompt)

        try:
            score = float(result.get("score", 0))
            # Clamp to valid range
            score = max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            score = 0.0

        return {
            "score": score,
            "commentary": str(result.get("reasoning", ""))[:500]
        }

    # ----- Main Analysis -----

    def analyze(self, content: str, title: str) -> SentimentScores:
        """
        Run FinBERT + LLM sentiment analysis on article.

        Args:
            content: Article text content
            title: Article title

        Returns:
            SentimentScores with FinBERT, LLM, and ensemble results
        """
        finbert = self._finbert_score(content)
        llm_result = self._llm_score(title, content)
        llm_score = llm_result["score"]

        # Ensemble: 50% FinBERT, 50% LLM
        ensemble = (finbert + llm_score) / 2

        # Agreement: how close are the two scores (0-1)
        agreement = 1 - abs(finbert - llm_score) / 2

        # Label
        if ensemble > 0.1:
            label = "positive"
        elif ensemble < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScores(
            finbert=round(finbert, 3),
            llm_score=round(llm_score, 3),
            llm_reasoning=llm_result["commentary"],
            ensemble=round(ensemble, 3),
            agreement=round(agreement, 2),
            label=label
        )

    def _chunk_text(self, text: str, size: int) -> list[str]:
        """Split text into chunks of approximately `size` characters."""
        if not text:
            return [""]

        chunks = []
        for i in range(0, len(text), size):
            chunk = text[i:i + size].strip()
            if chunk:
                chunks.append(chunk)

        return chunks if chunks else [""]
