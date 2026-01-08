"""
Sentiment Analyzer Agent - Multi-model sentiment analysis.

Uses 4 ML models + LLM:
1. FinBERT - Financial sentiment (trained on financial text)
2. VADER - Lexicon-based (good for intensity/hype detection)
3. TextBlob - General polarity + subjectivity measure
4. RoBERTa - General news sentiment

Then LLM provides:
- Its own sentiment score
- Commentary explaining why models agree/disagree
- Context-aware interpretation
"""

import os
import warnings

# Suppress transformers and torch warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from dataclasses import dataclass

from research.tools.llm import LLM


@dataclass
class SentimentScores:
    """Sentiment scores from all models for one article."""
    # ML Model scores (-1 to 1)
    finbert: float
    vader: float
    textblob_polarity: float
    textblob_subjectivity: float  # 0-1, opinion vs fact
    roberta: float

    # LLM analysis
    llm_score: float
    llm_commentary: str

    # Aggregated
    ensemble: float
    agreement: float  # 0-1, how much models agree
    label: str  # positive/negative/neutral


class SentimentAnalyzerAgent:
    """
    Multi-model sentiment analysis agent.

    Analyzes text using 4 different sentiment models plus LLM,
    then combines them into an ensemble score with commentary.
    """

    FINBERT_CHUNK_SIZE = 500

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()
        self.name = "Sentiment Analyzer"

        # Lazy-loaded models
        self._finbert = None
        self._roberta = None
        self._vader = None

    # ----- Lazy Loading -----

    @property
    def finbert(self):
        """Lazy load FinBERT model."""
        if self._finbert is None:
            print(f"  [{self.name}] Loading FinBERT...")
            from transformers import pipeline
            self._finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
        return self._finbert

    @property
    def roberta(self):
        """Lazy load RoBERTa model."""
        if self._roberta is None:
            print(f"  [{self.name}] Loading RoBERTa...")
            from transformers import pipeline
            self._roberta = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
        return self._roberta

    @property
    def vader(self):
        """Lazy load VADER analyzer."""
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    # ----- Individual Model Scores -----

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

    def _vader_score(self, text: str) -> float:
        """
        VADER: Lexicon-based sentiment (-1 to 1).
        Good at detecting intensity, caps, exclamation marks.
        """
        scores = self.vader.polarity_scores(text)
        return scores["compound"]  # Already -1 to 1

    def _textblob_score(self, text: str) -> tuple[float, float]:
        """
        TextBlob: Returns (polarity, subjectivity).
        - polarity: -1 to 1
        - subjectivity: 0 (fact) to 1 (opinion)
        """
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def _roberta_score(self, text: str) -> float:
        """
        RoBERTa: General news sentiment (-1 to 1).
        Truncate to ~512 tokens (roughly 2000 chars).
        """
        try:
            result = self.roberta(text[:2000])[0]
            label = result["label"].lower()
            score = result["score"]

            if "positive" in label:
                return score
            elif "negative" in label:
                return -score
            else:
                return 0.0
        except Exception:
            return 0.0

    def _llm_analyze(self, title: str, content: str, scores: dict) -> dict:
        """
        LLM: Own sentiment score + commentary on other models.
        """
        prompt = f"""Analyze this news article's sentiment.

ARTICLE TITLE: {title}

ARTICLE CONTENT (first 1500 chars):
{content[:1500]}

OTHER MODEL SCORES:
- FinBERT (financial sentiment): {scores['finbert']:.2f}
- VADER (intensity/hype): {scores['vader']:.2f}
- TextBlob polarity: {scores['textblob_polarity']:.2f}
- TextBlob subjectivity: {scores['textblob_subjectivity']:.2f} (0=fact, 1=opinion)
- RoBERTa (general): {scores['roberta']:.2f}

Provide your analysis as JSON:
{{
    "score": <your sentiment score from -1.0 to 1.0>,
    "commentary": "<2-3 sentences explaining: What's the overall sentiment? Why might models agree/disagree? Is this factual news or opinion?>"
}}

Be objective. Consider: Is this genuinely positive/negative news, or just neutral reporting?"""

        result = self.llm.generate_json(prompt)

        try:
            score = float(result.get("score", 0))
            # Clamp to valid range
            score = max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            score = 0.0

        return {
            "score": score,
            "commentary": str(result.get("commentary", ""))[:500]
        }

    # ----- Main Analysis -----

    def analyze(self, content: str, title: str) -> SentimentScores:
        """
        Run all sentiment models on article content.

        Args:
            content: Article text content
            title: Article title

        Returns:
            SentimentScores object with all model results
        """
        # 1. Run all ML models
        finbert = self._finbert_score(content)
        vader = self._vader_score(content)
        tb_polarity, tb_subjectivity = self._textblob_score(content)
        roberta = self._roberta_score(content)

        # 2. Run LLM with other scores as context
        ml_scores = {
            "finbert": finbert,
            "vader": vader,
            "textblob_polarity": tb_polarity,
            "textblob_subjectivity": tb_subjectivity,
            "roberta": roberta
        }
        llm_result = self._llm_analyze(title, content, ml_scores)

        # 3. Calculate ensemble (weighted average)
        # Weights: FinBERT 30%, VADER 15%, TextBlob 15%, RoBERTa 20%, LLM 20%
        ensemble = (
            finbert * 0.30 +
            vader * 0.15 +
            tb_polarity * 0.15 +
            roberta * 0.20 +
            llm_result["score"] * 0.20
        )

        # 4. Calculate agreement (inverse of standard deviation)
        scores_list = [finbert, vader, tb_polarity, roberta, llm_result["score"]]
        mean = sum(scores_list) / len(scores_list)
        variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
        std_dev = variance ** 0.5
        agreement = max(0, 1 - std_dev)  # 1 = perfect agreement

        # 5. Determine label
        if ensemble > 0.1:
            label = "positive"
        elif ensemble < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScores(
            finbert=round(finbert, 3),
            vader=round(vader, 3),
            textblob_polarity=round(tb_polarity, 3),
            textblob_subjectivity=round(tb_subjectivity, 3),
            roberta=round(roberta, 3),
            llm_score=round(llm_result["score"], 3),
            llm_commentary=llm_result["commentary"],
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
