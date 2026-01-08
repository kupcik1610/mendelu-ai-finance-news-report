# Company Sentiment Research Agent

A Django web application that provides AI-powered daily briefings on any company. Analyzes news sentiment using multiple ML models and generates comprehensive reports.

## Features

- **Company Research**: Gathers stock data, recent developments, and company overview
- **LLM-Powered Article Curation**: Evaluates and selects relevant articles from diverse sources
- **Multi-Model Sentiment Analysis**: Uses 4 ML models (FinBERT, VADER, TextBlob, RoBERTa) + LLM
- **Daily Briefing Format**: Clean, readable reports with stock charts and sentiment breakdown
- **Source Transparency**: Shows source type and credibility for each article

## Setup

### 1. Create Virtual Environment

```bash
cd company_sentiment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 3. Start Ollama

Make sure Ollama is installed and running:

```bash
# Install Ollama from https://ollama.ai
ollama serve

# In another terminal, pull the model:
ollama pull mistral
```

### 4. Initialize Database

```bash
python manage.py makemigrations research
python manage.py migrate
python manage.py createsuperuser  # Optional, for admin access
```

The database is SQLite and will be created at `company_sentiment/db.sqlite3`.

### 5. Run the Server

```bash
python manage.py runserver
```

Then open http://127.0.0.1:8000 in your browser.

## Usage

1. Enter a company name (e.g., "Tesla", "Apple", "Microsoft")
2. Wait for the analysis to complete (may take a few minutes)
3. View the daily briefing with:
   - Stock price and chart
   - Key developments
   - Executive summary
   - Sentiment breakdown
   - Individual article analysis

## Architecture

```
research/
├── agents/
│   ├── orchestrator.py      # Coordinates pipeline
│   ├── company_researcher.py # Stock + overview
│   ├── article_curator.py    # LLM-curated articles
│   └── sentiment_analyzer.py # Multi-model analysis
├── tools/
│   ├── llm.py               # Ollama wrapper
│   ├── web_search.py        # DuckDuckGo search
│   ├── web_scraper.py       # Article extraction
│   └── stock_data.py        # yfinance wrapper
└── templates/               # Frontend
```

## Sentiment Models

1. **FinBERT** - Financial sentiment (trained on financial text)
2. **VADER** - Lexicon-based (good for intensity detection)
3. **TextBlob** - General polarity + subjectivity
4. **RoBERTa** - General news sentiment
5. **LLM** - Context-aware interpretation + commentary

## Configuration

Edit `company_sentiment/settings.py`:

```python
OLLAMA_MODEL = "mistral"  # or "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
MAX_ARTICLES = 10
```

## MENDELU AI Project 2025
