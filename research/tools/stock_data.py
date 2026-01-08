"""
Stock data retrieval using yfinance.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf


@dataclass
class StockData:
    """Represents stock data for a company."""
    ticker: str
    price: float | None
    change_percent: float | None
    change_1w: float | None
    change_1m: float | None
    market_cap: str
    history: dict  # {"dates": [...], "prices": [...]}


def get_stock_data(company_name: str, ticker: str = None) -> StockData | None:
    """
    Get stock data for a company.

    Args:
        company_name: Company name (used to guess ticker if not provided)
        ticker: Optional stock ticker symbol

    Returns:
        StockData object or None if not found
    """
    # Try to find ticker if not provided
    if not ticker:
        ticker = _guess_ticker(company_name)
        if not ticker:
            return None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get current price
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not price:
            return None

        # Get price changes
        change_percent = info.get('regularMarketChangePercent')

        # Get historical data for chart and weekly/monthly changes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=35)  # Extra days for buffer

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        # Calculate 1-week and 1-month changes
        change_1w = None
        change_1m = None

        if len(hist) >= 5:
            week_ago_price = hist['Close'].iloc[-5]
            change_1w = round(((price - week_ago_price) / week_ago_price) * 100, 2)

        if len(hist) >= 20:
            month_ago_price = hist['Close'].iloc[-20]
            change_1m = round(((price - month_ago_price) / month_ago_price) * 100, 2)

        # Format market cap
        market_cap = info.get('marketCap', 0)
        market_cap_str = _format_market_cap(market_cap)

        # Prepare chart data (last 30 days)
        chart_data = hist.tail(30)
        history = {
            "dates": [d.strftime("%b %d") for d in chart_data.index],
            "prices": [round(p, 2) for p in chart_data['Close'].tolist()]
        }

        return StockData(
            ticker=ticker.upper(),
            price=round(price, 2),
            change_percent=round(change_percent, 2) if change_percent else None,
            change_1w=change_1w,
            change_1m=change_1m,
            market_cap=market_cap_str,
            history=history
        )

    except Exception as e:
        print(f"Failed to get stock data for {ticker}: {e}")
        return None


def _guess_ticker(company_name: str) -> str | None:
    """
    Try to guess the stock ticker from company name.

    This is a simple mapping - in production you might use a proper lookup API.
    """
    # Common company name to ticker mappings
    name_to_ticker = {
        "tesla": "TSLA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "intel": "INTC",
        "amd": "AMD",
        "ibm": "IBM",
        "oracle": "ORCL",
        "salesforce": "CRM",
        "adobe": "ADBE",
        "spotify": "SPOT",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "paypal": "PYPL",
        "square": "SQ",
        "block": "SQ",
        "shopify": "SHOP",
        "zoom": "ZM",
        "slack": "WORK",
        "twitter": "X",
        "snap": "SNAP",
        "snapchat": "SNAP",
        "pinterest": "PINS",
        "reddit": "RDDT",
        "disney": "DIS",
        "coca-cola": "KO",
        "coca cola": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "mcdonalds": "MCD",
        "mcdonald's": "MCD",
        "walmart": "WMT",
        "target": "TGT",
        "costco": "COST",
        "home depot": "HD",
        "lowes": "LOW",
        "nike": "NKE",
        "starbucks": "SBUX",
        "boeing": "BA",
        "lockheed": "LMT",
        "lockheed martin": "LMT",
        "general motors": "GM",
        "ford": "F",
        "toyota": "TM",
        "honda": "HMC",
        "volkswagen": "VWAGY",
        "bmw": "BMWYY",
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "chevron": "CVX",
        "shell": "SHEL",
        "bp": "BP",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "goldman sachs": "GS",
        "morgan stanley": "MS",
        "visa": "V",
        "mastercard": "MA",
        "american express": "AXP",
        "amex": "AXP",
        "berkshire": "BRK-B",
        "berkshire hathaway": "BRK-B",
        "johnson & johnson": "JNJ",
        "j&j": "JNJ",
        "pfizer": "PFE",
        "moderna": "MRNA",
        "merck": "MRK",
        "abbvie": "ABBV",
        "unitedhealth": "UNH",
        "cvs": "CVS",
        "walgreens": "WBA",
        "at&t": "T",
        "verizon": "VZ",
        "t-mobile": "TMUS",
        "comcast": "CMCSA",
        "verizon": "VZ",
    }

    company_lower = company_name.lower().strip()

    # Direct lookup
    if company_lower in name_to_ticker:
        return name_to_ticker[company_lower]

    # Partial match
    for name, ticker in name_to_ticker.items():
        if name in company_lower or company_lower in name:
            return ticker

    # Try using yfinance search (this is slow but more comprehensive)
    try:
        # Try the company name as ticker directly
        test_ticker = company_name.upper().replace(" ", "")
        stock = yf.Ticker(test_ticker)
        if stock.info.get('currentPrice'):
            return test_ticker
    except:
        pass

    return None


def _format_market_cap(market_cap: int) -> str:
    """Format market cap as human-readable string."""
    if not market_cap:
        return ""

    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.2f}M"
    else:
        return f"${market_cap:,}"
