"""
Company Research Agent - Gathers company overview, stock data, and key developments.
"""

from research.logging_config import get_logger
from research.tools.llm import LLM
from research.tools.web_search import search_web, search_news
from research.tools.stock_data import get_stock_data, StockData

logger = get_logger('company_researcher')


class CompanyResearchAgent:
    """
    Researches a company to gather:
    - Basic company info (industry, what they do)
    - Stock data (price, changes, historical chart)
    - Recent key developments
    - Company overview/writeup
    """

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()
        self.name = "Company Researcher"

    def run(self, company_name: str) -> dict:
        """
        Research a company and return overview data.

        Args:
            company_name: Name of the company to research

        Returns:
            Dictionary with company data
        """
        logger.info(f"Researching {company_name}...")

        # 1. Get stock data
        logger.info("Fetching stock data...")
        stock_data = get_stock_data(company_name)

        # 2. Search for company info
        logger.info("Searching for company information...")
        search_results = self._search_company_info(company_name)

        # 3. LLM generates overview
        logger.info("Generating company overview...")
        overview = self._generate_overview(company_name, stock_data, search_results)

        result = {
            'industry': overview.get('industry', ''),
            'overview': overview.get('overview', ''),
            'recent_developments': overview.get('developments', []),
            'stock_ticker': stock_data.ticker if stock_data else '',
            'stock_price': stock_data.price if stock_data else None,
            'stock_change_percent': stock_data.change_percent if stock_data else None,
            'stock_change_1w': stock_data.change_1w if stock_data else None,
            'stock_change_1m': stock_data.change_1m if stock_data else None,
            'market_cap': stock_data.market_cap if stock_data else '',
            'stock_history': stock_data.history if stock_data else {"dates": [], "prices": []},
        }

        logger.info("Research complete")
        return result

    def _search_company_info(self, company_name: str) -> list:
        """Search for company information via web search."""
        results = []

        # Search for general company info
        general_results = search_web(f"{company_name} company", max_results=5)
        results.extend(general_results)

        # Search for recent news
        news_results = search_news(f"{company_name}", max_results=5)
        results.extend(news_results)

        return results

    def _generate_overview(
        self,
        company_name: str,
        stock_data: StockData | None,
        search_results: list
    ) -> dict:
        """LLM generates company overview from gathered data."""

        # Format search results for prompt
        search_text = ""
        for i, r in enumerate(search_results[:10], 1):
            search_text += f"{i}. {r.title}\n   {r.snippet[:200] if r.snippet else 'No description'}\n\n"

        # Format stock data
        stock_text = "Not available (company may be private or ticker not found)"
        if stock_data:
            stock_text = f"""- Ticker: {stock_data.ticker}
- Current Price: ${stock_data.price}
- Today's Change: {stock_data.change_percent}%
- 1 Week Change: {stock_data.change_1w}%
- 1 Month Change: {stock_data.change_1m}%
- Market Cap: {stock_data.market_cap}"""

        prompt = f"""You are researching the company: {company_name}

STOCK DATA:
{stock_text}

SEARCH RESULTS:
{search_text}

Based on this information, provide a JSON response with:
{{
    "industry": "<primary industry/sector the company operates in>",
    "overview": "<2-3 paragraphs about the company: what they do, their market position, recent performance, any notable news or trends. Write in a professional, informative tone suitable for a business briefing.>",
    "developments": ["<recent development 1>", "<recent development 2>", "<recent development 3>", "<recent development 4>", "<recent development 5>"]
}}

Guidelines:
- Industry should be specific (e.g., "Electric Vehicles & Clean Energy" not just "Automotive")
- Overview should be factual and based on the search results
- Developments should be recent, specific events or news items
- If stock data is available, mention it in the overview
- Keep the tone professional and objective"""

        result = self.llm.generate_json(prompt)

        # Ensure we have all required fields
        return {
            'industry': result.get('industry', 'Unknown'),
            'overview': result.get('overview', f'{company_name} is a company. No detailed information available.'),
            'developments': result.get('developments', [])[:5]
        }
