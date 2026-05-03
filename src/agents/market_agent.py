import yfinance as yf
from langchain_core.tools import tool
from src.agents.base_agent import BaseAgent
from src.core.config import config

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic info for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "currentPrice" not in info:
            # Fallback to history
            hist = ticker.history(period="1d")
            if hist.empty:
                return f"Could not find data for symbol: {symbol}"
            price = hist["Close"].iloc[-1]
            return f"{symbol.upper()}: ${price:.2f} (latest close)"

        name = info.get("shortName", symbol.upper())
        price = info.get("currentPrice", 0)
        change = info.get("regularMarketChangePercent", 0)
        high_52 = info.get("fiftyTwoWeekHigh", "N/A")
        low_52 = info.get("fiftyTwoWeekLow", "N/A")
        market_cap = info.get("marketCap", 0)
        pe_ratio = info.get("trailingPE", "N/A")

        # Format market cap
        if market_cap >= 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        else:
            cap_str = f"${market_cap/1e6:.2f}M"

        direction = "📈" if change >= 0 else "📉"

        return (
            f"{direction} {name} ({symbol.upper()})\n"
            f"Price: ${price:.2f} ({change:+.2f}%)\n"
            f"52-Week Range: ${low_52} - ${high_52}\n"
            f"Market Cap: {cap_str}\n"
            f"P/E Ratio: {pe_ratio}"
        )
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"


class MarketAgent(BaseAgent):
    """Provides real-time market data and analysis."""

    def __init__(self):
        system_prompt = f"""You are {config['app']['name']}, a market analysis assistant.

Your role:
- Provide real-time stock data using the get_stock_price tool
- Explain market metrics in beginner-friendly terms
- Help users understand what the numbers mean

Rules:
- Never recommend buying or selling specific stocks
- {config['app']['disclaimer']}
- Explain what P/E ratio, market cap, and 52-week range mean when relevant
- If a user asks about a stock, always use the tool to get current data"""

        super().__init__(
            name="Market Analysis Agent",
            system_prompt=system_prompt,
            tools=[get_stock_price],
        )
