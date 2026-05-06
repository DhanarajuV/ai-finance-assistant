from mcp.server.fastmcp import FastMCP
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.rag.knowledge_base import load_vector_store
from src.core.config import config

# Initialize MCP server
mcp = FastMCP("Finnie - AI Finance Assistant")

# Load vector store once
_vector_store = load_vector_store()


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """Get current stock price and key metrics for a ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "currentPrice" not in info:
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


@mcp.tool()
def analyze_portfolio(holdings: str) -> str:
    """Analyze a portfolio for allocation, diversification, and risk.

    Args:
        holdings: Portfolio holdings as "NAME PERCENTAGE" pairs separated by commas.
                  Example: "AAPL 40%, GOOGL 30%, BND 30%"
    """
    lines = [h.strip() for h in holdings.replace(",", "\n").split("\n") if h.strip()]
    parsed = []
    total = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        name = parts[0]
        value = 0
        for p in parts[1:]:
            cleaned = p.replace("%", "").replace("$", "").replace(",", "")
            try:
                value = float(cleaned)
                break
            except ValueError:
                continue
        parsed.append({"name": name, "value": value})
        total += value

    if not parsed:
        return "Could not parse portfolio. Please provide holdings like: AAPL 40%, GOOGL 30%, BND 30%"

    analysis = "📊 Portfolio Analysis\n\nHoldings:\n"
    stock_pct = 0
    bond_pct = 0
    bond_keywords = {"bnd", "agg", "bond", "bonds", "treasury", "fixed"}

    for h in parsed:
        pct = (h["value"] / total * 100) if total > 0 else 0
        analysis += f"  • {h['name']}: {pct:.1f}%\n"
        if h["name"].lower() in bond_keywords:
            bond_pct += pct
        else:
            stock_pct += pct

    div_score = min(len(parsed) * 2, 10)

    if stock_pct >= 80:
        risk = "Aggressive (high risk, high potential return)"
    elif stock_pct >= 50:
        risk = "Moderate (balanced risk and return)"
    else:
        risk = "Conservative (lower risk, steady returns)"

    analysis += f"\nDiversification Score: {div_score}/10"
    analysis += f"\nRisk Level: {risk}"
    analysis += f"\nStock/Equity Allocation: {stock_pct:.1f}%"
    analysis += f"\nBond/Fixed Income Allocation: {bond_pct:.1f}%"

    if len(parsed) < 5:
        analysis += "\n\n⚠️ Suggestion: Consider adding more holdings for better diversification."
    if bond_pct == 0:
        analysis += "\n⚠️ Suggestion: Consider adding bonds to reduce portfolio volatility."

    return analysis


@mcp.tool()
def project_investment(monthly_amount: float, years: int, annual_return: float = 7.0) -> str:
    """Project future value of regular monthly investments using compound interest.

    Args:
        monthly_amount: Amount invested per month in dollars.
        years: Number of years to invest.
        annual_return: Expected annual return percentage (default 7% for stock market average).
    """
    monthly_rate = annual_return / 100 / 12
    months = years * 12
    total_contributed = monthly_amount * months

    if monthly_rate > 0:
        future_value = monthly_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    else:
        future_value = total_contributed

    earnings = future_value - total_contributed

    result = f"💰 Investment Projection\n\n"
    result += f"Monthly investment: ${monthly_amount:,.2f}\n"
    result += f"Time horizon: {years} years\n"
    result += f"Assumed annual return: {annual_return}%\n\n"
    result += f"Total contributed: ${total_contributed:,.2f}\n"
    result += f"Investment earnings: ${earnings:,.2f}\n"
    result += f"Projected value: ${future_value:,.2f}\n\n"

    result += "Milestones:\n"
    for milestone in [5, 10, 20, 30]:
        if milestone <= years:
            m = milestone * 12
            fv = monthly_amount * (((1 + monthly_rate) ** m - 1) / monthly_rate)
            result += f"  Year {milestone}: ${fv:,.2f}\n"

    return result


@mcp.tool()
def search_financial_knowledge(query: str) -> str:
    """Search the financial education knowledge base for information about investing concepts.

    Args:
        query: The financial topic or question to search for (e.g., "what is diversification", "how do ETFs work")
    """
    results = _vector_store.similarity_search(query, k=config["rag"]["top_k"])

    if not results:
        return "No relevant articles found in the knowledge base."

    output = ""
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("title", "Unknown")
        output += f"\n[Source {i}: {source}]\n{doc.page_content}\n"

    return output


@mcp.tool()
def get_financial_news(topic: str) -> str:
    """Get recent financial news headlines for a stock or company.

    Args:
        topic: A stock ticker symbol (e.g., AAPL, MSFT, TSLA)
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(topic.upper())
        news = ticker.news

        if not news:
            return f"No recent news found for {topic.upper()}."

        output = f"📰 Recent News for {topic.upper()}:\n\n"
        for i, item in enumerate(news[:5], 1):
            content = item.get("content", {})
            title = content.get("title", "No title")
            publisher = content.get("provider", {}).get("displayName", "Unknown")
            link = content.get("canonicalUrl", {}).get("url", "")
            summary = content.get("summary", "")[:200]
            output += f"{i}. **{title}**\n   Source: {publisher}\n   {summary}\n   Link: {link}\n\n"

        return output
    except Exception as e:
        return f"Error fetching news for {topic}: {str(e)}"


if __name__ == "__main__":
    mcp.run()