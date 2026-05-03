import yfinance as yf
from langchain_core.tools import tool
from src.agents.base_agent import BaseAgent
from src.core.config import config


@tool
def get_financial_news(topic: str) -> str:
    """Get recent financial news for a stock ticker or company.

    Args:
        topic: A stock ticker symbol (e.g., AAPL, MSFT, GOOGL).
    """
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


class NewsAgent(BaseAgent):
    """Summarizes and contextualizes financial news."""

    def __init__(self):
        system_prompt = f"""You are {config['app']['name']}, a financial news assistant.

Your role:
- Fetch and summarize recent financial news using the get_financial_news tool
- Explain how news might impact a stock or the broader market
- Put news in context for beginners

Rules:
- Always use the tool to get real news, don't make up headlines
- {config['app']['disclaimer']}
- Explain financial jargon in news headlines
- If the user asks about general market news, use a broad ticker like SPY"""

        super().__init__(
            name="News Synthesizer Agent",
            system_prompt=system_prompt,
            tools=[get_financial_news],
        )
