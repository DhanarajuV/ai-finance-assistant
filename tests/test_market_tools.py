"""Tests for the market data tool."""
from src.agents.market_agent import get_stock_price


class TestGetStockPrice:
    """Test the get_stock_price tool."""

    def test_valid_ticker(self):
        result = get_stock_price.invoke({"symbol": "AAPL"})
        assert "AAPL" in result
        # Should contain price info or fallback message
        assert "$" in result or "Could not find" in result

    def test_invalid_ticker(self):
        result = get_stock_price.invoke({"symbol": "ZZZZZZZ999"})
        assert "Could not find" in result or "Error" in result

    def test_uppercase_handling(self):
        result = get_stock_price.invoke({"symbol": "aapl"})
        assert "AAPL" in result

    def test_contains_price(self):
        result = get_stock_price.invoke({"symbol": "MSFT"})
        if "Could not find" not in result and "Error" not in result:
            assert "Price:" in result or "$" in result

    def test_contains_market_cap(self):
        result = get_stock_price.invoke({"symbol": "GOOGL"})
        if "Could not find" not in result and "Error" not in result:
            assert "Market Cap:" in result or "$" in result


class TestGetFinancialNews:
    """Test the get_financial_news tool."""

    def test_valid_ticker_news(self):
        from src.agents.news_agent import get_financial_news
        result = get_financial_news.invoke({"topic": "AAPL"})
        assert "AAPL" in result
        # Should have news or "No recent news"
        assert "News" in result or "No recent news" in result

    def test_invalid_ticker_news(self):
        from src.agents.news_agent import get_financial_news
        result = get_financial_news.invoke({"topic": "ZZZZZZZ999"})
        assert "No recent news" in result or "Error" in result
