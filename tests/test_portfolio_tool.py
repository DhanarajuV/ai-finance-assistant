"""Tests for the portfolio analysis tool."""
from src.agents.portfolio_agent import analyze_portfolio


class TestAnalyzePortfolio:
    """Test the analyze_portfolio tool function."""

    def test_basic_percentage_format(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 40%, GOOGL 30%, BND 30%"})
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "BND" in result
        assert "Diversification Score" in result
        assert "Risk Level" in result

    def test_single_holding(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 100%"})
        assert "AAPL: 100.0%" in result
        assert "Diversification Score: 2/10" in result
        assert "Aggressive" in result

    def test_bond_detection(self):
        result = analyze_portfolio.invoke({"holdings": "BND 60%, AAPL 40%"})
        assert "Bond/Fixed Income Allocation: 60.0%" in result
        assert "Stock/Equity Allocation: 40.0%" in result
        assert "Conservative" in result

    def test_moderate_risk(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 50%, BND 50%"})
        assert "Moderate" in result

    def test_aggressive_risk(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 50%, GOOGL 30%, MSFT 20%"})
        assert "Aggressive" in result

    def test_diversification_score_max(self):
        result = analyze_portfolio.invoke({"holdings": "A 10%, B 10%, C 10%, D 10%, E 10%, F 50%"})
        assert "Diversification Score: 10/10" in result

    def test_low_diversification_suggestion(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 50%, GOOGL 50%"})
        assert "Consider adding more holdings" in result

    def test_no_bonds_suggestion(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL 50%, GOOGL 50%"})
        assert "Consider adding bonds" in result

    def test_empty_input(self):
        result = analyze_portfolio.invoke({"holdings": ""})
        assert "Could not parse" in result

    def test_invalid_input(self):
        result = analyze_portfolio.invoke({"holdings": "hello world"})
        # "hello" becomes name, "world" can't be parsed as number, value=0
        # total=0, so pct calculation handles division
        assert "Portfolio Analysis" in result or "Could not parse" in result

    def test_dollar_format(self):
        result = analyze_portfolio.invoke({"holdings": "AAPL $5000, BND $5000"})
        assert "AAPL: 50.0%" in result
        assert "BND: 50.0%" in result

    def test_multiple_bond_keywords(self):
        result = analyze_portfolio.invoke({"holdings": "AGG 30%, TREASURY 20%, AAPL 50%"})
        assert "Bond/Fixed Income Allocation: 50.0%" in result
