"""Tests for the router module."""
from unittest.mock import patch, MagicMock
from src.workflow.router import route_query, AGENT_TYPES


class TestRouter:
    """Test the query router."""

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_finance_qa(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="finance_qa")
        mock_llm_class.return_value = mock_llm

        result = route_query("What is an ETF?")
        assert result == ["finance_qa"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_portfolio(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="portfolio")
        mock_llm_class.return_value = mock_llm

        result = route_query("Analyze my portfolio")
        assert result == ["portfolio"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_market(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="market")
        mock_llm_class.return_value = mock_llm

        result = route_query("How is AAPL doing?")
        assert result == ["market"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_goal_planning(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="goal_planning")
        mock_llm_class.return_value = mock_llm

        result = route_query("I want to retire in 20 years")
        assert result == ["goal_planning"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_news(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="news")
        mock_llm_class.return_value = mock_llm

        result = route_query("Latest news about Tesla")
        assert result == ["news"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_routes_tax(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="tax")
        mock_llm_class.return_value = mock_llm

        result = route_query("How are capital gains taxed?")
        assert result == ["tax"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_multi_agent_routing(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="market,portfolio,goal_planning")
        mock_llm_class.return_value = mock_llm

        result = route_query("Analyze Apple and plan my 100k")
        assert "market" in result
        assert "portfolio" in result
        assert "goal_planning" in result

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_fallback_on_invalid_response(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="invalid_category")
        mock_llm_class.return_value = mock_llm

        result = route_query("random gibberish")
        assert result == ["finance_qa"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_fallback_on_empty_response(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        mock_llm_class.return_value = mock_llm

        result = route_query("hello")
        assert result == ["finance_qa"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_strips_whitespace(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  market  ")
        mock_llm_class.return_value = mock_llm

        result = route_query("How is AAPL?")
        assert result == ["market"]

    @patch("src.workflow.router.ChatGoogleGenerativeAI")
    def test_handles_uppercase_response(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="MARKET")
        mock_llm_class.return_value = mock_llm

        result = route_query("How is AAPL?")
        assert result == ["market"]

    def test_agent_types_complete(self):
        expected = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]
        assert AGENT_TYPES == expected
