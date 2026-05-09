"""Tests for the LangGraph workflow."""
from unittest.mock import patch, MagicMock
from src.workflow.graph import AGENT_PRIORITY, chat


class TestAgentPriority:
    """Test agent execution priority ordering."""

    def test_market_runs_before_portfolio(self):
        assert AGENT_PRIORITY["market"] < AGENT_PRIORITY["portfolio"]

    def test_news_runs_before_portfolio(self):
        assert AGENT_PRIORITY["news"] < AGENT_PRIORITY["portfolio"]

    def test_portfolio_runs_before_goal(self):
        assert AGENT_PRIORITY["portfolio"] < AGENT_PRIORITY["goal_planning"]

    def test_market_is_first(self):
        assert AGENT_PRIORITY["market"] == 1

    def test_goal_planning_is_last(self):
        assert AGENT_PRIORITY["goal_planning"] == max(AGENT_PRIORITY.values())

    def test_all_agents_have_priority(self):
        expected_agents = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]
        for agent in expected_agents:
            assert agent in AGENT_PRIORITY


class TestChatFunction:
    """Test the main chat() entry point."""

    @patch("src.workflow.graph.app")
    def test_returns_tuple_of_three(self, mock_app):
        mock_app.invoke.return_value = {
            "response": "Test response",
            "agent_types": ["finance_qa"],
            "chat_history": [],
        }

        response, agent_label, history = chat("Hello")
        assert response == "Test response"
        assert agent_label == "finance_qa"
        assert history == []

    @patch("src.workflow.graph.app")
    def test_multi_agent_label(self, mock_app):
        mock_app.invoke.return_value = {
            "response": "Combined response",
            "agent_types": ["market", "portfolio"],
            "chat_history": [],
        }

        _, agent_label, _ = chat("Analyze AAPL portfolio")
        assert agent_label == "market, portfolio"

    @patch("src.workflow.graph.app")
    def test_default_empty_history(self, mock_app):
        mock_app.invoke.return_value = {
            "response": "Response",
            "agent_types": ["finance_qa"],
            "chat_history": [],
        }

        chat("Hello")
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["chat_history"] == []

    @patch("src.workflow.graph.app")
    def test_passes_existing_history(self, mock_app):
        mock_app.invoke.return_value = {
            "response": "Response",
            "agent_types": ["finance_qa"],
            "chat_history": ["msg1", "msg2"],
        }

        existing_history = ["prev_msg"]
        chat("Hello", existing_history)
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["chat_history"] == ["prev_msg"]
