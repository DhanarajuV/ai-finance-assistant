"""Tests for the base agent class."""
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agents.base_agent import BaseAgent


class TestBaseAgent:
    """Test BaseAgent functionality."""

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_init_without_tools(self, mock_llm_class):
        agent = BaseAgent(name="Test", system_prompt="You are a test agent.")
        assert agent.name == "Test"
        assert agent.tools == []
        assert agent.llm_with_tools == agent.llm

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_init_with_tools(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        mock_llm_class.return_value = mock_llm

        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        agent = BaseAgent(name="Test", system_prompt="Test", tools=[dummy_tool])
        assert len(agent.tools) == 1
        mock_llm.bind_tools.assert_called_once()

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_run_returns_tuple(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = BaseAgent(name="Test", system_prompt="You are a test agent.")
        response, history = agent.run("Hello")

        assert response == "Test response"
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_run_preserves_history(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = BaseAgent(name="Test", system_prompt="Test")
        existing_history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
        ]

        _, history = agent.run("New question", existing_history)
        assert len(history) == 4  # 2 existing + 2 new

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_run_handles_none_history(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = BaseAgent(name="Test", system_prompt="Test")
        response, history = agent.run("Hello", None)

        assert response == "Response"
        assert len(history) == 2

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_run_handles_list_content(self, mock_llm_class):
        """Test that list-type content from Gemini is handled."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "Hello from Gemini", "extras": {}}]
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = BaseAgent(name="Test", system_prompt="Test")
        response, _ = agent.run("Hello")

        assert response == "Hello from Gemini"

    @patch("src.agents.base_agent.ChatGoogleGenerativeAI")
    def test_run_with_tool_calls(self, mock_llm_class):
        """Test the tool execution loop."""
        mock_llm = MagicMock()

        # First call returns tool_calls
        first_response = MagicMock()
        first_response.content = ""
        first_response.tool_calls = [{"name": "dummy", "args": {"x": "test"}, "id": "123"}]

        # Second call returns final response
        second_response = MagicMock()
        second_response.content = "Final answer"
        second_response.tool_calls = []

        mock_llm.invoke.side_effect = [first_response, second_response]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_class.return_value = mock_llm

        from langchain_core.tools import tool

        @tool
        def dummy(x: str) -> str:
            """Dummy tool."""
            return f"Result: {x}"

        agent = BaseAgent(name="Test", system_prompt="Test", tools=[dummy])
        response, _ = agent.run("Use the tool")

        assert response == "Final answer"
        assert mock_llm.invoke.call_count == 2
