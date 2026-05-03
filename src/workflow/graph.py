from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.workflow.router import route_query
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.market_agent import MarketAgent
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.news_agent import NewsAgent
from src.agents.tax_agent import TaxEducationAgent


class AgentState(TypedDict):
    user_message: str
    chat_history: list
    agent_type: str
    response: str


agents = {
    "finance_qa": FinanceQAAgent(),
    "portfolio": PortfolioAgent(),
    "market": MarketAgent(),
    "goal_planning": GoalPlanningAgent(),
    "news": NewsAgent(),
    "tax": TaxEducationAgent(),
}


def router_node(state: AgentState) -> dict:
    agent_type = route_query(state["user_message"])
    return {"agent_type": agent_type}


def agent_node(state: AgentState) -> dict:
    agent = agents[state["agent_type"]]
    response, updated_history = agent.run(
        state["user_message"],
        state["chat_history"],
    )
    return {"response": response, "chat_history": updated_history}


workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("router")
workflow.add_edge("router", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()


def chat(user_message: str, chat_history: list = None) -> tuple[str, str, list]:
    if chat_history is None:
        chat_history = []

    result = app.invoke({
        "user_message": user_message,
        "chat_history": chat_history,
        "agent_type": "",
        "response": "",
    })

    return result["response"], result["agent_type"], result["chat_history"]