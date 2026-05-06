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
    agent_types: list       # Now a list
    response: str


agents = {
    "finance_qa": FinanceQAAgent(),
    "portfolio": PortfolioAgent(),
    "market": MarketAgent(),
    "goal_planning": GoalPlanningAgent(),
    "news": NewsAgent(),
    "tax": TaxEducationAgent(),
}
# Define execution order — agents that gather data run first
AGENT_PRIORITY = {
    "market": 1,
    "news": 2,
    "finance_qa": 3,
    "tax": 4,
    "portfolio": 5,
    "goal_planning": 6,
}

def router_node(state: AgentState) -> dict:
    agent_types = route_query(state["user_message"])
    return {"agent_types": agent_types}


def agent_node(state: AgentState) -> dict:
    agent_types = sorted(state["agent_types"], key=lambda a: AGENT_PRIORITY.get(a, 99))
    responses = []
    history = state["chat_history"]

    # Build context from previous agents' responses
    accumulated_context = ""

    for agent_type in agent_types:
        agent = agents[agent_type]

        # If we have context from previous agents, append it to the query
        if accumulated_context:
            enriched_message = (
                f"{state['user_message']}\n\n"
                f"Context from previous analysis:\n{accumulated_context}"
            )
        else:
            enriched_message = state["user_message"]

        response, history = agent.run(enriched_message, history)
        responses.append(f"**[{agent.name}]**\n{response}")
        accumulated_context += f"\n{agent.name}: {response}\n"

    combined = "\n\n---\n\n".join(responses)
    return {"response": combined, "chat_history": history}


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
        "agent_types": [],
        "response": "",
    })

    # Return comma-joined agent types for display
    agent_label = ", ".join(result["agent_types"])
    return result["response"], agent_label, result["chat_history"]