from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import config
from dotenv import load_dotenv

load_dotenv()

AGENT_TYPES = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]

ROUTER_PROMPT = """You are a query router for a financial education assistant.
Your ONLY job is to classify the user's message into one of these categories:

- finance_qa: General financial education questions (what is a stock, how do bonds work, explain diversification, what is a Roth IRA)
- portfolio: User wants to analyze their portfolio, discuss their holdings, or get allocation advice
- market: User asks about specific stock prices, market trends, or how a particular company is doing
- goal_planning: User wants to plan financial goals, project savings, discuss retirement planning, or calculate investment growth
- news: User asks about recent news, headlines, or current events about a stock or the market
- tax: User asks specifically about taxes on investments, capital gains, tax-loss harvesting, tax brackets, or tax-advantaged account strategies

Respond with ONLY the category name, nothing else."""


def route_query(user_message: str) -> str:
    """Classify a user message and return the appropriate agent type."""
    llm = ChatGoogleGenerativeAI(
        model=config["llm"]["model"],
        temperature=0,
    )

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_message),
    ])

    agent_type = response.content.strip().lower()

    if agent_type not in AGENT_TYPES:
        return "finance_qa"

    return agent_type


if __name__ == "__main__":
    tests = [
        "What is an ETF?",
        "My portfolio is 60% AAPL and 40% GOOGL",
        "How is Tesla stock doing today?",
        "I want to retire in 20 years with $1 million",
        "What's the latest news about Apple?",
        "How are capital gains taxed?",
        "What is a Roth IRA?",
    ]

    for q in tests:
        result = route_query(q)
        print(f"{result:20s} ← {q}")