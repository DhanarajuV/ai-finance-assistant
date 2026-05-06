from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import config
from dotenv import load_dotenv

load_dotenv()

AGENT_TYPES = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]

ROUTER_PROMPT = """You are a query router for a financial education assistant.
Your job is to classify the user's message into one or more categories.

Categories:
- finance_qa: General financial education questions (what is a stock, how do bonds work, explain diversification)
- portfolio: User wants to analyze their portfolio, discuss holdings, or get allocation advice
- market: User asks about specific stock prices, market trends, or how a particular company is doing
- goal_planning: User wants to plan financial goals, project savings, or calculate investment growth
- news: User asks about recent news, headlines, or current events about a stock or the market
- tax: User asks specifically about taxes on investments, capital gains, or tax-advantaged accounts

Rules:
- If the query involves MULTIPLE categories, return ALL that apply separated by commas
- If only one category applies, return just that one
- Examples:
  - "What is an ETF?" → finance_qa
  - "How is AAPL doing and what's the latest news?" → market,news
  - "Analyze Apple and Microsoft and plan my 100k among them" → market,portfolio,goal_planning

Respond with ONLY the category name(s), nothing else."""


def route_query(user_message: str) -> list[str]:
    """Classify a user message. Returns list of agent types."""
    llm = ChatGoogleGenerativeAI(
        model=config["llm"]["model"],
        temperature=0,
    )

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = response.content.strip().lower()
    agent_types = [a.strip() for a in raw.split(",")]

    # Filter to valid types only
    valid = [a for a in agent_types if a in AGENT_TYPES]

    return valid if valid else ["finance_qa"]


if __name__ == "__main__":
    tests = [
        "What is an ETF?",
        "My portfolio is 60% AAPL and 40% GOOGL",
        "How is Tesla stock doing today?",
        "I want to retire in 20 years with $1 million",
        "What's the latest news about Apple?",
        "How are capital gains taxed?",
        "Analyze Apple and Microsoft and plan my 100k among them",
        "How is AAPL doing and what's the latest news?",
    ]

    for q in tests:
        result = route_query(q)
        print(f"{str(result):45s} ← {q}")