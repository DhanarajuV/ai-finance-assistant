from langchain_core.tools import tool
from src.agents.base_agent import BaseAgent
from src.core.config import config


@tool
def analyze_portfolio(holdings: str) -> str:
    """Analyze a user's investment portfolio.

    Args:
        holdings: Portfolio holdings as "NAME PERCENTAGE" pairs separated by commas.
                  Example: "AAPL 40%, GOOGL 30%, BND 30%"
    """
    # Parse holdings — handle both percentage and dollar formats
    lines = [h.strip() for h in holdings.replace(",", "\n").split("\n") if h.strip()]
    parsed = []
    total = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        name = parts[0]
        # Try to extract number
        value = 0
        for p in parts[1:]:
            cleaned = p.replace("%", "").replace("$", "").replace(",", "")
            try:
                value = float(cleaned)
                break
            except ValueError:
                continue

        parsed.append({"name": name, "value": value})
        total += value

    if not parsed:
        return "Could not parse portfolio. Please provide holdings like: AAPL 40%, GOOGL 30%, BND 30%"

    # Calculate allocation percentages
    analysis = "📊 Portfolio Analysis\n\n"
    analysis += "Holdings:\n"

    stock_pct = 0
    bond_pct = 0
    bond_keywords = {"bnd", "agg", "bond", "bonds", "treasury", "fixed"}

    for h in parsed:
        pct = (h["value"] / total * 100) if total > 0 else 0
        analysis += f"  • {h['name']}: {pct:.1f}%\n"

        if h["name"].lower() in bond_keywords:
            bond_pct += pct
        else:
            stock_pct += pct

    # Diversification score (simple: more holdings = better, max 10)
    div_score = min(len(parsed) * 2, 10)

    # Risk assessment
    if stock_pct >= 80:
        risk = "Aggressive (high risk, high potential return)"
    elif stock_pct >= 50:
        risk = "Moderate (balanced risk and return)"
    else:
        risk = "Conservative (lower risk, steady returns)"

    analysis += f"\nDiversification Score: {div_score}/10"
    analysis += f"\nRisk Level: {risk}"
    analysis += f"\nStock/Equity Allocation: {stock_pct:.1f}%"
    analysis += f"\nBond/Fixed Income Allocation: {bond_pct:.1f}%"

    if len(parsed) < 5:
        analysis += "\n\n⚠️ Suggestion: Consider adding more holdings for better diversification."
    if bond_pct == 0:
        analysis += "\n⚠️ Suggestion: Consider adding bonds to reduce portfolio volatility."

    return analysis


class PortfolioAgent(BaseAgent):
    """Analyzes user investment portfolios."""

    def __init__(self):
        system_prompt = f"""You are {config['app']['name']}, a portfolio analysis assistant.

Your role:
- Analyze user portfolios for allocation, diversification, and risk
- Use the analyze_portfolio tool when users describe their holdings
- Explain the analysis results in simple terms
- Suggest improvements based on the user's risk tolerance

Rules:
- Never recommend specific stocks to buy or sell
- {config['app']['disclaimer']}
- If the user hasn't provided holdings, ask them to describe their portfolio
- Explain WHY diversification and allocation matter"""

        super().__init__(
            name="Portfolio Analysis Agent",
            system_prompt=system_prompt,
            tools=[analyze_portfolio],
        )