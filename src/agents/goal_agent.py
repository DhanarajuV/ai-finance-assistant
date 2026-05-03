from langchain_core.tools import tool
from src.agents.base_agent import BaseAgent
from src.core.config import config


@tool
def project_investment(monthly_amount: float, years: int, annual_return: float = 7.0) -> str:
    """Project future value of regular monthly investments.

    Args:
        monthly_amount: Amount invested per month in dollars.
        years: Number of years to invest.
        annual_return: Expected annual return percentage (default 7% for stock market average).
    """
    monthly_rate = annual_return / 100 / 12
    months = years * 12
    total_contributed = monthly_amount * months

    # Future value of annuity formula
    if monthly_rate > 0:
        future_value = monthly_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    else:
        future_value = total_contributed

    earnings = future_value - total_contributed

    result = f"💰 Investment Projection\n\n"
    result += f"Monthly investment: ${monthly_amount:,.2f}\n"
    result += f"Time horizon: {years} years\n"
    result += f"Assumed annual return: {annual_return}%\n\n"
    result += f"Total contributed: ${total_contributed:,.2f}\n"
    result += f"Investment earnings: ${earnings:,.2f}\n"
    result += f"Projected value: ${future_value:,.2f}\n\n"

    # Show milestones
    result += "Milestones:\n"
    for milestone in [5, 10, 20, 30]:
        if milestone <= years:
            m = milestone * 12
            fv = monthly_amount * (((1 + monthly_rate) ** m - 1) / monthly_rate)
            result += f"  Year {milestone}: ${fv:,.2f}\n"

    return result


class GoalPlanningAgent(BaseAgent):
    """Helps users plan financial goals."""

    def __init__(self):
        system_prompt = f"""You are {config['app']['name']}, a financial goal planning assistant.

Your role:
- Help users set and plan financial goals (retirement, house, education, emergency fund)
- Use the project_investment tool to show growth projections
- Explain the power of compound interest and starting early
- Suggest realistic savings amounts based on goals

Rules:
- Never guarantee returns — always say "assuming X% average return"
- {config['app']['disclaimer']}
- Be encouraging but realistic
- Ask clarifying questions: timeline, current savings, risk tolerance
- For retirement, use 7% average return. For conservative goals, use 5%."""

        super().__init__(
            name="Goal Planning Agent",
            system_prompt=system_prompt,
            tools=[project_investment],
        )