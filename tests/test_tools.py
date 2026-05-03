import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Step 1: Define a tool — it's just a Python function with a docstring
@tool
def calculate_simple_interest(principal: float, rate: float, years: int) -> str:
    """Calculate simple interest on an investment.

    Args:
        principal: The initial investment amount in dollars.
        rate: Annual interest rate as a percentage (e.g., 5 for 5%).
        years: Number of years to invest.
    """
    interest = principal * (rate / 100) * years
    total = principal + interest
    return f"Principal: ${principal:,.2f}, Rate: {rate}%, Years: {years}\nInterest earned: ${interest:,.2f}\nTotal value: ${total:,.2f}"


# Step 2: Bind the tool to the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools([calculate_simple_interest])

# Step 3: Ask something that requires the tool
messages = [
    SystemMessage(content="You are a financial calculator assistant."),
    HumanMessage(content="If I invest $10,000 at 5% for 3 years, how much interest will I earn?"),
]

response = llm_with_tools.invoke(messages)

# Step 4: See what the LLM decided to do
print("Content:", response.content)
print("---")
print("Tool calls:", response.tool_calls)