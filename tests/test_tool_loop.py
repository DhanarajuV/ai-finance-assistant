import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()


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


# Map tool names to functions — you'll need this when you have many tools
tools = {"calculate_simple_interest": calculate_simple_interest}

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools([calculate_simple_interest])

# Step 1: User asks a question
messages = [
    SystemMessage(content="You are Finnie, a friendly financial education assistant."),
    HumanMessage(content="If I invest $10,000 at 5% for 3 years, how much simple interest do I earn?"),
]

# Step 2: LLM responds (may request tool calls)
response = llm_with_tools.invoke(messages)
print("Step 2 - LLM wants to call:", [tc["name"] for tc in response.tool_calls])

# Step 3: Add the LLM's response to conversation history
messages.append(response)

# Step 4: Execute each tool call and add results to conversation
for tool_call in response.tool_calls:
    # Look up the function and run it
    func = tools[tool_call["name"]]
    result = func.invoke(tool_call["args"])

    # Add the result as a ToolMessage (linked by tool_call_id)
    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    print(f"Step 4 - Tool result: {result}")

# Step 5: Send everything back to LLM for final answer
final_response = llm_with_tools.invoke(messages)
print("\nStep 5 - Finnie's answer:")
print(final_response.content)