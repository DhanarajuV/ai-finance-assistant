from src.agents.finance_qa_agent import FinanceQAAgent

agent = FinanceQAAgent()
history = []

# Turn 1
print("User: What's an ETF?")
response, history = agent.run("What's an ETF?", history)
print(f"Finnie: {response}\n")

# Turn 2 — references "it" from turn 1
print("User: How is it different from a mutual fund?")
response, history = agent.run("How is it different from a mutual fund?", history)
print(f"Finnie: {response}\n")

# Turn 3 — references both previous turns
print("User: Which one is better for a beginner like me?")
response, history = agent.run("Which one is better for a beginner like me?", history)
print(f"Finnie: {response}\n")

# Show what history looks like
print(f"--- History has {len(history)} messages ---")
for msg in history:
    print(f"  {msg.__class__.__name__}: {msg.content[:60]}...")