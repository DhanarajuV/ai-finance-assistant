from src.agents.finance_qa_agent import FinanceQAAgent

agent = FinanceQAAgent()

print("=== Test 1: Should use knowledge base ===")
response, _ = agent.run("What's the difference between an ETF and a mutual fund?")
print(response)
print()

print("=== Test 2: Should cite sources ===")
response, _ = agent.run("How does a Roth IRA work?")
print(response)