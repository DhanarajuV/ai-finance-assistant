from src.agents.finance_qa_agent import FinanceQAAgent

agent = FinanceQAAgent()

# Test 1: Basic concept
print("=== Test 1: Basic concept ===")
print(agent.run("What's the difference between a stock and a bond?"))
print()

# Test 2: Beginner question
print("=== Test 2: Beginner question ===")
print(agent.run("I have $1000 and I'm scared to invest. What should I know?"))