import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# A conversation is just a list of messages
messages = [
    SystemMessage(content="You are Finnie, a friendly financial education assistant. "
                          "Explain concepts simply for beginners. "
                          "Always add a disclaimer that this is educational, not financial advice."),
    HumanMessage(content="What is an ETF and why would I buy one?"),
]

response = llm.invoke(messages)

print("Type:", type(response))
print("---")
print(response.content)