import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env file
load_dotenv()

# Create the LLM instance
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# Send a message and get a response
response = llm.invoke("What is compound interest? Explain in 2 sentences.")

print(response.content)