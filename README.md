# 💰 Finnie — AI Personal Finance Assistant

Finnie is a multi-agent AI-powered personal finance education assistant that helps beginners learn about investing, analyze portfolios, track markets, plan financial goals, and understand taxes — all through natural conversation.

Built with LangGraph, Google Gemini, FAISS, and Streamlit.

**Live App**: [https://ai-finance-assistant-knyh3c5fzws5eupkrgdnxc.streamlit.app/](https://ai-finance-assistant-knyh3c5fzws5eupkrgdnxc.streamlit.app/)

> ⚠️ **Disclaimer**: This is for educational purposes only, not financial advice. Always consult a qualified financial advisor for investment decisions.

---

## Architecture Overview

Finnie uses a multi-agent architecture orchestrated by LangGraph. A central LLM-based router classifies each user query and dispatches it to the appropriate specialized agent.

```
User Query
    │
    ▼
┌──────────────────┐
│   LLM Router     │  Classifies query (temperature=0)
│  (Gemini Flash)  │
└────────┬─────────┘
         │
    ┌────┴────┬──────────┬──────────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Finance ││Portfo- ││Market  ││Goal    ││News    ││Tax     │
│Q&A     ││lio     ││Analysis││Planning││Synth.  ││Educatn │
│Agent   ││Agent   ││Agent   ││Agent   ││Agent   ││Agent   │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │         │
  RAG/     Portfolio  yFinance  Investment yFinance   RAG/
  FAISS    Calculator   API     Projector  News API   FAISS
```

### Agents

| Agent | What It Does | Tools |
|-------|-------------|-------|
| **Finance Q&A** | Explains financial concepts using a curated knowledge base | RAG search (FAISS) |
| **Portfolio Analysis** | Analyzes holdings for allocation, diversification, and risk | Portfolio calculator |
| **Market Analysis** | Fetches real-time stock prices and explains metrics | yFinance API |
| **Goal Planning** | Projects investment growth with compound interest | Investment projector |
| **News Synthesizer** | Fetches and summarizes recent financial news | yFinance news API |
| **Tax Education** | Explains investment-related tax concepts | RAG search (FAISS) |

### Key Design Decisions

- **LLM-based routing over keyword matching**: More accurate classification of natural language queries. Cost is negligible (~20 tokens per routing call).
- **FAISS over cloud vector DBs**: Runs locally, zero cost, no external dependency. Sufficient for 50 articles.
- **Shared BaseAgent class**: All agents inherit common LLM setup, tool execution loop, and memory management. Adding a new agent requires only a system prompt and tools.
- **Hybrid UI**: Chat tab demonstrates agentic routing; dashboard tabs showcase visualizations.

For detailed architecture documentation, see [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangGraph (state graph) |
| Agent Framework | LangChain |
| Embeddings | Gemini Embedding 001 |
| Vector Store | FAISS (local) |
| Market Data | yFinance |
| UI | Streamlit |
| Charts | Plotly |
| Deployment | Streamlit Community Cloud |

---

## Setup Instructions

### Prerequisites

- Python 3.13+
- Google AI Studio API key ([get one free](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/DhanarajuV/ai-finance-assistant.git
cd ai-finance-assistant

# Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

2. Build the RAG knowledge base index:

```bash
python src/rag/knowledge_base.py
```

You should see:
```
Loaded 50 articles
Split into ~200 chunks
Vector store saved to src/data/faiss_index
```

### Run the Application

```bash
streamlit run src/web_app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage Examples

### Chat Tab (Primary Interface)

The chat tab is the main interface. Type naturally — the router automatically dispatches to the right agent.

| What You Type | Agent Triggered | What Happens |
|--------------|----------------|-------------|
| "What is an ETF?" | Finance Q&A | Searches knowledge base, explains with citations |
| "Analyze my portfolio: AAPL 40%, GOOGL 30%, BND 30%" | Portfolio | Calculates allocation, diversification score, risk level + pie chart |
| "How is Apple stock doing?" | Market | Fetches real-time price, 52-week range, P/E ratio + price chart |
| "I want to save $500/month for 20 years" | Goal Planning | Projects future value with compound interest milestones |
| "Latest news about Tesla?" | News | Fetches top 5 recent headlines with sources |
| "How are capital gains taxed?" | Tax Education | Searches tax articles in knowledge base, explains with disclaimer |
| "Tell me more about that" | (same as previous) | Multi-turn memory — Finnie remembers conversation context |

### Portfolio Tab

Enter holdings in the text area (e.g., `AAPL 40%, GOOGL 30%, BND 20%, VTI 10%`) and click **Analyze Portfolio**. You'll see:
- Interactive pie chart of allocation
- AI-generated analysis with diversification score and risk assessment

### Market Tab

Enter a stock ticker (e.g., `AAPL`) and select a time period. Click **Look Up** to see:
- Interactive price history chart
- AI-generated analysis with key metrics explained

---

## API Documentation

### Core Function

```python
from src.workflow.graph import chat

response, agent_type, updated_history = chat(
    user_message="What is an ETF?",
    chat_history=[]  # Pass previous history for multi-turn conversations
)
# response: str — The agent's answer
# agent_type: str — Which agent handled it ("finance_qa", "portfolio", etc.)
# updated_history: list — Updated conversation history to pass back next turn
```

### Router

```python
from src.workflow.router import route_query

agent_type = route_query("How is AAPL doing?")
# Returns: "market"
```

### Individual Agents

```python
from src.agents.finance_qa_agent import FinanceQAAgent

agent = FinanceQAAgent()
response, history = agent.run("What is compound interest?", chat_history=[])
```

All agents follow the same interface:
```python
agent.run(user_message: str, chat_history: list = None) -> tuple[str, list]
```

### RAG Search

```python
from src.rag.knowledge_base import load_vector_store

vs = load_vector_store()
results = vs.similarity_search("retirement accounts", k=3)
for doc in results:
    print(doc.metadata["title"], doc.page_content[:100])
```

### Tools (Standalone)

```python
from src.agents.market_agent import get_stock_price
from src.agents.portfolio_agent import analyze_portfolio
from src.agents.goal_agent import project_investment
from src.agents.news_agent import get_financial_news

# Each tool can be invoked directly
result = get_stock_price.invoke({"symbol": "AAPL"})
result = analyze_portfolio.invoke({"holdings": "AAPL 40%, BND 60%"})
result = project_investment.invoke({"monthly_amount": 500, "years": 20, "annual_return": 7.0})
result = get_financial_news.invoke({"topic": "TSLA"})
```

---

## Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Base class — LLM setup, tool loop, memory
│   │   ├── finance_qa_agent.py    # Financial education Q&A (RAG)
│   │   ├── portfolio_agent.py     # Portfolio analysis + calculator tool
│   │   ├── market_agent.py        # Real-time stock data (yFinance)
│   │   ├── goal_agent.py          # Investment growth projections
│   │   ├── news_agent.py          # Financial news summaries (yFinance)
│   │   └── tax_agent.py           # Tax education (RAG)
│   ├── core/
│   │   └── config.py              # Centralized YAML config loader
│   ├── data/
│   │   ├── knowledge_base/        # 50 financial education articles (.md)
│   │   └── faiss_index/           # Persisted FAISS vector store
│   ├── rag/
│   │   ├── knowledge_base.py      # Article loading, chunking, embedding, indexing
│   │   └── retriever.py           # RAG search tool for agents
│   ├── web_app/
│   │   ├── app.py                 # Streamlit UI — chat + portfolio + market tabs
│   │   └── charts.py              # Plotly chart generators (pie, line)
│   ├── utils/                     # Shared utilities
│   └── workflow/
│       ├── router.py              # LLM-based query classifier
│       └── graph.py               # LangGraph state graph orchestration
├── tests/                         # Test files
├── config.yaml                    # Application configuration
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup for imports
├── Dockerfile                     # Docker configuration
├── .env                           # API keys (gitignored)
├── .gitignore
├── README.md                      # This file
└── TECHNICAL_DESIGN.md            # Detailed architecture document
```

---

## Configuration

All tunable parameters are in `config.yaml`:

```yaml
llm:
  model: "gemini-2.5-flash"       # LLM model for all agents
  temperature: 0.3                 # Lower = more factual, higher = more creative

embedding:
  model: "models/gemini-embedding-001"  # Embedding model for RAG

rag:
  chunk_size: 500                  # Characters per document chunk
  chunk_overlap: 50                # Overlap between chunks
  top_k: 3                        # Number of chunks retrieved per query

app:
  name: "Finnie"
  disclaimer: "This is for educational purposes only, not financial advice."
```

To switch LLM providers (e.g., to OpenAI), change the model in `config.yaml` and update the import in `base_agent.py` from `ChatGoogleGenerativeAI` to `ChatOpenAI`.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

The project needs to be installed as a package. Run:
```bash
pip install -e .
```
Or if using a venv, ensure the `.pth` file exists:
```bash
echo "$(pwd)" > venv/lib/python3.13/site-packages/finnie.pth
```

### "GOOGLE_API_KEY not found" or authentication errors

1. Verify `.env` file exists in the project root with `GOOGLE_API_KEY=your_key`
2. Verify the key is valid at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
3. On Streamlit Cloud: check Secrets in app settings

### "Model not found" (404 errors)

Google periodically deprecates model names. Check available models:
```bash
python -c "
from dotenv import load_dotenv; import os; load_dotenv()
from google import genai
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
for m in client.models.list():
    print(m.name)
"
```
Update `config.yaml` with a valid model name.

### "Rate limit exceeded" (429 errors)

Gemini free tier: 15 requests/minute, 1M tokens/day. If exceeded:
- Wait 30-60 seconds and retry
- Enable billing at Google AI Studio (free tier still applies, billing is a fallback)

### FAISS index errors or stale results

Rebuild the index after adding/modifying knowledge base articles:
```bash
rm -rf src/data/faiss_index
python src/rag/knowledge_base.py
```

### yFinance returns no data

- Verify the ticker symbol is valid (e.g., `AAPL` not `APPLE`)
- yFinance may be temporarily unavailable — the app will show an error message and continue
- Some tickers (OTC, international) may have limited data

### Streamlit app won't start

```bash
# Verify you're in the right directory with venv activated
cd ~/Documents/playground/Capstone/ai_finance_assistant
source venv/bin/activate

# Verify dependencies
pip install -r requirements.txt

# Run with verbose output
streamlit run src/web_app/app.py --logger.level=debug
```

### Charts not appearing

- Portfolio pie chart requires the agent response to contain bullet-formatted holdings (e.g., `• AAPL: 40.0%`)
- Market chart requires a valid ticker symbol that yFinance recognizes
- Clear chat history (sidebar button) and retry if state is corrupted

---

## Testing

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run with coverage report

```bash
python -m pytest tests/ --cov=src --cov-report=term
```

This prints coverage percentage per module and a total at the bottom.

### Test summary

- **89 tests** covering tools, agents, router, workflow, config, charts, and knowledge base
- Core logic coverage: 80%+
- Tests use mocks for LLM calls (no API key needed to run tests)

---

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Create new app → select repo → set main file to `src/web_app/app.py`
4. Add `GOOGLE_API_KEY` in Advanced Settings → Secrets
5. Deploy

The app auto-redeploys on every `git push`.

### Docker

```bash
docker build -t finnie .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key_here finnie
```

Open [http://localhost:8501](http://localhost:8501).

### Local

```bash
source venv/bin/activate
streamlit run src/web_app/app.py
```

---

## Adding New Knowledge Base Articles

1. Create a markdown file in `src/data/knowledge_base/`:

```markdown
---
title: Your Article Title
category: basics
---

# Your Article Title

Content here...
```

Valid categories: `basics`, `strategy`, `planning`, `tax`, `analysis`, `alternatives`, `advanced`, `reference`

2. Rebuild the FAISS index:
```bash
python src/rag/knowledge_base.py
```

3. Restart the app. The RAG agents will now search the new content.

---

## Adding a New Agent

1. Create `src/agents/your_agent.py` inheriting from `BaseAgent`:

```python
from src.agents.base_agent import BaseAgent
from src.core.config import config

class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Your Agent",
            system_prompt="Your system prompt here...",
            tools=[your_tool],  # optional
        )
```

2. Add the agent type to `AGENT_TYPES` and `ROUTER_PROMPT` in `src/workflow/router.py`
3. Import and register the agent in `src/workflow/graph.py`'s `agents` dict
4. Restart the app

---

## License

This project is for educational purposes as part of the Applied Agentic AI for SWEs capstone program.
