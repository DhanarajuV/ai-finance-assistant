# Finnie — AI Personal Finance Assistant: Technical Design Document

## 1. Executive Summary

Finnie is a multi-agent AI-powered personal finance education assistant that democratizes financial literacy through intelligent conversational AI. It uses a LangGraph-orchestrated workflow of 6 specialized agents, a RAG-enhanced knowledge base of 50 financial education articles, real-time market data integration, and a Streamlit-based hybrid UI combining chat and dashboard interfaces.

**Live Application**: https://ai-finance-assistant-knyh3c5fzws5eupkrgdnxc.streamlit.app/  
**GitHub Repository**: https://github.com/DhanarajuV/ai-finance-assistant

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Chat Tab │  │ Portfolio Tab│  │  Market Tab   │     │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘     │
│       └────────────────┼──────────────────┘             │
│                        ▼                                │
│              ┌──────────────────┐                       │
│              │  chat() function │                       │
│              └────────┬─────────┘                       │
└───────────────────────┼─────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  LangGraph Workflow                        │
│                                                           │
│   START ──▶ ┌──────────────┐                              │
│             │ Router Node  │ (LLM-based classification)   │
│             └──────┬───────┘                              │
│                    ▼                                      │
│             ┌──────────────┐                              │
│             │  Agent Node  │ (dispatches to selected agent)│
│             └──────┬───────┘                              │
│                    ▼                                      │
│                   END                                     │
└───────────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────────────────┐
          ▼             ▼             ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Finance QA │ │Portfolio │ │  Market  │ │   Goal   │
   │   Agent    │ │  Agent   │ │  Agent   │ │ Planning │
   └─────┬──────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │              │            │             │
   ┌─────┴──────┐      │      ┌─────┴─────┐ ┌────┴──────┐
   │ RAG Search │      │      │ yFinance  │ │Investment │
   │   (FAISS)  │      │      │   API     │ │ Projector │
   └────────────┘      │      └───────────┘ └───────────┘
                 ┌─────┴──────┐
                 │ Portfolio  │
                 │ Calculator │
                 └────────────┘

   ┌────────────┐ ┌──────────┐
   │   News     │ │   Tax    │
   │   Agent    │ │  Agent   │
   └─────┬──────┘ └────┬─────┘
   ┌─────┴──────┐ ┌────┴──────┐
   │ yFinance   │ │ RAG Search│
   │   News     │ │  (FAISS)  │
   └────────────┘ └───────────┘
```

### 2.2 Data Flow

```
User Input (natural language)
    │
    ▼
Router Node
    │  LLM classifies query into one of 6 categories
    │  (temperature=0 for deterministic routing)
    ▼
Agent Node
    │  Selected agent receives user message + chat history
    │  Agent builds message list: SystemMessage + history + HumanMessage
    │  LLM may request tool calls
    │  Tools execute and return results
    │  LLM generates final response
    ▼
Response returned to UI
    │  UI renders text + optional charts
    │  Chat history updated for multi-turn memory
    ▼
User sees response
```

### 2.3 State Management

LangGraph manages a typed state dictionary that flows through the graph:

```python
class AgentState(TypedDict):
    user_message: str      # Current user input
    chat_history: list     # List of HumanMessage/AIMessage objects
    agent_type: str        # Router's classification result
    response: str          # Agent's final response text
```

The router node writes `agent_type`. The agent node writes `response` and updates `chat_history`. The UI reads all fields.

---

## 3. Agent Architecture

### 3.1 Base Agent Pattern

All agents inherit from `BaseAgent`, which provides:
- LLM initialization from centralized config
- Tool binding via LangChain's `bind_tools()`
- The complete tool execution loop (invoke LLM → execute tool calls → re-invoke LLM)
- Conversation history management (appends HumanMessage + AIMessage per turn)

Each specialized agent defines only two things:
1. **System prompt** — personality, rules, and domain expertise
2. **Tools** — Python functions decorated with `@tool`

### 3.2 Agent Details

| Agent | System Prompt Focus | Tools | Use Case |
|-------|-------------------|-------|----------|
| **Finance Q&A** | Beginner-friendly explanations, no jargon, cite sources | `search_knowledge_base` (RAG) | "What is an ETF?", "Explain diversification" |
| **Portfolio Analysis** | Analyze allocation, risk, diversification | `analyze_portfolio` (calculator) | "Analyze my portfolio: AAPL 40%, BND 60%" |
| **Market Analysis** | Real-time data, explain metrics for beginners | `get_stock_price` (yFinance) | "How is Apple stock doing?" |
| **Goal Planning** | Retirement/savings projections, compound growth | `project_investment` (calculator) | "Save $500/month for 20 years" |
| **News Synthesizer** | Summarize headlines, explain market impact | `get_financial_news` (yFinance) | "Latest news about Tesla?" |
| **Tax Education** | Tax concepts, recommend consulting professionals | `search_knowledge_base` (RAG) | "How are capital gains taxed?" |

### 3.3 Router Design

The router is an LLM call with `temperature=0` (deterministic) and a system prompt that maps query types to agent categories. It returns a single category string.

**Classification categories**: `finance_qa`, `portfolio`, `market`, `goal_planning`, `news`, `tax`

**Fallback**: If the LLM returns an unrecognized category, the router defaults to `finance_qa` — the safest general-purpose agent.

**Design tradeoff**: Some overlap exists between agents (e.g., "What is a Roth IRA?" could go to `finance_qa` or `tax`). This is acceptable because both agents can handle such queries — the difference is the system prompt emphasis. Perfect routing is not the goal; graceful handling is.

---

## 4. RAG Implementation

### 4.1 Knowledge Base

- **50 markdown articles** covering: basics, strategy, planning, tax, analysis, alternatives, advanced topics, and reference glossaries
- Each article has YAML frontmatter with `title` and `category` metadata
- Articles are original educational content, beginner-friendly, 300-600 words each

**Categories covered**:
- Basics (16): Stocks, bonds, ETFs, mutual funds, index funds, compound interest, S&P 500, dividends, inflation, expense ratios, order types, brokerage accounts, investment fees, bond advanced, bear/bull markets, stock market history
- Strategy (8): Diversification, dollar-cost averaging, risk tolerance, asset allocation, behavioral finance, value vs growth, international investing, passive vs active, sector investing, investment mistakes
- Planning (8): Emergency fund, retirement planning, college savings, estate planning, budgeting, net worth, social security, financial advisor
- Tax (4): Tax basics, tax-advantaged accounts, tax filing for investments, tax-efficient investing
- Analysis (3): P/E ratio, market cap, financial statements
- Alternatives (2): Real estate, cryptocurrency
- Advanced (2): Options basics, margin trading
- Reference (2): Financial glossary A-M, glossary N-Z

### 4.2 Embedding & Indexing Pipeline

```
Markdown files
    │
    ▼  load_articles()
Document objects (text + metadata)
    │
    ▼  RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
Chunks (~200+ chunks from 50 articles)
    │
    ▼  GoogleGenerativeAIEmbeddings (gemini-embedding-001)
Vector embeddings
    │
    ▼  FAISS.from_documents()
FAISS index (saved to disk)
```

### 4.3 Retrieval

The `search_knowledge_base` tool:
1. Receives a query string from the LLM
2. Embeds the query using the same embedding model
3. Performs similarity search in FAISS (`top_k=3`)
4. Returns formatted results with source attribution: `[Source 1: Article Title]\n content...`
5. The LLM uses these sources to generate a grounded, cited response

### 4.4 Design Decisions

- **Chunk size 500 with 50 overlap**: Balances precision (smaller chunks match more specifically) with context (enough text for the LLM to understand)
- **top_k=3**: Returns 3 most relevant chunks. Enough context without overwhelming the prompt.
- **FAISS over Pinecone/ChromaDB**: Runs locally, no external service, zero cost, fast enough for this scale
- **Persistent index**: Saved to disk so the app doesn't rebuild on every restart

---

## 5. Tool Design

### 5.1 Tool Pattern

Tools are Python functions with the `@tool` decorator. The docstring serves as a contract with the LLM — it tells the LLM when and how to use the tool.

```
User says natural language
    → LLM reads tool docstring
    → LLM decides to call tool with structured arguments
    → Code executes the function
    → Result returned as ToolMessage
    → LLM generates final human-readable response
```

### 5.2 Tool Inventory

| Tool | Agent | Input | Output | External API |
|------|-------|-------|--------|-------------|
| `search_knowledge_base` | Finance QA, Tax | query string | Formatted article chunks with sources | None (local FAISS) |
| `analyze_portfolio` | Portfolio | holdings string (e.g., "AAPL 40%, BND 60%") | Allocation %, diversification score, risk level | None (calculation) |
| `get_stock_price` | Market | ticker symbol | Price, change %, 52-week range, market cap, P/E | yFinance API |
| `project_investment` | Goal Planning | monthly amount, years, return rate | Future value, milestones, total contributed vs earned | None (calculation) |
| `get_financial_news` | News | ticker symbol | Top 5 headlines with source and summary | yFinance API |

### 5.3 Error Handling

- All API-calling tools wrap external calls in try/except and return user-friendly error messages
- The portfolio parser validates input and returns guidance if parsing fails
- Market data tool falls back to historical close price if real-time data is unavailable

---

## 6. User Interface

### 6.1 Hybrid Design

The UI combines a **chat-first** interface with **dashboard tabs**:

| Tab | Purpose | Key Features |
|-----|---------|-------------|
| **💬 Chat** | Primary interface — natural language conversation | Multi-turn memory, agent type display, inline charts |
| **📊 Portfolio** | Dedicated portfolio analysis | Text input for holdings, pie chart visualization, AI analysis |
| **📈 Market** | Stock lookup dashboard | Ticker input, time period selector, price history chart, AI analysis |

### 6.2 Chart Integration

- **Portfolio pie chart**: Parsed from agent response using regex (`[•\-\*]\s*(\w+):\s*([\d.]+)%`). Falls back to parsing user input.
- **Stock price chart**: Plotly line chart from yFinance historical data. Configurable time period (1mo to 5y).
- **Chart persistence**: Chart source data stored in `session_state` so charts survive Streamlit reruns.

### 6.3 Session Management

Streamlit reruns the entire script on every interaction. Two separate histories maintain state:
- `chat_history`: List of LangChain `HumanMessage`/`AIMessage` objects — sent to the LLM for context
- `display_messages`: List of dicts with role, content, agent type, and chart data — used for UI rendering

---

## 7. Configuration Management

All tunable parameters are centralized in `config.yaml`:

```yaml
llm:
  model: "gemini-2.5-flash"
  temperature: 0.3

embedding:
  model: "models/gemini-embedding-001"

rag:
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3

app:
  name: "Finnie"
  description: "Your AI-powered personal finance education assistant"
  disclaimer: "This is for educational purposes only, not financial advice."
```

**Secrets** (API keys) are stored separately in `.env` (local) or Streamlit Cloud secrets (production). Never in config or code.

---

## 8. Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Google Gemini 2.5 Flash | Free tier, fast, good quality for conversational + RAG |
| Orchestration | LangGraph | State graph model, explicit control over agent routing |
| Agent Framework | LangChain | Consistent LLM interface, tool binding, message types |
| Embeddings | Gemini Embedding 001 | Same provider as LLM, no additional API key |
| Vector Store | FAISS | Local, fast, zero cost, sufficient for 50 articles |
| Market Data | yFinance | Free, no API key required, real-time stock data + news |
| UI | Streamlit | Rapid prototyping, built-in chat components, free cloud hosting |
| Charts | Plotly | Interactive charts, good Streamlit integration |
| Config | PyYAML + python-dotenv | Clean separation of settings and secrets |

---

## 9. Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Base class with LLM + tool loop
│   │   ├── finance_qa_agent.py    # General financial education (RAG)
│   │   ├── portfolio_agent.py     # Portfolio analysis + calculator tool
│   │   ├── market_agent.py        # Real-time market data (yFinance)
│   │   ├── goal_agent.py          # Investment projection calculator
│   │   ├── news_agent.py          # Financial news (yFinance)
│   │   └── tax_agent.py           # Tax education (RAG)
│   ├── core/
│   │   └── config.py              # YAML config loader
│   ├── data/
│   │   ├── knowledge_base/        # 50 markdown articles
│   │   └── faiss_index/           # Persisted FAISS vector store
│   ├── rag/
│   │   ├── knowledge_base.py      # Article loading, chunking, indexing
│   │   └── retriever.py           # Search tool for agents
│   ├── web_app/
│   │   ├── app.py                 # Streamlit UI (chat + tabs)
│   │   └── charts.py              # Plotly chart generators
│   └── workflow/
│       ├── router.py              # LLM-based query classifier
│       └── graph.py               # LangGraph state graph
├── tests/
├── config.yaml
├── requirements.txt
├── setup.py
├── .env                           # API keys (gitignored)
├── .gitignore
└── README.md
```

---

## 10. Deployment

### 10.1 Architecture

```
Developer laptop
    │  git push
    ▼
GitHub (public repo)
    │  auto-detect by Streamlit Cloud
    ▼
Streamlit Community Cloud
    │  pip install -r requirements.txt
    │  streamlit run src/web_app/app.py
    ▼
https://ai-finance-assistant-knyh3c5fzws5eupkrgdnxc.streamlit.app/
```

### 10.2 Secrets Management

- **Local**: `.env` file with `GOOGLE_API_KEY` (gitignored)
- **Production**: Streamlit Cloud Secrets dashboard (encrypted, injected as environment variables)

### 10.3 Package Resolution

The project uses `setup.py` + `-e .` in `requirements.txt` to make `src.*` imports work on Streamlit Cloud (equivalent to `pip install -e .`).

---

## 11. Performance Considerations

- **Router cost**: Each query makes 2 LLM calls (1 for routing, 1 for the agent). The router call is cheap (~20 tokens response).
- **RAG embedding**: Query embedding happens on every RAG search. Cached FAISS index avoids re-embedding articles.
- **yFinance**: No rate limits, but network latency adds ~1-2s per stock lookup. No caching implemented (acceptable for demo scale).
- **Chat history growth**: All messages sent to LLM on every turn. For long conversations, token usage grows linearly. Acceptable for demo; production would need history trimming or summarization.

---

## 12. Regulatory & Ethical Considerations

- Every agent's system prompt includes the disclaimer: *"This is for educational purposes only, not financial advice."*
- Agents are instructed to never recommend specific stocks to buy or sell
- The Tax Education Agent explicitly recommends consulting a tax professional
- Source attribution is built into RAG responses for transparency
- The sidebar displays the educational disclaimer prominently

---

## 13. Demo Plan (5-10 Minutes)

### Opening (30 seconds)
- Show the live app URL
- Briefly explain: "Finnie is a multi-agent AI finance assistant with 6 specialized agents orchestrated by LangGraph"

### Chat Tab — Agent Routing Demo (4 minutes)
Show each agent being triggered by natural language:

1. **Finance Q&A Agent**: "What is an ETF and how is it different from a mutual fund?"
   - Point out: RAG retrieval, source citations, beginner-friendly language

2. **Market Analysis Agent**: "How is Apple stock doing today?"
   - Point out: Real-time data from yFinance, inline price chart, metric explanations

3. **Portfolio Analysis Agent**: "Analyze my portfolio with 20000 in AAPL, 10000 in BND, 5000 in GOOGL"
   - Point out: Inline pie chart, diversification score, risk assessment

4. **Goal Planning Agent**: "I'm 30 years old and want to save $500 per month for retirement"
   - Point out: Compound growth projection, milestones, realistic assumptions

5. **News Agent**: "What's the latest news about Tesla?"
   - Point out: Real headlines from yFinance, source attribution

6. **Tax Agent**: "How are capital gains taxed?"
   - Point out: RAG retrieval from tax articles, disclaimer about consulting professionals

### Multi-Turn Memory (1 minute)
- Follow up on a previous question: "Which one would you recommend for a beginner?" (referencing ETF vs mutual fund)
- Show that Finnie remembers context

### Portfolio Tab (1 minute)
- Enter holdings: "AAPL 40%, GOOGL 30%, BND 20%, VTI 10%"
- Show pie chart + AI analysis

### Market Tab (1 minute)
- Look up MSFT with 1-year time period
- Show price history chart + AI analysis

### Architecture Walkthrough (1 minute)
- Show the project structure briefly
- Explain: "User query → LLM router → specialized agent → tool execution → response"
- Mention: 50 articles in RAG knowledge base, FAISS vector store, LangGraph state graph

### Closing (30 seconds)
- Mention the tech stack: Gemini 2.5 Flash, LangGraph, FAISS, Streamlit, yFinance
- Show the GitHub repo
- "All educational, not financial advice"

---

## 14. Future Enhancements

- **Voice interface** for accessibility
- **History trimming/summarization** for long conversations
- **Caching layer** for market data (30-min TTL) to reduce API calls
- **User profiles** with saved portfolios and goals
- **Monte Carlo simulations** for goal planning risk analysis
- **MCP server** for Claude Desktop integration
- **Docker containerization** for portable deployment
- **Comprehensive test suite** with unit and integration tests
