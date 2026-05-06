import re
import streamlit as st
from src.workflow.graph import chat
from src.web_app.charts import portfolio_pie_chart, stock_price_chart

st.set_page_config(page_title="Finnie - Finance Assistant", page_icon="💰", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

tab_chat, tab_portfolio, tab_market = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Market"])

# ---- CHAT TAB ----
with tab_chat:
    st.title("💰 Finnie")
    st.caption("Your AI-powered personal finance education assistant")

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("agent"):
                st.caption(f"Handled by: {msg['agent']}")
            # Re-render charts from stored data
            if msg.get("chart_data"):
                fig = portfolio_pie_chart(msg["chart_data"])
                if fig:
                    st.plotly_chart(fig, width="stretch")
            if msg.get("stock_symbol"):
                fig = stock_price_chart(msg["stock_symbol"])
                if fig:
                    st.plotly_chart(fig, width="stretch")

    if prompt := st.chat_input("Ask me anything about finance..."):
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, agent_type, st.session_state.chat_history = chat(
                    prompt, st.session_state.chat_history
                )
            st.markdown(response)
            st.caption(f"Handled by: {agent_type}")

            chart_data = None
            stock_symbol = None

            if agent_type == "portfolio":
                # Only chart if response contains bullet-point holdings (actual analysis)
                fig = None
                if response and isinstance(response, str) and "•" in response:
                    fig = portfolio_pie_chart(response)
                if fig:
                    st.plotly_chart(fig, width="stretch")
                    chart_data = response

            elif agent_type == "market":
                words = prompt.upper().split()
                for w in words:
                    w = w.strip(".,!?")
                    if w.isalpha() and 1 <= len(w) <= 5:
                        fig = stock_price_chart(w)
                        if fig:
                            st.plotly_chart(fig, width="stretch")
                            stock_symbol = w
                            break

        st.session_state.display_messages.append({
            "role": "assistant", "content": response, "agent": agent_type,
            "chart_data": chart_data, "stock_symbol": stock_symbol,
        })

# ---- PORTFOLIO TAB ----
with tab_portfolio:
    st.title("📊 Portfolio Analysis")

    holdings_input = st.text_area(
        "Describe your portfolio",
        placeholder="Example: AAPL 40%, GOOGL 30%, BND 20%, VTI 10%",
    )

    if st.button("Analyze Portfolio"):
        if holdings_input:
            with st.spinner("Analyzing..."):
                response, _, _ = chat(f"Analyze this portfolio: {holdings_input}", [])

            # Try chart from response first, then from input
            fig = portfolio_pie_chart(response)
            if not fig:
                fig = portfolio_pie_chart(holdings_input)
            if fig:
                st.plotly_chart(fig, width="stretch")

            st.markdown(response)
        else:
            st.warning("Please enter your holdings first.")

# ---- MARKET TAB ----
with tab_market:
    st.title("📈 Market Lookup")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter stock ticker", placeholder="e.g., AAPL, GOOGL, MSFT")
    with col2:
        period = st.selectbox("Time period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)

    if st.button("Look Up"):
        if symbol:
            fig = stock_price_chart(symbol, period)
            if fig:
                st.plotly_chart(fig, width="stretch")

            with st.spinner(f"Analyzing {symbol.upper()}..."):
                response, _, _ = chat(f"How is {symbol} stock doing today?", [])
            st.markdown(response)
        else:
            st.warning("Please enter a ticker symbol.")

# ---- SIDEBAR ----
with st.sidebar:
    st.title("About Finnie")
    st.markdown("""
    **Finnie** is an AI-powered personal finance education assistant.
    
    **Features:**
    - 💬 Chat with specialized AI agents
    - 📊 Portfolio analysis & visualization
    - 📈 Real-time market data
    - 🎯 Financial goal planning
    
    ⚠️ *This is for educational purposes only, not financial advice.*
    """)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()